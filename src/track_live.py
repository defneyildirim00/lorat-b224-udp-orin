"""Live UDP tracker subscriber.

Reads the H.264/MPEG-TS stream published by publish_udp.sh, lets you draw a box
with the mouse, then tracks that object in real time with the selected tracker
(OSTrack / DropTrack / AQATrack, optionally their TensorRT engines, or LoRAT
B-224/B-378/L-224/L-378).

Launch via track_live.sh (picks the right venv). Direct use:
    python track_live.py --tracker ostrack_trt --url udp://127.0.0.1:1234

Controls:  drag a box any time to (re-)start tracking that object ·
           --write: type "x y w h" in the terminal instead ·
           r = drop the target · f = freeze/unfreeze the picture while aiming ·
           q/ESC = quit

Three threads, so the picture NEVER stops moving:

  grabber  — drains the UDP socket from the moment it opens and keeps only the
             latest frame. Draining must never pause (not even during a slow
             mouse selection) or the socket overruns and the decoder emits a
             flood of "corrupted macroblock" errors once reading resumes.
  tracker  — builds the tracker (slow: ~13 s of CPU plus ~600 MB off the SD
             card) and then runs it on whatever the freshest frame is, as fast
             as it can. It never touches the GUI.
  main     — owns the window: paints the newest frame plus the newest known box
             and handles the mouse/keys, at display rate.

Decoupling those is the whole point. Display speed is independent of tracker
speed, so lorat_l378 at 6 FPS still shows a smooth 30 fps stream (with a box
that updates at 6 Hz) instead of a 6 fps slideshow. Selection happens on the
LIVE picture rather than a frozen snapshot, and the stream is already flowing
while the tracker is still loading.
"""
import os
# must be set BEFORE cv2/ffmpeg import: hide the h264 decode-error spam that is
# normal when joining a live stream mid-GOP (keep fatal only).
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
# low-latency UDP: no input buffering, always keep the freshest packet
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "fflags;nobuffer|flags;low_delay|reorder_queue_size;0")

import sys
import time
import argparse
import threading
import subprocess

import numpy as np
import cv2


def wait_for_publisher(url, timeout=60.0, on_status=None):
    """Block until UDP packets actually arrive on the subscribe port.

    cv2.VideoCapture on a udp:// source with no traffic blocks deep inside
    ffmpeg's probe for a long time and ignores our retry loop, so we confirm the
    publisher is live with a cheap datagram socket first. The probe socket is
    closed before VideoCapture opens, so the two never compete for packets.
    Returns True if traffic was seen, False on timeout (caller still tries).
    """
    if not url.startswith("udp://"):
        return True
    import socket
    from urllib.parse import urlparse
    p = urlparse(url.split("?", 1)[0])
    host, port = p.hostname or "127.0.0.1", p.port
    if port is None:
        return True
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except (AttributeError, OSError):
        pass
    try:
        s.bind(("" if host in ("0.0.0.0", "") else host, port))
    except OSError:
        s.close()
        return True                       # port busy: let VideoCapture decide
    s.settimeout(0.5)
    t0 = time.time()
    try:
        while time.time() - t0 < timeout:
            try:
                s.recv(4096)
                return True
            except socket.timeout:
                if on_status:
                    on_status(f"no publisher on :{port} yet ({time.time() - t0:.0f}s)")
    finally:
        s.close()
    return False


def open_stream(url, retries=15, delay=1.0, on_status=None):
    # for a raw udp:// url, enlarge the receive fifo and survive overruns
    # (localhost bursts overrun the default circular buffer otherwise)
    # reuse=1 -> SO_REUSEADDR/REUSEPORT so a stale receiver socket (e.g. left by a
    #            previous run that was killed mid-stream) doesn't block binding.
    if url.startswith("udp://") and "overrun_nonfatal" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}reuse=1&overrun_nonfatal=1&fifo_size=5000000"
    # VideoCapture can fail to open a udp:// source if it probes before any
    # packet has arrived (publisher just started / mid-GOP). Retry a few times.
    for attempt in range(retries):
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            return cap
        cap.release()
        if attempt == 0:
            print(f"[sub] waiting for stream {url} ...")
        if on_status:
            on_status(f"waiting for publisher ({attempt + 1}/{retries})")
        time.sleep(delay)
    raise SystemExit(f"cannot open stream {url!r} — is the publisher running?")


def grabber(cap, state, max_consec_fail=400):
    """Continuously read; keep only the LATEST frame (drop the rest) so the
    tracker always sees the freshest frame — real live-stream semantics. Runs
    from socket-open through selection and tracking, so the UDP buffer never
    overruns. Transient decode failures (mid-GOP join / packet loss) return
    ok=False; only declare the stream dead after many in a row."""
    fails = 0
    while state["run"]:
        ok, frame = cap.read()
        if not ok or frame is None:
            fails += 1
            if fails >= max_consec_fail:
                state["alive"] = False
                break
            time.sleep(0.002)
            continue
        fails = 0
        with state["lock"]:
            state["frame"] = frame
            state["idx"] += 1
            # when this frame became available — the clock the end-to-end lag is
            # measured against (see tracker_thread)
            state["t"] = time.time()


def wait_for_frame(state, timeout=15.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with state["lock"]:
            if state["frame"] is not None:
                return state["frame"].copy()
        if not state["alive"]:
            break
        time.sleep(0.01)
    raise SystemExit("no frames received from stream — is the publisher running?")


def typed_box_thread(tstate):
    """Read init boxes typed as `x y w h` on stdin (--write), in parallel with
    everything else.

    A thread on purpose: the window, the socket and the model build all keep
    running while you type. Blocking any of them would freeze the picture and —
    worse — stall the grabber, which is the one thing that must never pause.

    A typed box differs from a drawn one in a way that matters. A drawn box is
    tied to the frame you were looking at, so it goes stale in seconds. A typed
    box comes from an annotation, not from a picture, so it carries no frame with
    it and no staleness rule: it is applied against the freshest frame at the
    moment it CAN be applied. That is what makes typing it before the publisher
    even starts work — it simply waits for frame 1."""
    print("[box] --write: type the init box as  x y w h  (or x,y,w,h), then Enter."
          "  Drawing with the mouse keeps working.")
    while tstate["run"]:
        try:
            line = sys.stdin.readline()
        except Exception:
            return
        if not line:                       # stdin closed (nohup, </dev/null, ...)
            return
        line = line.strip().strip("[]()").replace(",", " ")
        if not line:
            continue
        try:
            vals = [float(v) for v in line.split()]
        except ValueError:
            print("[box] not a box: %r — 4 numbers please:  x y w h" % line)
            continue
        if len(vals) != 4:
            print("[box] need exactly 4 numbers (x y w h), got %d" % len(vals))
            continue
        if vals[2] < 5 or vals[3] < 5:
            print("[box] rejected %gx%g — needs 5x5 px at least" % (vals[2], vals[3]))
            continue
        with tstate["lock"]:
            tstate["typed"] = tuple(vals)
        print("[box] queued (xywh) = %s — goes in as soon as the tracker and a "
              "frame are both ready" % [int(v) for v in vals])


# ---------------------------------------------------------------- startup UI --
# Opening the stream takes seconds (and the tracker build takes ~13 s more). If
# that happens on the main thread AFTER the window exists, Qt's event loop never
# spins (it only runs inside cv2.waitKey/imshow) and the window manager paints a
# dead grey window — the "opens frozen" symptom. So the slow work runs in a
# loader thread while the main thread paints a progress splash at ~30 Hz.

_GREY, _YELLOW, _GREEN, _RED = (150, 150, 150), (0, 220, 255), (90, 230, 90), (80, 80, 255)


def _splash(lines, w=880, h=320):
    img = np.full((h, w, 3), 30, np.uint8)
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "live tracker", (36, 62), f, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(img, (36, 84), (w - 36, 84), (70, 70, 70), 1)
    y = 132
    for text, col in lines:
        cv2.putText(img, text, (36, y), f, 0.62, col, 1, cv2.LINE_AA)
        y += 36
    cv2.putText(img, "q = cancel", (36, h - 24), f, 0.55, (110, 110, 110), 1, cv2.LINE_AA)
    return img




def _open_stream_ui(args, state, gui, win):
    """Phase 1: wait for the publisher, open the stream, start the grabber and
    get the first frame — in a loader thread, while the caller pumps the GUI so
    the window is alive and cancellable. Returns (cap, first_frame)."""
    st = {"stream": ("stream: connecting", _YELLOW),
          "frame": ("first frame: waiting", _GREY),
          "out": None, "err": None, "done": False}

    def load():
        try:
            note = lambda m: st.__setitem__("stream", ("stream: " + m, _YELLOW))
            st["stream"] = ("stream: waiting for publisher", _YELLOW)
            if not wait_for_publisher(args.url, timeout=args.wait, on_status=note):
                raise SystemExit(f"no UDP traffic on {args.url} after {args.wait:.0f}s "
                                 f"— is publish_udp.sh running?")
            note("opening " + args.url)
            cap = open_stream(args.url, on_status=note)
            st["stream"] = ("stream: open  " + args.url, _GREEN)
            print(f"[sub] stream open: {args.url}")

            # drain the socket from now on — never pause, or it overruns
            state["cap"] = cap
            th = threading.Thread(target=grabber, args=(cap, state), daemon=True)
            th.start()
            state["thread"] = th

            frame = wait_for_frame(state)
            st["frame"] = (f"first frame: {frame.shape[1]}x{frame.shape[0]}", _GREEN)
            st["out"] = (cap, frame)
        except BaseException as e:                  # incl. SystemExit from helpers
            st["err"] = e
        finally:
            st["done"] = True

    loader = threading.Thread(target=load, daemon=True)
    loader.start()

    t0 = time.time()
    while not st["done"]:
        if gui:
            cv2.imshow(win, _splash([st["stream"], st["frame"],
                                     (f"elapsed {time.time() - t0:4.1f}s", _GREY)]))
            if (cv2.waitKey(30) & 0xFF) in (27, ord("q")):
                state["run"] = False
                raise SystemExit("cancelled during startup")
        else:
            time.sleep(0.05)
    loader.join(timeout=1.0)

    if st["err"] is not None:
        if gui:                                     # leave the reason on screen briefly
            cv2.imshow(win, _splash([st["stream"], st["frame"],
                                     (f"failed: {st['err']}", _RED)]))
            cv2.waitKey(1500)
        raise st["err"]
    return st["out"]


# ------------------------------------------------------------- tracker thread --

def tracker_thread(args, state, tstate):
    """Build the tracker, then track the freshest frame forever. Never touches
    the GUI: it only publishes the latest box into tstate, which the display
    loop reads. Re-init requests arrive through tstate["init"]."""
    try:
        print(f"[sub] building tracker {args.tracker} "
              f"(first build loads/merges weights) ...")
        t0 = time.time()
        import tracker_api                          # heavy: torch/TensorRT import
        trk, cfg = tracker_api.build(args.tracker, args.config, args.device, not args.no_amp)
        # CUDA picks its kernels on the FIRST forward pass: that call measured
        # 2.3 s against ~100 ms for every one after it. Left unpaid, the cost
        # lands on the first frame after YOUR box — the tracker's opening move is
        # then a 2.3 s old picture, the target has walked out of the search
        # region, and it drops the target immediately. Paying it here on a dummy
        # box (live_benchmark.py does the same before timing anything) means the
        # box you draw starts tracking on the very next frame.
        try:
            warm = None
            for _ in range(60):
                with state["lock"]:
                    warm = state["frame"]
                if warm is not None:
                    break
                time.sleep(0.05)
            if warm is None:
                warm = np.zeros((720, 1280, 3), np.uint8)
            rgb = cv2.cvtColor(warm, cv2.COLOR_BGR2RGB)
            h, w = warm.shape[:2]
            tw = time.time()
            trk.init(rgb, [w / 2 - 40, h / 2 - 40, 80, 80])
            trk.track(rgb)
            trk.track(rgb)
            print(f"[sub] warm-up {time.time() - tw:.1f}s (dummy box; "
                  f"your first box now starts immediately)")
        except Exception as e:                       # never block startup on this
            print(f"[sub] warm-up skipped: {e}")

        with tstate["lock"]:
            tstate["trk"], tstate["cfg"] = trk, cfg
            tstate["build_s"] = time.time() - t0
        print(f"[sub] tracker ready: {cfg} ({time.time() - t0:.1f}s)")
    except BaseException as e:
        with tstate["lock"]:
            tstate["err"] = e
        return

    last_idx = -1
    while tstate["run"] and state["run"]:
        with tstate["lock"]:
            req, active = tstate.pop("init", None), tstate["active"]
        if req is not None:
            frame, bbox = req
            trk.init(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), bbox)
            with tstate["lock"]:
                tstate["box"], tstate["score"] = list(bbox), None
                tstate["active"], tstate["fps"], tstate["n"] = True, None, 0
                # the latency history belongs to one target; a re-init starts over
                # (and its first frame is another CUDA warm-up outlier)
                tstate["steps"], tstate["lags"] = [], []
            with tstate["lock"]:
                tstate["n_init"] += 1
                n_init = tstate["n_init"]
            print(f"[sub] INIT #{n_init} done, box (xywh) = {[int(v) for v in bbox]}"
                  f" — tracking from here")
            last_idx = -1
            continue
        if not active:
            time.sleep(0.005)
            continue

        with state["lock"]:
            idx, frame, t_grab = state["idx"], state["frame"], state["t"]
        if frame is None or idx == last_idx:
            if not state["alive"]:
                break
            time.sleep(0.001)
            continue
        last_idx = idx

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # acquisition, not timed
        t = time.time()
        box, score = trk.track(rgb)
        now = time.time()
        dt = now - t
        fps = 1.0 / dt if dt > 0 else 0.0
        # Two different latencies, and on a live feed they are NOT the same:
        #   step — how long trk.track() took on this frame. The model's own cost.
        #   lag  — how old the frame was when its box existed (now - grab time).
        #          It also contains the time the frame sat in the single slot
        #          while the tracker was still busy with an older one, so it is
        #          what an operator actually sees. step is a lower bound on it.
        step_ms, lag_ms = dt * 1000.0, (now - t_grab) * 1000.0
        with tstate["lock"]:
            tstate["box"], tstate["score"] = box, score
            tstate["fps"] = fps if tstate["fps"] is None else 0.9 * tstate["fps"] + 0.1 * fps
            tstate["n"] += 1
            tstate["steps"].append(step_ms)
            tstate["lags"].append(lag_ms)
            n, fps_ema = tstate["n"], tstate["fps"]
        if args.print_box and n % args.print_box == 0:
            x, y, bw, bh = (int(round(v)) for v in box)
            print(f"[box] #{n:<6d} xywh={x},{y},{bw},{bh}"
                  f"  center={x + bw // 2},{y + bh // 2}"
                  + (f"  conf={score:.2f}" if score is not None else "")
                  + f"  step={step_ms:.0f}ms  lag={lag_ms:.0f}ms", flush=True)
        elif args.headless and not args.print_box and n % 30 == 0:
            print(f"[sub] {n} frames  box(xywh)={[int(v) for v in box]}  "
                  f"step {step_ms:.0f} ms  lag {lag_ms:.0f} ms  {fps_ema:.1f} FPS"
                  + (f"  conf {score:.2f}" if score is not None else ""))


# ------------------------------------------------------------------ recorder --

class Recorder:
    """--record without slowing the picture down: encoding a 720p frame costs
    more than drawing one, and doing it inline halved the display rate. Frames go
    to an encoder thread through a short queue; if the encoder falls behind the
    frame is dropped rather than stalling the display."""

    def __init__(self, path, size, fps=30):
        import queue
        self._q = queue.Queue(maxsize=8)
        self._full = queue.Full
        self._w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        self.dropped = 0
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while True:
            f = self._q.get()
            if f is None:
                return
            self._w.write(f)

    def write(self, frame):        # frame is freshly built each iteration: no copy
        try:
            self._q.put_nowait(frame)
        except self._full:
            self.dropped += 1

    def release(self):
        self._q.put(None)
        self._t.join(timeout=10.0)
        self._w.release()
        if self.dropped:
            print(f"[sub] recorder dropped {self.dropped} frame(s) to keep the display smooth")


# ------------------------------------------------------ annotated UDP output --

class UdpOut:
    """Re-publish the annotated picture over UDP so another machine can watch.

    This is the return leg of the deployed shape: the tracking box usually has no
    monitor, so the only way to see what it is doing is to send the drawn frames
    somewhere. Encoding happens in a child ffmpeg fed through a pipe, on its own
    thread with a short queue — a frame is DROPPED rather than stalling the
    tracker, exactly like the inbound side drops packets.

    ffmpeg is spawned as an argv list, never through a shell: a shell parent makes
    terminate() kill the shell while ffmpeg keeps the socket, and the next run
    then fails to bind. And no `-nostdin` here — unlike every other ffmpeg in this
    repo, this one's input IS stdin.
    """

    def __init__(self, dest, size, fps=30.0):
        import queue
        w, h = size
        self.size = (w, h)
        fps = max(1.0, float(fps))
        self._q = queue.Queue(maxsize=4)
        self._full = queue.Full
        g = str(int(round(fps)))
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (w, h),
               "-r", "%.4f" % fps, "-i", "pipe:0", "-an",
               "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
               "-pix_fmt", "yuv420p", "-g", g,
               # inline SPS/PPS + frequent PAT/PMT so a viewer that starts late
               # (the normal case) recovers at the next keyframe instead of
               # showing nothing
               "-x264-params", "keyint=%s:min-keyint=%s:scenecut=0:repeat-headers=1" % (g, g),
               "-flush_packets", "1", "-mpegts_flags", "+resend_headers",
               "-f", "mpegts", "udp://%s?pkt_size=1316" % dest]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.dropped = 0
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        print("[sub] annotated stream -> udp://%s  (%dx%d @%.0f fps)" % (dest, w, h, fps))

    def _run(self):
        while True:
            f = self._q.get()
            if f is None:
                return
            try:
                self.proc.stdin.write(f.tobytes())
            except (BrokenPipeError, ValueError, OSError):
                return

    def write(self, frame):
        if (frame.shape[1], frame.shape[0]) != self.size:
            return
        try:
            self._q.put_nowait(frame)
        except self._full:
            self.dropped += 1

    def close(self):
        self._q.put(None)
        self._t.join(timeout=5.0)
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5.0)
        except Exception:
            self.proc.kill()
        if self.dropped:
            print("[sub] outbound stream dropped %d frame(s) to keep tracking smooth"
                  % self.dropped)


def out_thread(args, state, tstate, dest, size, fps):
    """Draw the current box on the freshest frame and push it out, at a steady
    rate. Runs with or without a window — the headless case is the real one, a
    Jetson on an aircraft has no display. It re-sends even when the frame has not
    changed, so a viewer that joins late still gets a picture."""
    pub = UdpOut(dest, size, fps)
    period = 1.0 / max(1.0, float(fps))
    nxt = time.time()
    try:
        while state["run"]:
            now = time.time()
            if now < nxt:
                time.sleep(min(0.005, nxt - now))
                continue
            nxt = max(now, nxt + period)
            with state["lock"]:
                frame = state["frame"]
            if frame is None:
                continue
            with tstate["lock"]:
                box, active, score = tstate["box"], tstate["active"], tstate["score"]
                steps = list(tstate["steps"][-31:])
                lags = list(tstate["lags"][-31:])
            img = frame.copy()
            head = args.tracker
            if active and box is not None:
                x, y, w, h = (int(round(v)) for v in box)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                ms = _window_median(steps)
                if ms > 0:
                    head += "   step %.0f ms   lag %.0f ms" % (ms, _window_median(lags))
                if score is not None:
                    head += "   conf %.2f" % score
            else:
                head += "   no target"
            _label(img, head, 30, 0.7)
            pub.write(img)
    finally:
        pub.close()


# ------------------------------------------------------------------ latency --

def _window_median(vals, n=30):
    """Median of the last n samples, with the very first one dropped.

    The first tracked frame pays CUDA context setup and lazy kernel selection —
    seconds, against tens of milliseconds afterwards. An EMA seeded with it is
    still carrying it dozens of frames later, which is how an overlay ends up
    showing a latency and a frame rate that contradict each other. A median over
    a short window has neither problem."""
    v = (vals[1:] if len(vals) > 1 else vals)[-n:]
    return float(np.median(v)) if v else 0.0


def latency_summary(tstate):
    with tstate["lock"]:
        steps, lags = list(tstate["steps"]), list(tstate["lags"])
    if not steps:
        return
    warm = steps[0]
    for name, vals in (("step (model)", steps), ("lag  (end-to-end)", lags)):
        a = np.sort(np.asarray(vals[1:] if len(vals) > 1 else vals, dtype=np.float64))
        p = lambda q: float(a[min(len(a) - 1, int(round(q * (len(a) - 1))))])
        print(f"[sub] {name}: median {p(0.5):6.1f} ms   p95 {p(0.95):6.1f}   "
              f"min {a[0]:6.1f}   max {a[-1]:6.1f}")
    print(f"[sub] first frame {warm:.0f} ms (CUDA warm-up, excluded above)")


# -------------------------------------------------------------- display loop --

def _label(disp, text, y, scale=0.7, col=(0, 255, 255)):
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(disp, text, (10, y), f, scale, (0, 0, 0), int(scale * 5) + 1, cv2.LINE_AA)
    cv2.putText(disp, text, (10, y), f, scale, col, max(1, int(scale * 2)), cv2.LINE_AA)


def _handle_key(k, state, tstate, roi, frame):
    """Single place where keys act, so the same handling applies whether the key
    arrived during a render or during an idle wait. Returns "quit" to stop."""
    if k in (27, ord("q")):
        return "quit"
    if k == ord("r"):                  # drop the target / cancel a half-made box
        with tstate["lock"]:
            tstate["active"], tstate["box"] = False, None
        roi["p0"] = roi["p1"] = roi["click0"] = roi["down"] = None
        roi["snap"] = roi["pending"] = None
        roi["drag"] = False
        roi["dirty"] = True
    elif k == ord("f"):                # freeze/unfreeze the displayed picture
        state["frozen"] = not state["frozen"]
        state["snap"] = frame.copy() if (state["frozen"] and frame is not None) else None
        roi["dirty"] = True
    return None


def display_loop(args, state, tstate, win, writer):
    """Owns the window. Paints the freshest frame at display rate with the
    newest box on top — so the picture keeps moving no matter how slow the
    tracker is — and handles the mouse (drag = (re-)init) and keys."""
    # snap    — the frame the picture is FROZEN on while you drag (see below)
    # pending — a finished box waiting for the tracker to exist
    # msg     — a short-lived note on screen, so a rejected drag is never silent
    # snap    — the frame the picture is FROZEN on while you select
    # click0  — first corner of a TWO-CLICK selection (see on_mouse)
    # pending — a finished box waiting for the tracker to exist
    # msg     — a short-lived note on screen, so a rejected attempt is never silent
    roi = {"p0": None, "p1": None, "drag": False, "dirty": True, "down": None,
           "click0": None, "click0_t": 0.0, "snap": None, "pending": None,
           "msg": None, "n_sent": 0, "pending_t": 0.0}
    s = args.scale

    def note(text):
        roi["msg"] = (text, time.time())
        roi["dirty"] = True

    def displayed_frame():
        """The picture you are actually LOOKING AT.

        This is the whole ball game: the box must be handed to the tracker
        together with the frame it was drawn over. Taking state["frame"] here was
        wrong whenever the picture was frozen with 'f' — you aim carefully at the
        frozen image, the code inits on the live one, the box lands on empty road
        and the tracker drops it instantly. That reads exactly like "it did not
        accept my box"."""
        with state["lock"]:
            src = state["snap"] if (state["frozen"] and state["snap"] is not None) \
                else state["frame"]
            return None if src is None else src.copy()

    def freeze_now():
        roi["snap"] = displayed_frame()

    def accept(a, b):
        box = (min(a[0], b[0]), min(a[1], b[1]), abs(b[0] - a[0]), abs(b[1] - a[1]))
        src = roi["snap"] if roi["snap"] is not None else displayed_frame()
        if src is None:
            note("no picture yet")
        elif box[2] < 5 or box[3] < 5:
            note("box too thin (%dx%d) - needs 5x5 px at least" % (box[2], box[3]))
            print("[sel] REJECTED %dx%d box (too thin)" % (box[2], box[3]))
        else:
            # Queue it. It used to be DISCARDED when the tracker was not built yet
            # (~12 s of weights off the SD card) with nothing on screen saying so.
            roi["pending"] = (src, box)
            roi["pending_t"] = time.time()
            roi["n_sent"] += 1
            print("[sel] box #%d sent to tracker (xywh) = %s"
                  % (roi["n_sent"], [int(v) for v in box]))
        roi["p0"] = roi["p1"] = roi["click0"] = roi["down"] = None
        roi["snap"] = None
        roi["dirty"] = True

    def on_mouse(event, x, y, flags, _):
        # --scale shrinks the PICTURE, so undo it here: the tracker must get the
        # box in stream coordinates, not window ones.
        x, y = x / s, y / s
        if args.debug_mouse and event != cv2.EVENT_MOUSEMOVE:
            print("[mouse] event=%d at (%.0f,%.0f) flags=%d drag=%s click0=%s"
                  % (event, x, y, flags, roi["drag"], roi["click0"]))
        if event == cv2.EVENT_LBUTTONDOWN:
            roi["down"] = (x, y)
            roi["drag"] = True
            if roi["click0"] is None:          # start of a fresh selection
                roi["p0"] = (x, y)
                freeze_now()
            roi["p1"] = (x, y)
            roi["dirty"] = True
        elif event == cv2.EVENT_MOUSEMOVE:
            # follow the pointer while dragging AND between the two clicks
            if roi["drag"] or roi["click0"] is not None:
                roi["p1"] = (x, y)
                roi["dirty"] = True
        elif event == cv2.EVENT_LBUTTONUP:
            roi["drag"] = False
            down = roi["down"]
            roi["down"] = None
            if down is None:                   # an UP with no matching DOWN
                return
            # ANY real movement counts as a drag attempt. Requiring both axes
            # turned a thin drag into "corner 1 of a two-click selection", so the
            # NEXT attempt closed a nonsense box spanning the two tries.
            moved = max(abs(x - down[0]), abs(y - down[1])) >= 5
            if moved:
                # A drag is ALWAYS a fresh selection, anchored where the button
                # went down. Anchoring it at a leftover two-click corner made the
                # box span from a previous attempt to this one — a nonsense
                # rectangle that the tracker then dutifully failed to follow.
                accept(down, (x, y))
            elif roi["click0"] is None:
                # Not a drag but a click: take it as the FIRST corner of a
                # two-click selection. Dragging on a live stream is fiddly and
                # depends on the whole press-move-release sequence surviving; two
                # clicks need only two press events, so this is the reliable path
                # when the drag keeps "not registering".
                roi["click0"] = down
                roi["click0_t"] = time.time()
                roi["p0"], roi["p1"] = down, (x, y)
                note("corner 1 set - now click the opposite corner  (r = cancel)")
            else:
                accept(roi["click0"], (x, y))

    cv2.setMouseCallback(win, on_mouse)
    last_idx, t_open = -1, time.time()
    # Three different rates, kept apart on purpose:
    #   fps_arrive  EMA of 1/(gap between new stream frames). This is the STREAM's
    #               rate, not ours — it reads absurdly high when the publisher
    #               bursts (ffmpeg dumping a backlog at a -stream_loop restart),
    #               so it is shown live in the overlay but never reported as a
    #               display rate.
    #   n_render    actual cv2.imshow calls -> the true render rate over the run.
    #   n_stream    distinct frames taken from the grabber -> honest mean arrival
    #               rate over the run (burst-insensitive, unlike the EMA).
    fps_arrive, t_prev = None, time.time()
    n_render = n_stream = 0
    # grabber counter at the moment tracking actually started. Without it the
    # skipped-frame ratio counts the ~12 s of frames that arrived while the
    # tracker was still building, when there was no target at all — that read
    # "518 of 618 skipped (83.8%)" for a tracker that really drops about half.
    idx_track_start = None
    t_loop = time.time()

    while True:
        with state["lock"]:
            idx, frame = state["idx"], state["frame"]
        fresh = idx != last_idx and frame is not None

        with tstate["lock"]:
            trk, cfg, box, score = tstate["trk"], tstate["cfg"], tstate["box"], tstate["score"]
            tfps, active, err, n = tstate["fps"], tstate["active"], tstate["err"], tstate["n"]
            steps_w, lags_w = list(tstate["steps"][-31:]), list(tstate["lags"][-31:])
            n_init = tstate["n_init"]
        if err is not None:
            raise err

        # A stray click arms corner 1; without a timeout a click minutes later
        # would complete a nonsense box from it. Forget it after 10 s.
        if roi["click0"] is not None and time.time() - roi["click0_t"] > 10.0:
            roi["click0"] = roi["p0"] = roi["p1"] = None
            roi["snap"] = None
            note("selection timed out - start again")

        # A box drawn while the tracker was still loading fires here, the moment
        # it exists — instead of having been thrown away on release.
        if roi["pending"] is not None and trk is not None:
            src, b = roi["pending"]
            age = time.time() - roi["pending_t"]
            roi["pending"] = None
            # ...but only if it is still CURRENT. A box drawn during the ~12 s
            # build carries the frame it was drawn on; initialising on a frame
            # that old and then tracking the live one means the target left the
            # search region long ago — a guaranteed miss that looks exactly like
            # "it accepted my box and then did nothing". Better to say so.
            if age > 2.0:
                roi["n_sent"] -= 1
                note("tracker is ready - pick the box again (that one was %.0f s old)"
                     % age)
                print("[sel] dropped a %.0f s old box: the picture has moved on "
                      "since it was drawn" % age)
            else:
                with tstate["lock"]:
                    tstate["init"] = (src, b)
                note("tracking %dx%d box" % (int(b[2]), int(b[3])))

        # A TYPED box (--write). No staleness check here, unlike the drawn box
        # above: it was never tied to a displayed frame, so it goes in against
        # the freshest one the moment the tracker exists.
        typed = None
        if frame is not None and trk is not None:
            with tstate["lock"]:
                typed, tstate["typed"] = tstate["typed"], None
        if typed is not None:
            fh, fw = frame.shape[:2]
            if (typed[0] + typed[2] <= 0 or typed[1] + typed[3] <= 0
                    or typed[0] >= fw or typed[1] >= fh):
                note("typed box falls outside the %dx%d picture" % (fw, fh))
                print("[box] REJECTED %s — outside the %dx%d picture"
                      % ([int(v) for v in typed], fw, fh))
            else:
                with state["lock"]:
                    src = state["frame"].copy()
                with tstate["lock"]:
                    tstate["init"] = (src, typed)
                roi["n_sent"] += 1
                note("typed box %dx%d" % (int(typed[2]), int(typed[3])))
                print("[sel] box #%d sent to tracker, TYPED (xywh) = %s"
                      % (roi["n_sent"], [int(v) for v in typed]))

        # Nothing new to draw: idle on waitKey so the window stays responsive.
        # The key still falls through to the single handler at the bottom —
        # handling keys only in the render path silently swallowed an 'r'/'f'
        # pressed in an idle gap.
        if not fresh and not roi["dirty"]:
            if not state["alive"]:
                break
            k = cv2.waitKey(5) & 0xFF
            if _handle_key(k, state, tstate, roi, frame) == "quit":
                break
            continue

        if active and idx_track_start is None:
            idx_track_start = idx
        if fresh:
            now = time.time()
            f = 1.0 / (now - t_prev) if now > t_prev else 0.0
            fps_arrive = f if fps_arrive is None else 0.9 * fps_arrive + 0.1 * f
            t_prev = now
            last_idx = idx
            n_stream += 1
        roi["dirty"] = False

        # What to paint, in priority order: the frame the drag froze on, then an
        # 'f' freeze, then the live frame. The grabber keeps draining throughout —
        # freezing is a display decision only, the socket must never pause.
        base = roi["snap"] if roi["snap"] is not None else (
            state["snap"] if state["frozen"] else frame)
        if base is None:
            base = frame
        disp = base.copy()
        if s != 1.0:
            disp = cv2.resize(disp, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

        if active and box is not None:
            x, y, w, h = (int(round(v * s)) for v in box)
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if roi["p0"] and roi["p1"]:
            cv2.rectangle(disp, tuple(int(v * s) for v in roi["p0"]),
                          tuple(int(v * s) for v in roi["p1"]), (255, 200, 0), 2)
        if roi["click0"] is not None:      # mark the corner already committed
            cx, cy = (int(v * s) for v in roi["click0"])
            cv2.drawMarker(disp, (cx, cy), (255, 200, 0), cv2.MARKER_CROSS, 22, 2)

        if trk is None:
            _label(disp, f"loading {args.tracker} ...  {time.time() - t_open:4.1f}s "
                         f"(stream is live)", 30)
            _label(disp, "box saved - but pick it again once loading ends"
                   if roi["pending"] is not None else
                   ("you can pick your box already (drag, click 2 corners, or "
                    "type x y w h in the terminal)" if args.write else
                    "you can pick your box already (drag, or click 2 corners)"),
                   62, 0.6, _GREEN)
        else:
            head = f"{args.tracker}   stream {fps_arrive or 0:4.1f} FPS"
            if active:
                # median of a short window, and the FPS derived from the same
                # number so the two can never disagree on screen
                ms = _window_median(steps_w)
                if ms > 0:
                    head += f"   step {ms:4.0f} ms ({1000.0 / ms:4.1f} FPS)"
                    head += f"   lag {_window_median(lags_w):4.0f} ms"
                else:
                    head += f"   track {tfps or 0:4.1f} FPS"
                if score is not None:
                    head += f"   conf {score:.2f}"
            _label(disp, head, 30, 0.8)
            _label(disp, "drag a box OR click 2 corners = (re)track"
                         + ("   r = drop/cancel" if (active or roi["click0"]) else "")
                         + ("   f = unfreeze" if state["frozen"] else "   f = freeze")
                         + "   q = quit", disp.shape[0] - 12, 0.6)
            if roi["click0"] is not None:
                _label(disp, "CORNER 1 SET - click the opposite corner", 62, 0.8,
                       (255, 200, 0))
            elif roi["drag"]:
                _label(disp, "SELECTING - picture held", 62, 0.8, (255, 200, 0))
            elif not active:
                _label(disp, "no target - type x y w h in the terminal, drag a "
                             "box, or click its two corners" if args.write else
                             "no target - drag a box, or click its two corners",
                       62, 0.7, _YELLOW)
        if roi["n_sent"] or n_init:
            # If these two disagree, the box is being lost between the window and
            # the tracker. If they agree but nothing is tracked, the tracker took
            # the box and lost the target — a different problem with a different
            # fix, and this line is what tells the two apart.
            _label(disp, "sel: sent %d / init %d" % (roi["n_sent"], n_init),
                   disp.shape[0] - 78, 0.55,
                   _GREEN if roi["n_sent"] == n_init else _RED)
        if roi["msg"] is not None:
            text, t0 = roi["msg"]
            if time.time() - t0 < 2.5:
                _label(disp, text, disp.shape[0] - 44, 0.7, _YELLOW)
            else:
                roi["msg"] = None

        if writer is not None and fresh:
            writer.write(disp)
        cv2.imshow(win, disp)
        n_render += 1

        if args.max_frames and n >= args.max_frames:
            break

        k = cv2.waitKey(1) & 0xFF
        if _handle_key(k, state, tstate, roi, frame) == "quit":
            break
        if k == 255 and not state["alive"]:
            break

    el = max(1e-6, time.time() - t_loop)
    print(f"[sub] render {n_render / el:.1f} FPS ({n_render} frames drawn)   "
          f"stream {n_stream / el:.1f} FPS in   "
          f"tracker {tfps or 0:.1f} FPS ({n} tracked)   over {el:.1f}s")
    # What the latest-frame-wins policy actually cost, counted from the moment a
    # target existed. The denominator is the GRABBER's counter (every frame the
    # socket delivered), not n_stream (what the display loop happened to pick up)
    # — only the former is "frames that arrived".
    with state["lock"]:
        n_arrived = state["idx"] - (idx_track_start if idx_track_start is not None
                                    else state["idx"])
    if n_arrived > 0 and n:
        skipped = max(0, n_arrived - n)
        print(f"[sub] while tracking: {skipped} of {n_arrived} arriving frame(s) "
              f"never reached the tracker ({100.0 * skipped / n_arrived:.1f}%) — it "
              f"always takes the freshest frame")
    latency_summary(tstate)


def headless_loop(args, state, tstate):
    """No window: wait for the tracker, init from --bbox (or from a box typed
    under --write), let tracker_thread run."""
    while tstate["trk"] is None and tstate["err"] is None:
        time.sleep(0.05)
    if tstate["err"] is not None:
        raise tstate["err"]
    frame = wait_for_frame(state)
    if args.bbox:
        bbox = tuple(float(v) for v in args.bbox.split(","))
        assert len(bbox) == 4, "--bbox needs x,y,w,h"
        with tstate["lock"]:
            tstate["init"] = (frame, bbox)
    while state["alive"] and tstate["err"] is None:
        # a typed box (--write) may arrive at any time, first one or a re-init
        with tstate["lock"]:
            typed, tstate["typed"] = tstate["typed"], None
        if typed is not None:
            with state["lock"]:
                src = state["frame"]
            if src is not None:
                with tstate["lock"]:
                    tstate["init"] = (src.copy(), typed)
                print("[sel] TYPED box (xywh) = %s" % [int(v) for v in typed])
        with tstate["lock"]:
            n, tfps = tstate["n"], tstate["fps"]
        if args.max_frames and n >= args.max_frames:
            print(f"[sub] reached --max-frames {args.max_frames}; mean {tfps or 0:.1f} FPS")
            latency_summary(tstate)
            return
        time.sleep(0.02)
    if tstate["err"] is not None:
        raise tstate["err"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", required=True)
    ap.add_argument("--url", default="udp://127.0.0.1:1234")
    ap.add_argument("--config", default=None, help="override tracker config name")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true", help="disable fp16 autocast (LoRAT)")
    ap.add_argument("--record", default=None, help="optional path to save annotated mp4")
    ap.add_argument("--out", default=None,
                    help="re-publish the annotated picture over UDP for another "
                         "machine to watch: HOST:PORT (e.g. 192.168.1.50:1235). "
                         "Works headless too — that is the point.")
    ap.add_argument("--out-fps", type=float, default=30.0,
                    help="frame rate of the outgoing annotated stream (default 30)")
    ap.add_argument("--print-box", nargs="?", type=int, const=1, default=0,
                    metavar="EVERY",
                    help="print the tracked box to the terminal: bare --print-box "
                         "prints every frame, --print-box 10 every 10th")
    ap.add_argument("--debug-mouse", action="store_true",
                    help="print every mouse event and every accepted box — use this "
                         "when a selection does not seem to register")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="shrink the WINDOW (e.g. 0.5) when the stream is bigger "
                         "than your screen. Do this instead of dragging the window "
                         "corner: the box you draw is mapped back to stream "
                         "coordinates, a hand-resized window is not.")
    ap.add_argument("--bbox", default=None, help="preset init box x,y,w,h (skip the mouse)")
    ap.add_argument("--write", action="store_true",
                    help="also accept the init box TYPED in the terminal as "
                         "\"x y w h\" (or x,y,w,h) — for annotated coordinates. "
                         "The mouse keeps working; you can type before the "
                         "publisher starts, and type again later to re-init.")
    ap.add_argument("--headless", action="store_true",
                    help="no window (needs --bbox, or --write to type the box)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N tracked frames")
    ap.add_argument("--wait", type=float, default=60.0,
                    help="seconds to wait for the publisher before giving up")
    args = ap.parse_args()

    gui = not args.headless
    if not gui and not (args.bbox or args.write):
        raise SystemExit("--headless requires --bbox x,y,w,h (or --write to type it)")
    if args.out:
        print("[sub] watch it with:  ffplay -fflags nobuffer -flags low_delay "
              "-i udp://%s   (or VLC: udp://@%s)"
              % (args.out.split("//", 1)[-1], args.out.split("//", 1)[-1]))
    win = "live tracker"
    if gui:
        # create + paint the window FIRST so it is alive and responsive from the
        # very first moment; every slow step runs off the main thread.
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 880, 320)
        cv2.imshow(win, _splash([("starting ...", _YELLOW)]))
        cv2.waitKey(1)

    state = {"frame": None, "idx": 0, "run": True, "alive": True, "lock": threading.Lock(),
             "cap": None, "thread": None, "frozen": False, "snap": None, "t": time.time()}
    tstate = {"trk": None, "cfg": None, "box": None, "score": None, "fps": None,
              "n": 0, "active": False, "err": None, "run": True,
              "build_s": None, "steps": [], "lags": [], "n_init": 0,
              "typed": None, "lock": threading.Lock()}

    # Started BEFORE the stream opens on purpose: --write's whole point is being
    # able to type the box while the publisher is not even running yet.
    if args.write:
        threading.Thread(target=typed_box_thread, args=(tstate,), daemon=True).start()

    cap, first = _open_stream_ui(args, state, gui, win)

    # The tracker builds in the background — the stream is already on screen.
    tthread = threading.Thread(target=tracker_thread, args=(args, state, tstate), daemon=True)
    tthread.start()

    writer = None
    if args.record:
        h, w = first.shape[:2]
        writer = Recorder(args.record, (w, h))

    othread = None
    if args.out:
        dest = args.out.split("//", 1)[-1].split("?", 1)[0]   # accept udp://h:p too
        h, w = first.shape[:2]
        othread = threading.Thread(target=out_thread,
                                   args=(args, state, tstate, dest, (w, h), args.out_fps),
                                   daemon=True)
        othread.start()

    try:
        if gui:
            h0, w0 = first.shape[:2]
            cv2.resizeWindow(win, int(w0 * args.scale), int(h0 * args.scale))
            if args.bbox:                   # optional: skip the mouse
                bbox = tuple(float(v) for v in args.bbox.split(","))
                while tstate["trk"] is None and tstate["err"] is None:
                    time.sleep(0.05)
                with tstate["lock"]:
                    tstate["init"] = (wait_for_frame(state), bbox)
            display_loop(args, state, tstate, win, writer)
        else:
            headless_loop(args, state, tstate)
    finally:
        # ordered teardown: stop + join the worker threads BEFORE releasing cap,
        # otherwise a daemon thread reads a freed VideoCapture -> segfault at exit.
        tstate["run"] = state["run"] = False
        tthread.join(timeout=5.0)
        if othread is not None:
            othread.join(timeout=8.0)      # lets its ffmpeg flush and exit
        if state["thread"] is not None:
            state["thread"].join(timeout=3.0)
        if writer is not None:
            writer.release()
        cap.release()
        if gui:
            cv2.destroyAllWindows()
    print("[sub] done")


if __name__ == "__main__":
    main()
