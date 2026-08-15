"""Terminal 3: watch the annotated stream the tracker publishes with --out.

Deliberately built on the same OpenCV/ffmpeg reader the tracker itself uses on
the inbound side, because that combination is known to work on this Jetson.
`ffplay` is present but its own build is unreliable here — with `-fflags
nobuffer` it never opens a window at all, and without it it logs
"non-existing PPS 0 referenced / no frame!" continuously on a stream that plain
`ffmpeg` decodes at 25 fps. This viewer sidesteps that entirely.

    python view_udp.py                 # udp://0.0.0.0:1235
    python view_udp.py --url udp://0.0.0.0:1236
    python view_udp.py --count 50      # no window: receive 50 frames and report

Latest-frame-wins, like the tracker: a reader thread drains the socket and keeps
only the newest frame, so a slow display can never make the socket overrun.
"""
import os
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")     # hide mid-GOP join spam
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "fflags;nobuffer|flags;low_delay|reorder_queue_size;0")

import sys
import time
import argparse
import threading

import numpy as np
import cv2


def open_stream(url, wait_s, on_status=None):
    if url.startswith("udp://") and "overrun_nonfatal" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}reuse=1&overrun_nonfatal=1&fifo_size=5000000"
    t0 = time.time()
    attempt = 0
    while time.time() - t0 < wait_s:
        attempt += 1
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            return cap
        cap.release()
        if on_status:
            on_status(attempt)
        time.sleep(1.0)
    raise SystemExit(f"[view] {url} açılamadı — yayıncı/tracker çalışıyor mu?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="udp://0.0.0.0:1235")
    ap.add_argument("--wait", type=float, default=60.0,
                    help="seconds to keep trying before giving up")
    ap.add_argument("--count", type=int, default=0,
                    help="no window: receive N frames, print the rate, exit")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    print(f"[view] izleniyor: {args.url}")
    cap = open_stream(args.url, args.wait,
                      on_status=lambda a: print(f"[view] yayın bekleniyor ({a}) ..."))

    state = {"frame": None, "n": 0, "run": True, "lock": threading.Lock()}

    def reader():
        fails = 0
        while state["run"]:
            ok, f = cap.read()
            if not ok or f is None:
                fails += 1
                if fails > 400:
                    state["run"] = False
                    break
                time.sleep(0.002)
                continue
            fails = 0
            with state["lock"]:
                state["frame"], state["n"] = f, state["n"] + 1

    th = threading.Thread(target=reader, daemon=True)
    th.start()

    t0 = time.time()
    if args.count:
        while state["run"] and state["n"] < args.count and time.time() - t0 < args.wait:
            time.sleep(0.02)
        el = max(1e-6, time.time() - t0)
        with state["lock"]:
            n, f = state["n"], state["frame"]
        print(f"[view] {n} kare alındı, {el:.1f}s içinde ({n / el:.1f} FPS)"
              + (f", {f.shape[1]}x{f.shape[0]}" if f is not None else ""))
        state["run"] = False
        th.join(timeout=2.0)
        cap.release()
        sys.exit(0 if n >= args.count else 1)

    win = args.title or f"tracker view - {args.url}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    splash = np.full((240, 720, 3), 30, np.uint8)
    cv2.putText(splash, "waiting for video ...", (30, 130), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.imshow(win, splash)
    cv2.waitKey(1)

    last, sized, shown = -1, False, 0
    fps, t_prev = None, time.time()
    while True:
        with state["lock"]:
            n, f = state["n"], state["frame"]
        if f is not None and n != last:
            if not sized:
                cv2.resizeWindow(win, f.shape[1], f.shape[0])
                sized = True
            now = time.time()
            inst = 1.0 / (now - t_prev) if now > t_prev else 0.0
            fps = inst if fps is None else 0.9 * fps + 0.1 * inst
            t_prev, last = now, n
            img = f.copy()
            cv2.putText(img, f"{fps:4.1f} FPS   {n} frames   q = quit",
                        (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, f"{fps:4.1f} FPS   {n} frames   q = quit",
                        (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (150, 150, 150), 1, cv2.LINE_AA)
            cv2.imshow(win, img)
            shown += 1
        if (cv2.waitKey(5) & 0xFF) in (27, ord("q")):
            break
        if not state["run"]:
            print("[view] yayın kesildi")
            break

    state["run"] = False
    th.join(timeout=2.0)
    cap.release()
    cv2.destroyAllWindows()
    el = max(1e-6, time.time() - t0)
    print(f"[view] {shown} kare gösterildi ({shown / el:.1f} FPS), {el:.1f}s")


if __name__ == "__main__":
    main()
