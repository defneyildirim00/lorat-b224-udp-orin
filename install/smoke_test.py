"""Duman testi: model gerçekten kuruluyor ve bir kare işliyor mu.

setup_model.sh sonunda çalışır. Ağı, UDP'yi, ffmpeg'i hiç ilgilendirmez — sadece
"ağırlıklar yerinde mi, model kuruluyor mu, ileri geçiş çalışıyor mu".
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("REPO_ROOT", os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

KEY = os.environ.get("TRACKER_KEY", "lorat_b224")


def main():
    import tracker_api
    print("[smoke] %s kuruluyor ..." % KEY)
    t0 = time.time()
    trk, cfg = tracker_api.build(KEY, None, "cuda", True)
    print("[smoke] kuruldu: %s (%.1f sn)" % (cfg, time.time() - t0))
    if "NO WEIGHTS" in cfg:
        raise SystemExit("[smoke] ağırlıklar bulunamadı — weights/ dizinine bak")

    # 640x360'lık düz bir kare: burada doğruluk değil, boru hattının çalışması
    # ölçülüyor. İlk track() CUDA çekirdeklerini seçtiği için yavaştır; asıl hız
    # ikinci çağrıdır, o yüzden ikisi de yazdırılıyor.
    img = np.zeros((360, 640, 3), np.uint8)
    img[150:230, 280:360] = 200
    trk.init(img, [280, 150, 80, 80])
    t = time.time(); trk.track(img); warm = time.time() - t
    t = time.time(); box, score = trk.track(img); step = time.time() - t
    print("[smoke] ilk kare %.0f ms (CUDA ısınması), sonraki %.0f ms" % (warm * 1000, step * 1000))
    print("[smoke] kutu(xywh) = %s%s" % ([int(v) for v in box],
                                         "  skor %.2f" % score if score is not None else ""))
    print("[smoke] TAMAM")


if __name__ == "__main__":
    main()
