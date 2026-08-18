#!/usr/bin/env bash
# TERMİNAL 2 — TRACKER.
#
# UDP'den gelen kareleri okur, hedefi takip eder, kutuyu çizip sonucu tekrar
# UDP'ye basar (izleyici için) ve her kutuyu terminale yazar.
#
#   ./track.sh                          # pencere açılır, kutuyu fareyle çizersin
#   ./track.sh --write                  # kutuyu terminale "x y w h" diye de yazabilirsin
#   ./track.sh --engine                 # TensorRT FP16 motoru (önce ./setup_engine.sh)
#   VIEW=10.0.0.9:1235 ./track.sh       # kutulu görüntüyü BAŞKA makineye yolla
#   ./track.sh --headless --bbox x,y,w,h   # ekransız (uçaktaki durum)
#
# Adresler:
#   dinlenen  : udp://$BIND:$IN_PORT      (BIND=0.0.0.0 -> her arayüz)
#   gönderilen: $VIEW                     (VIEW=off -> hiç gönderme)
set -u
cd "$(dirname "$0")"
source ./env.sh
need_venv

ARGS=(--tracker "${TRACKER_KEY:-lorat_b224}" --url "udp://$BIND:$IN_PORT" --print-box)
[ "$VIEW" != "off" ] && ARGS+=(--out "$VIEW" --out-fps "$FPS")

say "tracker=${TRACKER_KEY:-lorat_b224}   dinleniyor=udp://$BIND:$IN_PORT   gönderiliyor=$VIEW"
say "model yükleniyor (~10-15 sn), bittiğinde 'no target' yazacak — kutuyu ondan sonra çiz"

exec "$PY" "$SRC/track_live.py" "${ARGS[@]}" "$@"
