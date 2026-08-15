#!/usr/bin/env bash
# TERMİNAL 1 — YAYINCI (kamera rolü).
#
# Videoyu gerçek zamanlı (-re) H.264/MPEG-TS olarak UDP'ye basar. Uçakta bu bacak
# kameranın kendisidir; burada bir mp4 ile taklit ediyoruz. Taşıma birebir aynı.
#
#   ./publish.sh                      # videos/ içindeki ilk mp4, 127.0.0.1:1234
#   ./publish.sh test_target.mp4      # videos/ içinden seç
#   ./publish.sh /yol/video.mp4       # herhangi bir dosya
#   ./publish.sh video.mp4 10.0.0.7:1234   # tracker BAŞKA makinedeyse onun IP'si
#
# Ortam: FPS=30  LOOP=1 (0 = bir kez oynat)
set -u
cd "$(dirname "$0")"
source ./env.sh

VIDEO="${1:-}"
DEST="${2:-127.0.0.1:$IN_PORT}"
LOOP="${LOOP:-1}"

if [ -z "$VIDEO" ]; then
  VIDEO=$(ls "$VIDEO_DIR"/*.mp4 2>/dev/null | head -1) \
    || die "videos/ boş. önce: ./setup_stream.sh"
  [ -n "$VIDEO" ] || die "videos/ boş. önce: ./setup_stream.sh"
elif [ ! -f "$VIDEO" ] && [ -f "$VIDEO_DIR/$VIDEO" ]; then
  VIDEO="$VIDEO_DIR/$VIDEO"
fi
[ -f "$VIDEO" ] || die "video yok: $VIDEO"

LOOPARG=""; [ "$LOOP" = "1" ] && LOOPARG="-stream_loop -1"
say "$(basename "$VIDEO")  ->  udp://$DEST   (${FPS} fps, loop=$LOOP)"
say "durdurmak için Ctrl-C"

# repeat-headers=1 : her IDR karesi SPS/PPS taşır
# +resend_headers  : PAT/PMT sık sık tekrarlanır
# keyint=$FPS      : saniyede bir keyframe
# Üçü birlikte, akışa SONRADAN katılan bir alıcının (normal durum) bir saniye
# içinde görüntü yakalamasını sağlar. Olmazsa alıcı "non-existing PPS" der ve
# hiç kare çözemez.
exec ffmpeg -hide_banner -loglevel warning -re $LOOPARG -i "$VIDEO" \
  -an -r "$FPS" -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p \
  -g "$FPS" -x264-params "keyint=$FPS:min-keyint=$FPS:scenecut=0:repeat-headers=1" \
  -flush_packets 1 -mpegts_flags +resend_headers \
  -f mpegts "udp://$DEST?pkt_size=1316"
