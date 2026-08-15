#!/usr/bin/env bash
# ADIM 2/2 — AKIŞ KURULUMU.
#
# Modele dokunmaz. ffmpeg'i doğrular, portları kontrol eder ve elinde video yoksa
# test için bir tane üretir.
#
#   ./setup_stream.sh                    # doğrula + test videosu üret
#   ./setup_stream.sh /yol/video.mp4     # kendi videonu kaydet
set -u
cd "$(dirname "$0")"
source ./env.sh

USER_VIDEO="${1:-}"

# ------------------------------------------------------------------ ffmpeg --
say "ffmpeg"
command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg yok:  sudo apt install ffmpeg"
ok "$(ffmpeg -version 2>/dev/null | head -1)"
ffmpeg -hide_banner -encoders 2>/dev/null | grep -q " libx264 " \
  || die "ffmpeg'de libx264 yok. Yayıncı H.264 üretemez; libx264'lü bir ffmpeg kur."
ok "libx264 encoder var"
ffmpeg -hide_banner -protocols 2>/dev/null | grep -qx "  udp" \
  || warn "ffmpeg protokol listesinde udp görünmüyor (yine de çalışabilir)"

# ------------------------------------------------------------------ portlar --
say "portlar"
for p in "$IN_PORT" "${VIEW##*:}"; do
  if ss -lun 2>/dev/null | grep -q ":$p "; then
    warn "UDP $p şu an kullanımda — eski bir ffmpeg kalmış olabilir:  pkill -x ffmpeg"
  else
    ok "UDP $p boş"
  fi
done

# Soket alım tamponu: 720p bir keyframe varsayılan tamponu taşırabilir ve
# alıcıda "Part of datagram lost" olarak görünür. Kalıcı yapmak sudo ister.
RMEM=$(sysctl -n net.core.rmem_max 2>/dev/null || echo 0)
if [ "$RMEM" -lt 2000000 ] 2>/dev/null; then
  warn "net.core.rmem_max = $RMEM (küçük). Büyük karelerde paket düşebilir. İstersen:"
  echo "        sudo sysctl -w net.core.rmem_max=8388608"
else
  ok "net.core.rmem_max = $RMEM"
fi

# ------------------------------------------------------------------- video --
mkdir -p "$VIDEO_DIR"
if [ -n "$USER_VIDEO" ]; then
  [ -f "$USER_VIDEO" ] || die "video bulunamadı: $USER_VIDEO"
  cp -n "$USER_VIDEO" "$VIDEO_DIR/" 2>/dev/null || true
  ok "video eklendi: $VIDEO_DIR/$(basename "$USER_VIDEO")"
fi

if ! ls "$VIDEO_DIR"/*.mp4 >/dev/null 2>&1; then
  say "elde video yok, test klibi üretiliyor (30 fps, 20 sn, hareketli hedef)"
  # Sentetik ama takip edilebilir: düz zemin üzerinde çapraz giden bir kare.
  # Gerçek bir kayıt yerine geçmez, hattı uçtan uca denemek içindir.
  ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i "color=c=0x3a5f3a:s=1280x720:r=30:d=20" \
    -f lavfi -i "color=c=0xffcc00:s=90x140:d=20" \
    -filter_complex "[0][1]overlay=x='120+(W-240)*t/20':y='200+120*sin(2*PI*t/7)'" \
    -c:v libx264 -preset medium -pix_fmt yuv420p "$VIDEO_DIR/test_target.mp4" \
    || die "test videosu üretilemedi"
  ok "üretildi: $VIDEO_DIR/test_target.mp4"
fi

echo
say "kullanılabilir videolar:"
for f in "$VIDEO_DIR"/*.mp4; do
  info=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,avg_frame_rate -of csv=p=0 "$f" 2>/dev/null)
  printf "   %-28s %s\n" "$(basename "$f")" "$info"
done

echo
ok "AKIŞ HAZIR"
echo "   3 terminal:"
echo "     ${B}1)${N} ./publish.sh $(basename "$(ls "$VIDEO_DIR"/*.mp4 | head -1)")"
echo "     ${B}2)${N} ./track.sh"
echo "     ${B}3)${N} ./view.sh"
