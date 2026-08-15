#!/usr/bin/env bash
# TERMİNAL 3 — İZLEYİCİ.
#
# Tracker'ın bastığı kutulu görüntüyü gösterir. Bu komut, görüntünün GİTTİĞİ
# makinede çalışır — yani track.sh'deki VIEW adresi hangi makineyse orada.
#
#   ./view.sh              # udp://0.0.0.0:1235
#   ./view.sh 1236         # başka port
#
# Repo'nun olmadığı bir makinede (laptop vb.) aynı işi VLC yapar:
#   vlc udp://@:1235
set -u
cd "$(dirname "$0")"
source ./env.sh
need_venv

PORT="${1:-${VIEW##*:}}"
[ $# -gt 0 ] && shift
say "izleniyor: udp://0.0.0.0:$PORT   (pencerede q = çık)"
exec "$PY" "$SRC/view_udp.py" --url "udp://0.0.0.0:$PORT" "$@"
