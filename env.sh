#!/usr/bin/env bash
# Sourced by every other script. Resolves everything from where this repo was
# cloned, so nothing here depends on the machine it was built on.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT
export VENV="$REPO_ROOT/venv"
export PY="$VENV/bin/python"
export SRC="$REPO_ROOT/src"
export WEIGHTS="$REPO_ROOT/weights"
export THIRD_PARTY="$REPO_ROOT/third_party"
export LORAT_ROOT="$THIRD_PARTY/LoRAT"
export SAMURAI_ROOT="$THIRD_PARTY/SAMURAI"
export ENGINE_DIR="$WEIGHTS/lorat_trt"
export VIDEO_DIR="$REPO_ROOT/videos"

# Defaults for the pipeline. Override any of them in the environment:
#   IN_PORT=1234  BIND=0.0.0.0  VIEW=127.0.0.1:1235  FPS=30
export IN_PORT="${IN_PORT:-1234}"
export BIND="${BIND:-0.0.0.0}"
export VIEW="${VIEW:-127.0.0.1:1235}"
export FPS="${FPS:-30}"

# Python must find the modules in src/ whichever directory you call this from.
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"

if [ -t 1 ]; then
  C=$'\e[36m'; G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; D=$'\e[90m'; B=$'\e[1m'; N=$'\e[0m'
else
  C=""; G=""; Y=""; R=""; D=""; B=""; N=""
fi
say()  { echo "${C}[$(basename "$0" .sh)]${N} $*"; }
ok()   { echo "${G}  ok${N}   $*"; }
warn() { echo "${Y}  uyarı${N} $*" >&2; }
die()  { echo "${R}  HATA${N} $*" >&2; exit 1; }

need_venv() {
  [ -x "$PY" ] || die "venv yok. önce: ./setup_model.sh"
}
