#!/usr/bin/env bash
# ADIM 1/2 — MODEL KURULUMU.
#
# Sadece modeli hazırlar: venv, python bağımlılıkları, üstkaynak kodu, ağırlıklar
# ve bir duman testi. Akışla (ffmpeg/UDP) ilgili hiçbir şeye dokunmaz — o
# ./setup_stream.sh içinde.
#
#   ./setup_model.sh              # her şeyi kur
#   ./setup_model.sh --check      # sadece doğrula, hiçbir şey indirme
#
# Jetson AGX Orin / JetPack 5.1 (L4T R35.x) hedefli: torch ve TensorRT
# NVIDIA'nın sistem paketlerinden gelir, o yüzden venv --system-site-packages ile
# kurulur. Python 3.8 zorunlu: JetPack 5.1'in TensorRT bağlamaları sadece 3.8.
set -u
cd "$(dirname "$0")"
source ./env.sh

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

# --------------------------------------------------------------- ön koşullar --
say "ön koşullar"
command -v python3.8 >/dev/null 2>&1 || die "python3.8 yok (JetPack 5.1 ile gelir)"
ok "python3.8: $(python3.8 -V 2>&1)"
python3.8 -c "import torch" 2>/dev/null \
  || die "sistem torch'u yok. JetPack'in torch paketini kur (NVIDIA wheel), sonra tekrar dene."
python3.8 - <<'EOF' || die "torch CUDA görmüyor"
import torch, sys
print("  torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
sys.exit(0 if torch.cuda.is_available() else 1)
EOF
ok "torch + CUDA"

# ---------------------------------------------------------------------- venv --
if [ "$CHECK_ONLY" = 0 ]; then
  if [ ! -x "$PY" ]; then
    say "venv oluşturuluyor ($VENV)"
    # --system-site-packages: torch/torchvision/tensorrt NVIDIA'dan sistem
    # paketleri olarak gelir, venv içine tekrar kurulmaz (ve kurulamaz).
    python3.8 -m venv --system-site-packages "$VENV" || die "venv oluşturulamadı"
  fi
  ok "venv: $($PY -V 2>&1)"

  say "python bağımlılıkları"
  "$VENV/bin/pip" install --upgrade pip >/dev/null 2>&1 || true
  "$VENV/bin/pip" install -r requirements.txt || die "pip install başarısız"
  ok "bağımlılıklar kuruldu"
fi
need_venv

# ------------------------------------------------------------- üstkaynak kod --
mkdir -p "$THIRD_PARTY"
clone_pinned() {   # clone_pinned <url> <dizin> <commit>
  local url=$1 dir=$2 sha=$3
  if [ -d "$dir/.git" ]; then
    ok "$(basename "$dir") zaten var"
    return
  fi
  [ "$CHECK_ONLY" = 1 ] && die "$(basename "$dir") eksik ( --check modunda indirmiyorum )"
  say "$(basename "$dir") indiriliyor"
  git clone --filter=blob:none "$url" "$dir" || die "git clone başarısız: $url"
  # Sabit commit: üstkaynak değişse de bu repo çalışmaya devam etsin.
  ( cd "$dir" && git checkout --quiet "$sha" ) || warn "commit $sha bulunamadı, HEAD kullanılıyor"
  ok "$(basename "$dir") @ ${sha:0:12}"
}
clone_pinned https://github.com/LitingLin/LoRAT.git "$LORAT_ROOT" 5260744ff0a65207a73289bbda1788c377d23bcc


# ---------------------------------------------------------------- ağırlıklar --
mkdir -p "$WEIGHTS/lorat"
DINOV2="$WEIGHTS/dinov2"
HUB="${TORCH_HOME:-$HOME/.cache/torch}/hub/checkpoints"
mkdir -p "$HUB"

fetch() {   # fetch <url> <hedef>
  local url=$1 dst=$2
  [ -s "$dst" ] && { ok "$(basename "$dst") zaten var ($(du -h "$dst" | cut -f1))"; return; }
  [ "$CHECK_ONLY" = 1 ] && die "$(basename "$dst") eksik ( --check modunda indirmiyorum )"
  say "indiriliyor: $(basename "$dst")"
  curl -fL --retry 5 --retry-delay 5 -o "$dst.part" "$url" || { rm -f "$dst.part"; die "indirilemedi: $url"; }
  mv "$dst.part" "$dst"
  ok "$(basename "$dst") ($(du -h "$dst" | cut -f1))"
}

# DINOv2 sırt kemiği: LoRAT modeli kurulurken torch hub önbelleğinden okur.
fetch https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth \
      "$HUB/dinov2_vitb14_pretrain.pth"

# LoRAT'ın yayınlanmış ağırlıkları Google Drive klasöründe duruyor. gdown klasörü
# olduğu gibi indirir — bize sadece base.bin (52 MB) lazım, gerisi hemen silinir.
LORAT_DRIVE_FOLDER="https://drive.google.com/drive/folders/1FvViP0MCSiAu2FSrNjg7XEORn74yOBdD"
if [ ! -s "$WEIGHTS/lorat/base.bin" ]; then
  [ "$CHECK_ONLY" = 1 ] && die "weights/lorat/base.bin eksik"
  # Klasörde bugün on iki dosya var (GOT10k varyantları da eklenmiş): ~2 GB iner,
  # 52 MB'ı kalır. Tekil dosya ID'si yayınlanmadığı için tek dosya indirmenin
  # güvenilir bir yolu yok.
  say "LoRAT ağırlıkları indiriliyor (Google Drive klasörü, ~2 GB iner, 52 MB kalır)"
  "$VENV/bin/gdown" --folder "$LORAT_DRIVE_FOLDER" -O "$WEIGHTS/lorat" --remaining-ok \
    || die "gdown başarısız. Elle indir: $LORAT_DRIVE_FOLDER -> $WEIGHTS/lorat/base.bin"
  # gdown klasörü bir alt dizine açabilir; base.bin'i yukarı taşı.
  if [ ! -s "$WEIGHTS/lorat/base.bin" ]; then
    found=$(find "$WEIGHTS/lorat" -name "base.bin" -o -name "LoRAT-B-224.bin" | head -1)
    [ -n "$found" ] && mv "$found" "$WEIGHTS/lorat/base.bin"
  fi
  [ -s "$WEIGHTS/lorat/base.bin" ] || die "base.bin bulunamadı; elle indir: $LORAT_DRIVE_FOLDER"
  # base.bin dışında NE VARSA sil. İsim isim silmek yetmiyordu: Drive klasörüne
  # sonradan giant varyantları ve bir GOT10k/ alt dizini eklenmiş, ikisi de
  # listede yoktu ve 1.5 GB diskte kalıyordu. -delete derinlik önceliklidir,
  # yani alt dizinler de temizlenir.
  find "$WEIGHTS/lorat" -mindepth 1 -not -name base.bin -delete 2>/dev/null || true
fi
# Beklenen boyut ~52 MB. Çok küçükse gdown büyük ihtimalle bir HTML onay
# sayfasını kaydetmiştir (Drive'ın virüs taraması uyarısı) — sessizce bozuk bir
# dosyayla devam etmektense burada durmak daha iyi.
sz=$(stat -c %s "$WEIGHTS/lorat/base.bin" 2>/dev/null || echo 0)
[ "$sz" -gt 40000000 ] || die "weights/lorat/base.bin şüpheli küçük ($sz bayt) — elle indir: $LORAT_DRIVE_FOLDER"
ok "LoRAT B-224: $(du -h "$WEIGHTS/lorat/base.bin" | cut -f1)"
# Uzantısı .bin ama içerik safetensors; ve dosya sadece lora+head+token_type
# ağırlıklarını taşır (sırt kemiği DINOv2'den gelir, pos_embed türetilir).


# ------------------------------------------------------------- duman testi --
say "duman testi (model kurulup bir kare işleyecek)"
REPO_ROOT="$REPO_ROOT" "$PY" "$SRC/../install/smoke_test.py" || die "duman testi başarısız"

echo
ok "MODEL HAZIR"
echo "   sıradaki adım:  ${B}./setup_stream.sh${N}"
