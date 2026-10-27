#!/usr/bin/env bash
# Hi3D Studio installer — Blackwell (sm_120) + Ampere (sm_86) stack.
#
#   ./install.sh                 full install
#   ./install.sh --skip-weights  everything except the ~8 GB Hi3D download
#   ./install.sh --only-verify   re-run the GPU stack check
#
# Safe to re-run: every step checks whether it is already done.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HI3D_SRC_DEFAULT="$HOME/Downloads/hi3d-v1.0/hi3d-v1.0-main"
TRELLIS_STUDIO_DEFAULT="$(dirname "$ROOT")/trellis_AI_to_3d"

HI3D_SRC="${HI3D_SRC:-$HI3D_SRC_DEFAULT}"
TRELLIS_MODELS="${TRELLIS_MODELS:-$TRELLIS_STUDIO_DEFAULT/studio-data/models/trellis-cpp}"
ENGINE_REPO="${ENGINE_REPO:-https://github.com/raven38/pixal3d.cpp.git}"

TORCH_VER="2.8.0"; TV_VER="0.23.0"; TORCH_INDEX="https://download.pytorch.org/whl/cu128"
SPCONV_BASE="https://github.com/fafraob/spconv-wheels/releases/download/v2.3.8"
FA_VER="2.8.3"
FA_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FA_VER}/flash_attn-${FA_VER}+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"

SKIP_WEIGHTS=0; ONLY_VERIFY=0
for a in "$@"; do
  case "$a" in
    --skip-weights) SKIP_WEIGHTS=1 ;;
    --only-verify)  ONLY_VERIFY=1 ;;
    -h|--help) sed -n 2,9p "$0"; exit 0 ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/install-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

c_ok(){ printf '\e[32m✔ %s\e[0m\n' "$*"; }
c_warn(){ printf '\e[33m⚠ %s\e[0m\n' "$*"; }
c_err(){ printf '\e[31m✘ %s\e[0m\n' "$*"; }
step(){ printf '\n\e[1;36m== %s ==\e[0m\n' "$*"; }
die(){ c_err "$*"; echo "Full log: $LOG"; exit 1; }

# ---------------------------------------------------------------- preflight
step "Preflight"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found — install the NVIDIA driver first."
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv,noheader
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "$DRIVER_MAJOR" -ge 570 ] || die "Driver $DRIVER_MAJOR is too old for Blackwell + CUDA 12.8 (need ≥ 570)."

# Pick the Blackwell card (compute 12.x); fall back to the card with the most VRAM.
GPU_INDEX=$(nvidia-smi --query-gpu=index,compute_cap,memory.total --format=csv,noheader,nounits \
  | awk -F', ' '{cc=$2+0; if (cc>=12 && !b) {b=$1} if ($3>m) {m=$3; i=$1}} END {print (b!="")?b:i}')
GPU_INDEX="${HI3D_GPU:-$GPU_INDEX}"   # e.g. HI3D_GPU=0 bash install.sh --only-verify
GPU_NAME=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=name --format=csv,noheader)
c_ok "Using GPU $GPU_INDEX: $GPU_NAME"

missing=()
for t in git cmake ninja gcc g++ curl tar; do command -v "$t" >/dev/null || missing+=("$t"); done
if [ ${#missing[@]} -gt 0 ]; then
  c_err "Missing tools: ${missing[*]}"
  if command -v dnf >/dev/null; then echo "  sudo dnf install git cmake ninja-build gcc gcc-c++ curl tar"
  elif command -v apt >/dev/null; then echo "  sudo apt install git cmake ninja-build build-essential curl tar"
  elif command -v pacman >/dev/null; then echo "  sudo pacman -S git cmake ninja base-devel curl tar"; fi
  exit 1
fi

# CUDA toolkit (needed for building trellis.cpp and, if the wheel fails, flash-attn).
NVCC=""
for c in "${CUDA_HOME:-}/bin/nvcc" "$(command -v nvcc 2>/dev/null || true)" /usr/local/cuda/bin/nvcc /usr/local/cuda-1*/bin/nvcc /opt/cuda/bin/nvcc; do
  [ -n "$c" ] && [ -x "$c" ] && { NVCC="$c"; break; }
done
NVCC_OK=0
if [ -n "$NVCC" ]; then
  NVCC_VER=$("$NVCC" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
  export CUDA_HOME="$(dirname "$(dirname "$NVCC")")"; export PATH="$CUDA_HOME/bin:$PATH"
  if awk "BEGIN{exit !($NVCC_VER >= 12.8)}"; then NVCC_OK=1; c_ok "nvcc $NVCC_VER at $NVCC"
  else c_warn "nvcc $NVCC_VER is older than 12.8 — cannot compile sm_120 code; will use prebuilt binaries."; fi
else
  c_warn "No CUDA toolkit found — will use prebuilt binaries only. (Fedora: sudo dnf install cuda-toolkit-12-9 from NVIDIA's repo)"
fi

# nvcc refuses host compilers newer than it supports (CUDA 12.8/12.9 → gcc ≤ 14).
HOST_CXX=""
if [ "$NVCC_OK" = 1 ]; then
  GCC_MAJOR=$(gcc -dumpversion | cut -d. -f1)
  MAX_GCC=14; awk "BEGIN{exit !($NVCC_VER >= 13.0)}" && MAX_GCC=15
  if [ "$GCC_MAJOR" -gt "$MAX_GCC" ]; then
    for cand in g++-14 g++-13 cuda-g++ /usr/bin/g++-14 /usr/bin/g++-13; do
      command -v "$cand" >/dev/null && { HOST_CXX="$(command -v "$cand")"; break; }
    done
    if [ -n "$HOST_CXX" ]; then
      c_ok "gcc $GCC_MAJOR is too new for nvcc $NVCC_VER — using $HOST_CXX as CUDA host compiler"
      export NVCC_PREPEND_FLAGS="-ccbin $HOST_CXX"
    else
      c_warn "gcc $GCC_MAJOR is newer than nvcc $NVCC_VER supports and no gcc-14 found; CUDA source builds will fail."
      c_warn "Fedora: sudo dnf install gcc14 gcc14-c++   |  or install CUDA 13 which accepts gcc 15"
      NVCC_OK=0
    fi
  fi
fi

[ -d "$TRELLIS_MODELS" ] || die "TRELLIS GGUF folder not found: $TRELLIS_MODELS (set TRELLIS_MODELS=...)"
for f in dinov3 ss_flow ss_dec shape_flow_512 shape_flow_1024 shape_dec tex_flow_512 tex_flow_1024 tex_dec; do
  [ -f "$TRELLIS_MODELS/$f.gguf" ] || die "Missing $TRELLIS_MODELS/$f.gguf"
done
c_ok "TRELLIS.2 GGUFs found in $TRELLIS_MODELS (reused, not copied)"

# The New Volume drive is NTFS. Venvs and compiled extensions need exec + symlinks there;
# probe it rather than assume, and fall back to the home drive if it cannot.
FS_TYPE=$(stat -f -c %T "$ROOT" 2>/dev/null || echo unknown)
VENV="$ROOT/.venv"
probe="$ROOT/.exec-probe.sh"
printf '#!/bin/sh\nexit 0\n' > "$probe"; chmod +x "$probe" 2>/dev/null || true
if ! "$probe" 2>/dev/null || ! ln -sf "$probe" "$ROOT/.ln-probe" 2>/dev/null; then
  VENV="$HOME/.local/share/hi3d-studio/venv"
  c_warn "$ROOT ($FS_TYPE) does not allow exec/symlinks — putting the Python env at $VENV"
fi
rm -f "$probe" "$ROOT/.ln-probe"
c_ok "Filesystem: $FS_TYPE   Python env: $VENV"
PY="$VENV/bin/python"
export HF_HOME="$ROOT/hf-cache"
export UV_LINK_MODE=copy
# Keep every big download/build off the system drive: wheel cache, temp extraction,
# flash-attn/CUDA builds, Triton kernels and the rembg model all live on this volume.
mkdir -p "$ROOT/.cache/uv" "$ROOT/.cache/tmp" "$ROOT/.cache/pip" "$ROOT/.cache/triton" "$ROOT/.cache/u2net"
export UV_CACHE_DIR="$ROOT/.cache/uv"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export TMPDIR="$ROOT/.cache/tmp"
export TRITON_CACHE_DIR="$ROOT/.cache/triton"
export U2NET_HOME="$ROOT/.cache/u2net"
export UV_PYTHON_INSTALL_DIR="$ROOT/.cache/python"
FREE_GB=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
[ "$FREE_GB" -ge 40 ] || c_warn "Only ${FREE_GB} GB free on $ROOT — the full install needs ~35 GB."
c_ok "Caches and temp on $ROOT/.cache (${FREE_GB} GB free there)"

verify_stack() {
  step "Verifying GPU stack on GPU $GPU_INDEX"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PY" "$ROOT/scripts/verify_stack.py"
}

if [ "$ONLY_VERIFY" = 1 ]; then verify_stack; exit $?; fi

# ---------------------------------------------------------------- uv + venv
step "Python environment"
if ! command -v uv >/dev/null; then
  [ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
[ -x "$PY" ] || uv venv "$VENV" --python 3.10 --seed
"$PY" -c 'import sys; assert sys.version_info[:2]==(3,10), sys.version' || die "venv is not Python 3.10"
UVPIP=(uv pip install --python "$PY")
c_ok "Python $("$PY" -V)"

step "PyTorch $TORCH_VER + CUDA 12.8 (Blackwell-capable)"
if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('$TORCH_VER') and torch.version.cuda.startswith('12.8') else 1)" 2>/dev/null; then
  "${UVPIP[@]}" "torch==${TORCH_VER}+cu128" "torchvision==${TV_VER}+cu128" --extra-index-url "$TORCH_INDEX"
fi
"$PY" -c "import torch; print('torch', torch.__version__, 'arches', torch.cuda.get_arch_list())"

SITE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
cp "$ROOT/scripts/_hi3d_cuda_preload.py" "$ROOT/scripts/hi3d_cuda_preload.pth" "$SITE/"

step "spconv 2.3.8 with native sm_120 kernels"
if ! "$PY" -c "import torch, spconv; assert spconv.__version__=='2.3.8'" 2>/dev/null; then
  "${UVPIP[@]}" "$SPCONV_BASE/cumm_cu128-0.8.2-cp310-cp310-linux_x86_64.whl" \
                 "$SPCONV_BASE/spconv_cu128-2.3.8-cp310-cp310-linux_x86_64.whl"
fi

step "Hi3D source + Python dependencies"
mkdir -p "$ROOT/src"
if [ ! -f "$ROOT/src/hi3d/pyproject.toml" ]; then
  [ -f "$HI3D_SRC/pyproject.toml" ] || die "Hi3D source not found at $HI3D_SRC (set HI3D_SRC=...)"
  cp -r "$HI3D_SRC" "$ROOT/src/hi3d"
fi
# Hi3D pins spconv-cu120/flash-attn 2.7.4 (pre-Blackwell). Install it without those pins
# and supply its runtime deps explicitly at the versions its weights were verified with.
"${UVPIP[@]}" -e "$ROOT/src/hi3d" --no-deps
"${UVPIP[@]}" "transformers==4.46.3" "numpy==1.26.4" "safetensors>=0.4.3" "huggingface-hub>=0.28,<1" \
  "pillow>=10.3" "trimesh>=4.4,<5" "pyfqmr>=0.4" "accelerate>=0.31,<2" scipy \
  "fastapi>=0.115" "uvicorn[standard]>=0.30" "python-multipart>=0.0.9" requests rembg onnxruntime \
  packaging ninja psutil pymeshlab scikit-image

step "flash-attn $FA_VER"
fa_ok() { CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PY" - <<'EOF'
import torch, flash_attn
q = torch.randn(64, 3, 4, 64, device="cuda", dtype=torch.float16)
cu = torch.tensor([0, 64], device="cuda", dtype=torch.int32)
flash_attn.flash_attn_varlen_qkvpacked_func(q, cu, 64)
torch.cuda.synchronize(); print("flash-attn kernel OK on", torch.cuda.get_device_name())
EOF
}
if ! fa_ok 2>/dev/null; then
  echo "Trying prebuilt wheel…"
  "${UVPIP[@]}" "$FA_WHEEL" || true
  if ! fa_ok; then
    [ "$NVCC_OK" = 1 ] || die "Prebuilt flash-attn has no kernel for this GPU and no CUDA ≥12.8 toolkit is available to build one."
    c_warn "Building flash-attn from source for sm_80 (covers 3060) + sm_120 (5070 Ti). This takes 20–60 min."
    MEM_GB=$(awk '/MemTotal/ {print int($2/1048576)}' /proc/meminfo); JOBS=$(( MEM_GB / 6 )); [ $JOBS -lt 1 ] && JOBS=1; [ $JOBS -gt 8 ] && JOBS=8
    FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS="80;120" TORCH_CUDA_ARCH_LIST="8.6;12.0" MAX_JOBS=$JOBS \
      "${UVPIP[@]}" --no-build-isolation --reinstall "flash-attn==$FA_VER"
    fa_ok || die "flash-attn built but its kernel still fails — see log."
  fi
fi
c_ok "flash-attn ready"

# ---------------------------------------------------------------- coarse engine
step "Coarse engine: pixal3d.cpp (trellis.cpp fork with multi-image TRELLIS.2)"
ENG="$ROOT/engines/pixal3d.cpp"
SERVER_BIN=""
if [ "$NVCC_OK" = 1 ]; then
  if [ ! -d "$ENG/.git" ]; then git clone --recursive "$ENGINE_REPO" "$ENG"
  else git -C "$ENG" checkout -- . 2>/dev/null || true; git -C "$ENG" pull --ff-only || true; git -C "$ENG" submodule update --init --recursive; fi
  CMAKE_EXTRA=()
  [ -n "$HOST_CXX" ] && CMAKE_EXTRA+=("-DCMAKE_CUDA_HOST_COMPILER=$HOST_CXX")
  # GCC 15 no longer pulls <algorithm> in transitively; some fork sources use the
  # initializer-list std::max/std::min without including it. Patch them in place.
  grep -rlE 'std::(max|min|minmax)\(\{' "$ENG/src" "$ENG/include" 2>/dev/null | while read -r f; do
    grep -q '#include <algorithm>' "$f" || { sed -i '1i #include <algorithm>' "$f"; echo "  patched $(basename "$f")"; }
  done
  # Only the binaries the app uses (post-replay = TRELLIS UV + PBR bake for the refined mesh).
  if cmake -S "$ENG" -B "$ENG/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
       -DCMAKE_CUDA_ARCHITECTURES="86;120" "${CMAKE_EXTRA[@]}" \
     && cmake --build "$ENG/build" -j "$(nproc)" --target trellis-server trellis-cli post-replay; then
    SERVER_BIN=$(find "$ENG/build" -type f -name trellis-server -perm -u+x | head -1)
  else
    c_warn "Source build failed — falling back to the prebuilt release."
  fi
fi
if [ -z "$SERVER_BIN" ]; then
  # raven38 publishes no binary releases; upstream trellis.cpp does (single-image only).
  c_warn "Using upstream trellis.cpp prebuilt — multi-image coarse mode will be unavailable."
  curl -fsSL https://raw.githubusercontent.com/pwilkin/trellis.cpp/main/install/install.sh -o "$ROOT/engines/upstream-install.sh"
  bash "$ROOT/engines/upstream-install.sh" --backend cuda --gpu 0 --dest "$ROOT/engines/prebuilt" \
       --models-dir "$TRELLIS_MODELS" --skip-models --skip-app -y
  SERVER_BIN=$(find "$ROOT/engines/prebuilt" -type f -name trellis-server -perm -u+x | head -1)
fi
[ -n "$SERVER_BIN" ] || die "trellis-server binary not found after install."
c_ok "trellis-server: $SERVER_BIN"
"$SERVER_BIN" --help > "$ROOT/engines/trellis-server-help.txt" 2>&1 || true
CLI_BIN="$(dirname "$SERVER_BIN")/trellis-cli"
[ -x "$CLI_BIN" ] && { "$CLI_BIN" --help > "$ROOT/engines/trellis-cli-help.txt" 2>&1 || true; }
# Record the multi-image endpoint's handler so the backend can read its exact field names.
if [ -d "$ENG" ]; then
  grep -rn --include='*.cpp' --include='*.h' -A60 'generate-trellis2-mv' "$ENG" \
    --exclude-dir=build --exclude-dir=thirdparty > "$ROOT/engines/mv-endpoint.txt" 2>/dev/null || true
fi

# ---------------------------------------------------------------- weights
if [ "$SKIP_WEIGHTS" = 0 ]; then
  step "Hi3D v1.0 weights (~7.7 GB) + DINOv2/CLIP encoders"
  "$PY" - "$ROOT/models/Hi3Dv1.0" <<'EOF'
import sys
from huggingface_hub import snapshot_download
files = ["model_index.json", "scheduler/scheduler_config.json", "vae/config.json", "vae/model.safetensors",
         "transformer_512/config.json", "transformer_512/model.safetensors",
         "transformer_1024/config.json", "transformer_1024/model.safetensors"]
snapshot_download("hi3d-official/Hi3Dv1.0", local_dir=sys.argv[1], allow_patterns=files)
for repo in ("facebook/dinov2-large", "openai/clip-vit-large-patch14"):
    snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt"])
print("weights ready")
EOF
else
  c_warn "Skipped weights (--skip-weights)"
fi

# ---------------------------------------------------------------- config + launcher
step "Config and desktop launcher"
"$PY" - "$ROOT/config.json" "$GPU_INDEX" "$SERVER_BIN" "$TRELLIS_MODELS" "$ROOT/models/Hi3Dv1.0" "$HF_HOME" "$VENV" <<'EOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
cfg = json.loads(p.read_text()) if p.exists() else {}
cfg.update(gpu_index=int(sys.argv[2]), trellis_server=sys.argv[3], trellis_models=sys.argv[4],
           hi3d_model=sys.argv[5], hf_home=sys.argv[6], venv=sys.argv[7])
cfg.setdefault("port", 7860)
p.write_text(json.dumps(cfg, indent=2))
print(p.read_text())
EOF

step "Viewer libraries (three.js, vendored for offline use)"
if [ ! -f "$ROOT/app/static/vendor/three/build/three.module.js" ]; then
  tmp=$(mktemp -d)
  curl -fsSL https://registry.npmjs.org/three/-/three-0.170.0.tgz -o "$tmp/three.tgz"
  tar -xzf "$tmp/three.tgz" -C "$tmp"
  mkdir -p "$ROOT/app/static/vendor/three"
  cp -r "$tmp/package/build" "$tmp/package/examples" "$ROOT/app/static/vendor/three/"
  rm -rf "$tmp"
fi

chmod +x "$ROOT/hi3d-studio.sh"
mkdir -p "$HOME/.local/share/applications"
cat > "$HOME/.local/share/applications/hi3d-studio.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Hi3D Studio
Comment=Image to high-detail 3D (TRELLIS.2 + Hi3D)
Exec="$ROOT/hi3d-studio.sh"
Icon=$ROOT/app/static/icon.svg
Terminal=false
Categories=Graphics;3DGraphics;
EOF
cp "$HOME/.local/share/applications/hi3d-studio.desktop" "$HOME/Desktop/" 2>/dev/null && chmod +x "$HOME/Desktop/hi3d-studio.desktop" || true
c_ok "Launcher installed (app menu + Desktop)"

verify_stack || c_warn "Stack check reported problems — see above."
step "Done"
echo "Start it from your app menu (Hi3D Studio) or: \"$ROOT/hi3d-studio.sh\""
echo "Log: $LOG"
