"""Make pip-installed CUDA 12.8 libraries visible to spconv/cumm.

torch dlopens its bundled NVRTC by path, but cumm's extension links
libnvrtc-builtins.so.12.8 by soname, which the loader can't find in
site-packages/nvidia/*. Loading them RTLD_GLOBAL at interpreter start lets the
soname resolve. Installed into the venv as a .pth hook by install.sh.
"""

import ctypes
import glob
import os


def _preload() -> None:
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvidia")
    for pattern in ("cuda_runtime/lib/libcudart.so.12",
                    "cuda_nvrtc/lib/libnvrtc.so.12",
                    "cuda_nvrtc/lib/libnvrtc-builtins.so.12*"):
        for path in sorted(glob.glob(os.path.join(base, pattern))):
            if ".alt." in path:
                continue
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_preload()
