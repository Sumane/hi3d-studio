"""Exercise every GPU kernel family Hi3D needs on the selected card.

Imports succeeding proves nothing on a new architecture: each check below runs a
real kernel so a missing sm_120 build shows up here instead of mid-generation.
"""

from __future__ import annotations

import sys
import traceback

results: list[tuple[str, bool, str]] = []


def check(name):
    def wrap(fn):
        try:
            detail = fn() or ""
            results.append((name, True, detail))
        except Exception as exc:  # noqa: BLE001 - report everything
            results.append((name, False, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
        return fn
    return wrap


@check("torch + CUDA")
def _torch():
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    cap = torch.cuda.get_device_capability()
    arch = f"sm_{cap[0]}{cap[1]}"
    archs = torch.cuda.get_arch_list()
    assert arch in archs or f"compute_{cap[0]}{cap[1]}" in archs, f"{arch} not in torch build {archs}"
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    (x @ x).sum().item()
    free, total = torch.cuda.mem_get_info()
    return f"{torch.__version__} on {torch.cuda.get_device_name()} ({arch}), {free/2**30:.1f}/{total/2**30:.1f} GiB free"


@check("triton kernel compile")
def _triton():
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def add(x, y, out, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        tl.store(out + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)

    a = torch.ones(4096, device="cuda"); b = torch.ones(4096, device="cuda"); o = torch.empty_like(a)
    add[(4,)](a, b, o, 4096, BLOCK=1024)
    assert float(o.sum()) == 8192.0
    return triton.__version__


@check("spconv submanifold conv (fp16 + fp32)")
def _spconv():
    import torch
    import spconv.pytorch as sp
    coords = torch.unique(torch.randint(0, 32, (2000, 3), device="cuda", dtype=torch.int32), dim=0)
    coords = torch.cat([torch.zeros(len(coords), 1, dtype=torch.int32, device="cuda"), coords], 1)
    for dtype in (torch.float32, torch.float16):
        feats = torch.randn(len(coords), 32, device="cuda", dtype=dtype)
        t = sp.SparseConvTensor(feats, coords, [32, 32, 32], 1)
        conv = sp.SubMConv3d(32, 32, 3, indice_key="k").cuda().to(dtype)
        out = conv(t)
        assert out.features.shape == (len(coords), 32)
    import spconv
    return spconv.__version__


@check("flash-attn varlen (self + cross)")
def _flash():
    import torch
    import flash_attn
    q = torch.randn(300, 3, 22, 96, device="cuda", dtype=torch.float16)
    cu = torch.tensor([0, 300], device="cuda", dtype=torch.int32)
    flash_attn.flash_attn_varlen_qkvpacked_func(q, cu, 300)
    qq = torch.randn(300, 22, 96, device="cuda", dtype=torch.float16)
    kv = torch.randn(2917, 2, 22, 96, device="cuda", dtype=torch.float16)
    ck = torch.tensor([0, 2917], device="cuda", dtype=torch.int32)
    flash_attn.flash_attn_varlen_kvpacked_func(qq, kv, cu, ck, 300, 2917)
    return flash_attn.__version__


@check("Hi3D package + GPU voxelizer")
def _hi3d():
    import torch
    import trimesh
    import sparc3d
    from sparc3d.ops import adaptive_vox
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    coords = adaptive_vox(mesh, resolution=64, target_num=80_000, device=torch.device("cuda"))
    assert coords.shape[1] == 3 and len(coords) > 100
    return f"sparc3d {getattr(sparc3d, '__version__', '?')}, {len(coords)} voxels"


width = max(len(n) for n, _, _ in results)
print()
for name, ok, detail in results:
    mark = "\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name.ljust(width)}  {detail}")
sys.exit(0 if all(ok for _, ok, _ in results) else 1)
