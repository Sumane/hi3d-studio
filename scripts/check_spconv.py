#!/usr/bin/env python3
"""Check spconv's sparse convolution against a plain-PyTorch reference on this GPU.

  .venv/bin/python scripts/check_spconv.py

Hi3D's decoder (the step that turns latents into the SDF the mesh is cut from)
is built from these convolutions. Hi3D validated them on spconv-cu120 2.3.6;
this install uses a cu128 build on Blackwell. If the two paths disagree, every
mesh comes out noisy, terraced, and layered.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from app import config  # noqa: E402

os.environ.update({k: v for k, v in config.gpu_env(config.load()).items() if k.startswith("CUDA")})


def main() -> int:
    import spconv.pytorch as spconv
    from sparc3d.ops import sparse_backend as sb

    dev = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
    torch.manual_seed(0)
    res = 128
    # A thin shell of voxels, like Hi3D's sparse surface band.
    g = torch.stack(torch.meshgrid(*[torch.arange(res)] * 3, indexing="ij"), -1).reshape(-1, 3).float()
    r = (g - res / 2).norm(dim=1)
    coords = g[(r > 40) & (r < 43)].int()
    idx = torch.cat([torch.zeros(len(coords), 1, dtype=torch.int32), coords], 1).to(dev)
    print("voxels:", len(coords))

    worst = 0.0
    for cin, cout in [(64, 64), (128, 64), (64, 16)]:
        Conv = sb._submanifold_conv_class()
        conv = Conv(cin, cout, 3, bias=True, indice_key=f"t{cin}{cout}").to(dev)
        for dtype in (torch.float32, torch.float16):
            conv = conv.to(dtype)
            feats = torch.randn(len(coords), cin, device=dev, dtype=dtype)
            x = spconv.SparseConvTensor(feats, idx, [res] * 3, 1)
            with torch.no_grad():
                sb._SPCONV_INT32_BYTE_LIMIT = 2**31 - 1
                a = conv(x).features.float()
                x2 = spconv.SparseConvTensor(feats, idx, [res] * 3, 1)
                sb._SPCONV_INT32_BYTE_LIMIT = 0          # force the pure-PyTorch path
                b = conv(x2).features.float()
                sb._SPCONV_INT32_BYTE_LIMIT = 2**31 - 1
            rel = ((a - b).norm() / b.norm()).item()
            worst = max(worst, rel)
            print(f"  {cin:>3}->{cout:<3} {str(dtype):14} relative error spconv vs torch: {rel:.2e}"
                  f"   max abs: {(a - b).abs().max().item():.3e}")

    print("\nflash-attn vs PyTorch attention (Hi3D's transformer and decoder use flash-attn):")
    import flash_attn
    import torch.nn.functional as F
    worst_fa = 0.0
    for H, D, lens in [(16, 64, [4096, 1500]), (12, 64, [1024] * 4), (16, 128, [3000])]:
        n = sum(lens)
        qkv = torch.randn(n, 3, H, D, device=dev, dtype=torch.float16)
        cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device=dev, dtype=torch.int32)
        out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv, cu, max(lens)).float()
        ref = []
        for s, e in zip(cu[:-1].tolist(), cu[1:].tolist()):
            q, k, v = (qkv[s:e, i].float().transpose(0, 1)[None] for i in range(3))
            ref.append(F.scaled_dot_product_attention(q, k, v)[0].transpose(0, 1))
        ref = torch.cat(ref)
        rel = ((out - ref).norm() / ref.norm()).item()
        worst_fa = max(worst_fa, rel)
        print(f"  heads {H:>2} dim {D:>3} seqs {lens}: relative error {rel:.2e}")
    print("PASS  flash-attn matches" if worst_fa < 5e-3 else f"FAIL  flash-attn disagrees (worst {worst_fa:.1e})")

    tol = 5e-3
    print()
    if worst < tol:
        print(f"PASS  spconv matches the reference (worst {worst:.1e}). The decoder is not the cause.")
        return 0
    print(f"FAIL  spconv disagrees with the reference (worst {worst:.1e}).")
    print("      Run jobs with the PyTorch convolution path: tick 'Safe decode' in Hi3D Studio,")
    print("      or set HI3D_TORCH_CONV=1 before launching it.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
