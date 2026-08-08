"""
Unified efficiency measurement for PRIN and open-source LLIE baselines
(paper Table 6).

Measures, per model (random weights -- checkpoints are NOT needed for
params/FLOPs/latency):
  - Params: full sum(p.numel()) AND thop-counted params (the paper's Table 6
    figure 24.412M for PRIN is the thop convention, which skips e.g.
    GroupNorm affine parameters)
  - FLOPs (G, thop MACs convention -- same convention as the numbers reported
    by Retinexformer/HVI-CIDNet et al.) at 256x256 and at 600x400 (LOL native
    resolution; models that need size multiples are reflect-padded inside the
    wrapper, so FLOPs at "600x400" are counted on the padded tensor -- footnote
    this in the paper)
  - GPU latency (ms) / FPS at 600x400 (padded the same way), warmup + averaged
    timed runs. Diffusion baselines (DiffLL) run their FULL default sampling
    loop inside forward, so latency reflects real inference.

Baseline configs follow each method's official LOL configuration (the exact
source config is cited in a comment inside each builder).

NOTE on baselines: the third-party builders require the official repos to be
cloned into experiments/efficiency/repos/ (third-party code is not
redistributed here), e.g. repos/Retinexformer, repos/HVI-CIDNet, ...
`python measure.py --model PRIN` works standalone without any of them.

Pacing: each invocation measures ONE model (a few seconds of GPU work) and
appends its row incrementally to the results CSV
(experiments/efficiency/results/efficiency.csv); run_all_gentle.py sleeps
between models. This keeps each burst of sustained GPU load short -- an
optional thermal safeguard for thermally limited machines (e.g. laptop GPUs).

Usage (from this folder; prepend experiments/efficiency/ when running from
the repo root):
  python measure.py --model PRIN
  python measure.py --model Retinexformer --overwrite
  python measure.py --list
"""

import argparse
import csv
import importlib.util
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))   # experiments/efficiency
REPO_ROOT = os.path.dirname(os.path.dirname(ROOT))  # repository root
REPOS = os.path.join(ROOT, "repos")
OUT_CSV = os.path.join(ROOT, "results", "efficiency.csv")

WARMUP = 3
TIMED = 15


def load_module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class PadWrapper(nn.Module):
    """Reflect-pads input to a size multiple, runs `call`, crops back.

    square=True additionally pads to a square (Uformer's window blocks
    recover H=W=sqrt(L), so it only accepts square inputs; its own test
    scripts pad to square the same way).
    """

    def __init__(self, net, multiple, call=None, square=False):
        super().__init__()
        self.net = net
        self.multiple = multiple
        self.square = square
        self.call = call or (lambda net, x: net(x))

    def forward(self, x):
        B, C, H, W = x.shape
        m = self.multiple
        th = -(-H // m) * m
        tw = -(-W // m) * m
        if self.square:
            th = tw = max(th, tw)
        if th - H or tw - W:
            x = F.pad(x, (0, tw - W, 0, th - H), mode="reflect")
        out = self.call(self.net, x)
        return out[:, :, :H, :W]


# ---------------------------------------------------------------- builders --
# Each builder returns an nn.Module whose forward(x) runs one full inference
# for x in [0,1] of any measured size. One builder call per subprocess, so
# same-named packages (basicsr/models/utils) across repos never collide.

def build_prin(device):
    sys.path.insert(0, REPO_ROOT)
    from models.prin import PRIN
    net = PRIN()  # pads internally to multiples of 32
    return PadWrapper(net, 1, lambda n, x: n(x)[2])


def build_retinexformer(device):
    arch = load_module_from_file(
        "retinexformer_arch",
        os.path.join(REPOS, "Retinexformer", "basicsr", "models", "archs",
                     "RetinexFormer_arch.py"))
    # LOL config (Options/RetinexFormer_LOL_v1.yml): n_feat=40, stage=1
    net = arch.RetinexFormer(in_channels=3, out_channels=3, n_feat=40,
                             stage=1, num_blocks=[1, 2, 2])
    return PadWrapper(net, 4)


def build_snr(device):
    sys.path.insert(0, os.path.join(REPOS, "SNR-Aware-Low-Light-Enhance"))
    from models.archs.low_light_transformer import low_light_transformer
    # LOL config (options/test/LOLv1.yml network_G)
    net = low_light_transformer(nf=64, nframes=5, groups=8, front_RBs=1,
                                back_RBs=1, center=None, predeblur=True,
                                HR_in=True, w_TSA=True)

    def call(n, x):
        # SNR map exactly as in models/Video_base_model3.py (blur via 5x5
        # box filter standing in for cv2.blur)
        dark = (x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114)
        light_rgb = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        light = (light_rgb[:, 0:1] * 0.299 + light_rgb[:, 1:2] * 0.587
                 + light_rgb[:, 2:3] * 0.114)
        noise = torch.abs(dark - light)
        mask = torch.div(light, noise + 0.0001)
        mask_max = mask.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1)
        mask = torch.clamp(mask / (mask_max + 0.0001), 0, 1).float()
        return n(x, mask)

    return PadWrapper(net, 16, call)  # two stride-2 convs + stride-4 unfold


def build_llformer(device):
    arch = load_module_from_file(
        "llformer_arch", os.path.join(REPOS, "LLFormer", "model", "LLFormer.py"))
    # config used by the repo's train.py/test.py (NOT the class defaults)
    net = arch.LLFormer(inp_channels=3, out_channels=3, dim=16,
                        num_blocks=[2, 4, 8, 16], num_refinement_blocks=2,
                        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                        bias=False, LayerNorm_type='WithBias',
                        attention=True, skip=False)
    return PadWrapper(net, 16)  # 4 downsampling levels


def build_mirnet(device):
    sys.path.insert(0, os.path.join(REPOS, "MIRNet"))
    from networks.MIRNet_model import MIRNet  # needs repo's utils.antialias
    net = MIRNet()  # defaults = config of test_enhancement.py (LOL)
    return PadWrapper(net, 4)


def build_uformer(device):
    # timm >= 0.9 moved timm.models.layers -> timm.layers
    try:
        import timm.models.layers  # noqa: F401
    except Exception:
        import timm.layers
        sys.modules["timm.models.layers"] = timm.layers
    arch = load_module_from_file(
        "uformer_model", os.path.join(REPOS, "Uformer", "model.py"))
    # Uformer-T (embed_dim=16), the config used in LLIE comparison tables
    net = arch.Uformer(img_size=256, embed_dim=16,
                       depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], win_size=8,
                       mlp_ratio=4., token_projection='linear',
                       token_mlp='leff', modulator=True, shift_flag=False)
    # 4 downsamples x win_size 8 = multiple of 128; square input required
    return PadWrapper(net, 128, square=True)


def build_restormer(device):
    arch = load_module_from_file(
        "restormer_arch",
        os.path.join(REPOS, "Restormer", "basicsr", "models", "archs",
                     "restormer_arch.py"))
    net = arch.Restormer()  # defaults = standard config in LLIE tables
    return PadWrapper(net, 8)


def build_diffll(device):
    import yaml
    repo = os.path.join(REPOS, "Diffusion-Low-Light")
    sys.path.insert(0, repo)
    from models.ddm import Net

    def ns(d):
        out = argparse.Namespace()
        for k, v in d.items():
            setattr(out, k, ns(v) if isinstance(v, dict) else v)
        return out

    with open(os.path.join(repo, "configs", "LOLv1.yml")) as f:
        config = ns(yaml.safe_load(f))
    config.device = device
    args = argparse.Namespace(sampling_timesteps=10)  # paper/repo default S=10
    net = Net(args, config)
    # eval-mode forward internally runs the FULL 10-step DDIM sampling on the
    # LL-LL subband and returns {'pred_x': enhanced}. Two DWT levels + 3 UNet
    # downsamplings on the LL-LL subband -> input must be a multiple of 32.
    return PadWrapper(net, 32, lambda n, x: n(x)["pred_x"])


def build_hvi_cidnet(device):
    sys.path.insert(0, os.path.join(REPOS, "HVI-CIDNet"))
    from net.CIDNet import CIDNet
    net = CIDNet()  # defaults, as instantiated by the repo's eval.py
    return PadWrapper(net, 8)


def build_retinexmamba(device):
    """Params/FLOPs only. The fused selective-scan CUDA kernel (mamba_ssm /
    causal_conv1d) has no Windows build, so the scan is replaced by a
    shape-faithful stub. thop never counts custom kernels or functional
    einsums anyway (same treatment as attention matmuls in the transformer
    baselines), so params and thop-FLOPs are identical to a Linux run --
    but LATENCY WITH THE STUB IS MEANINGLESS and is skipped (NO_LATENCY)."""
    import types
    archs = os.path.join(REPOS, "RetinexMamba", "basicsr", "models", "archs")
    pkg = types.ModuleType("rmarchs")
    pkg.__path__ = [archs]
    sys.modules["rmarchs"] = pkg

    ss2d = load_module_from_file("rmarchs.SS2D_arch",
                                 os.path.join(archs, "SS2D_arch.py"))

    def _scan_stub(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                   delta_softplus=False, return_last_state=False):
        return torch.zeros_like(u, dtype=torch.float32)

    ss2d.selective_scan_fn = _scan_stub
    load_module_from_file("rmarchs.IFA_arch", os.path.join(archs, "IFA_arch.py"))
    rm = load_module_from_file("rmarchs.RetinexMamba_arch",
                               os.path.join(archs, "RetinexMamba_arch.py"))
    # LOL config (Options/RetinexMamba_LOL_v1.yml): n_feat=40, stage=1
    net = rm.RetinexMamba(in_channels=3, out_channels=3, n_feat=40,
                          stage=1, num_blocks=[1, 2, 2])
    return PadWrapper(net, 4)


def build_quadprior(device):
    """QuadPrior (CVPR 2024): SD1.5 UNet + ControlNet + bypass VAE, official
    inference = DPM-Solver++ (10 steps, order 3) with CFG scale 9 (2 NFE per
    step), fp16, input resized so the short side is 512 (rounded to /64), as
    in the repo's test.py. Random init; cond stage is '__is_unconditional__'
    (precomputed empty_embedding.pkl), so nothing is downloaded."""
    import types
    import importlib.machinery
    repo = os.path.join(REPOS, "QuadPrior")
    cwd = os.getcwd()
    os.chdir(repo)  # ControlLDM.__init__ opens empty_embedding.pkl via cwd
    try:
        sys.path.insert(0, repo)
        # hack first: it imports transformers, which must probe deepspeed as absent
        from cldm.hack import disable_verbosity
        disable_verbosity()
        # then shim deepspeed (training-only top-level import in cldm/cldm.py)
        for name in ["deepspeed", "deepspeed.ops", "deepspeed.ops.adam"]:
            mod = types.ModuleType(name)
            mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            sys.modules[name] = mod

        class _DummyOpt:  # noqa: N801
            pass

        sys.modules["deepspeed.ops.adam"].FusedAdam = _DummyOpt
        sys.modules["deepspeed.ops.adam"].DeepSpeedCPUAdam = _DummyOpt

        from cldm.model import create_model
        model = create_model(os.path.join(repo, "models", "cldm_v15.yaml")).cpu()
        model.add_new_layers()
        from my_vae.autoencoder import AutoencoderKL as MyVAE
        model.first_stage_model = MyVAE(load_checkpoint=False)
    finally:
        os.chdir(cwd)

    if device.type == "cuda":
        model = model.to(torch.float16)  # official default (--use_float16)
    model.eval()

    from ldm.models.diffusion.dpm_solver import DPMSolverSampler

    class QuadPriorWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            m = self.m
            B, C, H, W = x.shape
            # official resize_image: short side -> 512, round both to /64
            k = 512.0 / min(H, W)
            H2 = int(round(H * k / 64.0)) * 64
            W2 = int(round(W * k / 64.0)) * 64
            dtype = next(m.parameters()).dtype
            control = F.interpolate(x, size=(H2, W2), mode="bilinear",
                                    align_corners=False).to(dtype)
            ae_hs = m.encode_first_stage(control * 2 - 1)[1]
            uc = m.get_unconditional_conditioning(B)
            cond = {"c_concat": [control], "c_crossattn": [uc]}
            un_cond = {"c_concat": [control], "c_crossattn": [uc]}
            m.control_scales = [1.0] * 13
            sampler = DPMSolverSampler(m)
            samples, _ = sampler.sample(10, B, (4, H2 // 8, W2 // 8), cond,
                                        verbose=False, eta=0.0,
                                        unconditional_guidance_scale=9.0,
                                        unconditional_conditioning=un_cond,
                                        dmp_order=3)
            out = m.decode_new_first_stage(samples.to(dtype), ae_hs)
            out = (out.float() + 1) / 2
            return F.interpolate(out, size=(H, W), mode="bilinear",
                                 align_corners=False)

    return QuadPriorWrapper(model)


BUILDERS = {
    "PRIN": build_prin,
    "MIRNet": build_mirnet,
    "SNR-Net": build_snr,
    "Uformer": build_uformer,
    "Restormer": build_restormer,
    "LLFormer": build_llformer,
    "Retinexformer": build_retinexformer,
    "DiffLL": build_diffll,
    "HVI-CIDNet": build_hvi_cidnet,
    "RetinexMamba": build_retinexmamba,
    "QuadPrior": build_quadprior,
}

# models whose forward uses a stubbed kernel: params/FLOPs are exact, but
# wall-clock timing would be meaningless -> latency columns left empty
NO_LATENCY = {"RetinexMamba"}

# (warmup, timed_runs, sleep_between_runs_s) overrides for heavy models, so a
# single measurement keeps each burst of sustained GPU load short (optional
# thermal safeguard for thermally limited machines)
TIMING_OVERRIDE = {"QuadPrior": (1, 5, 3.0)}


# ------------------------------------------------------------- measurement --

FIELDS = ["model", "params_M", "params_thop_M", "flops_G_256x256",
          "flops_G_600x400", "latency_ms_600x400", "fps_600x400", "device"]


def measure_one(name, device):
    wrapper = BUILDERS[name](device).to(device).eval()

    row = {"model": name,
           "params_M": round(sum(p.numel() for p in wrapper.parameters()) / 1e6, 3)}

    with torch.no_grad():
        thop_params = None
        for tag, (h, w) in {"256x256": (256, 256), "600x400": (400, 600)}.items():
            x = torch.rand(1, 3, h, w, device=device)
            try:
                # profile a throwaway deepcopy: thop leaves stale hooks/buffers
                # on the profiled module (breaks later plain forwards, e.g.
                # MIRNet's PReLU), so never profile the instance we time
                import copy
                from thop import profile
                target = copy.deepcopy(wrapper)
                flops, params = profile(target, inputs=(x,), verbose=False)
                row[f"flops_G_{tag}"] = round(flops / 1e9, 2)
                thop_params = params
                del target
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001 - record and continue
                row[f"flops_G_{tag}"] = ""
                print(f"[WARN] thop failed for {name}@{tag}: {e}")
            time.sleep(1.0)
        row["params_thop_M"] = round(thop_params / 1e6, 3) if thop_params else ""

        if name in NO_LATENCY:
            row["latency_ms_600x400"] = ""
            row["fps_600x400"] = ""
            row["device"] = (torch.cuda.get_device_name(0)
                             if device.type == "cuda" else "cpu")
            return row

        # latency at 600x400 (LOL native), warmup + timed
        warmup, timed, gap = TIMING_OVERRIDE.get(name, (WARMUP, TIMED, 0.0))
        x = torch.rand(1, 3, 400, 600, device=device)
        for _ in range(warmup):
            wrapper(x)
            if gap:
                time.sleep(gap)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_s = 0.0
        for _ in range(timed):
            t0 = time.perf_counter()
            wrapper(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_s += time.perf_counter() - t0
            if gap:
                time.sleep(gap)
        lat_ms = total_s / timed * 1000
        row["latency_ms_600x400"] = round(lat_ms, 2)
        row["fps_600x400"] = round(1000.0 / lat_ms, 2)

    row["device"] = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    return row


def append_row(row):
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    exists = os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def existing_models():
    if not os.path.exists(OUT_CSV):
        return set()
    with open(OUT_CSV, newline="") as f:
        return {r["model"] for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("Available:", ", ".join(BUILDERS))
        return

    if not args.model or args.model not in BUILDERS:
        ap.error(f"--model must be one of: {', '.join(BUILDERS)}")

    if not args.overwrite and args.model in existing_models():
        print(f"[SKIP] {args.model} already in {OUT_CSV}")
        return

    device = torch.device("cpu" if args.cpu else "cuda")
    row = measure_one(args.model, device)
    if args.overwrite and os.path.exists(OUT_CSV):
        with open(OUT_CSV, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["model"] != args.model]
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    append_row(row)
    print("[OK]", row)


if __name__ == "__main__":
    main()
