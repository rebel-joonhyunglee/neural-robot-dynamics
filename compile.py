#!/usr/bin/env python3
"""
Compile a pretrained NeRD model for RBLN (rebel compile) or to a TorchScript
artifact. Example (RBLN compile without Warp):

  python3 compile.py \
    --cfg pretrained_models/NeRD_models/Franka/model/cfg.yaml \
    --checkpoint pretrained_models/NeRD_models/Franka/model/nn/model.pt \
    --device rbln:0 \
    --backend rbln \
    --mode manual \
    --output pretrained_models/NeRD_models/Franka/model/nn/model_compiled.rbln

Example (TorchScript export on CUDA):

python compile.py \
    --cfg pretrained_models/NeRD_models/Franka/model/cfg.yaml \
    --checkpoint pretrained_models/NeRD_models/Franka/model/nn/model.pt \
    --output pretrained_models/NeRD_models/Franka/model/nn/model_compiled.pt \
    --device cuda:0 \
    --backend torchscript
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.serialization import add_safe_globals

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from utils.python_utils import handle_cfg_overrides  # noqa: E402
from models.models import ModelMixedInput  # noqa: E402


class _EvalWrapper(torch.nn.Module):
    """Wrap ModelMixedInput.evaluate so Torch/JIT sees a single forward."""

    def __init__(self, model, input_names):
        super().__init__()
        self.model = model
        self.input_names = input_names

    def forward(self, *inputs):
        """
        Accept either a single dict or positional tensors matching input_names.
        This makes it compatible with torch.jit.trace and rebel.compile_from_torch.
        """
        if len(inputs) == 1 and isinstance(inputs[0], dict):
            input_dict = inputs[0]
        elif len(inputs) == len(self.input_names):
            input_dict = {name: tensor for name, tensor in zip(self.input_names, inputs)}
        else:
            raise TypeError(
                f"Expected 1 dict or {len(self.input_names)} tensors, got {len(inputs)}"
            )
        return self.model.evaluate(input_dict, deterministic=True)


def load_cfg(cfg_path: Path) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def build_env(cfg: dict, device: str, num_envs: int):
    try:
        from envs.neural_environment import NeuralEnvironment  # type: ignore
    except ModuleNotFoundError as exc:
        if exc.name == "warp":
            raise SystemExit(
                "warp-lang is required to construct the NeuralEnvironment. "
                "Install with: pip install warp-lang==1.8.0"
            ) from exc
        raise
    env_cfg = cfg.get("env", {}).copy()
    env_cfg["num_envs"] = num_envs
    env_cfg["render"] = False
    return NeuralEnvironment(device=device, **env_cfg)


def load_model(model_path: Path, device: str) -> torch.nn.Module:
    add_safe_globals([ModelMixedInput])
    # Load on CPU to avoid unknown storage tags (e.g., saved on CUDA/RBLN),
    # then move to target device when supported.
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = checkpoint[0]
    # Only move when device is a standard torch device.
    if not str(device).startswith("rbln"):
        model.to(device)
    model.eval()
    return model


def make_manual_inputs(model: ModelMixedInput, seq_len: int, device: str, state_dim: int | None = None):
    """
    Build dummy inputs without Warp by inferring dims from the loaded model.
    Assumes low_dim inputs are ['states_embedding', 'joint_acts'].
    """
    low_dim_size = model.encoders["low_dim"].out_features
    if state_dim is None:
        state_dim = model.model.output_net.out_features
    joint_act_dim = low_dim_size - state_dim
    assert joint_act_dim > 0, "Could not infer joint action dim from model."

    sample_inputs = {
        "states_embedding": torch.zeros((1, seq_len, state_dim), device=device, dtype=torch.float32),
        "joint_acts": torch.zeros((1, seq_len, joint_act_dim), device=device, dtype=torch.float32),
    }
    return sample_inputs, state_dim, joint_act_dim


def main():
    parser = argparse.ArgumentParser(description="Compile NeRD checkpoint for rbln.")
    parser.add_argument("--cfg", required=True, help="Path to the training cfg.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained model.pt")
    parser.add_argument(
        "--output",
        default="model_compiled.rbln",
        help="Output path. For backend=rbln this will be the .rbln file; for "
        "backend=torchscript this will be the traced module path.",
    )
    parser.add_argument("--device", default="rbln:0", help="Device to load/compile the model on")
    parser.add_argument(
        "--backend",
        choices=["torchscript", "rbln"],
        default="rbln",
        help="Compilation target. 'rbln' emits an RBLN artifact via rebel; "
        "'torchscript' saves a traced module.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of envs to instantiate for tracing input shapes (default: 1). "
        "Ignored when --mode manual.",
    )
    parser.add_argument(
        "--cfg-overrides",
        default="",
        help="Optional overrides string, same format as training (optional)",
    )
    parser.add_argument(
        "--torchscript-out",
        default=None,
        help="Optional TorchScript fallback path (only used when backend=rbln).",
    )
    parser.add_argument(
        "--mode",
        choices=["env", "manual"],
        default="manual",
        help="How to build sample inputs. 'env' uses Warp NeuralEnvironment; "
        "'manual' infers input dims from the checkpoint (no Warp required).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Sequence length (defaults to algorithm.sample_sequence_length in cfg or 1).",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=None,
        help="State embedding dim; if omitted, inferred from model output dim.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.cfg)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    assert cfg_path.exists(), f"cfg not found: {cfg_path}"
    assert checkpoint_path.exists(), f"checkpoint not found: {checkpoint_path}"

    cfg = load_cfg(cfg_path)
    if args.cfg_overrides:
        handle_cfg_overrides(args.cfg_overrides, cfg)

    # Keep everything on the requested device for tracing/compilation.
    device = args.device
    torch_device = "cpu" if str(device).startswith("rbln") else device
    torch.set_grad_enabled(False)

    seq_len = (
        args.seq_len
        if args.seq_len is not None
        else cfg.get("algorithm", {}).get("sample_sequence_length", 1)
    )

    neural_env = None
    if args.mode == "env":
        if str(device).startswith("rbln"):
            raise SystemExit("Warp env mode does not support rbln devices; use --mode manual.")
        neural_env = build_env(cfg, device=device, num_envs=args.num_envs)
    model = load_model(checkpoint_path, device=torch_device)

    # Build sample inputs either from the NeuralEnvironment (Warp) or inferred dims.
    if args.mode == "env":
        sample_inputs = neural_env.integrator_neural.get_neural_model_inputs()
        sample_inputs = {k: v.to(device) for k, v in sample_inputs.items()}
    else:
        sample_inputs, state_dim, joint_act_dim = make_manual_inputs(
            model, seq_len=seq_len, device=torch_device, state_dim=args.state_dim
        )
        print(
            f"[manual] seq_len={seq_len}, state_dim={state_dim}, joint_act_dim={joint_act_dim}, "
            f"low_dim_size={model.encoders['low_dim'].out_features}"
        )

    input_names = list(sample_inputs.keys())
    wrapper = _EvalWrapper(model, input_names)

    if args.backend == "rbln":
        try:
            import rebel  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "rebel (RBLN compiler/runtime) is required for RBLN compilation. "
                "Install with: pip install --extra-index-url https://pypi.rbln.ai/simple/ rebel-compiler"
            ) from exc

        # Build input specs for rebel.compile_from_torch
        input_specs = []
        for name, tensor in sample_inputs.items():
            input_specs.append((name, list(tensor.shape), tensor.dtype))

        compiled = rebel.compile_from_torch(wrapper, input_specs)
        compiled.save(output_path)
        print(f"[rbln] RBLN artifact saved to: {output_path}")

        if args.torchscript_out:
            scripted = torch.jit.trace(wrapper, tuple(sample_inputs.values()), strict=False)
            scripted.save(args.torchscript_out)
            print(f"[rbln] TorchScript fallback saved to: {args.torchscript_out}")
    else:
        scripted = torch.jit.trace(wrapper, (sample_inputs,), strict=False)
        scripted.save(output_path)
        print(f"[rbln] TorchScript model saved to: {output_path}")


if __name__ == "__main__":
    main()
