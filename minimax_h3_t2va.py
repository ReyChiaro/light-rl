"""MiniMax-H3 text-to-video-and-audio inference with Diffusers blocks.

Run from the project root: python -m pipelines.minimax_h3.t2va --help
"""

import argparse
import importlib.util
from pathlib import Path

import torch
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    MiniMaxH3ModularPipeline,
    align_num_frames,
)
from diffusers.utils.export_utils import encode_video


def create_pipeline(model: str) -> MiniMaxH3ModularPipeline:
    """Read the index and select t2va without loading any model weights."""
    root = Path(model).expanduser()
    if root.is_absolute() and not root.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {root}")
    if root.is_dir():
        root = root.resolve()
        model = str(root)
        if not any(
            (root / name).is_file()
            for name in ("modular_model_index.json", "model_index.json")
        ):
            raise FileNotFoundError(f"No Diffusers pipeline index in {root}")

    pipe = MiniMaxH3ModularPipeline.from_pretrained(model, workflow="t2va")

    for name in pipe.pretrained_component_names:
        spec = pipe.get_component_spec(name)
        if not spec.pretrained_model_name_or_path:
            raise ValueError(f"Pipeline index has no loading source for {name}")
    return pipe


def component_sources(pipe: MiniMaxH3ModularPipeline) -> dict:
    # Downloaded modular indexes can still refer to their original Hub repo.
    # Pass sources to load_components; update_components takes model instances.
    origin = pipe._pretrained_model_name_or_path
    root = Path(origin) if origin else None
    sources = {}
    for name in pipe.pretrained_component_names:
        spec = pipe.get_component_spec(name)
        source = spec.pretrained_model_name_or_path
        if root is not None and root.is_dir() and spec.subfolder:
            if (root / spec.subfolder).is_dir():
                source = str(root)
        sources[name] = source
    return sources


def load_models(pipe: MiniMaxH3ModularPipeline, device: str) -> None:
    """Load weights separately, retaining the checkpoint's mixed precision."""
    pipe.load_components(
        pretrained_model_name_or_path=component_sources(pipe),
        dtype={
            "text_encoder": torch.bfloat16,
            "transformer": torch.bfloat16,
            "vae": torch.float32,
            "audio_vae": torch.float32,
        },
    )
    # load_components logs individual failures and continues; fail before infer.
    missing = [name for name, value in pipe.components.items() if value is None]
    if missing:
        raise RuntimeError(
            f"Components failed to load: {', '.join(missing)}. "
            "See the Diffusers loading errors above."
        )
    for component in pipe.components.values():
        if isinstance(component, torch.nn.Module):
            component.eval()
    # Do not pass dtype here: that would cast the transformer's FP32 modules.
    pipe.to(device)


def generate(
    pipe: MiniMaxH3ModularPipeline,
    prompt: str,
    *,
    seed: int = 42,
    num_frames: int = 124,
    num_inference_steps: int = 50,
    height: int | None = None,
    width: int | None = None,
) -> dict:
    """Return one video and its jointly generated stereo soundtrack."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    return pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        output_type="pil",
        output=["videos", "audio", "sampling_rate"],
    )


def save_result(result: dict, output: Path, fps: int = 24) -> None:
    """Mux the first (and only) batch item into an MP4 with audio."""
    output.parent.mkdir(parents=True, exist_ok=True)
    encode_video(
        result["videos"][0],
        fps=fps,
        output_path=str(output),
        audio=result["audio"][0].detach().float().cpu(),
        audio_sample_rate=result["sampling_rate"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/root/models/MiniMax-H3")
    parser.add_argument("--prompt")
    parser.add_argument("--output", type=Path, default=Path("outputs/t2va.mp4"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print the execution blocks and loading sources, without weights.",
    )
    args = parser.parse_args()
    if not args.inspect:
        if not args.prompt or not args.prompt.strip():
            parser.error("--prompt is required for generation")
        if args.num_inference_steps < 2:
            parser.error(
                "--num-inference-steps must be >= 2 (sigma=0 included)"
            )
        if args.num_frames < 1:
            parser.error("--num-frames must be positive")
        aligned = align_num_frames(args.num_frames, 17, 5)
        if not 5 <= aligned / 24 <= 15:
            parser.error(f"Aligned frames ({aligned}) must span 5–15 seconds")
        if (args.height is None) != (args.width is None):
            parser.error("--height and --width must be passed together")
        if args.height is not None and any(
            size <= 0 or size % 32 for size in (args.height, args.width)
        ):
            parser.error("Height and width must be positive multiples of 32")
        if args.output.suffix.lower() != ".mp4":
            parser.error("--output must end in .mp4")
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            parser.error("CUDA is unavailable; run on the model's GPU host")
        if importlib.util.find_spec("av") is None:
            parser.error("MP4 export requires PyAV: uv pip install av")

    pipe = create_pipeline(args.model)
    print("Workflow: t2va", flush=True)
    print("Blocks: " + " -> ".join(pipe.blocks.sub_blocks), flush=True)
    sources = component_sources(pipe)
    for name in pipe.pretrained_component_names:
        spec = pipe.get_component_spec(name)
        print(
            f"  {name}: {sources[name]} (subfolder={spec.subfolder!r})",
            flush=True,
        )
    if args.inspect:
        return

    load_models(pipe, args.device)
    result = generate(
        pipe,
        args.prompt,
        seed=args.seed,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
    )
    save_result(result, args.output, fps=pipe.fps)
    print(f"Saved video + audio: {args.output.resolve()}")


if __name__ == "__main__":
    main()
