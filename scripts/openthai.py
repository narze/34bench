# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "huggingface-hub>=1.24.0,<2",
#   # OpenThai2.0-OCR-Text (runtime/line_engine.py)
#   "openvino>=2025.3",
#   "opencv-python-headless>=4.10",
#   "pyclipper>=1.3",
#   "numpy==2.2.6",
#   # OpenThai2.0-OCR-Markdown (the dependencies of its pyproject.toml)
#   "torch>=2.11.0,<2.14",
#   "torchvision>=0.26.0,<0.29",
#   "transformers>=5.14.1,<5.20",
#   "accelerate==1.12.0",
#   "timm==1.0.22",
#   "einops==0.8.1",
#   "Pillow>=12.2.0,<13",
#   "safetensors==0.8.0",
#   "beautifulsoup4==4.13.3",
#   "httpx==0.28.1",
# ]
# ///
"""Runner for the iApp OpenThai2.0 OCR models, which are open weights. They run on this computer.

- `text`: iapp/OpenThai2.0-OCR-Text, a detector and a line recognizer on the CPU (OpenVINO). It is
  the engine behind iApp Document OCR v3. Plain text output, with no prompt.
- `markdown`: iapp/OpenThai2.0-OCR-Markdown, a Thai fine-tune of NVIDIA Nemotron 3.5 Parse, on the
  GPU. It runs with the Transformers backend of the `DocumentParser` that ships with the model, in
  box mode (`mode="single"`), with the default generation settings. The model card says to use box
  mode.

Each model is downloaded at a fixed revision. The time is for one page, without the time to load
the model and one warm-up page that is not saved (the first page on the GPU is about 4 times slower). It saves the same files as src/run.ts.

    uv run scripts/openthai.py --model text --runs 5
    uv run scripts/openthai.py --model markdown --runs 5
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
IMAGE = ROOT / "data/route34/route34.png"
RUNS = 5

MODELS = {
    "text": {
        "repo": "iapp/OpenThai2.0-OCR-Text",
        "revision": "39fa41bdd49f067ba41b1d9de79c06ab506d67d4",
        "ignore": ["pytorch/*"],
    },
    "markdown": {
        "repo": "iapp/OpenThai2.0-OCR-Markdown",
        "revision": "7a153712e5b50fac7e4a9d42ccf9ec41fb4202f0",
        "ignore": ["reference/*", "dflash2/*", "assets/*"],
    },
}


def to_json(value):
    """Make numpy values in a result JSON serializable."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def cpu_name() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def load_text(model_dir: Path):
    sys.path.insert(0, str(model_dir))
    from runtime.line_engine import LineOCR

    ocr = LineOCR(str(model_dir), threads=1)
    image = IMAGE.read_bytes()

    def run():
        result = ocr.read(image)
        return result["text"], result, {"prompt": None, "finishReason": None, "usage": None}

    return run, f"Local CPU, 1 thread ({cpu_name()}), {ocr.backend}"


def load_markdown(model_dir: Path):
    import torch
    from transformers import AutoModel, AutoProcessor, GenerationConfig

    sys.path.insert(0, str(model_dir))
    from nemotron_parse.inference import PROMPT, DocumentParser, TransformersBackend

    if not torch.cuda.is_available():
        raise SystemExit("OpenThai2.0-OCR-Markdown needs a CUDA GPU.")
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.bfloat16)
    model = model.to("cuda:0").eval()
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
    parser = DocumentParser(backend=TransformersBackend(model, processor, config))

    def run():
        document = parser.parse(str(IMAGE), mode="single")
        fullpage = document["stages"]["fullpage"]
        return document["markdown"], document, {
            "prompt": PROMPT,
            "finishReason": fullpage.get("finish_reason"),
            "usage": fullpage.get("usage"),
        }

    return run, f"Local GPU ({torch.cuda.get_device_name(0)}), Transformers"


def main():
    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument("--model", choices=MODELS, required=True)
    args.add_argument("--runs", type=int, default=RUNS)
    args = args.parse_args()

    spec = MODELS[args.model]
    config_dir = ROOT / "results" / spec["repo"].replace("/", "~")
    todo = [n for n in range(1, args.runs + 1) if not (config_dir / str(n) / "response.md").exists()]
    if not todo:
        return

    model_dir = Path(snapshot_download(
        spec["repo"], revision=spec["revision"], ignore_patterns=spec["ignore"],
    ))
    run, provider = (load_text if args.model == "text" else load_markdown)(model_dir)
    run()  # Warm-up, not saved.

    for n in todo:
        tag = f"[{config_dir.name}] run {n}/{args.runs}"
        print(f"{tag}...", file=sys.stderr)
        date = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        started = time.perf_counter()
        content, raw, extra = run()
        duration_ms = round((time.perf_counter() - started) * 1000)

        meta = {
            "model": spec["repo"],
            "revision": spec["revision"],
            "effort": None,
            "requestedProvider": None,
            "image": "full",
            "prompt": extra["prompt"],
            "date": date,
            "durationMs": duration_ms,
            # Open weights on this computer: no price.
            "cost": None,
            "credits": None,
            "usage": extra["usage"],
            "provider": provider,
            "finishReason": extra["finishReason"],
        }

        run_dir = config_dir / str(n)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "response.md").write_text(content, encoding="utf-8")
        (run_dir / "raw.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2, default=to_json) + "\n", encoding="utf-8",
        )
        (run_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        print(f"{tag} done in {duration_ms / 1000:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
