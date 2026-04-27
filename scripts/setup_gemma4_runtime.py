import os
import subprocess
import sys
import venv
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = APP_ROOT / "data" / "models" / "gemma-4-runtime"
MODEL_REPO_ID = "unsloth/gemma-4-E4B-it-GGUF"
MODEL_FILES = {
    "Q4_K_M": "gemma-4-E4B-it-Q4_K_M.gguf",
    "Q6_K_M": "gemma-4-E4B-it-Q6_K.gguf",
}
DEFAULT_QUANTIZATION = "Q4_K_M"
MODEL_DIR = APP_ROOT / "data" / "models" / "gemma-4-E4B-it-GGUF"


def runtime_python() -> Path:
    if os.name == "nt":
        return RUNTIME_DIR / "Scripts" / "python.exe"
    return RUNTIME_DIR / "bin" / "python"


def run(cmd):
    print(" ".join(str(part) for part in cmd))
    subprocess.check_call(cmd)


def should_download_model() -> bool:
    return any(arg == "--download-model" for arg in sys.argv[1:]) or os.environ.get(
        "BALLOONTRANS_DOWNLOAD_GEMMA4_GGUF", ""
    ).strip().lower() in {"1", "true", "yes"}


def selected_quantization() -> str:
    args = sys.argv[1:]
    quantization = os.environ.get("BALLOONTRANS_GEMMA4_GGUF_QUANT", DEFAULT_QUANTIZATION)
    for idx, arg in enumerate(args):
        if arg.startswith("--quant="):
            quantization = arg.split("=", 1)[1]
        elif arg == "--quant" and idx + 1 < len(args):
            quantization = args[idx + 1]

    quantization = quantization.upper()
    if quantization == "Q6_K":
        quantization = "Q6_K_M"
    if quantization not in MODEL_FILES:
        valid = ", ".join(MODEL_FILES)
        raise ValueError(f"Unsupported Gemma4 GGUF quantization: {quantization}. Valid options: {valid}")
    return quantization


def download_model(py: Path):
    quantization = selected_quantization()
    model_filename = MODEL_FILES[quantization]
    model_path = MODEL_DIR / model_filename
    if model_path.exists():
        print(f"Gemma4 GGUF model already exists: {model_path}")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    code = (
        "from huggingface_hub import hf_hub_download; "
        f"hf_hub_download(repo_id={MODEL_REPO_ID!r}, filename={model_filename!r}, "
        f"local_dir={str(MODEL_DIR)!r}, local_dir_use_symlinks=False)"
    )
    print(f"Downloading Gemma4 GGUF {quantization}: {MODEL_REPO_ID}/{model_filename}")
    run([py, "-c", code])


def main():
    RUNTIME_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not runtime_python().exists():
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(RUNTIME_DIR)

    py = runtime_python()
    run([py, "-m", "pip", "install", "--upgrade", "--prefer-binary", "--disable-pip-version-check", "pip"])
    run([
        py,
        "-m",
        "pip",
        "install",
        "--prefer-binary",
        "--disable-pip-version-check",
        "numpy<2.4",
        "llama-cpp-python>=0.3.16",
        "huggingface_hub>=0.34.0",
    ])
    if should_download_model():
        download_model(py)
    print(f"Gemma4 runtime ready: {py}")


if __name__ == "__main__":
    main()
