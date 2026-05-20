import os
import subprocess
import sys
import venv
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = APP_ROOT / "data" / "models" / "gemma-4-runtime"
DEFAULT_QUANTIZATION = "Q4_K_M"
MODEL_CONFIGS = {
    "gemma4": {
        "display_name": "Gemma4 GGUF",
        "repo_id": "unsloth/gemma-4-E4B-it-GGUF",
        "model_dir": APP_ROOT / "data" / "models" / "gemma-4-E4B-it-GGUF",
        "files": {
            "Q4_K_M": "gemma-4-E4B-it-Q4_K_M.gguf",
            "Q6_K_M": "gemma-4-E4B-it-Q6_K.gguf",
        },
        "download_env": "BALLOONTRANS_DOWNLOAD_GEMMA4_GGUF",
        "quant_env": "BALLOONTRANS_GEMMA4_GGUF_QUANT",
        "aliases": {"gemma", "gemma4", "gemma-4", "gemma-4-e4b-it"},
    },
}


def runtime_python() -> Path:
    if os.name == "nt":
        return RUNTIME_DIR / "Scripts" / "python.exe"
    return RUNTIME_DIR / "bin" / "python"


def run(cmd):
    print(" ".join(str(part) for part in cmd))
    subprocess.check_call(cmd)


def should_download_model() -> bool:
    if any(arg == "--download-model" for arg in sys.argv[1:]):
        return True
    return any(
        os.environ.get(config["download_env"], "").strip().lower() in {"1", "true", "yes"}
        for config in MODEL_CONFIGS.values()
    )


def selected_model_key() -> str:
    args = sys.argv[1:]
    model_name = os.environ.get("BALLOONTRANS_GGUF_MODEL", "")
    for idx, arg in enumerate(args):
        if arg.startswith("--model="):
            model_name = arg.split("=", 1)[1]
        elif arg == "--model" and idx + 1 < len(args):
            model_name = args[idx + 1]

    if not model_name:
        for model_key, config in MODEL_CONFIGS.items():
            if os.environ.get(config["download_env"], "").strip().lower() in {"1", "true", "yes"}:
                return model_key
        return "gemma4"

    normalized = model_name.strip().lower()
    for model_key, config in MODEL_CONFIGS.items():
        if normalized == model_key or normalized in config["aliases"]:
            return model_key
    valid = ", ".join(sorted({alias for config in MODEL_CONFIGS.values() for alias in config["aliases"]}))
    raise ValueError(f"Unsupported GGUF model: {model_name}. Valid options: {valid}")


def selected_quantization(model_key: str) -> str:
    args = sys.argv[1:]
    config = MODEL_CONFIGS[model_key]
    quantization = os.environ.get(config["quant_env"], DEFAULT_QUANTIZATION)
    for idx, arg in enumerate(args):
        if arg.startswith("--quant="):
            quantization = arg.split("=", 1)[1]
        elif arg == "--quant" and idx + 1 < len(args):
            quantization = args[idx + 1]

    quantization = quantization.upper()
    if model_key == "gemma4" and quantization == "Q6_K":
        quantization = "Q6_K_M"
    if quantization not in config["files"]:
        valid = ", ".join(config["files"])
        raise ValueError(f"Unsupported {config['display_name']} quantization: {quantization}. Valid options: {valid}")
    return quantization


def download_model(py: Path, model_key: str):
    config = MODEL_CONFIGS[model_key]
    quantization = selected_quantization(model_key)
    model_filename = config["files"][quantization]
    model_dir = config["model_dir"]
    model_path = model_dir / model_filename
    if model_path.exists():
        print(f"{config['display_name']} model already exists: {model_path}")
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    code = (
        "from huggingface_hub import hf_hub_download; "
        f"hf_hub_download(repo_id={config['repo_id']!r}, filename={model_filename!r}, "
        f"local_dir={str(model_dir)!r}, local_dir_use_symlinks=False)"
    )
    print(f"Downloading {config['display_name']} {quantization}: {config['repo_id']}/{model_filename}")
    run([py, "-c", code])


def main():
    model_key = selected_model_key()
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
        download_model(py, model_key)
    print(f"GGUF runtime ready: {py}")


if __name__ == "__main__":
    main()
