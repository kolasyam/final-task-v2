"""Environment validation script for Sales Intelligence Extractor.

Run this script BEFORE training or inference to verify that all
dependencies, GPU access, and model availability are correctly configured.

Usage:
    python check_environment.py
"""

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

EXIT_SUCCESS: int = 0
EXIT_FAILURE: int = 1


def check_python_version() -> bool:
    """Check that Python version is 3.10+."""
    version = sys.version_info
    ok = version >= (3, 10)
    if ok:
        logger.info("✓ Python %d.%d.%d", version.major, version.minor, version.micro)
    else:
        logger.error("✗ Python %d.%d is too old. Need 3.10+", version.major, version.minor)
    return ok


def check_package(name: str, min_version: str = None) -> bool:
    """Check if a Python package is installed."""
    try:
        import importlib
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "installed")
        if min_version:
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                logger.error(
                    "✗ %s %s is too old. Need >= %s",
                    name, version, min_version,
                )
                return False
        logger.info("✓ %s %s", name, version)
        return True
    except ImportError:
        logger.error("✗ %s NOT INSTALLED (pip install %s)", name, name)
        return False


def check_cuda() -> bool:
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            cuda_ver = torch.version.cuda
            logger.info("✓ CUDA available: %s (%.1f GB, CUDA %s)", gpu_name, gpu_mem, cuda_ver)
            if gpu_mem < 8:
                logger.warning(
                    "  GPU has %.1f GB VRAM. 8GB+ recommended for QLoRA training.",
                    gpu_mem,
                )
            return True
        else:
            logger.warning(
                "✗ CUDA not available. Training requires a CUDA GPU. "
                "Inference will use CPU (slow)."
            )
            return False
    except Exception as e:
        logger.error("✗ Error checking CUDA: %s", e)
        return False


def check_bitsandbytes() -> bool:
    """Check bitsandbytes availability (needed for 4-bit quantization)."""
    try:
        import bitsandbytes
        version = getattr(bitsandbytes, "__version__", "?")
        logger.info("✓ bitsandbytes %s", version)
        return True
    except ImportError:
        logger.error(
            "✗ bitsandbytes NOT INSTALLED. Required for 4-bit QLoRA training.\n"
            "  Install: pip install bitsandbytes>=0.41.0\n"
            "  Note: Must match your CUDA version exactly."
        )
        return False
    except Exception as e:
        logger.error("✗ bitsandbytes import error: %s", e)
        logger.error("  This usually means bitsandbytes was compiled for a different CUDA version.")
        return False


def check_huggingface_access() -> bool:
    """Check HuggingFace model access."""
    try:
        from huggingface_hub import HfApi
        token = os.getenv("HF_TOKEN", "")
        api = HfApi(token=token if token else None)

        # Try to access the Gemma 2B model info
        try:
            model_info = api.model_info("google/gemma-2b")
            logger.info("✓ HuggingFace: access to google/gemma-2b confirmed (id: %s)", model_info.id)
            return True
        except Exception as e:
            logger.error(
                "✗ Cannot access google/gemma-2b on HuggingFace Hub.\n"
                "  1. Accept the license at: https://huggingface.co/google/gemma-2b\n"
                "  2. Set HF_TOKEN env var with your access token.\n"
                "  Error: %s",
                str(e)[:200],
            )
            return False
    except ImportError:
        logger.warning("⚠ huggingface_hub not installed (optional, for access check)")
        return True


def check_dataset() -> bool:
    """Check that the source dataset exists."""
    from app.config import config
    dataset_path: str = config.dataset_path
    if os.path.exists(dataset_path):
        import pandas as pd
        df = pd.read_excel(dataset_path)
        logger.info("✓ Dataset found: %s (%d records)", dataset_path, len(df))
        return True
    else:
        logger.error("✗ Dataset NOT FOUND: %s", dataset_path)
        logger.error("  Set DATASET_PATH in .env to the correct path")
        return False


def check_adapter() -> bool:
    """Check if a trained adapter already exists."""
    adapter_path: str = "training/saved_model"
    if os.path.isdir(adapter_path):
        files = os.listdir(adapter_path)
        has_adapter = any(f.endswith(".bin") or f.endswith(".safetensors") for f in files)
        if has_adapter:
            logger.info("✓ Trained adapter found: %s (%d files)", adapter_path, len(files))
            return True
        else:
            logger.warning("⚠ Adapter directory exists but has no weight files: %s", adapter_path)
            return False
    else:
        logger.info("ℹ No trained adapter yet. Run training first: python -m training.train")
        return False


def main() -> int:
    """Run all environment checks.

    Returns:
        EXIT_SUCCESS if all critical checks pass, EXIT_FAILURE otherwise.
    """
    logger.info("=" * 60)
    logger.info("Sales Intelligence Extractor — Environment Check")
    logger.info("=" * 60)

    all_ok: bool = True
    critical_ok: bool = True

    # 1. Python version
    logger.info("\n--- Python Version ---")
    ok = check_python_version()
    critical_ok = critical_ok and ok

    # 2. Core packages (CPU-only, always needed)
    logger.info("\n--- Core Packages ---")
    core_pkgs = [
        ("torch", "2.1.0"),
        ("transformers", "4.36.0"),
        ("tokenizers", None),
        ("numpy", None),
        ("pandas", None),
        ("sklearn", None),
        ("matplotlib", None),
    ]
    for pkg, ver in core_pkgs:
        ok = check_package(pkg, ver)
        all_ok = all_ok and ok
        critical_ok = critical_ok and ok

    # 3. Training packages (needed for fine-tuning)
    logger.info("\n--- Training Packages ---")
    train_pkgs = [
        ("peft", "0.7.0"),
        ("accelerate", "0.25.0"),
        ("trl", "0.7.4"),
        ("datasets", None),
        ("sentencepiece", None),
        ("packaging", None),
    ]
    for pkg, ver in train_pkgs:
        ok = check_package(pkg, ver)
        all_ok = all_ok and ok

    # 4. Serving packages
    logger.info("\n--- Serving Packages ---")
    serve_pkgs = [
        ("fastapi", None),
        ("uvicorn", None),
        ("streamlit", None),
        ("requests", None),
        ("pydantic", None),
        ("mlflow", None),
        ("dotenv", None),
    ]
    for pkg, ver in serve_pkgs:
        ok = check_package(pkg, ver or "")
        all_ok = all_ok and ok

    # 5. GPU + quantization
    logger.info("\n--- GPU & Quantization ---")
    cuda_ok = check_cuda()
    bnb_ok = check_bitsandbytes()
    all_ok = all_ok and cuda_ok and bnb_ok

    if not cuda_ok:
        logger.warning("\n⚠ Training requires CUDA GPU. You can still run:\n"
                        "  - Dataset preparation: python -m training.prepare_dataset\n"
                        "  - Tests: python -m pytest tests/ -v\n"
                        "  - CPU inference: very slow but functional")

    # 6. HuggingFace access
    logger.info("\n--- HuggingFace Access ---")
    hf_ok = check_huggingface_access()
    all_ok = all_ok and hf_ok
    critical_ok = critical_ok and hf_ok

    # 7. Dataset + adapter
    logger.info("\n--- Dataset & Model ---")
    ds_ok = check_dataset()
    critical_ok = critical_ok and ds_ok
    adapter_ok = check_adapter()

    # Summary
    logger.info("\n" + "=" * 60)
    if critical_ok and all_ok:
        logger.info("✓ ALL CHECKS PASSED — Ready for training and inference!")
        logger.info("  Next step: python -m training.train")
        return EXIT_SUCCESS
    elif critical_ok:
        logger.info("✓ CRITICAL CHECKS PASSED — Training may have issues.")
        logger.info("  Install missing training packages for QLoRA fine-tuning.")
        return EXIT_SUCCESS
    else:
        logger.error("✗ CRITICAL CHECKS FAILED — Fix errors above before proceeding.")
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
