"""modular_addition package.

On import, load this package's own ``.env`` so every entry point — the oracle
runner, the wandb training harness, ``push_to_hf``, and ``train.py`` — sees
``HF_TOKEN`` / ``WANDB_TOKEN`` without a manual ``huggingface-cli login`` or
``wandb login``. Real environment variables always win (``override=False``), so
RunPod / CI that export tokens directly are unaffected (this becomes a no-op).
"""
import os
from pathlib import Path

# Load <repo>/modular_addition/.env regardless of the current working directory.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ModuleNotFoundError:  # python-dotenv absent — fall back to the ambient env
    pass

# wandb authenticates with WANDB_API_KEY; mirror the WANDB_TOKEN name used in
# .env so a single token name works for both Hugging Face (HF_TOKEN, read
# natively by huggingface_hub) and Weights & Biases.
if os.environ.get("WANDB_TOKEN") and not os.environ.get("WANDB_API_KEY"):
    os.environ["WANDB_API_KEY"] = os.environ["WANDB_TOKEN"]
