# Third-party components and data

This repository’s **own source code** is licensed under **Apache License 2.0** (see repository root `LICENSE`), consistent with `pyproject.toml`.

The following are **not** part of this repository’s license grant; users must comply with each upstream’s terms when downloading or using them.

## Models (weights not included in git)

| Component | Source | Notes |
|-----------|--------|--------|
| **Gemma 4** (e.g. E4B / E4B-it) | [Google AI / Hugging Face Hub](https://huggingface.co/google) | Gemma 4 is distributed under **Apache 2.0** per Google’s documentation: [Gemma 4 license](https://ai.google.dev/gemma/apache_2). Older Gemma variants may fall under [Gemma Terms of Use](https://ai.google.dev/gemma/terms). **Do not commit** `model.safetensors` to git; download via Hub or local snapshot. |

## Python libraries

Runtime dependencies are listed in `pyproject.toml` (e.g. PyTorch, Hugging Face `transformers`, `datasets`). Each package has its own license (BSD, Apache, MIT, etc.). See each package’s PyPI or source repository for the exact text.

## Datasets (downloaded by scripts, not shipped in git)

| Dataset | Hub id (used by `scripts/download_experiment_data.py`) | Action |
|---------|----------------------------------------------------------|--------|
| MATH-500 eval split | `HuggingFaceH4/MATH-500` | Follow the dataset card and license on Hugging Face. |
| OpenMathInstruct-1 (subset) | `nvidia/OpenMathInstruct-1` | NVIDIA dataset license: see [dataset LICENSE on Hub](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1). |
| MATH train prompts (Stage2) | `EleutherAI/hendrycks_math` | Follow the dataset card on Hugging Face. |

## Citation

When publishing work that uses this codebase, cite the **CTS paper** (once public) and cite **Gemma**, **datasets**, and **libraries** as required by their respective authors.
