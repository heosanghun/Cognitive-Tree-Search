# Evaluation security

- **MATH / ARC-AGI:** use official or mirrored datasets per license; do not execute model-generated code unless inside an approved sandbox.
- **Default:** string-based grading only; no arbitrary code execution from model outputs.
- For any future code-execution benchmark, require explicit env flag `CTS_EVAL_ALLOW_CODE_EXEC=1` and container isolation.
