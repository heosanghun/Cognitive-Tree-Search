# Cognitive Tree Search (CTS)

Implementation scaffold for the CTS paper: **KV-cache-free DEQ transitions** + **neuromodulated MCTS** on a modular backbone (Gemma-style or `MockTinyBackbone`).

## License

- **This repository’s code:** [Apache License 2.0](LICENSE) (`NOTICE` for attribution boilerplate).
- **Third-party models, datasets, and PyPI packages:** not covered by the repo license alone — see [`doc/THIRD_PARTY_NOTICES.md`](doc/THIRD_PARTY_NOTICES.md).

## Docs

- Final plan: `doc/CTS_FINAL_DEVELOPMENT_PLAN.md`
- **별도 컴퓨트에서 표·학습 재현:** `doc/COMPUTE_AND_EXPERIMENT_RUNBOOK.md`
- **이 PC 사양 스냅샷(예시):** `artifacts/LOCAL_HARDWARE_SNAPSHOT.txt`
- **진행률 (완료 vs 미완료):** `doc/PAPER_ALIGNMENT_PROGRESS.md`
- Memory (M1/M2): `doc/memory_definitions.md`
- Improved spec: `doc/NeurIPS_2026_CTS_DEVELOPMENT_PLAN_IMPROVED.md`

## Install

```bash
pip install -e ".[dev]"
# 벤치·학습용 HF 데이터 (MATH-500 + OpenMath 스트리밍 서브셋)
pip install -e ".[data]"
python scripts/download_experiment_data.py
```

**Gemma 4 E4B:** requires a recent `transformers` build with `gemma4` support, e.g.:

```bash
pip install git+https://github.com/huggingface/transformers.git
```

Accept the model license on Hugging Face and set `HF_TOKEN` if the repo is gated.

**Disk / cache:** the full E4B checkpoint is large (~16GB). If drive `C:` is full, point the Hub cache to another disk (this repo defaults to `.hf_cache` under the project when `HF_HUB_CACHE` is unset):

```powershell
$env:HF_HUB_CACHE = "D:\AI\cts\.hf_cache"
```

### Local snapshots (full weights on disk)

Hub `main` for **google/gemma-4-E4B** exposes **8 files** (recursive tree); the only weight file is **`model.safetensors`** (~16GB). The same layout applies to **google/gemma-4-E4B-it**, plus **`chat_template.jinja`** (9 files).

After `snapshot_download`, this repo can use:

| Path | Contents |
|------|----------|
| `D:\AI\cts\gemma-4-E4B` | Base: `model.safetensors`, tokenizer, `config.json`, `processor_config.json`, `generation_config.json`, … |
| `D:\AI\cts\gemma-4-E4B-it` | Instruction-tuned: same + **`chat_template.jinja`** (recommended for dialogue / `<|think|>`-style runs) |

Load from disk (or set `CTS_GEMMA_MODEL_DIR` to one of the folders above when using the default Hub id):

```python
from cts.model.gemma_loader import load_gemma4_e4b
model, tok = load_gemma4_e4b(r"D:\AI\cts\gemma-4-E4B-it", device_map="cuda:0")
```

Smoke test (기본 **blend** — 가벼운 내부 맵):

```bash
python scripts/smoke_gemma_cts.py
```

### DEQ 내부 맵 단계별 실행 (권장)

| 단계 | 모드 | 설명 |
|------|------|------|
| **1** | `blend` (**기본값**) | 작은 어댑터만 사용 → 로드·`transition`·Broyden 수렴까지 빠르게 확인 |
| **2** | `parallel` | 논문 Eq.(5)에 가까운 **희소 병렬 모듈** (Broyden 한 번당 **GPU 부담 큼**) |
| (애블레이션) | `full` | 42층 순차 1패스 |

```powershell
$env:CTS_GEMMA_MODEL_DIR = "D:\AI\cts\gemma-4-E4B-it"

# 1단계: 동작 확인 (기본이 blend)
python scripts/run_cts_local_gemma.py

# 2단계: 논문에 가까운 inner map
python scripts/run_cts_local_gemma.py --parallel
```

환경 변수 `CTS_DEQ_MAP_MODE`는 스크립트에 `--map` / `--parallel` 을 주지 않을 때만 적용됩니다.

로컬 Gemma 한 스텝은 DEQ 후 **`--max-decode`** 로 AR 길이를 조절합니다 (기본 16).

### 표 1 스타일 프로파일 · 애블레이션

```bash
python -m cts.eval.profile_vram_latency --depths 1 5 10 15 --out artifacts/profile_table1.csv
python scripts/run_ablations.py --config ablation_no_ach
```

Stage1 DEQ warm-up **데모** (목업 백본 1스텝):

```bash
python -c "from cts.train.stage1_warmup import run_stage1_demo_step; print(run_stage1_demo_step())"
```

**논문에 가까운 풀 스택 (데이터 다운로드 후, GPU·시간 필요):** `pip install -e ".[data,train]"` → `python scripts/download_experiment_data.py` → `python scripts/run_stage1_openmath.py` → `python scripts/run_stage2_math_ppo.py --stage1-ckpt artifacts/stage1_last.pt`. 명령만 보려면 `python scripts/run_full_training_stack.py`.

**표 1·2 CSV + Stage1/2 GPU 스모크를 `artifacts/`에 한 번에:** `python scripts/run_paper_artifacts_pipeline.py` (옵션·티어는 `doc/COMPUTE_AND_EXPERIMENT_RUNBOOK.md` §7).

**최종 목표 자동 검증:** `python scripts/verify_cts_final_goal.py` · 파이프라인 실행 후 `python scripts/verify_cts_final_goal.py --check-artifacts`.

MATH/ARC 벤치는 **`--out-json path.json`** 으로 요약·문항별 결과를 UTF-8로 저장할 수 있습니다 (`run_math500.py`, `run_arc_agi_text.py`).

Iso-FLOP 리포트 (mock 전이 + Broyden 추정 FLOPs):

```bash
python -m cts.eval.report_isoflop --json
```

MATH / ARC JSONL pass@1 (데모: 고정 문자열; **Gemma** 로 실측 시 `--gemma`):

```bash
python scripts/run_math500.py --data your.jsonl --limit 100
python scripts/run_arc_agi_text.py --data your.jsonl --limit 100
# Greedy Gemma (가중치 필요, CTS_GEMMA_MODEL_DIR 권장)
python scripts/run_math500.py --data your.jsonl --limit 20 --gemma --max-new-tokens 256
# E4B-it 대화 템플릿 (가중치 로드와 동일 스크립트에서 --chat-template)
python scripts/run_math500.py --data your.jsonl --limit 5 --gemma --chat-template
# `<|think|>` 템플릿 문자열로 문제 감싸기 (토크나이저; `--think-prompt`)
python scripts/run_math500.py --data your.jsonl --limit 5 --gemma --think-prompt
```

MCTS 루트 확장 스모크 (mock 백본, W개 병렬 전이):

```bash
python scripts/run_mcts_episode_mock.py --prompt "2+2=?" -W 3
```

PUCT로 **한 자식만 선택** 후 한 번 `transition` (선택: `--meta`로 ν·prior):

```bash
python scripts/run_mcts_puct_once.py
python scripts/run_mcts_puct_once.py --meta
```

루트에서 **여러 번** PUCT→전이→**Q 평균 백업** (`mcts_root_rollouts`):

```bash
python scripts/run_mcts_root_rollouts.py --sims 8 --json
python scripts/run_mcts_root_rollouts.py --critic --sims 4
python scripts/run_two_ply_mcts.py --meta
```

**가중치가 필요한가?**

| 기능 | 가중치 |
|------|--------|
| `show_chat_template.py` / `prompt_format.format_user_prompt_chat_string` | **아니오** (토크나이저·템플릿만, ~수백 MB 이하 일반적) |
| `--gemma` 벤치 (generate) | **예** (`model.safetensors`) |
| `train_routing_proj_one_step.py` | **`--mock`이면 아니오** / 실제 Gemma면 **예** |

GPU에서 KV 텐서 **실측** 피크 (CUDA 없으면 CSV에 null):

```bash
python scripts/profile_kv_measured.py --depths 1 3 5 --out artifacts/kv_measured.csv
```

KV 행은 **해석적** O(depth) KV 증가 모델(`cts/baselines/mcts_kv_baseline.py`)이며, 실측 GPU KV-MCTS와 병행하면 논문 대비가 명확해집니다.

## Tests

```bash
pytest tests/ -q
```

## Config

See `configs/default.yaml` and `configs/README.md`.

## Status

v0.1 scaffold: core types, Broyden API, PUCT, mock transition, profiling entrypoint. Full Gemma 4 training/eval is gated on model weights and data.
