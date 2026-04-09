# 컴퓨트·실험 런북 (별도 머신에서 실행)

**GPU·데이터**는 로컬에서 실행해야 합니다. 아래 절차(및 **§7 일괄 파이프라인**)로 **동일 프로토콜**을 재현할 수 있습니다. 최종 점검은 **`python scripts/verify_cts_final_goal.py`** (`--check-artifacts` 선택).

## 1. 사전 조건

| 항목 | 권장 |
|------|------|
| GPU VRAM | 내부 개발 계획·요약 문서(`doc/NeurIPS_2026_*`) 기준 **단일 RTX 4090(24GB)** 또는 동급이 타깃으로 명시됨. 본 저장소 `CTS_FINAL_DEVELOPMENT_PLAN.md`는 **24GB급 GPU**로 서술. |
| 로컬 스냅샷 | `artifacts/LOCAL_HARDWARE_SNAPSHOT.txt` (4090 + 32GB RAM 예시 기록) |
| 디스크 | `model.safetensors` ~16GB + Hub 캐시 |
| 소프트웨어 | `pip install -e ".[dev]"`, Gemma4용 `transformers` 개발 빌드(README 참고) |
| 인증 | `HF_TOKEN` (게이트 모델), `CTS_GEMMA_MODEL_DIR` 로컬 스냅샷 권장 |

## 1.1 데이터 받기 (로컬 `data/`)

```powershell
cd D:\AI\cts
pip install "datasets>=2.16"
python scripts/download_experiment_data.py
# 선택: OpenMath 10만 행 + MATH 학습 프롬프트 5000 (기본 인자와 동일)
```

생성물: `data/math500/test.jsonl` (500), `data/openmath_instruct/train_100000.jsonl` (스트리밍 10만 행, 기본), `data/stage2/math_train_prompts_5000.jsonl`. 경로는 `configs/data_paths.yaml` 참고.

## 2. 표 1 유형 (메모리·지연 vs 깊이)

1. **CTS mock + 해석적 KV + (선택) 실측 KV**

```powershell
$env:HF_HUB_CACHE = "D:\AI\cts\.hf_cache"
python -m cts.eval.profile_vram_latency --depths 1 5 10 15 20 --out artifacts/table1_cts_kv.csv --cuda
python scripts/profile_kv_measured.py --depths 1 5 10 15 --out artifacts/table1_kv_measured.csv
```

2. 결과 CSV 병합·플롯은 외부 도구(Excel, pandas)에서 `tree_depth_proxy`, `approach`, `peak_vram_gb` 기준으로 정렬.

3. **논문 대비**: 동일 `W=3`, 동일 깊이 스윕, 동일 워밍업(`configs/default.yaml`의 `eval_warmup_runs`)을 맞출 것.

## 3. 표 2 유형 (MATH / ARC, Iso-FLOP)

1. 데이터셋 JSONL을 준비한 뒤:

```powershell
$env:CTS_GEMMA_MODEL_DIR = "D:\AI\cts\gemma-4-E4B-it"
python scripts/run_math500.py --data <math500.jsonl> --gemma --limit 500 --think-prompt --out-json artifacts/table2_math500_run.json
python scripts/run_arc_agi_text.py --data <arc.jsonl> --gemma --limit 200 --out-json artifacts/table2_arc_run.json
```

`--out-json` 은 요약·문항별 `match`·truncated `pred`/`gold` 를 UTF-8 JSON으로 저장합니다(재현·보고용).

**한 번에 (MATH 500 + 선택 ARC 200):** `python scripts/run_table2_full_bench.py` — 내부적으로 위 두 스크립트를 호출하고 `artifacts/table2_full_bench_manifest.txt`를 남깁니다. `--skip-gemma`로 명령만 확인 가능.

2. **Iso-FLOP 매칭**: 동일 쿼리당 `flops_broyden_estimate`(및 디코드 예산)를 로그에 남기고 `configs/README.md`의 DEQ vs 디코드 구분을 유지.

## 4. Stage 1 / 2 학습 (대규모 컴퓨트)

- **Stage 1 (OpenMathInstruct + 고정점 손실)**: `scripts/run_stage1_openmath.py` — JSONL(`question`) + `GemmaCTSBackbone` + `fixed_point_surrogate_loss`. 선택 `--lora` ( `pip install peft` ). 체크포인트 `artifacts/stage1_last.pt`.
- **Stage 2 (MATH 프롬프트 + PPO)**: `scripts/run_stage2_math_ppo.py` — `MetaPolicy` + `transition` 롤아웃 + `ppo_core` + 선택 `--use-critic-reward`. 기본 `CTS_DEQ_MAP_MODE=blend`(로컬 부담 완화); 논문에 가깝게 하려면 `--parallel-map`. `--stage1-ckpt`로 Stage1 가중치 로드 가능. 산출물 `artifacts/stage2_meta_value.pt`.
- **한 번에 보기만**: `python scripts/run_full_training_stack.py` (실행은 `--run`, GPU 시간·디스크 대용).
- **스모크**: `CTS_STAGE2_SMOKE=1` 이면 Stage2 스텝 상한이 짧아집니다.
- **멀티 GPU**: 단일 프로세스 기준. `torchrun` 분산은 이후 확장.

## 5. RoPE 앵커 vs inner `z` (계약 완료)

- **단일 참조:** `cts/backbone/rope_contract.py` — 앵커는 `encode_context`(HF RoPE), latent 업데이트는 `deq_step`에서 `z`·`context`만 사용.
- **옵션 Phase-2:** HF `language_model`에 `position_ids`/캐시 분기 커스텀은 `rope_contract.phase2_custom_forward_available()`가 True일 때만 해당(현재 False). `gemma_adapter` 모듈 docstring과 동일 선에서 동작한다.

## 6. 재현성

- `PYTHONHASHSEED`, `torch.manual_seed`, `CTS_*` 환경 변수를 실험 로그에 함께 기록하십시오.
- `artifacts/`에 CSV·로그를 버전 관리할지 여부는 팀 정책에 따릅니다(대용량은 git-lfs 또는 외부 저장소).

## 7. 일괄 실행 (표 1·2 CSV + Stage1/2 → `artifacts/`)

한 번에 프로파일·Iso-FLOP·(선택) MATH Gemma 평가·GPU 학습 스텝을 돌리고 `artifacts/RUN_MANIFEST.json`에 기록합니다.

```powershell
cd D:\AI\cts
$env:CTS_GEMMA_MODEL_DIR = "D:\AI\cts\gemma-4-E4B-it"
# 빠른 스모크(기본): 표1 CSV, Iso-FLOP JSON, MATH 24문항, Stage1 5스텝, Stage2 2스텝
python scripts/run_paper_artifacts_pipeline.py --tier quick --skip-download
# 더 긴 실행: --tier standard 또는 --tier full
```

`--skip-download`: 이미 `data/`가 있을 때 Hub 스트리밍을 생략합니다. HF 게이트 모델은 `HF_TOKEN`을 설정하십시오.

## 8. 최종 목표 검증 (코드 + 선택적 `artifacts/`)

```powershell
python scripts/verify_cts_final_goal.py
python scripts/verify_cts_final_goal.py --check-artifacts
```

이 런북은 **실행 책임이 있는 쪽(연구실/클라우드)**에서 그대로 따라 할 수 있도록 최소 단위로 유지합니다.
