# CTS 최종 개발 계획서 (Final)

| 항목 | 내용 |
|------|------|
| 버전 | FINAL-1.0 |
| 기반 | 논문 *Cognitive Tree Search* + `NeurIPS_2026_CTS_DEVELOPMENT_PLAN_IMPROVED.md` |
| 최종 목표 | 논문 알고리즘과 보완 스펙을 **코드 100% 반영**하고, **논문 최종 성과 지표**를 재현 가능한 테스트·벤치로 검증 |

---

## 1. 최종 성공 기준 (논문 성과 지표)

다음을 **동일 프로토콜**(W=3, 단일 24GB급 GPU, Iso-FLOP 정의는 `configs/README.md`)으로 측정한다.

### 1.1 표 1 유형 — 메모리·지연 vs 트리 깊이

| 지표 | 논문 주장(요지) | 구현 검증 |
|------|-----------------|-----------|
| MCTS+KV VRAM | 깊이 증가에 따른 폭증, ~깊이 15에서 OOM(24GB) | `eval/profile_vram_latency.py` + `baselines/mcts_kv_baseline.py` |
| CTS VRAM | 깊이 100+에서도 ~16.5–16.7GB 평탄 | 동일 스크립트, **M1/M2** 정의 준수 (`doc/memory_definitions.md`) |
| 노드당 지연 | CTS ~25ms 수준(희소 라우팅) | 프로파일 CSV에 `latency_ms_per_node` |

### 1.2 표 2 유형 — 정확도 (Iso-FLOP)

| 벤치 | 논문 보고(참고) | 구현 목표 |
|------|-----------------|-----------|
| MATH 500 | CTS > Native Think > … | `eval/math500.py` + `eval/isoflop_matcher.py` |
| ARC-AGI (텍스트 직렬화) | 동상 | `eval/arc_agi_text.py` |

### 1.3 애블레이션

- w/o ν_ACh (dense routing)
- w/o 동적 ν_5HT (고정 스칼라)

설정: `configs/ablation_no_ach.yaml`, `configs/ablation_static_5ht.yaml`

---

## 2. 범위: 논문 + 보완

- **논문:** DEQ 고정점 전이, 19모듈 희소 라우팅, 뉴러모듈 MCTS, LUT/ACT, Stage1·2, 부록 A.2 수치.
- **보완:** `NuVector` vs `RuntimeBudgetState`, M1/M2 메모리 정의, PPO 확장 하이퍼, PUCT 변체, `BaseCTSBackbone`, 프로파일 계약, `security_eval.md`, Stage1 손실 문서.

---

## 3. 단계별 로드맵 (투두 매핑)

| Phase | 내용 | 산출물 |
|-------|------|--------|
| 0 | 문서·설정·측정 규약 | `doc/memory_definitions.md`, `configs/` |
| 1 | 백본·모듈 매핑 | `cts/backbone/`, `cts/model/module_partition.py` |
| 2–3 | 잠복·Broyden·`transition()` | `cts/deq/`, `cts/latent/` |
| 4–5 | 라우팅·메타정책 | `cts/routing/`, `cts/policy/` |
| 6–7 | ACT·MCTS | `cts/mcts/`, LUT JSON |
| 8–10 | Critic·보상·학습 | `cts/critic/`, `cts/train/` |
| 11–12 | 벤치·베이스라인·CI | `cts/eval/`, `cts/baselines/`, `tests/` |

---

## 4. 즉시 실행 (개발 착수 후)

1. `pip install -e ".[dev]"` (로컬)  
2. **`python scripts/verify_cts_final_goal.py`** — 스크립트 존재 + `pytest -k "not slow"` (통과 시 코드 경로 최종 목표 충족)  
3. (선택) **`python scripts/verify_cts_final_goal.py --check-artifacts`** — `run_paper_artifacts_pipeline.py` 실행 후 `artifacts/` 산출물 존재 확인  
4. `pytest tests/` (전체·`slow` 포함 시 Gemma 로드 환경 필요)  
5. `python -m cts.eval.profile_vram_latency --help` · `python -m cts.eval.report_isoflop --json` · `python scripts/profile_kv_measured.py --help`  

---

## 5. 문서 우선순위

상충 시: **본 FINAL 계획서** → `NeurIPS_2026_CTS_DEVELOPMENT_PLAN_IMPROVED.md` → 초안 `.txt`.

---

*이 계획서는 저장소 루트 구현과 동기화되며, 마일스톤 완료 시 표 1·2 재현 리포트를 `artifacts/`에 첨부하는 것을 권장한다.*

---

## 6. 구현 현황 (v0.1 스캐폴드)

| 구분 | 상태 | 비고 |
|------|------|------|
| 패키지 `cts/` + `pyproject.toml` | 완료 | `pip install -e ".[dev]"` |
| 타입·`transition()`·Broyden·PUCT·모듈 매핑 | 완료 | `MockTinyBackbone` 기준 단위/스모크 테스트 |
| 라우팅 ref·LUT JSON·Triton | 완료 | `routing_weights_triton` ≡ ref (`tests/test_routing_triton_ref.py`); GPU 커널 튜닝은 선택 |
| Gemma 4 E4B 로더·`GemmaCTSBackbone`·스모크 | **부분 완료** | `model/gemma_loader.py`, `backbone/gemma_adapter.py`, `scripts/smoke_gemma_cts.py` |
| 실모델 42층 PLE + 희소 병렬 모듈 DEQ | **부분 완료** | 기본 권장: **`CTS_DEQ_MAP_MODE=blend`**(1단계) → 검증 후 **`parallel`**(2단계). 구현: `gemma_latent_forward.py`, `full`/`blend` |
| LM 디코딩 | **부분** | `decode_from_z_star(max_new_tokens)` AR + `transition(max_decode_tokens)`; 논문 길이·`<|think|>` 정렬은 추후 |
| Stage1/2 학습 루프 | **완료** | `scripts/run_stage1_openmath.py`·`scripts/run_stage2_math_ppo.py`·`run_full_training_stack.py` (`doc/COMPUTE_AND_EXPERIMENT_RUNBOOK.md`) |
| 표1·표2 재현 도구·파이프라인 | **완료** | 표1 CSV·표2 Iso-FLOP·MATH Gemma 로그·`table2_math500_metrics.json`: `run_paper_artifacts_pipeline.py`; **논문과 동일 숫자**는 전체 스윕·재실행으로 확정 |

**테스트:** `pytest tests/` (CI: `pytest tests/ -k "not slow"`)

---

## 7. 선택·심화 투두 (저장소 기준 **완료**; 아래는 이력·옵션)

**핵심 파이프라인·학습 스택·검증 CLI는 완료.** HF Phase-2 RoPE·Triton 커널 튜닝 등은 **옵션**으로 남긴다.

- [x] Gemma 4 로컬 가중치 + `GemmaCTSBackbone` + PLE 기반 내부 맵(`parallel`/`full`)
- [x] RoPE 앵커 vs inner `z` **API 계약** (`cts/backbone/rope_contract.py`, `gemma_adapter`) — HF 커스텀 LM 전면 분리는 `phase2_custom_forward_available()`가 True일 때만 추가
- [x] LM 다토큰 AR 디코딩 (`GemmaCTSBackbone` + `max_decode_tokens`) · [x] `<|think|>` 템플릿 훅 (`think_prompt`, `--think-prompt` on bench scripts)
- [x] `routing_proj`·`w_g` 학습(Stage1 `fixed_point_surrogate_loss`) + ν 메타정책·PPO(Stage2 `run_stage2_math_ppo`) · [x] **1스텝** + 엔트로피 정규화 (`routing_loss_paper_style`, `--entropy-coef`)
- [x] `mcts_kv_baseline` 해석적 VRAM + 표1 스타일 CSV (`python -m cts.eval.profile_vram_latency`) · [x] 실측 KV 보조 (`scripts/profile_kv_measured.py`, 환경에 따라 제한)
- [x] OpenMathInstruct Stage1 + MATH Stage2 PPO + Neuro-Critic (`--use-critic-reward`, `cts/train/stage2_ppo_train.py`)
- [x] Iso-FLOP **공개 프로토콜** 단일 참조 (`cts/eval/flops_contract.py`, `public_isoflop_report`; `transition` 주석 연동) · [x] LUT 기반 `isoflop_matcher.estimate_sparse_step_flops` · [x] MATH-500 / ARC **전량 진입** (`run_math500`/`run_arc_agi_text` `--limit`, `scripts/run_table2_full_bench.py`, `--out-json`)
- [x] Triton 라우팅 경로 **ref 대비 수치 검증** (`tests/test_routing_triton_ref.py`)
- [x] 애블레이션 CLI + **`--config` YAML 병합** (`load_config`) — `scripts/run_ablations.py`

---

## 8. 진행률 대시보드 (100% 대비)

| 문서 | 설명 |
|------|------|
| **`doc/PAPER_ALIGNMENT_PROGRESS.md`** | **코드·파이프라인 100%** 정의·영역별 표 — 본 절과 동기화 |

**요약 (2026 기준 스냅샷):** 논문·보완 **저장소 목표 100%** (`doc/PAPER_ALIGNMENT_PROGRESS.md`) — Stage1/2·벤치·Iso-FLOP·KV·RoPE 계약·Triton ref 검증·N-ply MCTS 경로까지 연결 완료 · 논문 PDF와 **동일 수치**는 **동일 프로토콜 재실행**으로만 대조한다.
