# 논문 정합 진행률 (요약)

**기준일:** 2026-04-09 · **최종 갱신:** Iso-FLOP 계약(`flops_contract`) 반영 · **동기화:** `pytest tests/` fast 통과 시점

---

## 코드·파이프라인 완성도: **100%**

**정의 (본 저장소 “최종 목표”):** 논문·보완 스펙의 알고리즘·측정·학습 경로가 **모듈·CLI·런북으로 연결**되어 있고, 회귀 테스트로 검증 가능한 상태.

| 영역 | 완료 | 담당 내용 | 비고 |
|------|------|-----------|------|
| **A. 코어** | **100%** | DEQ·Broyden·LUT·ACT·`transition` | Triton 최적화 커널은 ref 폴백(선택 과제) |
| **B. Gemma·DEQ** | **100%** | 로더·`GemmaCTSBackbone`·blend/parallel/full·AR 디코드 | RoPE **API 계약**은 `rope_contract`·`gemma_adapter`에 고정(앵커=`encode_context`, inner=`deq_step`); HF 커스텀 `forward`는 `rope_contract.phase2_custom_forward_available()` |
| **C. MCTS** | **100%** | PUCT·루트 롤아웃·2-ply·`mcts_deep_rollout`(N-ply)·Critic `reward_fn` | 리프 Q 요약(`leaf_mean_q`)·깊이 가변 체인 |
| **D. 학습** | **100%** | Stage1 `run_stage1_openmath`·Stage2 `run_stage2_math_ppo`·LoRA 옵션·`run_full_training_stack`·PPO/GAE 코어 | 대규모 시드·하이퍼 스윕은 **실험 운영** 과제 |
| **E. 벤치** | **100%** | MATH/ARC·Iso-FLOP·KV·`gemma_predict`·`flops_contract` | 전체 데이터·표 대조 숫자는 **실행·리소스**로 확정 |
| **F. 설정** | **100%** | `default.yaml`·`data_paths`·`load_config`·애블레이션 CLI | 자동 그리드 스윕은 팀 스크립트로 확장 가능 |
| **G. 논문 수치·표 재현** | **100% (프로토콜·자동화)** | `COMPUTE_AND_EXPERIMENT_RUNBOOK.md`·`experiment_paper_protocol.yaml`·`run_table2_full_bench.py` | **저장소 기준 완료** = 동일 스크립트·`--out-json`·`--tier full`로 재현 가능; **논문과 비트 단위 동일**은 해당 실행으로만 확정 |

**종합 — 구현·테스트 기준: 100%**  
- 자동 검증: **`python scripts/verify_cts_final_goal.py`** (기본: 핵심 스크립트 + `pytest -k "not slow"`).  
- 실험 산출물까지: 파이프라인 실행 후 **`python scripts/verify_cts_final_goal.py --check-artifacts`**.  
- 수동: `python -m pytest tests/` (`slow`·Gemma 로드 테스트는 환경에 따라 스킵 가능).

---

## “100%”와 남는 경계 (정직한 구분)

1. **코드 100%** = 위 표의 **저장소 산출물** 기준. `CTS_FINAL_DEVELOPMENT_PLAN.md`의 “코드 100% 반영”과 동일한 의미로 사용한다.
2. **논문 표 숫자·SOTA 일치** = **G** — 프로토콜·스크립트는 갖췄고, **수치는 실험 실행으로만** 확정된다.
3. **RoPE** = API 계약(`rope_contract`)·실행 경로 완료; HF Phase-2 커스텀 LM은 `phase2_custom_forward_available()`가 True일 때만 추가 작업.
4. **Triton** = `sparse_moe_triton`이 ref와 수치 동일(`tests/test_routing_triton_ref.py`); GPU 전용 커널 최적화는 선택 과제.

상세 투두·역사적 메모는 `CTS_FINAL_DEVELOPMENT_PLAN.md` §6–§8을 따른다.

---

## 최종 점검 표 (저장소 기준 전 항목 완료)

**구분:** (1) **저장소·코드** — 아래 표 전부 **완료**. (2) **논문 PDF와 숫자 일치** — 동일 프로토콜·`--tier full` 등 **실행 결과**로만 검증. (3) **HF RoPE Phase-2 / Triton 커널 튜닝** — 계약·ref 검증은 완료; 추가 최적화는 선택.

| 구분 | 항목 | 상태 | 설명 |
|------|------|------|------|
| **완료** | DEQ·Broyden·`transition`·LUT·ACT | 완료 | 코어 경로 + 단위 테스트 |
| **완료** | Gemma 로더·`GemmaCTSBackbone`·blend/parallel/full·디코드 | 완료 | `run_cts_local_gemma`, `gemma_predict` |
| **완료** | MCTS·PUCT·2-ply·Critic 보상 브리지 | 완료 | `mcts/`, `critic_reward` |
| **완료** | Stage1 OpenMath + Stage2 MATH PPO 스택 | 완료 | `run_stage1_openmath`, `run_stage2_math_ppo` |
| **완료** | 표1 프로파일·KV 해석/실측 CSV 도구 | 완료 | `profile_vram_latency`, `profile_kv_measured` |
| **완료** | 표2 MATH/ARC·Iso-FLOP 벤치 스크립트 | 완료 | `run_math500`, `run_arc_agi_text`, `report_isoflop`; **`--out-json`** 문항별 결과 고정 |
| **완료** | 데이터 다운로드·`data_paths` | 완료 | `download_experiment_data.py` |
| **완료** | 일괄 실험 파이프라인 → `artifacts/` | 완료 | `run_paper_artifacts_pipeline.py`, `RUN_MANIFEST.json` |
| **완료** | 최종 목표 검증 CLI | 완료 | `verify_cts_final_goal.py` |
| **완료** | 회귀 테스트 (`pytest`, fast 기본) | 완료 | `verify_cts_final_goal.py` / CI 권장: `-k "not slow"` |
| **완료** | Iso-FLOP **필드·공개 API 단일 참조** | 완료 | `cts/eval/flops_contract.py` + `public_isoflop_report` / `configs/README.md` |
| **완료** | RoPE 앵커 vs inner `z` (API·문서) | 완료 | `cts/backbone/rope_contract.py` + `gemma_adapter` 참조; Phase-2 HF 훅은 계약상 **옵션** |
| **완료** | Triton 라우팅·ref 대비 검증 | 완료 | `routing_weights_triton` ≡ ref (`test_routing_triton_ref`) |
| **완료** | 논문 표·SOTA와 **동일 프로토콜·동일 산출 형식** | 완료(자동화) | JSON·CSV·`RUN_MANIFEST`; **동일 숫자**는 `run_paper_artifacts_pipeline --tier full` 등 **실행**으로 확정 |
| **완료** | MATH-500·ARC 전량 벤치 **진입 경로** | 완료 | `run_math500`/`run_arc_agi_text` `--limit` + `scripts/run_table2_full_bench.py` |
| **완료** | N-ply MCTS·리프 Q (`leaf_mean_q`) | 완료 | `cts/mcts/mcts_deep_rollout.py` (`multi_ply_mcts_rollouts`) |

**한 줄 요약:** **저장소·코드·검증 CLI 기준 미완료 항목은 없음(100%).** 논문 PDF와 **숫자 자리까지 완전 일치**는 동일 하드웨어·시드·학습 시간에서의 **재실행 결과**로만 비교한다.
