# Cognitive Tree Search (CTS) 개발 계획서 — 분석·비평·개선 통합본

| 항목 | 내용 |
|------|------|
| 문서 ID | CTS-DEVPLAN-IMPROVED-002 |
| 기반 문서 | `NeurIPS_2026_CTS_FULL_DEVELOPMENT_PLAN_ko.txt` |
| 목적 | 기존 계획서 재분석 후 **문제점·보완점·제안**을 반영한 구현 기준서 |
| 대상 논문 | *Cognitive Tree Search: KV-Cache-Free O(1) Transitions for System 2 Inference via Neuromodulated Deep Equilibrium* (Preprint) |

---

## 목차

1. [선행 분석 요약](#1-선행-분석-요약)
2. [식별된 문제점](#2-식별된-문제점)
3. [보완 사항](#3-보완-사항)
4. [추가 제안](#4-추가-제안)
5. [개선된 실행 요약 및 성공 기준](#5-개선된-실행-요약-및-성공-기준)
6. [개선된 요구사항 추적 매트릭스](#6-개선된-요구사항-추적-매트릭스)
7. [개선된 저장소 구조](#7-개선된-저장소-구조)
8. [개선된 API·상태 모델](#8-개선된-api상태-모델)
9. [Phase별 작업 (개정)](#9-phase별-작업-개정)
10. [하이퍼파라미터·설정 (확장)](#10-하이퍼파라미터설정-확장)
11. [테스트·검증 (확장)](#11-테스트검증-확장)
12. [리스크 레지스터 (정정·확장)](#12-리스크-레지스터-정정확장)
13. [보안·데이터·윤리](#13-보안데이터윤리)
14. [마일스톤·의존성](#14-마일스톤의존성)
15. [Definition of Done](#15-definition-of-done)
16. [부록: 의사코드](#16-부록-의사코드)

---

## 1. 선행 분석 요약

기존 개발 계획서는 **모듈 분해·Phase 구조·부록 하이퍼 연계**가 명확하고, 논문의 DEQ–MCTS–ν–LUT 축을 코드 경로로 잘 매핑한다. 다만 아래 영역에서 **모호성·누락·내부 불일치**가 있어 구현 단계에서 결정 비용과 재작업 위험이 커진다.

- 메모리 주장의 **측정 단위**(트리 전체 vs 전이 1회) 정의 부족
- `NuVector`와 **런타임 누적 상태**(예: Adenosine)의 역할 혼선
- 디렉터리 구조와 REQ 매트릭스 간 **파일 누락**(`memory_accounting.py` 등)
- PPO·MCTS **세부 하이퍼** 일부 누락(GAE λ, clip ε 등)
- 논문 가정 모델(Gemma 4 E4B, 42층)과 **실제 공개 가중치** 불일치 가능성
- **보안/데이터/재현성**에 대한 공학 항목 부재

본 문서는 위 항목을 정정·보강한 **통합 개선본**이다.

---

## 2. 식별된 문제점

### 2.1 개념·서술

| ID | 문제 | 영향 |
|----|------|------|
| P-1 | 논문의 **O(1) 컨텍스트 메모리**는 “KV 캐시 누적 없는 전이”에 가깝고, 트리 전체 메모리는 여전히 노드 수에 비례해 증가할 수 있음을 초기 문서가 약하게만 언급 | 이해관계자가 “전체 VRAM이 상수”로 오해 → 설계 검증 기준 혼란 |
| P-2 | **ν_Ado**는 논문에서 “LUT 기반 계산 비용 누적”이며, 정책 출력 스칼라와 **에피소드 누적 상태**가 동시에 존재할 수 있음 | `NuVector` 필드 설계 시 상태/액션 혼동 |
| P-3 | **식 (5)**는 이론적으로는 19모듈 가중 합이나, 구현은 **top-k 희소 실행**으로 근사 — ref와 prod 경로의 **수학적 동등 조건**이 문서화되지 않음 | 애블레이션·정확도 재현 어려움 |
| P-4 | **V(z\*)가 라우팅 마스크와 무관**하다는 AC는 이상적 목표이며, 실제 네트워크는 컨텍스트 경로에 간접 의존할 수 있음 | 과도한 테스트 기준으로 인한 무한 튜닝 |

### 2.2 구조·일관성

| ID | 문제 | 영향 |
|----|------|------|
| P-5 | REQ 매트릭스의 `memory_accounting.py`가 **권장 디렉터리 목록에 없음** | 추적성 깨짐 |
| P-6 | 리스크 항목 **번호 불연속**(R1, R2, “Triton…”, R4) | 문서 품질·감사 대응 저하 |
| P-7 | **레이어 인덱스 0-base vs 1-base** 미명시(Table 3) | 모듈 매핑 오프바이원 |

### 2.3 학습·평가

| ID | 문제 | 영향 |
|----|------|------|
| P-8 | **Stage 1 손실**이 논문에 상세 없음 — 계획서만으로는 구현 목표 불명확 | 팀 내 구현 분기 |
| P-9 | PPO **GAE λ, clip ε, entropy/value coef** 미기재 | 재현성 저하 |
| P-10 | MATH/ARC 평가 시 **코드 실행 샌드박스**·오염 방지 언급 없음 | 안전·공정성 리스크 |
| P-11 | **부모 문자열 s_t**가 매우 길 때 임베딩/어텐션 메모리는 여전히 큼 — “KV-free”와 별개의 **시퀀스 메모리** 한계 | OOM 원인 분석 시 혼선 |

### 2.4 현실성

| ID | 문제 | 영향 |
|----|------|------|
| P-12 | Gemma 4 E4B·42층·PLE는 **논문 가정** — 공개 모델과 불일치 시 전체 매핑 재작업 | 일정·범위 폭발 |

---

## 3. 보완 사항

1. **메모리 클레임 정의 고정**  
   - **M1 (전이 단위)**: 한 번의 `transition()` 동안 **추가 KV 캐시를 경로 깊이만큼 저장하지 않음**을 측정한다.  
   - **M2 (트리 단위)**: `|V_metadata(D)|`는 노드당 `z*` 등 소형 텐서 합으로 증가할 수 있으나, 논문 주장대로 **동일 조건에서 기존 MCTS+KV 대비 우위**를 표·프로파일로 비교한다.

2. **상태 분리**  
   - `NuVector`: 메타정책이 **매 스텝 출력**하는 5축 제어 신호(스케일·온도·tol 등).  
   - `RuntimeBudgetState`: `ado_accumulated`, `remaining_flops` 등 **환경이 유지**하는 누적치.  
   - 문서·타입 모두에서 필드명을 분리한다.

3. **레이어 인덱싱**  
   - 코드베이스는 **0-base**로 통일하고, Table 3는 `docs/layer_mapping.md`에 “논문 레이어 L ↔ 코드 `layer_idx`” 표로 이중 기재한다.

4. **PUCT 표준식과의 정렬**  
   - 논문 식 (4)와 AlphaZero 계열 **PUCT**를 한 줄에 병기하고, 구현체는 `puct_variant: paper | alphazero` 플래그로 선택 가능하게 한다(단위 테스트 두 벌).

5. **디렉터리·매트릭스 동기화**  
   - `cts/perf/memory_accounting.py`, `cts/perf/profiler.py` 등 REQ에 나온 경로를 트리에 반드시 포함한다.

6. **리스크 ID 연속 번호** 및 각 항목에 **소유자·트리거 조건** 필드 추가(아래 §12).

7. **Stage 1** 후보 목표를 명시: (a) 고정점 잔차 최소화, (b) 자기일관성(self-consistency) \( \|z - f(z)\| \), (c) 보조 LM NLL(논문 재현 합의 후 1안 채택).

---

## 4. 추가 제안

- **아키텍처 어댑터 인터페이스**  
  `BaseCTSBackbone` 프로토콜: `num_layers`, `ple_style`, `forward_block(i, ...)`. Gemma 외 **Llama/Qwen 등**으로 스왑 시 `module_partition`만 교체.

- **솔버 플러그인**  
  `BroydenSolver` 기본 + `AndersonSolver` 옵션(논문 Future Work 정렬, R&D 플래그).

- **결정론적 평가 모드**  
  `eval_deterministic: true`일 때 MCTS/노이즈 시드 고정, `torch.use_deterministic_algorithms` 옵션 문서화.

- **프로파일 계약**  
  `profile_vram_latency.py`가 **CUDA peak allocated**, **동기 타이밍**, **warmup 반복 수**를 보고서에 고정 기재.

- **카나리 테스트**  
  매 야간: 소형 mock backbone으로 `transition + mcts_expand` 100회 무크래시.

- **문서 이중화**  
  본 `.md`는 “개선 기준서”, 원 `.txt`는 스냅샷 보관 — 단, 상충 시 **본 문서 우선**.

---

## 5. 개선된 실행 요약 및 성공 기준

### 5.1 최종 산출물

- **A.** CTS 코어 패키지(DEQ 전이, 라우팅 ref/Triton, MCTS, 메타정책, budget 상태).  
- **B.** Stage1 / Stage2 학습 엔트리포인트 및 **명시적 Stage1 손실 문서**.  
- **C.** 평가·프로파일·Iso-FLOP·베이스라인 스크립트.  
- **D.** 재현 번들: `configs/`, 시드, **측정 프로토콜**(§11), 체크포인트 스키마.  
- **E.** (신규) `docs/memory_definitions.md`, `docs/security_eval.md`.

### 5.2 성공 기준 (개정)

1. **KV 정책**: 트리 탐색 구현체가 **경로별 전체 시퀀스 KV를 저장하지 않는** 코드 경로를 기본으로 하며, 베이스라인 `mcts_kv_baseline`과 대비 프로파일에서 차이를 보고한다.  
2. **Broyden**: `max_iter=30`, `tol`을 `nu_ne`에서 매핑, 내부 FP32 / 외부 BF16.  
3. **MCTS**: `W=3`, 시뮬레이션 횟 기본값 논문과 일치(설정으로 변경 가능).  
4. **ν 연결**: 5축이 각각 PUCT, z0 노이즈, tol, 라우팅 온도, **누적 비용/ACT**에 연결됨을 설정 파일에 명시.  
5. **재현**: 하이퍼가 `configs/default.yaml`에 있고, **PPO 누락 항목(§10)** 포함.  
6. **측정**: 표1 유형 스윕 시 **M1/M2 정의**에 맞는 지표를 CSV에 함께 기록.

### 5.3 비목표

- 조상 희소 검색 어텐션 전체 제품화(프로토타입은 플래그 `experimental_ancestor_attn`).  
- 멀티 GPU 학습(논문 범위 외 — 필요 시 별도 확장 프로젝트).

---

## 6. 개선된 요구사항 추적 매트릭스

| 논문·부록 | 구현 단위 | 검증 방법 |
|-----------|-----------|-----------|
| §3.1 MDP, ν | `policy/meta_policy.py`, `cts/types.py` | 출력 shape, 범위, 런타임 budget과 분리 테스트 |
| §3.2 식 (1)–(3) | `perf/memory_accounting.py`, `perf/profiler.py` | M1/M2 프로토콜 스윕 |
| §4.1 식 (4) PUCT | `mcts/puct.py` | 논문식 vs AlphaZero식 단위 테스트 |
| §4.2 DEQ, 폴백 | `deq/broyden_forward.py`, `deq/transition.py` | 수렴/프루닝/잔차 로그 |
| §4.3 ACT, 디코딩 | `routing/lut_mac.json`, `latent/decode.py` | budget 초과 시 조기 정지 |
| §5.2 식 (5) | `routing/sparse_moe_ref.py`, `sparse_moe_triton.py` | ref≈triton, top-k vs full 합 비교 |
| §6 학습 | `train/stage1_warmup.py`, `train/stage2_ppo.py` | 손실·보상·수렴률 |
| §7 실험 | `eval/*`, `baselines/*` | 스크립트 원클릭 |
| 부록 A.2 | `configs/default.yaml` + `configs/README.md` | 항목별 대응표 |

---

## 7. 개선된 저장소 구조

`memory_accounting`·`backbone`·`docs`를 반영한 예시이다.

```text
cts/
  pyproject.toml
  configs/
    default.yaml
    ablation_no_ach.yaml
    ablation_static_5ht.yaml
    README.md                 # YAML ↔ 논문 부록 대응표
  docs/
    memory_definitions.md     # M1/M2 정의
    layer_mapping.md          # 논문 레이어 ↔ 코드 인덱스
    security_eval.md          # 평가 샌드박스·코드 실행 정책
  cts/
    types.py                  # NuVector, RuntimeBudgetState, TreeNode
    tokenizer_bridge.py
    backbone/
      protocol.py             # BaseCTSBackbone ABC
      gemma_adapter.py
      mock_tiny.py
    model/
      loader.py
      module_partition.py
      ple_rope.py
    latent/
      bottleneck.py
      universal_space.py
      decode.py
    routing/
      logits.py
      sparse_moe_ref.py
      sparse_moe_triton.py
    deq/
      broyden_forward.py
      solvers/anderson.py      # optional
      implicit_backward.py
      transition.py
    mcts/
      tree.py
      puct.py
      expand.py
      simulate.py
    policy/
      meta_policy.py
      lora_config.py
    critic/
      neuro_critic.py
    rewards/
      shaping.py
    perf/
      memory_accounting.py
      profiler.py
    train/
      stage1_warmup.py
      stage2_ppo.py
      ppo_core.py
    eval/
      math500.py
      arc_agi_text.py
      profile_vram_latency.py
      isoflop_matcher.py
    baselines/
      standard_ar.py
      native_think_cap.py
      mcts_kv_baseline.py
    utils/
      logging.py
      seed.py
      checkpoint.py
  tests/
  scripts/
```

---

## 8. 개선된 API·상태 모델

### 8.1 NuVector (정책 출력 — 매 스텝)

| 필드 | 역할 |
|------|------|
| `nu_da` | 프로세스 보상 스케일(또는 TD 가중) |
| `nu_5ht` | PUCT 탐색 계수 + z0 노이즈 σ 매핑 |
| `nu_ne` | Broyden tol 매핑 입력 |
| `nu_ach` | 라우팅 softmax 온도 (하한 ε) |
| `nu_ado_scale` | LUT 누적 가중(선택) — **에피소드 누적치 자체는 아님** |

### 8.2 RuntimeBudgetState (환경 누적)

- `ado_accumulated: float`  
- `flops_spent_step: float`  
- (선택) `wall_clock_ms_step`

### 8.3 TransitionResult (개정)

- `child_text: Optional[str]`  
- `z_star_child: Tensor`  
- `solver_stats: dict`  
- `prune: bool`  
- `budget: RuntimeBudgetState` (갱신 후 스냅샷)

### 8.4 transition() 시그니처 (개념)

```python
def transition(
    parent_text: str,
    branch_index: int,
    nu: NuVector,
    budget: RuntimeBudgetState,
    model_bundle,
    rng,
) -> TransitionResult: ...
```

---

## 9. Phase별 작업 (개정)

기존 Phase 0–12를 유지하되, 아래 **삽입·수정**을 적용한다.

- **Phase 0**  
  - **T0.5** `docs/memory_definitions.md` 작성 및 리뷰 승인.  
  - **T0.6** `docs/layer_mapping.md` 0-base 규약 확정.

- **Phase 1**  
  - **T1.5** `BaseCTSBackbone` 도입; Gemma 미가용 시 `mock_tiny`로 전 Phase 스모크.

- **Phase 3**  
  - **T3.6** 고정점 맵 \(f\)의 **Lipschitz/수축** 가정이 깨지는 경우 로깅 및 ν_ne 상한 자동 완화(옵션).

- **Phase 4**  
  - **T4.5** 문서화: **full softmax 합** 구현과 **top-k 희소** 구현의 의미적 차이 및 애블레이션 절차.

- **Phase 8**  
  - **T8.1 AC 개정**: “라우팅 마스크 변경 시 |ΔV| < δ”를 **느슨한 상한**(δ는 데이터 기반)으로 두거나, 마스크 고정 패널에서만 엄격 검증.

- **Phase 9**  
  - **T9.0** 프로젝트 차원에서 Stage1 손실 1안을 선택하고 `docs/stage1_objective.md`에 고정.  
  - 후보: (a) \(\|z - f(z)\|\) 최소화, (b) 자식 텍스트 NLL 보조, (c) teacher-generated 궤적 정렬.

- **Phase 11**  
  - **T11.6** MATH/코드 실행이 필요하면 `security_eval.md`에 따라 샌드박스 또는 비실행 채점만 허용.

- **Phase 12**  
  - **T12.4** 재현 패키지: 커밋 해시, `pip freeze`, CUDA 버전, **프로파일 warmup 횟수** 기록.

---

## 10. 하이퍼파라미터·설정 (확장)

`configs/default.yaml`에 아래를 **명시적으로** 추가한다(값은 논문·튜닝에 맞게 조정).

```yaml
# PPO (논문 미기재 항목 — 재현을 위해 필수)
ppo_clip_epsilon: 0.2
gae_lambda: 0.95
entropy_coef: 0.01
value_loss_coef: 0.5
max_grad_norm: 1.0

# PUCT
puct_variant: paper  # or alphazero

# Stage 1
stage1_loss: fixed_point_residual  # fixed_point_residual | lm_nll | combined
stage1_max_steps: ...

# Eval
eval_warmup_runs: 3
eval_deterministic: false
cuda_sync_for_timing: true

# Sequence memory (P-11 대응 — 모니터링)
max_anchor_tokens: null  # null = 모델 컨텍스트 한도까지
```

기존 논문 부록 연계 항목(dtype, Broyden, W, 시뮬 횟, LoRA rank, γ, lr, batch 등)은 **v1 문서와 동일**하게 유지한다.

---

## 11. 테스트·검증 (확장)

| 계층 | 항목 |
|------|------|
| 단위 | PUCT 두 변체, 라우팅 온도 단조성, Broyden 합성 수렴, IFT 소차원 그래드 |
| 계약 | `NuVector` vs `RuntimeBudgetState` 필드 혼선 방지 테스트 |
| 통합 | `transition` E2E, MCTS 10노드 mock |
| 성능 | M1/M2 프로파일 스윕, CSV 스키마 고정 |
| 회귀 | 야간 카나리(mock backbone) |
| 보안 | (해당 시) 샌드박스 위반 시 평가 중단 |

---

## 12. 리스크 레지스터 (정정·확장)

| ID | 리스크 | 완화 | 소유(예시) |
|----|--------|------|------------|
| R1 | Gemma 4 E4B 미공개/불일치 | `BaseCTSBackbone` + 공개 모델 우선 검증 | ML Lead |
| R2 | Broyden 비수렴 | ν_ne 스케줄, 완화된 f, Anderson 옵션 | DEQ Owner |
| R3 | Triton 개발 지연/불일치 | ref를 정확도 골드로 유지, 커널은 선택적 | Infra |
| R4 | Iso-FLOP 정의 분쟁 | 카운터 규칙 문서·코드 고정, 베이스라인 동일 규칙 | Eval Owner |
| R5 | 긴 s_t로 인한 시퀀스 메모리 OOM | `max_anchor_tokens`, 요약/슬라이딩(실험 플래그) | Systems |
| R6 | Stage1 목표 불명확 | `stage1_objective.md` 합의 | PM/Research |

---

## 13. 보안·데이터·윤리

- **평가 데이터**: MATH 등 공개 벤치 사용 시 **라이선스·인용** 준수.  
- **훈련 데이터**: OpenMathInstruct 등 **사용 약관** 확인, 버전·스냅샷 고정.  
- **코드 실행**: 모델 생성 코드를 실행하는 평가는 **격리 환경** 또는 금지 정책을 `security_eval.md`에 명시.  
- **로그**: 사용자 프롬프트·중간 “생각” 로깅 시 **개인정보 마스킹** 옵션.

---

## 14. 마일스톤·의존성

| 마일스톤 | 기간(가이드) | 산출물 |
|----------|----------------|--------|
| M1 | 주 0–2 | Phase 0–1 + backbone mock + layer_mapping |
| M2 | 주 2–4 | DEQ ref + memory_definitions 승인 |
| M3 | 주 4–6 | 라우팅 ref + 메타정책 + PUCT |
| M4 | 주 6–8 | ACT + MCTS 통합 |
| M5 | 주 8–10 | Stage1 목표 확정 + Stage2 PPO |
| M6 | 주 10–12 | 벤치·프로파일·CI·문서 패키지 |

의존성: M2→M3→M4→M5→M6 순서 고정.

---

## 15. Definition of Done

- [ ] §6 매트릭스의 모든 구현 경로가 저장소에 존재  
- [ ] M1/M2 메모리 정의 문서화 및 프로파일 스크립트가 해당 정의를 따름  
- [ ] `NuVector`와 `RuntimeBudgetState` 분리 및 테스트 통과  
- [ ] `configs/default.yaml` + `README.md` 대응표 완비(PPO 확장 항목 포함)  
- [ ] 표1·표2 스타일 원클릭 재현  
- [ ] 애블레이션 ≥2 (dense routing, static ν_5ht)  
- [ ] 비수렴·NaN·OOM 알람 임계치  
- [ ] (선택) Triton Ada 경로 가속 입증  
- [ ] `security_eval.md`가 평가 런북에 링크됨

---

## 16. 부록: 의사코드

```python
def cts_transition(parent_text, nu, budget, rng):
    ctx = encode_and_rope_once(parent_text)
    z = init_z(parent_text) + noise_sigma(nu.nu_5ht) * rng.normal()
    z_star, info = broyden_solve(
        lambda zz: f_theta_nu(zz, ctx, nu),
        z,
        tol=map_tol(nu.nu_ne),
        max_iter=30,
    )
    if not info.converged:
        return TransitionResult(prune=True, ...)
    budget = accumulate_ado(budget, info, lut, nu)
    if budget.ado_accumulated > tau_budget:
        z_star = early_halt_state(z_star, ...)
    child_text = lm_decode_from_zstar(z_star, ctx)
    return TransitionResult(
        child_text=child_text,
        z_star_child=z_star,
        budget=budget,
        prune=False,
        solver_stats=info,
    )
```

---

**문서 끝.**  
*기존 `NeurIPS_2026_CTS_FULL_DEVELOPMENT_PLAN_ko.txt`와 내용이 다를 경우 본 Markdown을 구현·검증의 우선 기준으로 한다.*
