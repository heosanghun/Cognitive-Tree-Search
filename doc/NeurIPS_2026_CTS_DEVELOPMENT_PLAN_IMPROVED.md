# Cognitive Tree Search (CTS) 개발 계획서 — 분석·비평·개선 통합본 v2.0

| 항목 | 내용 |
|------|------|
| 문서 ID | CTS-DEVPLAN-IMPROVED-003 |
| 기반 문서 | 논문 PDF + `NeurIPS_2026_CTS_FULL_DEVELOPMENT_PLAN_ko.txt` |
| 갱신일 | 2026-04-10 |
| 목적 | 논문 PDF를 최종 기준으로 삼아 **문제점·보완점·제안**을 반영한 구현 기준서 |
| 대상 논문 | *Cognitive Tree Search: KV-Cache-Free Per-Node O(1) Transitions for System 2 Inference via Deep Equilibrium Models* (NeurIPS 2026) |

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

기존 개발 계획서는 **모듈 분해·Phase 구조·부록 하이퍼 연계**가 명확하고, 논문의 DEQ–MCTS–ν–LUT 축을 코드 경로로 잘 매핑한다. **v2.0에서는 논문 PDF를 직접 대조**하여 아래 영역의 모호성·누락·내부 불일치를 정정한다.

### 1.1 논문 PDF 기반 핵심 변경사항 (v1→v2)

| 변경 영역 | v1 상태 | v2 정합 |
|-----------|---------|---------|
| **ν 벡터 명명법** | 신경전달물질 (DA, 5HT, NE, ACh, Ado) | **표준 ML**: νval, νexpl, νtol, νtemp, νact |
| **FAISS Context Window** | 미언급 | 논문 §4.4 — 핵심 컴포넌트로 추가 |
| **Wproj 프로젝션** | 미언급 | 논문 §4.5 — Latent-to-Text 디코딩 |
| **병렬 배치 DEQ** | 미명시 | 논문 §4.1 — W 형제 동시 솔브 |
| **L-Broyden FP32** | 미명시 | 논문 §5.2 — low-rank 버퍼 FP32 유지 |
| **보상 함수** | 불명확 | Eq.(5): Rtotal = 1{correct} − λhalt · T |
| **벤치마크 범위** | MATH, ARC | **GSM8K, HumanEval** 추가 (논문 표2) |
| **Vision/Audio 오프로딩** | 미언급 | ~0.9 GB 절약 (§7.1) |
| **수렴 통계** | 미언급 | 부록 C: 97.3% 수렴, 11.2 반복, γ≈0.92 |
| **K=64 검증** | 미언급 | 부록 H: 91.4% 심볼릭 복원율 |

---

## 2. 식별된 문제점

### 2.1 개념·서술

| ID | 문제 | 영향 | v2 해결 |
|----|------|------|---------|
| P-1 | O(1) 메모리 정의 부족 | 검증 기준 혼란 | M1/M2 정의 유지 + 논문 Eq.(3) 직접 인용 |
| P-2 | ν 벡터 명명법 불일치 | 코드-논문 괴리 | **논문 정합: νval/νexpl/νtol/νtemp/νact** |
| P-3 | 식 (6) top-k vs full 차이 미문서화 | 재현 어려움 | dense ref vs top-k 차이 테스트 유지 |
| P-4 | V(z\*) 라우팅 불변 가정 과도 | 무한 튜닝 | 완화된 AC (δ 상한) |
| **P-13** | **FAISS Context Window 미반영** | 논문 핵심 누락 | §4.4 기반 신규 모듈 |
| **P-14** | **Wproj 프로젝션 미반영** | 디코딩 경로 불완전 | §4.5 기반 구현 |
| **P-15** | **병렬 배치 DEQ 미반영** | 25ms 지연 미달 | §4.1 배치화 구현 |
| **P-16** | **L-Broyden FP32 미분리** | 수치 발산 위험 | §5.2 FP32 버퍼 분리 |
| **P-17** | **보상 함수 불명확** | 학습 목표 모호 | Eq.(5) 직접 구현 |
| **P-18** | **GSM8K/HumanEval 벤치 부재** | 논문 표2 미재현 | 벤치 스크립트 추가 |
| **P-19** | **Vision/Audio 오프로딩 미반영** | VRAM 예산 부정확 | gemma_loader 확장 |

### 2.2 구조·일관성

| ID | 문제 | 영향 |
|----|------|------|
| P-5 | REQ 매트릭스 `memory_accounting.py` 누락 | 추적성 깨짐 (해결됨) |
| P-6 | 리스크 번호 불연속 | 문서 품질 (해결됨) |
| P-7 | 레이어 인덱싱 미명시 | 오프바이원 (해결됨) |

### 2.3 학습·평가

| ID | 문제 | 영향 |
|----|------|------|
| P-8 | Stage 1 손실 상세 부족 | 구현 분기 (논문: IFT 잔차 ∥f(z*)−z*∥²₂) |
| P-9 | PPO 세부 하이퍼 미기재 | 재현성 (논문 Table 4로 보완) |
| P-10 | 코드 실행 샌드박스 미언급 | 안전 리스크 |
| P-11 | 긴 s_t 시퀀스 메모리 한계 | OOM 혼선 |

### 2.4 현실성

| ID | 문제 | 영향 |
|----|------|------|
| P-12 | Gemma 4 E4B 모델 가용성 | 매핑 재작업 |

---

## 3. 보완 사항

1. **메모리 클레임 정의 고정 (논문 Eq.3 기반)**
   - **M1 (전이 단위)**: `O_active(1)` — DEQ 전이 중 추가 KV 캐시 비저장
   - **M2 (트리 단위)**: `O_history(N)` — FAISS 인덱스 + 메타데이터, 노드당 수 KB
   - 논문: `V^CTS_Total(D) = V_Weights + V_Metadata(D) + O_active(1) + O_history(N)`

2. **ν 명명법 통일 (논문 §2.3)**
   - `NuVector` 필드: `nu_val`, `nu_expl`, `nu_tol`, `nu_temp`, `nu_act`
   - 레거시 별칭(DA, 5HT, NE, ACh, Ado) 코드에서 점진 교체
   - `RuntimeBudgetState`: `mac_accumulated`, `remaining_flops` 등 환경 누적

3. **FAISS Latent Space Context Window (논문 §4.4)**
   - 모든 z*를 mean-pool → FAISS IndexFlatIP에 추가
   - t > 10: Top-3 시맨틱 조상 벡터 검색
   - prepended global soft-prefix로 주입 (p-RoPE 격리)

4. **Latent-to-Text Wproj (논문 §4.5)**
   - Wproj ∈ R^{d×dmodel}: K=64 잠복 토큰 → continuous soft prompt
   - '<\|think\|>' 바이패스 → [Final Answer] 시맨틱 공간 직접 매핑

5. **병렬 배치 DEQ (논문 §4.1)**
   - W 형제에 독립 노이즈 주입 → 단일 배치 DEQ solve
   - Neuro-Critic 동기 평가 → Q(s,a) 산출

6. **L-Broyden FP32 (논문 §5.2)**
   - low-rank 역 야코비안 버퍼 FP32, 입출력 BF16
   - 부모→자식 역 야코비안 상속으로 수렴 가속

7. **보상 함수 (Eq.5)**
   - `R_total = 1{correct_answer} − λ_halt · T`
   - T = 실제 터미널 트리 깊이 (νact 직접 패널티 아님 — reward hacking 방지)

8. **레이어 인덱싱** — 0-base 통일, `docs/layer_mapping.md` 이중 기재 (유지)

9. **PUCT** — 논문 Eq.(4) + AlphaZero 변체 (유지)

10. **디렉터리·매트릭스 동기화** (유지)

---

## 4. 추가 제안

- **아키텍처 어댑터 인터페이스**
  `BaseCTSBackbone` 프로토콜 유지. Gemma 외 Llama/Qwen 등 스왑.

- **솔버 플러그인**
  `BroydenSolver` 기본 + `AndersonSolver` 옵션.

- **결정론적 평가 모드** (유지)

- **프로파일 계약** — CUDA peak allocated, 동기 타이밍, warmup 반복 수 (유지)

- **카나리 테스트** — mock backbone 100회 무크래시 (유지)

- **FAISS 의존성 관리**
  `pyproject.toml`에 `faiss-cpu` 추가 (선택: `faiss-gpu`). CI에서 mock FAISS로 대체.

- **벤치마크 확장**
  GSM8K (`datasets` HF), HumanEval (로컬 오프라인 `security_eval.md` 정책 준수).

---

## 5. 개선된 실행 요약 및 성공 기준

### 5.1 최종 산출물

- **A.** CTS 코어 패키지(DEQ, 라우팅 ref/Triton, MCTS, 메타정책, budget 상태, **FAISS Context Window**, **Wproj**)
- **B.** Stage1 / Stage2 학습 엔트리포인트 + Stage1 손실(IFT 잔차) + 보상 Eq.(5)
- **C.** 평가·프로파일·Iso-FLOP·베이스라인 스크립트 (**MATH/GSM8K/AIME/ARC/HumanEval 5종**)
- **D.** 재현 번들: `configs/`, 시드, 측정 프로토콜, 체크포인트 스키마
- **E.** `docs/memory_definitions.md`, `docs/security_eval.md`

### 5.2 성공 기준 (개정 v2)

1. **KV 정책**: 경로별 KV 비저장, FAISS로 이력 관리 → O(1) 활성 + O(N) 이력
2. **Broyden**: `max_iter=30`, `tol` = νtol 매핑, **L-Broyden FP32 버퍼**
3. **MCTS**: W=3, 병렬 배치 DEQ → ~25ms/node
4. **ν 연결**: 5축이 논문 명명법으로 PUCT(νexpl), 노이즈(νexpl), tol(νtol), 온도(νtemp), ACT(νact)에 연결
5. **재현**: 하이퍼가 `configs/default.yaml`에 논문 Table 4 전 항목 포함
6. **측정**: 표1 스윕 + 표2 5종 벤치마크(5 seeds, 95% CI)
7. **FAISS**: t>10 시 Top-3 검색, 노드당 메모리 < 10KB
8. **Wproj**: K=64 → soft prompt 주입, '<\|think\|>' 바이패스
9. **보상**: Eq.(5) Rtotal = 1{correct} − λhalt · T

### 5.3 비목표

- 멀티 GPU 학습 (논문 범위 외 — §8 Discussion)
- D > 200 초장기 탐색 시 FAISS 시맨틱 희석 문제 (논문 한계점)

---

## 6. 개선된 요구사항 추적 매트릭스

| 논문·부록 | 구현 단위 | 검증 방법 |
|-----------|-----------|-----------|
| §3.1 MDP, ν | `policy/meta_policy.py`, `cts/types.py` | νval/expl/tol/temp/act shape·범위 테스트 |
| §3.2 Eq.(1)–(3) | `perf/memory_accounting.py`, `perf/profiler.py` | M1/M2 프로토콜 스윕 |
| §4.1 Eq.(4) PUCT + **병렬배치** | `mcts/puct.py`, **`deq/transition.py` (batched)** | PUCT 단위 + W배치 지연 |
| §4.2 DEQ, L-Broyden | `deq/broyden_forward.py` | 수렴/FP32/잔차 로그 |
| §4.3 ACT, Eq.(5) 보상 | `routing/lut_mac.json`, `rewards/shaping.py` | budget 초과 시 조기 정지 |
| **§4.4 FAISS Context** | **`latent/faiss_context.py`** | Top-3 검색, 메모리 프로파일 |
| **§4.5 Wproj 프로젝션** | **`latent/bottleneck.py`** | shape, 정보 보존율 |
| §5.1 19-모듈 매핑 | `model/module_partition.py` | 42→19 블록 매핑 검증 |
| §5.2 Eq.(6) 라우팅 | `routing/sparse_moe_ref.py`, `sparse_moe_triton.py` | ref≈triton, top-k vs full |
| §5.3 Neuro-Critic νval | `critic/neuro_critic.py` | V(z*) 출력 검증 |
| §6.1 Stage1 (10K) | `train/stage1_warmup.py` | IFT 잔차 수렴 |
| §6.2 Stage2 PPO (5K) | `train/stage2_ppo.py` | PPO 보상 곡선 |
| §7 표1 VRAM/지연 | `eval/profile_vram_latency.py` | CSV 스키마 고정 |
| §7 표2 5종 벤치 | `eval/math500.py`, **`eval/gsm8k.py`**, `eval/arc_agi_text.py`, **`eval/humaneval.py`** | Iso-FLOP 보고서 |
| Table 4 하이퍼 | `configs/default.yaml` | 항목별 대응표 |
| 부록 C 수렴 | `deq/broyden_forward.py` | 97.3% 수렴, 11.2 iter |
| 부록 H K=64 | `latent/bottleneck.py` | 91.4% 복원율 테스트 |

---

## 7. 개선된 저장소 구조

`v2.0` — FAISS, Wproj, 벤치 확장 반영

```text
cts/
  pyproject.toml
  configs/
    default.yaml                # 논문 Table 4 전 항목
    ablation_no_ach.yaml
    ablation_static_5ht.yaml
    paper_parity.yaml
    experiment_paper_protocol.yaml
    data_paths.yaml
    README.md                   # YAML ↔ 논문 부록 대응표
  doc/
    memory_definitions.md       # M1/M2 + 논문 Eq.(3)
    layer_mapping.md            # 42블록 → 19모듈 (0-base)
    security_eval.md            # 평가 샌드박스 정책
    stage1_objective.md         # IFT 잔차 손실 확정
  cts/
    types.py                    # NuVector(val/expl/tol/temp/act), RuntimeBudgetState
    backbone/
      protocol.py               # BaseCTSBackbone ABC
      gemma_adapter.py
      mock_tiny.py
      rope_contract.py
    model/
      loader.py
      gemma_loader.py           # + offload_vision_audio 옵션
      module_partition.py
    latent/
      bottleneck.py             # + Wproj 프로젝션
      faiss_context.py          # ★ FAISS Latent Space Context Window
    routing/
      sparse_moe_ref.py
      sparse_moe_triton.py
      lut_mac.json
    deq/
      broyden_forward.py        # + L-Broyden FP32
      solvers/anderson.py
      transition.py             # + 병렬 배치 DEQ
      gemma_latent_forward.py
    mcts/
      tree.py
      puct.py
      episode.py
      deep_rollout.py
      mcts_deep_rollout.py
      critic_reward.py
    policy/
      meta_policy.py
    critic/
      neuro_critic.py
    rewards/
      shaping.py                # Eq.(5) R_total
    perf/
      memory_accounting.py
      profiler.py
    train/
      stage1_warmup.py
      stage1_openmath_train.py
      stage2_ppo.py
      stage2_ppo_train.py
      ppo_core.py
    eval/
      math500.py
      gsm8k.py                  # ★ 신규
      arc_agi_text.py
      humaneval.py              # ★ 신규
      profile_vram_latency.py
      isoflop_matcher.py
      report_isoflop.py
      flops_contract.py
      gemma_predict.py
      prompt_format.py
      think_prompt.py
      kv_measured.py
    baselines/
      mcts_kv_baseline.py
    utils/
      seed.py
      config.py
      repro_seed.py
      repro_snapshot.py
  tests/
  scripts/
```

---

## 8. 개선된 API·상태 모델

### 8.1 NuVector (정책 출력 — 매 스텝, 논문 §2.3 정합)

| 필드 | 논문 기호 | 역할 |
|------|-----------|------|
| `nu_val` | νval | Neuro-Critic V(z*) — State Value |
| `nu_expl` | νexpl | PUCT 탐색 계수 + z0 노이즈 σ 매핑 |
| `nu_tol` | νtol | Broyden 수렴 허용치 [10⁻⁴, 10⁻²] |
| `nu_temp` | νtemp | 라우팅 softmax 온도 (토큰별 적용) |
| `nu_act` | νact | ACT 중단 임계치 — τbudget · νact,t |

### 8.2 RuntimeBudgetState (환경 누적)

- `mac_accumulated: float` — 누적 MACs
- `terminal_depth: int` — 현재 트리 깊이 T
- `flops_spent_step: float`
- (선택) `wall_clock_ms_step`

### 8.3 TransitionResult (개정 v2)

- `child_text: Optional[str]`
- `z_star_child: Tensor` — K=64 잠복 토큰
- `solver_stats: dict` — iterations, converged, residual
- `prune: bool`
- `budget: RuntimeBudgetState` (갱신 후 스냅샷)
- `faiss_retrieved: Optional[Tensor]` — Top-3 조상 벡터 (t>10)

### 8.4 transition() 시그니처 (v2)

```python
def transition(
    parent_text: str,
    branch_index: int,
    nu: NuVector,
    budget: RuntimeBudgetState,
    model_bundle,
    faiss_index,        # ★ FAISS Context Window
    rng,
    batch_size: int = 1,  # ★ 병렬 배치 (W)
) -> Union[TransitionResult, List[TransitionResult]]: ...
```

### 8.5 FAISS Context Window API (신규)

```python
class LatentContextWindow:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.vectors = []

    def add(self, z_star: Tensor):
        pooled = z_star.mean(dim=-2).cpu().numpy()
        self.index.add(pooled)
        self.vectors.append(pooled)

    def retrieve(self, z_star: Tensor, k: int = 3) -> Optional[Tensor]:
        if self.index.ntotal <= 10:
            return None
        query = z_star.mean(dim=-2).cpu().numpy()
        _, indices = self.index.search(query, k)
        return torch.stack([torch.from_numpy(self.vectors[i]) for i in indices[0]])
```

---

## 9. Phase별 작업 (개정 v2)

기존 Phase 0–12를 유지하되, 논문 PDF 정합을 위한 **신규 Phase 삽입**.

- **Phase 0**
  - T0.5 `docs/memory_definitions.md` — Eq.(3) 기반 M1/M2 정의
  - T0.6 `docs/layer_mapping.md` — 42블록→19모듈 0-base
  - **T0.7** ν 명명법 통일 계획서 (DA→val 등 매핑 문서)

- **Phase 1**
  - T1.5 `BaseCTSBackbone` + mock
  - **T1.6** Vision/Audio 오프로딩 옵션 (`gemma_loader.py`)

- **Phase 3**
  - T3.6 contractive mapping 로깅
  - **T3.7** L-Broyden FP32 역 야코비안 버퍼 분리
  - **T3.8** 부모→자식 역 야코비안 상속 메커니즘

- **Phase 3.5 (신규) — FAISS Latent Space Context Window**
  - **T3.5.1** `latent/faiss_context.py` 구현
  - **T3.5.2** mean-pool → FAISS 인덱싱
  - **T3.5.3** Top-3 검색 + soft-prefix 주입
  - **T3.5.4** p-RoPE 격리 검증
  - **T3.5.5** 단위 테스트 (검색 정확도, 메모리 프로파일)

- **Phase 4**
  - T4.5 top-k vs full 문서화
  - **T4.6** 병렬 배치 DEQ 구현 (W 형제 동시 솔브)

- **Phase 5.5 (신규) — Latent-to-Text Wproj**
  - **T5.5.1** Wproj ∈ R^{d×dmodel} 프로젝션 레이어
  - **T5.5.2** '<\|think\|>' 바이패스 로직
  - **T5.5.3** K=64 정보 보존 테스트 (목표 91.4%)

- **Phase 8**
  - T8.1 AC 검증 완화 (유지)

- **Phase 9**
  - T9.0 Stage1 손실 확정 — **IFT 잔차 ∥f(z*)−z*∥²₂** (논문 §6.1)
  - **T9.1** 보상 함수 Eq.(5) 구현: Rtotal = 1{correct} − λhalt · T

- **Phase 11**
  - T11.6 security_eval.md (유지)
  - **T11.7** GSM8K 벤치마크 스크립트
  - **T11.8** HumanEval 벤치마크 스크립트 (오프라인, 샌드박스)
  - **T11.9** 5 seeds × 95% CI 자동화

- **Phase 12**
  - T12.4 재현 패키지 (유지)
  - **T12.5** ν 명명법 코드 전체 마이그레이션 완료 검증

---

## 10. 하이퍼파라미터·설정 (확장 — 논문 Table 4 정합)

`configs/default.yaml`에 아래를 **명시적으로** 추가한다.

```yaml
# === 논문 Table 4 하이퍼파라미터 ===
ppo_lr: 3e-5
critic_lr: 1e-4
ppo_clip_epsilon: 0.2
act_halting_penalty: 0.05     # λ_halt
discount_gamma: 0.99
gae_lambda: 0.95
lora_rank: 8
lora_alpha: 16

# === 보완 항목 ===
entropy_coef: 0.01
value_loss_coef: 0.5
max_grad_norm: 1.0

# === PUCT ===
puct_variant: paper  # or alphazero
branching_width: 3   # W

# === DEQ ===
latent_tokens: 64    # K
broyden_max_iter: 30
broyden_fp32_buffer: true   # ★ L-Broyden FP32
solver_tol_range: [1e-4, 1e-2]  # νtol 범위

# === FAISS Context Window ===
faiss_enabled: true
faiss_retrieval_k: 3
faiss_min_steps: 10   # t > 10에서 활성화

# === Wproj ===
wproj_enabled: true
think_mode_bypass: true  # '<|think|>' 바이패스

# === Vision/Audio ===
offload_vision_audio: true  # ~0.9 GB 절약

# === Stage 1 ===
stage1_loss: fixed_point_residual  # IFT 잔차 (논문 §6.1)
stage1_examples: 10000
stage1_data: openmathinstruct2

# === Stage 2 ===
stage2_prompts: 5000
stage2_data: math_aime

# === Eval ===
eval_warmup_runs: 3
eval_deterministic: false
eval_seeds: 5
eval_ci_level: 0.95
cuda_sync_for_timing: true

# === 보상 ===
reward_formula: binary_minus_depth  # Eq.(5)

# === Native Think ===
enable_thinking: false  # 논문 §7.1
max_new_tokens: 64      # CTS 표현 병목과 동일
```

---

## 11. 테스트·검증 (확장 v2)

| 계층 | 항목 |
|------|------|
| 단위 | PUCT 두 변체, 라우팅 온도 단조성, Broyden 합성 수렴, IFT 소차원 그래드 |
| 단위 (신규) | **FAISS 검색 정확도**, **Wproj shape 검증**, **병렬배치 W 일관성** |
| 계약 | `NuVector`(val/expl/tol/temp/act) vs `RuntimeBudgetState` 필드 분리 |
| 통합 | `transition` E2E + FAISS 컨텍스트, MCTS 10노드 mock |
| 성능 | M1/M2 프로파일 스윕, L-Broyden FP32 vs BF16 수렴 비교 |
| 벤치 | MATH/GSM8K/AIME/ARC/HumanEval 5종 × 5 seeds |
| 회귀 | 야간 카나리(mock backbone) |
| 보안 | 샌드박스 위반 시 평가 중단 |
| 수렴 | Broyden 수렴률 ≥ 97%, 평균 반복 ≤ 15 (부록 C) |
| 정보보존 | K=64 심볼릭 복원율 ≥ 90% (부록 H) |

---

## 12. 리스크 레지스터 (정정·확장 v2)

| ID | 리스크 | 완화 | 소유(예시) |
|----|--------|------|------------|
| R1 | Gemma 4 E4B 미공개/불일치 | `BaseCTSBackbone` + mock 우선 | ML Lead |
| R2 | Broyden 비수렴 | L-Broyden FP32, νtol 스케줄, Anderson | DEQ Owner |
| R3 | Triton 개발 지연 | ref 골드 유지 | Infra |
| R4 | Iso-FLOP 정의 분쟁 | 카운터 문서·코드 고정 | Eval Owner |
| R5 | 긴 s_t OOM | `max_anchor_tokens` | Systems |
| R6 | Stage1 목표 불명확 | IFT 잔차 (논문 확정) | Research |
| **R7** | **FAISS 시맨틱 희석 (D>200)** | 논문 §8 한계 인정, 검색 품질 모니터 | Research |
| **R8** | **K=64 정보 손실** | 91.4% 복원율 테스트, K=128 옵션 | Research |
| **R9** | **병렬배치 메모리 초과 (큰 W)** | W≤5 제한, 동적 배치 축소 | Systems |
| **R10** | **FAISS 의존성 (faiss-cpu/gpu)** | pip 옵션, mock fallback | Infra |

---

## 13. 보안·데이터·윤리

- **평가 데이터**: MATH, GSM8K, ARC, HumanEval 등 공개 벤치 **라이선스·인용** 준수
- **HumanEval**: 논문 §7.1에 따라 **로컬 오프라인 실행**, `security_eval.md` 정책 준수
- **훈련 데이터**: OpenMathInstruct-2 **사용 약관** 확인, 버전·스냅샷 고정
- **코드 실행**: 모델 생성 코드 실행은 **격리 환경** 또는 금지 정책
- **로그**: 개인정보 마스킹 옵션
- **Native Think 비활성화**: `enable_thinking=False` 시스템 프롬프트 (논문 §7.1)

---

## 14. 마일스톤·의존성 (개정 v2)

| 마일스톤 | 기간(가이드) | 산출물 |
|----------|-------------|--------|
| M1 | 주 0–2 | Phase 0–1 + backbone mock + layer_mapping + ν 명명법 |
| M2 | 주 2–4 | DEQ ref + L-Broyden FP32 + memory_definitions |
| **M2.5** | 주 3–5 | **FAISS Context Window** + **Wproj 프로젝션** |
| M3 | 주 4–6 | 라우팅 ref + 메타정책 + PUCT + 병렬배치 DEQ |
| M4 | 주 6–8 | ACT + MCTS 통합 + 보상 Eq.(5) |
| M5 | 주 8–10 | Stage1 (10K) + Stage2 PPO (5K) |
| M6 | 주 10–12 | 벤치 5종 + 프로파일 + CI + 문서 패키지 |

의존성: M2→M2.5→M3→M4→M5→M6 순서 고정.

---

## 15. Definition of Done (v2)

- [ ] §6 매트릭스의 모든 구현 경로가 저장소에 존재
- [ ] ν 명명법이 논문 정합 (νval/νexpl/νtol/νtemp/νact)
- [ ] M1/M2 메모리 정의 + Eq.(3) 직접 인용
- [ ] `NuVector`와 `RuntimeBudgetState` 분리 및 테스트 통과
- [ ] **FAISS Latent Space Context Window** 구현 + 테스트
- [ ] **Wproj Latent-to-Text Projection** 구현 + 91.4% 복원 테스트
- [ ] **병렬 배치 DEQ** (W 형제 동시 솔브) + 지연 프로파일
- [ ] **L-Broyden FP32** 버퍼 분리 + 수렴 안정성 테스트
- [ ] **보상 함수 Eq.(5)** 구현
- [ ] `configs/default.yaml` — 논문 Table 4 전 항목 + PPO 확장
- [ ] 표1 스타일 원클릭 재현 (VRAM 16.7GB 평탄)
- [ ] 표2 5종 벤치 (MATH/GSM8K/AIME/ARC/HumanEval) × 5 seeds × 95% CI
- [ ] 애블레이션 ≥3 (dense routing, static νexpl, w/o FAISS Context)
- [ ] Broyden 수렴률 ≥ 97%, 평균 반복 ≤ 15
- [ ] Vision/Audio 오프로딩 옵션
- [ ] `security_eval.md` 링크 + HumanEval 샌드박스
- [ ] 비수렴·NaN·OOM 알람 임계치

---

## 16. 부록: 의사코드 (논문 정합 v2)

```python
def cts_transition_batch(parent_text, nu, budget, faiss_ctx, rng, W=3):
    """논문 §4.1 병렬 배치 DEQ + §4.4 FAISS Context"""
    ctx = encode_and_rope_once(parent_text)

    # FAISS Context Window (§4.4) — t > 10
    history_prefix = faiss_ctx.retrieve(budget.current_z_star, k=3)
    if history_prefix is not None:
        ctx = prepend_soft_prefix(ctx, history_prefix)  # p-RoPE 격리

    # W개 형제에 독립 노이즈 주입 (§4.1)
    z_batch = []
    for _ in range(W):
        z = init_z(parent_text) + noise_sigma(nu.nu_expl) * rng.normal()
        z_batch.append(z)
    z_batch = torch.stack(z_batch)  # [W, K, d]

    # 병렬 배치 Broyden solve (L-Broyden, FP32 버퍼)
    z_star_batch, info_batch = broyden_solve_batch(
        lambda zz: f_theta_nu(zz, ctx, nu),
        z_batch,
        tol=map_tol(nu.nu_tol),
        max_iter=30,
        parent_jacobian=budget.parent_inv_jacobian,  # 상속
        fp32_buffer=True,  # L-Broyden
    )

    results = []
    for i in range(W):
        if not info_batch[i].converged:
            results.append(TransitionResult(prune=True, ...))
            continue

        budget_i = accumulate_mac(budget, info_batch[i], lut, nu)
        if budget_i.mac_accumulated > tau_budget * nu.nu_act:
            z_star_batch[i] = early_halt_state(z_star_batch[i])

        # FAISS에 새 z* 추가
        faiss_ctx.add(z_star_batch[i])

        # Latent-to-Text Wproj (§4.5) — '<|think|>' 바이패스
        child_text = wproj_decode(z_star_batch[i], ctx, model)

        results.append(TransitionResult(
            child_text=child_text,
            z_star_child=z_star_batch[i],
            budget=budget_i,
            prune=False,
            solver_stats=info_batch[i],
        ))

    # Neuro-Critic 동기 평가 (§5.3)
    q_values = neuro_critic(z_star_batch)  # V(z*) = νval

    return results, q_values
```

---

**문서 끝.**
*기존 문서와 내용이 다를 경우 **논문 PDF** → **본 문서** → 초안 `.txt` 순서를 구현·검증의 우선 기준으로 한다.*
