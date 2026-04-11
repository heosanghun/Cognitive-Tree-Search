# 논문 vs 코드 100% 정합성 검증 보고서

**검증 기준:** NeurIPS 2026 논문 PDF 원문 (13페이지 + 부록)
**검증 일시:** 2026-04-11
**검증 방법:** 논문의 모든 수식, 테이블, 하이퍼파라미터, 알고리즘을 코드와 1:1 대조

---

## 종합 결과

| 범주 | 일치 항목 | 불일치 항목 | 정합률 |
|------|:---------:|:----------:|:------:|
| 아키텍처 (§2~§5) | 18 | 4 | ~82% |
| 학습 (§6) | 6 | 3 | ~67% |
| 실험/평가 (§7) | 5 | 5 | ~50% |
| 하이퍼파라미터 (Table 4) | 13 | 0 | 100% |
| **전체** | **42** | **12** | **~78%** |

---

## A. 완전 일치 항목 (42개)

### 수식 (Eq.1~6)

| 수식 | 논문 | 코드 | 파일 | 상태 |
|------|------|------|------|:----:|
| Eq.(1) | V^MCTS = Σ V_KV-Cache | KV 분석적 계산 | `baselines/mcts_kv_baseline.py` | ✅ |
| Eq.(2) | z* = f_{θ,ν}(z*, s₀ ⊕ Hₜ) | `broyden_fixed_point(phi, z0)` | `deq/transition.py:96` | ✅ |
| Eq.(3) | V^CTS = V_W + V_M + O(1) + O(N) | FAISS O(N) + DEQ O(1) | `deq/transition.py` + `latent/faiss_context.py` | ✅ |
| Eq.(4) | PUCT with νexpl, P=1/W | `puct_score(..., nu_expl, prior)` | `mcts/puct.py` | ✅ |
| Eq.(5) | R = 1{correct} − λ·T | `paper_reward(correct, T, 0.05)` | `rewards/shaping.py` | ✅ |
| Eq.(6) | Softmax(Wg·z*/νtemp) Top-k | `routing_weights(z, w_g, nu_temp)` | `routing/sparse_moe_ref.py` | ✅ |

### 아키텍처 구성요소

| 항목 | 논문 | 코드 | 상태 |
|------|------|------|:----:|
| ν = [νval, νexpl, νtol, νtemp, νact] | §2.3 | `NuVector` 5필드 | ✅ |
| DEQ 고정점 반복 | §4.2 | L-Broyden solver | ✅ |
| FP32 버퍼 | §5.2 | `fp32_buffer=True` | ✅ |
| FAISS Top-3 검색 | §4.4 | `retrieval_k=3` | ✅ |
| t>10 최소 스텝 | §4.4 | `min_steps=10` | ✅ |
| Mean-pooled 인덱싱 | §4.4 | `z_star.mean(dim=0)` | ✅ |
| Soft-prefix 주입 | §4.4 | `prepend_soft_prefix()` | ✅ |
| Wproj ∈ R^{d×dmodel} | §4.5 | `LatentProjection(nn.Linear)` | ✅ |
| K=64 잠재 토큰 | §4.2 | `latent_tokens_K: 64` (config) | ✅ |
| 91.4% 정보 보존 | App. H | `threshold=0.914` | ✅ |
| '<\|think\|>' 바이패스 | §4.5 | `think_mode_bypass: true` | ✅ |
| 42블록 → 19모듈 | §5.1 | `module_partition.py` + `routing_proj[19,H]` | ✅ |
| Top-k=3 모듈 라우팅 | §5.2 | `top_k_modules: 3` | ✅ |
| νval = V(z*) | §5.3 | `NeuroCritic` on z* | ✅ |
| ACT 정지 조건 | §4.3 | `mac_accumulated > tau * nu_act` | ✅ |
| Vision/Audio 오프로드 | §7.1 | `offload_vision_audio=True` (~0.9 GB) | ✅ |
| enable_thinking=False | §7.1 | `enable_thinking: false` | ✅ |
| max_new_tokens=64 | §7.1 | `max_new_tokens: 64` | ✅ |

### Table 4 하이퍼파라미터 (13/13 완전 일치)

| 파라미터 | 논문 | `default.yaml` | 상태 |
|---------|:----:|:--------------:|:----:|
| PPO LR | 3×10⁻⁵ | `ppo_lr: 3.0e-5` | ✅ |
| Critic LR | 1×10⁻⁴ | `critic_lr: 1.0e-4` | ✅ |
| PPO Clip ε | 0.2 | `ppo_clip_epsilon: 0.2` | ✅ |
| λ_halt | 0.05 | `act_halting_penalty: 0.05` | ✅ |
| γ (discount) | 0.99 | `discount_gamma: 0.99` | ✅ |
| GAE λ | 0.95 | `gae_lambda: 0.95` | ✅ |
| LoRA r | 8 | `lora_rank: 8` | ✅ |
| LoRA α | 16 | `lora_alpha: 16` | ✅ |
| Broyden max iter | 30 | `broyden_max_iter: 30` | ✅ |
| K (latent tokens) | 64 | `latent_tokens_K: 64` | ✅ |
| W (branching) | 3 | `mcts_branching_W: 3` | ✅ |
| Top-k modules | 3 | `top_k_modules: 3` | ✅ |
| FAISS k | 3 | `faiss_retrieval_k: 3` | ✅ |

---

## B. 불일치 항목 (12개) — 심각도별 분류

### 🔴 Critical (논문 결과 재현에 직접 영향)

#### B-1. Stage 2 보상 함수가 Eq.(5)와 불일치

- **논문 Eq.(5):** `R = 1{correct} − 0.05 · T`
- **코드 실제 사용:** `default_transition_reward(r)` → DEQ 수렴 여부만 반환 (1.0/0.0)
- **위치:** `cts/train/stage2_ppo_train.py:180`, `cts/mcts/episode.py:25-29`
- **문제:** 정답 여부(correct)와 깊이 패널티(T)가 반영되지 않음
- **`paper_reward()` 함수**는 `rewards/shaping.py`에 구현되어 있으나, Stage 2 학습에서 **호출되지 않음**

#### B-2. transition() 기본 K=8 (논문 K=64)

- **논문:** K=64 latent tokens (§4.2)
- **코드:** `transition(... K=8 ...)` (기본값), `transition_batch(... K=8 ...)`
- **위치:** `cts/deq/transition.py:39`, `transition.py:176`
- **config**에는 `latent_tokens_K: 64`가 있으나, 실행 시 **config를 transition에 전달하는 로직 미비**

#### B-3. transition_batch()가 순차 실행 (논문: 병렬 배치)

- **논문 §4.1:** "W sibling branches are evaluated as a single parallel batch during the DEQ solve, maintaining ~25ms latency irrespective of branching width"
- **코드:** `for branch_index in range(W): r = transition(...)` — 순차 호출
- **위치:** `cts/deq/transition.py:194-214`
- **`broyden_fixed_point_batch()`** 함수는 존재하지만 `transition_batch()`에서 사용되지 않음

#### B-4. tau_flops_budget 기본값 불일치

- **논문 §7.1:** Iso-FLOP 10¹⁴ MACs
- **코드 transition():** `tau_flops_budget=1e15` (기본값, 10배 큼)
- **Stage 2 PPO:** `tau_flops_budget=float("inf")` (예산 무제한)
- **config:** `tau_flops_budget: 1.0e14` (올바름, 하지만 전달 미비)

### 🟡 Moderate (기능은 있으나 연결 미비)

#### B-5. 부모 J⁻¹ 상속 미연결

- **논문 §5.2:** "inheriting the parent node's inverse Jacobian approximation"
- **코드:** `broyden_fixed_point()`에 `parent_inv_jacobian` 파라미터 존재
- **하지만:** `transition()`에서 `broyden_fixed_point(phi, z0, ...)` 호출 시 **전달하지 않음**

#### B-6. Stage 2 PPO에서 GAE 미사용

- **논문 Table 4:** GAE λ=0.95
- **코드:** `ppo_core.py`에 `compute_gae()` 구현 있음
- **하지만:** `stage2_ppo_train.py`는 `advantages = rewards - values` (단순 차감)으로 대체

#### B-7. Stage 1 LoRA 기본 off

- **논문 §6.1:** "LoRA adapters (r=8) ... are injected"
- **코드:** `run_stage1_openmath.py`에서 `--lora`가 `store_true` (기본 False)
- **영향:** 실행 시 `--lora` 플래그를 명시적으로 줘야 함

#### B-8. Legacy config 충돌

- **`default.yaml`에 두 가지 K 값 공존:**
  - `latent_tokens_K: 64` (논문 정합)
  - `soft_thought_K: 8` (레거시)
- **`gamma: 0.95`** (레거시) vs **`discount_gamma: 0.99`** (논문)

### 🟢 Minor (결과에 미미한 영향)

#### B-9. L-Broyden limited-memory 구현 부정확

- **논문:** Limited-memory rank-1 updates
- **코드:** `memory_limit`으로 벡터 리스트만 자르지만, **전체 B 행렬은 매 스텝 갱신 누적**

#### B-10. VRAM 프로파일링 Mock 백본 사용

- **논문 Table 1:** CTS-Gemma 4 E4B → 16.5~16.7 GB
- **코드:** `profile_vram_latency.py` → MockTinyBackbone (79.4 MB)
- **실 모델 프로파일은 아직 미완**

#### B-11. p-RoPE 격리 불완전

- **논문 §4.4:** "retrieved vectors ... mathematically isolating their relative position indices"
- **코드:** `prepend_soft_prefix`는 단순 `torch.cat` — RoPE 위치 격리는 백본 의존

#### B-12. 라우팅이 토큰별이 아닌 시퀀스별

- **논문 Eq.(6):** "applied token-wise across the K latent sequence"
- **코드:** mean-pooled 단일 벡터 기준 라우팅

---

## C. 실험 결과 정합성

### 실제 실행된 실험

| 항목 | 논문 | 실행 결과 | 일치 |
|------|------|----------|:----:|
| Stage 1 학습 | 10K, LoRA | 2000 steps 완료, loss 수렴 | ⚠️ 스텝 수 다름 |
| Stage 2 PPO | 5K, Eq.(5) | 500 steps 완료, loss 수렴 | ⚠️ 보상함수 다름 |
| MATH-500 baseline | Greedy 45.2% | Raw Gemma 0.0% | ⚠️ 기대 미달 |
| VRAM O(1) | 79.4 MB 고정 | Mock backbone에서 검증 | ✅ O(1) 확인 |
| Broyden 수렴 | 12 iter, 3.42e-3 | Mock에서 확인 | ✅ |

### 논문 Table 2 목표 vs 현재

| 벤치마크 | 논문 목표 | 현재 상태 |
|---------|:---------:|---------|
| MATH 500 | 68.4 ± 0.8% | Raw Gemma baseline만 (0%) — CTS 파이프라인 미적용 |
| GSM8K | 92.1 ± 0.5% | 데이터셋 준비 완료, 미실행 |
| AIME 2026 | 56.4 ± 1.1% | 데이터셋/스크립트 미확인 |
| ARC-AGI-Text | 64.1 ± 0.9% | 스크립트 존재, 미실행 |
| HumanEval | 74.2 ± 0.6% | 스크립트 존재, 미실행 |

---

## D. 수정 우선순위 로드맵

### Phase 1: Critical 수정 (논문 핵심 주장에 필수)

1. **Stage 2 보상 → `paper_reward()` 연결** — Eq.(5) 정합
2. **`transition()` K 기본값 → 64** — config 연동
3. **`transition_batch()` → 진정한 병렬 배치 DEQ** — `broyden_fixed_point_batch` 연결
4. **`tau_flops_budget` 기본값 → config에서 로드** — 1e14 일관성

### Phase 2: Moderate 수정

5. **부모 J⁻¹ 전달** — transition에서 broyden으로
6. **GAE 연결** — stage2_ppo_train에서 compute_gae 사용
7. **LoRA 기본 on** — Stage 1 스크립트
8. **Legacy config 정리** — soft_thought_K, gamma 제거

### Phase 3: Full-scale 재현

9. **실 모델 VRAM 프로파일** — Table 1 정합
10. **5-seed 벤치마크 실행** — Table 2 수치 재현

---

## 결론

**현재 정합률: ~78% (42/54 항목)**

- **아키텍처/수식/하이퍼파라미터**: 높은 정합률 (~90%)
- **학습 파이프라인**: 구조는 있으나 핵심 연결(보상함수, K, 배치)이 미비 (~67%)
- **실험 결과**: 인프라는 준비됐으나 full-scale 재현은 미완 (~50%)

**가장 시급한 4가지 수정:**
1. Eq.(5) 보상함수 → Stage 2에 연결
2. K=8 → K=64 기본값 통일
3. transition_batch → 병렬 DEQ
4. tau_flops_budget → config 일관성
