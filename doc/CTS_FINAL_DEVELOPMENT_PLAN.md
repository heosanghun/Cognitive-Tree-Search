# CTS 최종 개발 계획서 (Final v2.0 — 논문 PDF 정합)

| 항목 | 내용 |
|------|------|
| 버전 | FINAL-2.0 |
| 기반 | 논문 *Cognitive Tree Search: KV-Cache-Free Per-Node O(1) Transitions for System 2 Inference via Deep Equilibrium Models* (NeurIPS 2026) |
| 갱신일 | 2026-04-10 |
| 최종 목표 | 논문 알고리즘과 보완 스펙을 **코드 100% 반영**하고, **논문 최종 성과 지표**를 재현 가능한 테스트·벤치로 검증 |

---

## 0. 논문 핵심 아키텍처 요약

### 0.1 CTS 프레임워크 개요

CTS는 **Outer-Loop (Neuromodulated MCTS)** + **Inner-Loop (DEQ Implicit Transition)** 이중 구조:

- **Outer-Loop**: Meta-Policy가 연속 제어 벡터 ν를 예측 → PUCT로 트리 탐색
- **Inner-Loop**: Broyden 솔버로 고정점 z*를 찾아 O(1) 전이 수행
- **Latent Space Context Window**: FAISS 인덱스로 과거 z* 벡터를 sub-linear 검색
- **Latent-to-Text Decoding**: Wproj 프로젝션으로 z*를 텍스트 디코딩

### 0.2 Meta-Policy ν 벡터 (논문 §2.3, §4)

논문은 생물학적 은유를 **표준 ML 파라미터**로 공식화한다:

| 논문 기호 | 역할 | 섹션 |
|-----------|------|------|
| **νval** | State Value — Neuro-Critic 출력 V(z*) | §5.3 |
| **νexpl** | Exploration Rate — PUCT 탐색 계수 + z0 노이즈 σ | §4.1 |
| **νtol** | Solver Tolerance — Broyden 수렴 허용치 [10⁻⁴, 10⁻²] | §4.2 |
| **νtemp** | Routing Temperature — 라우팅 softmax 온도 | §5.2 |
| **νact** | ACT Halting Threshold — 누적 MACs 기반 중단 | §4.3 |

**기존 코드 매핑 (레거시 → 논문 정합):**

| 레거시 (신경전달물질) | 논문 정합 | 역할 |
|----------------------|-----------|------|
| `nu_da` | `nu_val` | 프로세스 보상 스케일 / TD 가중 |
| `nu_5ht` | `nu_expl` | PUCT 탐색 + z0 노이즈 σ |
| `nu_ne` | `nu_tol` | Broyden tol |
| `nu_ach` | `nu_temp` | 라우팅 온도 |
| `nu_ado_scale` | `nu_act` | ACT 누적 가중 |

### 0.3 핵심 수식 (논문 기준)

**Eq.(2) DEQ 고정점 전이:**
```
z*_{t+1} = f_{θ,ν}(z*_{t+1}, s₀ ⊕ Hₜ)    (Broyden's method)
```

**Eq.(3) CTS 메모리:**
```
V^CTS_Total(D) = V_Weights + V_Metadata(D) + O_active(1) + O_history(N)
```

**Eq.(4) PUCT:**
```
a* = argmax_a [ Q(s,a) + νexpl · P(s,a) · √N(s) / (1 + N(s,a)) ]
```
여기서 P(s,a) = 1/W (균일 사전 분포)

**Eq.(5) 보상 함수:**
```
R_total = 1{correct_answer} − λ_halt · T
```
T = 실제 누적 터미널 트리 깊이 (또는 소모된 MACs)

**Eq.(6) 라우팅:**
```
z* = Σ_{i∈Top-k} Softmax(W_g · z*_t / νtemp)_i · m_i(z*, s₀ ⊕ Hₜ)
```

---

## 1. 최종 성공 기준 (논문 성과 지표)

다음을 **동일 프로토콜**(W=3, 단일 24GB급 GPU, Iso-FLOP ≤ 10¹⁴ MACs)으로 측정한다.

### 1.1 표 1 유형 — 메모리·지연 vs 트리 깊이

| 지표 | 논문 주장 | 구현 검증 |
|------|----------|-----------|
| Vanilla MCTS VRAM | Depth 1: 16.5GB → Depth 15: OOM (>24GB) | `eval/profile_vram_latency.py` + `baselines/mcts_kv_baseline.py` |
| MCTS + Prefix Caching | Depth 15: 18.2GB → Depth 35: OOM (>24GB) | 동일 |
| CTS VRAM | Depth 1~100+: **16.5–16.7 GB 평탄** | **M1/M2** 정의 (`doc/memory_definitions.md`) |
| Mamba/RWKV | 14.2 GB 고정 (단, MCTS 불가) | 참고 베이스라인 |
| 노드당 지연 | CTS **~25 ms** (깊이 무관) | 프로파일 CSV `latency_ms_per_node` |

### 1.2 표 2 유형 — 정확도 (Iso-FLOP, ≤ 10¹⁴ MACs, 5 seeds, 95% CI)

| 벤치마크 | Greedy | SC@14 | Native Think | MCTS(Vanilla) | **CTS (Ours)** |
|----------|--------|-------|-------------|---------------|----------------|
| **MATH 500** | 45.2 | 59.3±0.7 | 57.0±0.6 | 48.2±0.8 | **68.4±0.8** |
| **GSM8K** | 76.5 | 84.2±0.5 | 82.4±0.4 | 78.1±0.6 | **92.1±0.5** |
| **AIME 2026** | 28.3 | 34.8±0.9 | 42.5±0.9 | 31.1±1.0 | **56.4±1.1** |
| **ARC-AGI-Text** | 36.1 | 52.4±0.8 | 50.1±0.7 | 40.1±0.9 | **64.1±0.9** |
| **HumanEval** | 56.4 | 65.2±0.6 | 63.3±0.5 | 58.2±0.8 | **74.2±0.6** |
| Avg. MACs (×10¹⁴) | 0.05 | 1.0 | Internal | 0.2(OOM) | **0.65** |

**핵심 관찰:**
- CTS는 **ACT 중단**을 통해 할당 예산의 **65%만** 사용하면서 SOTA 달성
- Native Think 모드를 AIME 2026에서 **+13.9%p** 능가

### 1.3 애블레이션

- w/o Latent Context Window → **−4.9%** 성능 하락
- w/o ACT halting (νact) → MAC 과소비(예산 초과)
- K=32 → 심각한 symbolic information loss
- K=128 → Jacobian 복잡도 O(K²) 폭발, Iso-FLOP 위반
- Top-k < 2 → representation collapse
- Top-k > 3 → 무의미한 연산 비용 증가

설정: `configs/ablation_no_ach.yaml`, `configs/ablation_static_5ht.yaml`

### 1.4 Broyden 솔버 수렴 통계 (부록 C)

| 지표 | 값 (mean ± std) |
|------|-----------------|
| 수렴률 | 97.3 ± 0.4% |
| 평균 Broyden 반복 | 11.2 ± 2.8 |
| 폴백(프루닝)률 | 2.7 ± 0.4% |
| 유효 스펙트럼 반경 γ | ≈ 0.92 |

---

## 2. 범위: 논문 + 보완

### 2.1 논문 핵심 컴포넌트

| 논문 섹션 | 컴포넌트 | 구현 경로 |
|-----------|----------|-----------|
| §3.2 | DEQ 고정점 전이, O(1) 활성 메모리 | `cts/deq/` |
| §4.1 | Neuromodulated MCTS + PUCT(νexpl) | `cts/mcts/` |
| §4.2 | Broyden 솔버 + L-Broyden (FP32 버퍼) | `cts/deq/broyden_forward.py` |
| §4.3 | ACT + Energy-Constrained Halting (νact) | `cts/routing/lut_mac.json` |
| **§4.4** | **FAISS Latent Space Context Window** | `cts/latent/` |
| **§4.5** | **Latent-to-Text Decoding Projection (Wproj)** | `cts/latent/bottleneck.py` |
| §5.1 | 19-모듈 희소 라우팅 (42블록 → 19모듈) | `cts/routing/`, `cts/model/module_partition.py` |
| §5.2 | Routing Temperature (νtemp) + Triton 커널 | `cts/routing/sparse_moe_triton.py` |
| §5.3 | Neuro-Critic V(z*) (νval) | `cts/critic/neuro_critic.py` |
| §6.1 | Stage 1: DEQ Warm-Up (LoRA r=8, 10K examples) | `cts/train/stage1_warmup.py` |
| §6.2 | Stage 2: PPO + Outcome Rewards (5K prompts) | `cts/train/stage2_ppo.py` |

### 2.2 논문 신규 요소 (기존 문서에 부재/불충분)

| 요소 | 논문 내용 | 현황 |
|------|-----------|------|
| **FAISS Context Window** | t>10일 때 Top-3 조상 벡터를 mean-pooled z*로 FAISS 검색, prepended global soft-prefix로 주입 | **문서화 필요** |
| **Wproj 프로젝션** | Wproj ∈ R^{d×dmodel}, '<\|think\|>' 바이패스, [Final Answer] 시맨틱 공간 매핑 | **문서화 필요** |
| **병렬 배치 DEQ** | W개 형제 노드를 단일 배치로 동시 DEQ 솔브 → 25ms 유지 | **문서화 필요** |
| **Vision/Audio 오프로딩** | ~150M vision + ~300M audio 인코더 오프로드 → ~0.9 GB 절약 | **문서화 필요** |
| **Native Think 비활성화** | `enable_thinking=False` 시스템 프롬프트로 강제 비활성 | **문서화 필요** |
| **K=64 검증** | 91.4% 정확 심볼릭 방정식 복원율 (부록 H) | **문서화 필요** |
| **교차 도메인 전이** | 코딩 시 νexpl +15% 자동 상향 (부록 I) | **문서화 필요** |
| **GSM8K / HumanEval 벤치** | 표 2에 포함 | **스크립트 확장 필요** |

### 2.3 보완 (기존 유지)

- M1/M2 메모리 정의, PPO 확장 하이퍼, PUCT 변체, `BaseCTSBackbone`, 프로파일 계약, `security_eval.md`, Stage1 손실 문서

---

## 3. 단계별 로드맵 (투두 매핑)

| Phase | 내용 | 산출물 |
|-------|------|--------|
| 0 | 문서·설정·측정 규약 | `doc/memory_definitions.md`, `configs/` |
| 1 | 백본·모듈 매핑·PLE | `cts/backbone/`, `cts/model/module_partition.py` |
| 2–3 | 잠복 공간·Broyden·`transition()`·L-Broyden | `cts/deq/`, `cts/latent/` |
| **3.5** | **FAISS Latent Context Window** | `cts/latent/faiss_context.py` (신규) |
| 4–5 | 라우팅·메타정책·**병렬 배치 DEQ** | `cts/routing/`, `cts/policy/` |
| **5.5** | **Latent-to-Text Wproj 프로젝션** | `cts/latent/bottleneck.py` |
| 6–7 | ACT·MCTS·보상(Eq.5) | `cts/mcts/`, `cts/rewards/` |
| 8–10 | Critic·학습 (Stage1: 10K, Stage2: 5K PPO) | `cts/critic/`, `cts/train/` |
| 11–12 | 벤치 (MATH/GSM8K/AIME/ARC/HumanEval)·베이스라인·CI | `cts/eval/`, `cts/baselines/`, `tests/` |

---

## 4. 논문 정합 아키텍처 상세

### 4.1 FAISS Latent Space Context Window (§4.4)

논문의 핵심 메커니즘:
- 모든 고정점 z*를 mean-pooling하여 1D 벡터로 축소 → FAISS IndexFlatIP에 추가
- t > 10인 전이에서 Top-3 시맨틱 유사 조상 벡터를 검색
- 검색된 벡터를 **prepended global soft-prefix**로 주입
- p-RoPE 시퀀스와 **수학적으로 격리** (상대 위치 인덱스 분리)
- 메모리: O(N) — 노드당 수 KB 수준

```python
# 구현 방향 (의사코드)
def retrieve_context(z_star: Tensor, faiss_index, step: int, k: int = 3):
    if step <= 10:
        return None
    query = z_star.mean(dim=-2)  # mean-pool K tokens → 1D
    _, indices = faiss_index.search(query, k)
    return faiss_index.reconstruct_batch(indices)
```

### 4.2 Latent-to-Text Decoding Projection (§4.5)

- Wproj ∈ R^{d × dmodel}: K=64 잠복 토큰을 frozen 모델의 continuous soft prompt로 주입
- **'<|think|>' 포맷 완전 바이패스** → [Final Answer] 시맨틱 공간 직접 매핑
- 터미널 텍스트 생성: 1회 표준 AR 오버헤드 (~1.2s for 500 tokens)

### 4.3 병렬 배치 DEQ (§4.1)

- W개 형제 분기에 독립적 노이즈 주입 → W개 잠복 상태 생성
- **단일 배치**로 동시 DEQ 전이 → z*_{t+1} 고정점 탐색
- Neuro-Critic으로 동기 평가 → Q(s,a) 산출
- 결과: **브랜칭 폭 W에 무관하게 ~25ms 유지**

### 4.4 L-Broyden & 솔버 세부사항

- **Limited-memory Broyden**: low-rank 버퍼를 **FP32**로 유지 → BF16 수치 발산 방지
- 부모 노드의 역 야코비안 근사를 **상속** → 수렴 가속
- 루트 노드(t=0): s₀의 표준 forward pass로 z*₀ 부트스트랩, 역 야코비안은 scaled identity
- O((Kd)²) 메모리 폭발 방지

### 4.5 Vision/Audio 인코더 오프로딩

- Gemma 4 E4B의 vision (~150M) + audio (~300M) 인코더를 VRAM에서 제거
- 텍스트 기반 추론만 수행하므로 **~0.9 GB 절약**
- 16.7 GB 풋프린트는 텍스트 백본 전용

---

## 5. 학습 커리큘럼 (논문 §6)

### 5.1 Stage 1: DEQ Warm-Up

| 항목 | 값 |
|------|-----|
| 목표 | Broyden 솔버 안정화, IFT 잔차 손실 ∥f(z*) − z*∥²₂ 최소화 |
| 데이터 | OpenMathInstruct-2, **10,000 examples** |
| 가중치 | Gemma 4 **동결** |
| 학습 파라미터 | LoRA r=8 + 라우팅 프로젝션 Wg (총 **~18 MB**) |
| 손실 | `fixed_point_surrogate_loss` — 엄격한 균형 정규화 |

### 5.2 Stage 2: PPO + Outcome Rewards

| 항목 | 값 |
|------|-----|
| 목표 | Meta-Policy 학습 — 도메인 비의존적 탐색 역학 |
| 데이터 | MATH/AIME **5,000 prompts** (평가셋과 완전 분리) |
| 보상 | Rtotal = 1{correct} − λhalt · T (Eq.5) |
| 알고리즘 | PPO |

### 5.3 학습 하이퍼파라미터 (논문 Table 4 + 보완)

| 하이퍼파라미터 | 값 |
|---------------|-----|
| PPO Learning Rate | 3 × 10⁻⁵ |
| Critic Learning Rate | 1 × 10⁻⁴ |
| PPO Clip Ratio (ε) | 0.2 |
| ACT Halting Penalty (λhalt) | 0.05 |
| Discount Factor (γ) | 0.99 |
| GAE Parameter (λ) | 0.95 |
| LoRA Rank (r) | 8 |
| LoRA Alpha (α) | 16 |
| Entropy Coef (보완) | 0.01 |
| Value Loss Coef (보완) | 0.5 |
| Max Grad Norm (보완) | 1.0 |

---

## 6. 즉시 실행 (개발 착수 후)

1. `pip install -e ".[dev]"` (로컬)
2. **`python scripts/verify_cts_final_goal.py`** — 핵심 스크립트 + `pytest -k "not slow"` 통과 확인
3. (선택) **`python scripts/verify_cts_final_goal.py --check-artifacts`** — 산출물 존재 확인
4. `pytest tests/` (전체·`slow` 포함 시 Gemma 로드 환경 필요)
5. `python -m cts.eval.profile_vram_latency --help` · `python -m cts.eval.report_isoflop --json`

---

## 7. 문서 우선순위

상충 시: **논문 PDF** → **본 FINAL 계획서** → `NeurIPS_2026_CTS_DEVELOPMENT_PLAN_IMPROVED.md` → 초안 `.txt`.

---

## 8. 구현 현황 (v0.1 → v0.2 업그레이드 대상)

| 구분 | 상태 | 비고 |
|------|------|------|
| 패키지 `cts/` + `pyproject.toml` | 완료 | `pip install -e ".[dev]"` |
| 타입·`transition()`·Broyden·PUCT·모듈 매핑 | 완료 | `MockTinyBackbone` 기준 단위/스모크 테스트 |
| 라우팅 ref·LUT JSON·Triton | 완료 | `routing_weights_triton` ≡ ref (`tests/test_routing_triton_ref.py`) |
| Gemma 4 E4B 로더·`GemmaCTSBackbone`·스모크 | **부분 완료** | `model/gemma_loader.py`, `backbone/gemma_adapter.py` |
| 실모델 42층 PLE + 희소 병렬 모듈 DEQ | **부분 완료** | blend → parallel 2단계 |
| LM 디코딩 | **부분** | `decode_from_z_star` AR |
| Stage1/2 학습 루프 | **완료** | `run_stage1_openmath.py`·`run_stage2_math_ppo.py` |
| 표1·표2 재현 도구 | **완료** | `run_paper_artifacts_pipeline.py` |
| **FAISS Latent Context Window** | **⚠ 미구현** | 논문 §4.4 — `latent/faiss_context.py` 신규 필요 |
| **Wproj Decoding Projection** | **⚠ 미구현** | 논문 §4.5 — `latent/bottleneck.py` 확장 필요 |
| **병렬 배치 DEQ (W branches)** | **⚠ 미구현** | 논문 §4.1 — `deq/transition.py` 배치화 필요 |
| **L-Broyden FP32 버퍼** | **⚠ 미구현** | 논문 §5.2 — `deq/broyden_forward.py` FP32 분리 |
| **Vision/Audio 오프로딩** | **⚠ 미구현** | 논문 §7.1 — `model/gemma_loader.py` 확장 |
| **GSM8K 벤치** | **⚠ 미구현** | 논문 표2 — `eval/gsm8k.py` 신규 |
| **HumanEval 벤치** | **⚠ 미구현** | 논문 표2 — `eval/humaneval.py` 신규 |
| **ν 명명법 논문 정합** | **⚠ 필요** | DA→val, 5HT→expl, NE→tol, ACh→temp, Ado→act |

**테스트:** `pytest tests/` (CI: `pytest tests/ -k "not slow"`)

---

## 9. 신규 구현 항목 (논문 PDF 기반 추가)

### 9.1 FAISS Latent Space Context Window

| 항목 | 상세 |
|------|------|
| 파일 | `cts/latent/faiss_context.py` |
| 의존성 | `faiss-cpu` (또는 `faiss-gpu`) |
| 기능 | (1) z* mean-pool → 1D 벡터로 FAISS 인덱싱 (2) t>10 시 Top-3 검색 (3) soft-prefix 주입 |
| 테스트 | 검색 정확도, 메모리 증가율 (노드당 KB), p-RoPE 격리 |

### 9.2 Latent-to-Text Wproj

| 항목 | 상세 |
|------|------|
| 파일 | `cts/latent/bottleneck.py` 확장 |
| 구현 | Wproj ∈ R^{d×dmodel}, K=64 잠복 토큰 → continuous soft prompt |
| 핵심 | '<\|think\|>' 바이패스, [Final Answer] 직접 매핑 |
| 테스트 | 프로젝션 shape 검증, 정보 보존율 (목표 91.4%) |

### 9.3 병렬 배치 DEQ

| 항목 | 상세 |
|------|------|
| 파일 | `cts/deq/transition.py` 확장 |
| 구현 | W개 노이즈 주입 → 배치 차원으로 쌓기 → 단일 Broyden solve |
| 핵심 | 배치 크기 W에 무관한 ~25ms 달성 |
| 테스트 | W=1,3,5에서 지연 시간 비교 |

### 9.4 L-Broyden FP32 정밀도

| 항목 | 상세 |
|------|------|
| 파일 | `cts/deq/broyden_forward.py` 확장 |
| 구현 | low-rank 역 야코비안 버퍼를 FP32로 유지, 입출력은 BF16 |
| 핵심 | 수치 발산 방지, O((Kd)²) 메모리 억제 |
| 테스트 | FP32 vs BF16 수렴 안정성 비교 |

### 9.5 Vision/Audio 인코더 오프로딩

| 항목 | 상세 |
|------|------|
| 파일 | `cts/model/gemma_loader.py` 확장 |
| 구현 | `load_gemma4_e4b(offload_vision_audio=True)` 옵션 |
| 핵심 | ~0.9 GB VRAM 절약 |

### 9.6 GSM8K · HumanEval 벤치마크

| 항목 | 상세 |
|------|------|
| 파일 | `cts/eval/gsm8k.py`, `cts/eval/humaneval.py` 신규 |
| 데이터 | GSM8K (HF), HumanEval (로컬 오프라인 실행) |
| 핵심 | 논문 표2 완전 재현 — 5 seeds, 95% CI |

---

## 10. 진행률 대시보드 (100% 대비)

| 문서 | 설명 |
|------|------|
| **`doc/PAPER_ALIGNMENT_PROGRESS.md`** | **코드·파이프라인 진행률** 정의·영역별 표 — 본 절과 동기화 |

**요약 (2026-04-10 기준):**
- 기존 코어 프레임워크: **100%** (DEQ, Broyden, PUCT, MCTS, Stage1/2)
- **논문 PDF 신규 요소**: ~60% (FAISS, Wproj, 병렬배치, L-Broyden, 벤치 확장 미완)
- **명명법 정합**: 진행 중 (ν 벡터 용어 통일)

---

## 11. 선택·심화 투두

**핵심 파이프라인·학습 스택·검증 CLI는 완료.** 아래는 논문 PDF 정합을 위한 **신규 필수 항목**이다.

- [x] 기존 코어 DEQ·Broyden·transition·PUCT·모듈 매핑
- [x] Stage1/2 학습 루프 + PPO/GAE
- [x] 표1·표2 프로파일·벤치 자동화
- [x] Triton ref 검증
- [ ] **FAISS Latent Space Context Window** (논문 §4.4)
- [ ] **Wproj Latent-to-Text Projection** (논문 §4.5)
- [ ] **병렬 배치 DEQ** — W 형제 동시 솔브 (논문 §4.1)
- [ ] **L-Broyden FP32 정밀도 분리** (논문 §5.2)
- [ ] **Vision/Audio 인코더 오프로딩** (논문 §7.1)
- [ ] **GSM8K 벤치마크 스크립트** (논문 표2)
- [ ] **HumanEval 벤치마크 스크립트** (논문 표2)
- [ ] **ν 명명법 코드 전체 정합** (νval, νexpl, νtol, νtemp, νact)
- [ ] **보상 함수 Eq.(5) 정합** — Rtotal = 1{correct} − λhalt · T
- [ ] **K=64 정보 보존 검증 테스트** (목표 91.4%)
- [ ] **교차 도메인 ν 분포 분석** (부록 I)

---

*이 계획서는 논문 PDF를 최종 기준으로 하며, 마일스톤 완료 시 표 1·2 재현 리포트를 `artifacts/`에 첨부하는 것을 권장한다.*
