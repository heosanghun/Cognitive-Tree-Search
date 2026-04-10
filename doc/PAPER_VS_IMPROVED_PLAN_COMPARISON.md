# 논문 PDF vs 개선 개발 계획 — 한눈에 보는 비교표 (v2.0)

> **결론:** `NeurIPS_2026_CTS_DEVELOPMENT_PLAN_IMPROVED.md` v2.0은 논문 PDF를 **최종 기준**으로 직접 대조하여, v1에서 누락된 핵심 아키텍처(FAISS Context Window, Wproj, 병렬배치 DEQ, L-Broyden)와 ν 명명법을 **완전히 정합**시킨 것이다.

---

## v1 → v2 주요 변경 요약

| 영역 | v1 (기존) | v2 (논문 PDF 정합) |
|------|-----------|-------------------|
| ν 명명법 | 신경전달물질 (DA, 5HT, NE, ACh, Ado) | **νval, νexpl, νtol, νtemp, νact** |
| FAISS Context Window | 미언급 | **§4.4 기반 구현 계획** |
| Wproj 프로젝션 | 미언급 | **§4.5 기반 구현 계획** |
| 병렬배치 DEQ | 미명시 | **§4.1 W 형제 동시 솔브** |
| L-Broyden | 미명시 | **§5.2 FP32 역 야코비안** |
| 보상 함수 | 불명확 | **Eq.(5) 정합** |
| 벤치마크 | MATH, ARC | **+ GSM8K, HumanEval** |
| Vision/Audio | 미언급 | **오프로딩 옵션** |
| 수렴 통계 | 미언급 | **부록 C: 97.3%, 11.2 iter** |
| K=64 검증 | 미언급 | **부록 H: 91.4% 복원율** |

---

## 핵심 비교표

| 구분 | 논문(기준) | 개선 계획·코드 방향 (v2) | 한 줄 요지 |
|------|------------|--------------------------|------------|
| O(1) 메모리 의미 | Eq.(3): V^CTS = V_Weights + V_Metadata + O_active(1) + O_history(N) | M1/M2 + Eq.(3) 직접 인용 | 수식 기반 검증 |
| **ν 벡터 명명법** | νval, νexpl, νtol, νtemp, νact (§2.3) | **코드 전체 정합** (레거시 별칭 교체) | 논문-코드 1:1 |
| **FAISS Context Window** | §4.4: mean-pool z* → Top-3 검색 → soft-prefix | **`latent/faiss_context.py`** 신규 | 핵심 아키텍처 |
| **Wproj 프로젝션** | §4.5: Wproj ∈ R^{d×dmodel}, '<\|think\|>' 바이패스 | **`latent/bottleneck.py`** 확장 | 디코딩 핵심 |
| **병렬배치 DEQ** | §4.1: W 형제 단일 배치 → ~25ms | **`deq/transition.py`** 배치화 | 지연 핵심 |
| **L-Broyden FP32** | §5.2: low-rank FP32 버퍼, 부모 상속 | **`deq/broyden_forward.py`** 확장 | 수치 안정성 |
| **보상 함수** | Eq.(5): R = 1{correct} − λhalt · T | **`rewards/shaping.py`** 정합 | 학습 목표 |
| 식 (6) vs top-k | 수학: softmax 가중합 / 구현: top-k 희소 | dense ref vs top-k 테스트 (유지) | 재현성 |
| Neuro-Critic νval | §5.3: V(z*) = νval, 별도 linear head | `critic/neuro_critic.py` (유지) | AC 검증 |
| Stage 1 손실 | §6.1: IFT 잔차 ∥f(z*)−z*∥²₂, 10K examples | IFT 잔차 확정 (유지) | 명확 |
| **Stage 2** | §6.2: PPO, 5K MATH/AIME, Eq.(5) 보상 | **Eq.(5) 보상 연동** | 학습 정합 |
| PPO 하이퍼 | Table 4: lr=3e-5, clip=0.2, γ=0.99, λ=0.95 등 | YAML 명시 + 보완(entropy, value coef) | 재현성 |
| **벤치마크** | 표2: MATH/GSM8K/AIME/ARC/HumanEval 5종 | **GSM8K, HumanEval 스크립트 추가** | 완전 재현 |
| 레이어 인덱스 | Table 3 (1-based) | 0-base 코드 + layer_mapping (유지) | 일관성 |
| PUCT | Eq.(4): P(s,a)=1/W, νexpl 대체 c=1.0 | paper/alphazero 변체 (유지) | 디버깅 |
| **Vision/Audio** | §7.1: ~150M+300M 오프로드 → ~0.9GB 절약 | **gemma_loader 옵션** | VRAM 절약 |
| **수렴 통계** | 부록C: 97.3%, 11.2 iter, 2.7% 폴백 | **Broyden 통계 추적·테스트** | 품질 검증 |
| **K=64 검증** | 부록H: 91.4% 심볼릭 복원 | **복원율 테스트** | 정보 보존 |
| **γ ≈ 0.92** | 부록G: 스펙트럼 반경 경험적 바운드 | **contractive 추적** | 안정성 |
| **교차 도메인** | 부록I: 코딩 시 νexpl +15% | **분포 분석 로깅** | 범용성 |
| **Native Think** | §7.1: enable_thinking=False | **시스템 프롬프트 강제 비활성** | 공정성 |
| ACT 예산 사용 | 65% 평균 (0.65×10¹⁴ MACs) | **MAC 추적 + 조기 중단** | 효율성 |
| 백본 모델 | Gemma 4 E4B, 42층, PLE | `BaseCTSBackbone` + mock (유지) | 유연성 |
| 메모리 검증 | (구현됨) | `perf/memory_accounting.py` (유지) | REQ 일치 |
| 솔버 | Broyden + L-Broyden | + Anderson 옵션 (유지) | Future work |
| 프로파일 계약 | (구현됨) | warmup, cuda sync, peak VRAM (유지) | 재현 |
| 평가 안전 | (구현됨) | security_eval.md (유지) | 안전 |
| CI/회귀 | (구현됨) | 카나리 테스트 (유지) | 품질 |

---

## 상세 비교표 (항목별 — v2 갱신)

| ID | 주제 | 논문 상태 | 개선 계획 보완 (v2) |
|----|------|----------|---------------------|
| C-1 | 메모리 정의 | Eq.(3) 명시 | M1/M2 + Eq.(3) 직접 인용 |
| C-2 | ν vs budget 상태 | νact vs 누적 MACs 분리 | `NuVector`(val/expl/tol/temp/act) + `RuntimeBudgetState` |
| C-3 | top-k vs 식(6) | softmax 가중합 | dense ref vs top-k 테스트 |
| C-4 | Critic AC | 별도 linear head | 완화된 AC (δ 상한) |
| C-5 | Stage1 손실 | IFT 잔차 ∥f(z*)−z*∥²₂ | 확정·문서화 |
| C-6 | PPO 전 항목 | Table 4 명시 | YAML + 보완 |
| C-7 | PUCT 변체 | Eq.(4) 단일 | paper/alphazero 두 변체 |
| C-8 | 0-base 매핑 | 1-based 자연수 | 0-base + layer_mapping |
| C-9 | perf 모듈 | — | `memory_accounting.py` |
| C-10 | Backbone 추상화 | Gemma 특정 | `BaseCTSBackbone` |
| C-11 | 긴 s_t 모니터 | KV 중심 | `max_anchor_tokens` |
| C-12 | security_eval | — | 샌드박스 정책 |
| C-13 | 프로파일 계약 | 표1 결과 | warmup/sync/peak 고정 |
| C-14 | Anderson 옵션 | Future work | 솔버 플러그인 |
| C-15 | 결정론 평가 | — | seed/deterministic 옵션 |
| C-16–C20 | 운영·문서 | — | 리스크/CI/문서우선순위 |
| **C-21** | **FAISS Context Window** | **§4.4 핵심** | **`latent/faiss_context.py` 신규** |
| **C-22** | **Wproj 프로젝션** | **§4.5 핵심** | **`latent/bottleneck.py` 확장** |
| **C-23** | **병렬배치 DEQ** | **§4.1 핵심** | **`deq/transition.py` 배치화** |
| **C-24** | **L-Broyden FP32** | **§5.2 핵심** | **`deq/broyden_forward.py` FP32** |
| **C-25** | **Vision/Audio 오프로딩** | **§7.1** | **gemma_loader 옵션** |
| **C-26** | **GSM8K 벤치** | **표2** | **`eval/gsm8k.py` 신규** |
| **C-27** | **HumanEval 벤치** | **표2** | **`eval/humaneval.py` 신규** |
| **C-28** | **보상 Eq.(5)** | **§4.3** | **rewards/shaping.py 정합** |
| **C-29** | **ν 명명법** | **§2.3** | **코드 전체 교체** |
| **C-30** | **수렴 통계** | **부록C** | **97.3%/11.2 iter 추적** |
| **C-31** | **K=64 복원율** | **부록H** | **91.4% 테스트** |
| **C-32** | **교차 도메인 ν** | **부록I** | **분포 분석 로깅** |
| **C-33** | **Native Think 비활성** | **§7.1** | **enable_thinking=False** |

---

## "논문에 있음 / 개선안이 추가함" 빠른 범례

| 표기 의미 | 설명 |
|-----------|------|
| 논문 **있음** | 핵심 아이디어·수식·주요 하이퍼가 본문 또는 부록에 존재 |
| 논문 **부분** | 개념은 있으나 구현 디테일 부족 |
| 논문 **없음** | 공학·운영·안전·재현 항목으로 논문에 없음 |

| ID | 논문 | v1 보완 | v2 추가 |
|----|:----:|:-------:|:-------:|
| C-1 메모리 정의 | 있음 | ● | 강화 |
| C-2 ν vs budget | 부분 | ● | **명명법 정합** |
| C-3 top-k vs 식(6) | 부분 | ● | 유지 |
| C-4 Critic AC 완화 | 없음 | ● | 유지 |
| C-5 Stage1 손실 | 있음 | ● | 확정 |
| C-6 PPO 전 항목 | 있음 | ● | Table 4 동기화 |
| C-7 PUCT 변체 | 없음 | ● | 유지 |
| C-8 0-base 매핑 | 없음 | ● | 유지 |
| C-9 perf 모듈 | 없음 | ● | 유지 |
| C-10 Backbone | 없음 | ● | 유지 |
| C-11 긴 s_t | 부분 | ● | 유지 |
| C-12 security | 없음 | ● | HumanEval 확장 |
| C-13 프로파일 | 부분 | ● | 유지 |
| C-14 Anderson | 부분 | ● | 유지 |
| C-15–C20 운영 | 없음 | ● | 유지 |
| **C-21 FAISS** | **있음** | — | **● 신규** |
| **C-22 Wproj** | **있음** | — | **● 신규** |
| **C-23 병렬배치** | **있음** | — | **● 신규** |
| **C-24 L-Broyden** | **있음** | — | **● 신규** |
| **C-25 V/A 오프로드** | **있음** | — | **● 신규** |
| **C-26 GSM8K** | **있음** | — | **● 신규** |
| **C-27 HumanEval** | **있음** | — | **● 신규** |
| **C-28 Eq.(5) 보상** | **있음** | — | **● 신규** |
| **C-29 ν 명명법** | **있음** | — | **● 신규** |
| **C-30 수렴 통계** | **있음** | — | **● 신규** |
| **C-31 K=64 복원** | **있음** | — | **● 신규** |
| **C-32 교차 도메인** | **있음** | — | **● 신규** |
| **C-33 Think 비활성** | **있음** | — | **● 신규** |

● = 해당 버전에서 **보완/추가**한 축.

---

## 관련 파일

- 논문 PDF: `doc/NeurIPS_2026_Cognitive_Tree_Search__KV_Cache_Free_Per_Node_O_1_Transitions_for_System_2_Inference_via_Deep_Equilibrium_Models.pdf`
- 개선 개발 계획 본문: `doc/NeurIPS_2026_CTS_DEVELOPMENT_PLAN_IMPROVED.md` (v2.0)
- 최종 개발 계획서: `doc/CTS_FINAL_DEVELOPMENT_PLAN.md` (v2.0)
- 진행률: `doc/PAPER_ALIGNMENT_PROGRESS.md` (v2.0)
- (참고) 초안 전문: `doc/NeurIPS_2026_CTS_FULL_DEVELOPMENT_PLAN_ko.txt`

---

*본 비교표는 논문 PDF 전문을 직접 대조하여 v2.0으로 갱신했으며, 구현 진행에 따라 상태를 업데이트한다.*
