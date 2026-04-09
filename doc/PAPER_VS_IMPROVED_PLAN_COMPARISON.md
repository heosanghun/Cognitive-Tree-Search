# 논문 vs 개선 개발 계획 — 한눈에 보는 비교표

> **결론:** `NeurIPS_2026_CTS_DEVELOPMENT_PLAN_IMPROVED.md`는 논문이 **의도적으로 생략했거나 짧게만 적힌 공학·재현·안전** 요소를, 구현 가능한 수준으로 **명시·분리·측정 규약**으로 보완한 것이다. 아래 표에서 **「개선(계획·코드 방향)」** 은 아직 전부 구현된 코드가 아니라, **개선 계획서가 요구하는 산출물·스펙**을 뜻한다.

---

## 요약 표 (핵심만)

| 구분 | 논문(기준) | 개선 계획·코드 방향 | 한 줄 요지 |
|------|------------|---------------------|------------|
| O(1) 메모리 의미 | KV-free 전이·트리 메타데이터는 짧게 언급 | **M1(전이 단위)** / **M2(트리 단위)** 로 측정 정의 고정 | 오해 없이 검증 가능 |
| ν_Ado 역할 | LUT 누적·ACT로 서술 | **`NuVector`(정책 출력)** vs **`RuntimeBudgetState`(누적 상태)** 분리 | 타입/API 혼선 제거 |
| 식 (5) vs 구현 | 전체 19모듈 가중 합 | **full 합**과 **top-k 희소** 관계·애블레이션 문서화 | ref/prod 재현성 |
| Neuro-Critic | z\* 앵커, 스킵 불변 강조 | 검증 기준을 **완화 가능(δ)** 로 현실화 | 무한 튜닝 방지 |
| Stage 1 손실 | “워밍업” 수준만 | **고정점 잔차 / NLL / 결합** 후보 중 1안을 문서로 확정 | 구현 목표 단일화 |
| PPO 세부 | lr, γ, batch, LoRA 등 일부만(부록) | **clip ε, GAE λ, entropy/value coef, grad clip** 등 YAML 명시 | 재현성 |
| 레이어 인덱스 | Table 3 “Layers 1–42” | **0-base 코드 규약** + `layer_mapping.md` 이중 표 | 오프바이원 방지 |
| PUCT | 식 (4) 한 종류 | **`paper` / `alphazero` 변체** + 단위 테스트 | 문헌·디버깅 정합 |
| 메모리 검증 모듈 | (코드 경로 없음) | **`perf/memory_accounting.py`**, **`profiler.py`** | REQ-코드 일치 |
| 백본 모델 | Gemma 4 E4B·42층 가정 | **`BaseCTSBackbone`** + mock/다른 SLM 스왑 | 가용성 리스크 완화 |
| 긴 컨텍스트 s_t | KV와 혼동 가능 | **`max_anchor_tokens`** 등 모니터링·옵션 | OOM 원인 분리 |
| 평가 안전 | (언급 없음) | **`security_eval.md`**, 샌드박스/비실행 채점 | 안전·공정성 |
| 프로파일 | 표1 스타일 결과 | **warmup, cuda sync, peak VRAM** 계약 | 숫자 재현 |
| 솔버 | Broyden + NE 폴백 | **Anderson 등 플러그인(옵션)** | Future work 정렬 |
| 문서 우선순위 | — | **`.md` 개선본 > 초안 `.txt`** | 단일 기준 |

---

## 상세 비교표 (항목별)

| ID | 주제 | 논문에서의 상태 | 개선 계획·코드(산출물)에서의 보완 |
|----|------|-----------------|-----------------------------------|
| C-1 | **메모리 복잡도 서술** | KV 누적 제거·O(1) 컨텍스트를 강조하나, “무엇을 O(1)로 부를지” 측정 단위가 한 문장으로만 연결됨 | `docs/memory_definitions.md`: **M1**=전이당 KV 비누적, **M2**=트리 메타데이터 vs MCTS+KV 비교 프로토콜 |
| C-2 | **Adenosine(ν_Ado)** | 누적 비용·LUT·ACT는 있으나 “정책이 매 스텝 내는 값”과 “환경이 쌓는 값” 구분 없음 | `RuntimeBudgetState.ado_accumulated` + (선택) `nu_ado_scale`; `transition(..., budget)` API |
| C-3 | **라우팅 식 (5)** | 수학적으로 softmax 가중 합 | **dense ref** vs **top-k 실행** 차이를 문서·테스트로 고정; Triton은 ref 대비 오차 상한 |
| C-4 | **Critic 불변성** | Value head를 z\*에 앵커, 라우팅과 무관하게 평가한다는 취지 | 동일 z\*에서 마스크 바꿔도 **|ΔV|<δ** 등 완화된 AC 또는 패널 테스트 |
| C-5 | **Stage 1 목표** | OpenMath 1만 예·“워밍업” 수준 | `docs/stage1_objective.md`에 **잔차 최소화 / LM NLL / 결합** 중 채택안 명시 |
| C-6 | **PPO 하이퍼** | 부록: lr, batch, γ, LoRA r, 스텝 수 등 | `default.yaml`에 **ppo_clip_epsilon, gae_lambda, entropy_coef, value_loss_coef, max_grad_norm** 추가 |
| C-7 | **PUCT** | 식 (4) 단일 형태 | `puct_variant: paper \| alphazero`; 두 식 단위 테스트 |
| C-8 | **레이어↔모듈 매핑** | Table 3 (1-based 자연수 레이어) | 코드 **0-base** 통일 + `docs/layer_mapping.md` 대응표 |
| C-9 | **요구사항↔파일 일치** | (해당 없음) | REQ의 `memory_accounting`을 **`cts/perf/`** 트리에 반드시 포함 |
| C-10 | **베이스 모델** | Gemma 4 E4B, PLE, 42층 특정 | **`backbone/protocol.py`**, `gemma_adapter.py`, **`mock_tiny.py`** 로 교체 가능 |
| C-11 | **시퀀스 메모리** | KV 병목 중심 서술 | 긴 `s_t`에 대한 **앵커 토큰 상한·모니터링** 설정으로 “KV-free와 별개 병목” 명시 |
| C-12 | **평가 파이프라인** | MATH·ARC·Iso-FLOP 결과 중심 | **`security_eval.md`**: 코드 실행 평가 시 샌드박스 또는 금지 정책 |
| C-13 | **프로파일 재현** | 표1 형태 결과 | **warmup 횟수, cuda sync, peak allocated**를 보고서 스키마에 고정 |
| C-14 | **암시적 솔버** | Broyden, NE tol, 폴백; Future work에 Anderson | **`deq/solvers/anderson.py`** 옵션 플래그 |
| C-15 | **결정론적 평가** | (명시 없음) | `eval_deterministic`, 시드·`torch` 결정론 옵션 문서화 |
| C-16 | **리스크 관리** | (해당 없음) | R1–R6 연속 ID, **소유자·트리거** 컬럼 |
| C-17 | **데이터·라이선스** | 데이터셋 이름 수준 | 훈련/벤치 **약관·버전 스냅샷·인용** 체크리스트 |
| C-18 | **CI/회귀** | (해당 없음) | mock backbone **카나리**(`transition`+`mcts_expand` 스모크) |
| C-19 | **설정↔논문 추적** | 부록 A.2 나열 | `configs/README.md` **항목별 대응표**(YAML 키 ↔ 논문 문단) |
| C-20 | **문서 우선순위** | — | 상충 시 **본 개선 `.md` 우선** 규칙 명시 |

---

## “논문에 있음 / 개선안이 추가함” 빠른 범례

| 표기 의미 | 설명 |
|-----------|------|
| 논문 **있음** | 핵심 아이디어·수식·주요 하이퍼가 본문 또는 부록에 존재 |
| 논문 **부분** | 개념은 있으나 구현에 필요한 디테일 부족 |
| 논문 **없음** | 공학·운영·안전·재현을 위한 항목으로 논문에 거의 없음 |

| ID | 논문 | 개선안 추가 |
|----|:----:|:-----------:|
| C-1 메모리 정의 | 부분 | ● |
| C-2 ν vs budget 상태 | 부분 | ● |
| C-3 top-k vs 식(5) | 부분 | ● |
| C-4 Critic AC 완화 | 없음 | ● |
| C-5 Stage1 손실 명시 | 부분 | ● |
| C-6 PPO 전 항목 | 부분 | ● |
| C-7 PUCT 변체 | 없음 | ● |
| C-8 0-base 매핑 문서 | 없음 | ● |
| C-9 perf 모듈 경로 | 없음 | ● |
| C-10 Backbone 추상화 | 없음 | ● |
| C-11 긴 s_t 모니터 | 부분 | ● |
| C-12 security_eval | 없음 | ● |
| C-13 프로파일 계약 | 부분 | ● |
| C-14 Anderson 옵션 | 부분(FW) | ● |
| C-15 결정론 평가 | 없음 | ● |
| C-16–C20 운영·문서 | 없음 | ● |

● = 개선 계획서에서 **새로 두껍게 보완**한 축.

---

## 관련 파일

- 개선 개발 계획 본문: `doc/NeurIPS_2026_CTS_DEVELOPMENT_PLAN_IMPROVED.md`
- (참고) 초안 전문: `doc/NeurIPS_2026_CTS_FULL_DEVELOPMENT_PLAN_ko.txt`

---

*본 비교표는 논문 PDF와 개선 개발 계획서를 대조해 정리했으며, 논문이 정식 출판되거나 코드가 공개되면 항목을 그에 맞게 갱신하는 것이 좋다.*
