# 논문 조건 vs 저장소(현재) 조건 — 동일 재현용 비교표

**목적:** 논문(NeurIPS 2026 CTS 프리프린트 요약·부록 A.2 정합 `configs/`)과 **이 저장소의 기본값·실행 경로**를 나란히 두고, **동일 조건 재현**에 필요한 조정을 한눈에 본다.

**전제:** “논문 PDF와 숫자까지 동일”은 아래 **논문 열**에 맞춰 환경·인자·학습 길이를 맞춘 뒤 **실제 실행**으로만 검증한다.

---

## 1. 하드웨어·소프트웨어

| 항목 | 논문(요약·개발계획) | 저장소 기본/전제 | 동일 재현 시 |
|------|---------------------|------------------|--------------|
| GPU | 단일 **RTX 4090 (24GB)** 등 소비자급 타깃 | 런북·스냅샷 예시 동일 가정 (`doc/COMPUTE_AND_EXPERIMENT_RUNBOOK.md`) | 동일 GPU·드라이버·CUDA 가능하면 동일 |
| VRAM·OOM 논의 | 표준 MCTS는 깊이 ~15 OOM, CTS는 100+에서도 ~16.5–16.7GB 등 | 프로파일 스크립트로 측정 (`profile_vram_latency`) | 동일 `W`, 깊이 스윕, 워밍업(`eval_warmup_runs`) |
| PyTorch / Transformers | PyTorch 2.x, 최신 `transformers`(Gemma4) | README: `pip install git+https://github.com/huggingface/transformers.git` | 논문 실험 시점과 **동일 빌드 고정** 권장(커밋/버전 기록) |
| Gemma 가중치 | **Gemma 4 E4B** BF16 | `google/gemma-4-E4B(-it)` 또는 로컬 `CTS_GEMMA_MODEL_DIR` | **동일 체크포인트**(base vs `-it` 구분) |
| HF 인증 | 게이트 모델 시 토큰 | `HF_TOKEN` | 동일 계정·동일 스냅샷 |

---

## 2. DEQ·Broyden·MCTS (코어 하이퍼)

| 항목 | 논문(부록 A.2 / 요약) | `configs/default.yaml` | 동일 재현 시 |
|------|------------------------|-------------------------|--------------|
| 가중치 dtype | BF16 로드 | `dtype_weights: bfloat16` | 동일 |
| Broyden 최대 반복 | **30** | `broyden_max_iter: 30` | 동일 |
| Broyden tol | ν_NE에 따른 **동적** [1e-4, 1e-2] | `broyden_tol_min/max` | 동일 스케줄 쓰는지 코드 경로 확인 |
| Broyden 내부 FP32 | 부록: 내부 FP32 | 구현은 모듈별 | 재현 시 동일 구현 경로 고정 |
| MCTS 분기 폭 **W** | **3** 고정 | `mcts_branching_W: 3` | 동일 |
| 시뮬레이션/스텝 | 예: **3** (요약) | `mcts_simulations_per_step: 3` | 동일 |
| PUCT 변형 | 논문 식 | `puct_variant: paper` | 동일 |
| 희소 top-k | **2~3** 모듈 | `top_k_modules: 3` | 동일 |
| Iso-FLOP 예산 τ | ~**1e14** / 쿼리 (Sec 7.3 요지) | `tau_flops_budget: 1.0e14` | 동일 |
| DEQ inner 맵 | 논문: 희소 병렬 모듈 경로 강조 | 기본 **`CTS_DEQ_MAP_MODE=blend`** (가벼운 선형 어댑터) | 논문에 가깝게: **`parallel`** 또는 `full` + 문서화된 부담 |
| Soft thought **K×d** | 가변·설정 가능 | `soft_thought_K: 8`, `soft_thought_d: 64` | 논문/부록과 동일 값으로 고정 |

---

## 3. 학습 (Stage 1 / 2)

| 항목 | 논문(요약·부록) | 저장소 | 동일 재현 시 |
|------|------------------|--------|--------------|
| Stage1 데이터 | OpenMathInstruct **약 1만 예** 워밍업 | `download_experiment_data.py` 기본 **10만 행** JSONL 스트리밍; 학습은 JSONL 순환 | 데이터 **출처·행 수**를 논문과 맞출지(1만 vs 10만) 결정 후 고정 |
| Stage1 **스텝 수** | 요약은 “1만 예” 서술(해석: 예제 수 또는 스텝) | `stage1_max_steps: **2000**`; `stage1_openmath_n: 10000`은 YAML에만 있고 **학습 루프에서 미사용** | 논문이 **옵티마이저 스텝 기준**이면 `--max-steps`를 부록과 맞춤(예: 10000) |
| Stage2 프롬프트 | MATH **5k** 서브셋 등 | `stage2_math_prompts_n: 5000`, `data/stage2/math_train_prompts_5000.jsonl` | 동일 JSONL·동일 행 수 |
| PPO 총 스텝 | **1만 스텝** (요약) | `stage2_total_ppo_steps: 10000` (`run_stage2_math_ppo` 기본) | 동일 |
| PPO lr / batch / γ | lr **3e-5**, batch **16**, γ **0.95** | 동일 키 `default.yaml` | 동일 |
| LoRA | rank **8**, Q/V | `lora_rank: 8`, `lora_target: [q_proj, v_proj]` | 동일 |
| Stage2 inner 맵 | 논문 스타일은 parallel 쪽 | 기본 `parallel_map=False` → **blend** | `parallel_map=True` 또는 `CTS_DEQ_MAP_MODE=parallel`로 정렬 |

---

## 4. 평가·벤치 (표 1·2)

| 항목 | 논문 | 저장소 | 동일 재현 시 |
|------|------|--------|--------------|
| 표1 깊이 스윕 | 논문 표와 동일 축 | `experiment_paper_protocol.yaml`: `profile_depths` 등 | 동일 depth 리스트 |
| MATH | **500** (MATH-500) | `run_math500.py --limit`; `data_paths` `eval_math_limit: null` = 전체 | `--limit 500`, 전체 JSONL |
| ARC | 논문 표2 (예: **200** 문항 등 본문 기준) | `run_arc_agi_text.py --limit 200`, 데이터는 **별도 JSONL** | 동일 데이터·동일 `limit` |
| Think / chat | Native Think·템플릿 비교 | `--think-prompt`, `--chat-template` 옵션 | 논문에서 켠 설정과 동일하게 |
| 파이프라인 **tier** | — | 기본 **`quick`**: MATH 24, 짧은 학습 등 | 논문에 가깝게: **`--tier full`** (그래도 Stage 스텝은 아래 참고) |
| `run_paper_artifacts_pipeline --tier full` | 논문 전체 학습 길이와 동일하다고 보장하지 않음 | Stage1 **500** 스텝, Stage2 **100** 스텝 등 **파이프라인 고정 단축** | **장시간 재현**은 `run_stage1_openmath` / `run_stage2_math_ppo`를 **config만** 쓰고 스텝은 YAML·CLI로 논문과 맞춤 |

---

## 5. 재현성·로그 (공통)

| 항목 | 논문 | 저장소 | 동일 재현 시 |
|------|------|--------|--------------|
| 시드 | 부록/본문 기준 | `eval_deterministic: false` 기본 | 보고용: `PYTHONHASHSEED`, `torch.manual_seed`, 샘플링 끄기 등 **명시적 고정** |
| 로그·산출 | 표·CSV | `--out-json`, `artifacts/`, `RUN_MANIFEST.json` | 동일 스크립트·동일 tier가 아니면 숫자 비교 불가 |

---

## 6. 한 줄 정리

| 구분 | 의미 |
|------|------|
| **이미 맞춰 둔 것** | W=3, Broyden 30/tol, PPO·LoRA·τ·top-k·데이터 소스(MATH-500 등) 대부분 `default.yaml` / 런북과 논문 서술 정합 |
| **반드시 맞출 선택** | (1) **Gemma 스냅샷 base vs -it**, (2) **DEQ `blend` vs `parallel`**, (3) **Stage1 스텝 수**(2000 vs 10000 등), (4) **파이프라인 tier vs 직접 학습 스크립트**, (5) **시드·결정적 평가** |
| **숫자 100% 일치** | 위 표를 논문 조건으로 맞춘 뒤 **동일 머신에서 전체 학습·평가를 재실행**했을 때만 검증 가능 |

---

## 7. 논문과 동일 설정으로 **전량 실험**을 돌리기 위한 필요 항목 (전체 리스트)

아래는 **하나라도 빠지면 “논문과 동일 조건”이라고 말하기 어려운 항목**을 빠짐없이 모은 체크리스트다. 실행 순서는 대략 **환경 → 데이터 → 설정 고정 → 학습 → 평가 → 기록**이다.

### A. 계정·라이선스·디스크

- [ ] Hugging Face **계정** 및 Gemma **라이선스 동의** (게이트 모델이면 `HF_TOKEN` 발급·환경 변수)
- [ ] **Gemma 4 E4B** (또는 논문과 동일한) **체크포인트** 확보: Hub 스냅샷 또는 로컬 폴더 (`CTS_GEMMA_MODEL_DIR`)
- [ ] 논문 실험과 **동일 변형** 사용 여부 확정: **`gemma-4-E4B`** vs **`gemma-4-E4B-it`** (벤치·채팅 템플릿·`<|think|>` 경로가 달라짐)
- [ ] 가중치·캐시용 **디스크** (~16GB+ 모델, Hub 캐시), 필요 시 `HF_HUB_CACHE`를 여유 드라이브로 지정
- [ ] `pip install -e ".[dev]"` 및 **`transformers` Gemma4 지원 빌드** (README의 git 설치 등)
- [ ] 선택: `pip install peft` (LoRA 학습 시), 데이터셋용 `pip install -e ".[data]"` 또는 `datasets`

### B. 하드웨어

- [ ] 논문과 **동일 또는 상정된 GPU** (요약: **단일 RTX 4090 24GB** 등) — VRAM·속도 수치 비교 시 중요
- [ ] 동일 실험 내에서 **GPU·드라이버·CUDA** 고정 (가능하면 버전 문자열까지 로그에 남김)
- [ ] Stage2·전체 벤치용 **충분한 wall-clock 시간** (수 시간~수 일 가능)

### C. 데이터 (로컬 경로 고정)

- [ ] `python scripts/download_experiment_data.py` (또는 논문과 동일 출처·행 수로 수동 준비)
- [ ] **MATH-500** 평가용 JSONL: `data/math500/test.jsonl` (500행) — 논문 표2 재현 시 **전량**
- [ ] **OpenMathInstruct** 학습용: 논문이 말한 **행 수(예: 1만 vs 10만)** 와 **동일한 서브샘플 규칙** 결정 후 JSONL 경로 고정
- [ ] **Stage2 MATH 프롬프트** 5000행: `data/stage2/math_train_prompts_5000.jsonl` (또는 동일 구축 스크립트)
- [ ] **ARC**가 표2에 포함되면: 논문과 동일한 **ARC JSONL** 확보 + `run_arc_agi_text.py`에 넘길 경로
- [ ] `configs/data_paths.yaml`과 실제 파일 존재 여부 일치 확인

### D. 설정 파일·하이퍼 (논문·부록과 1:1)

- [ ] 기준 설정: `configs/default.yaml` (또는 복사본)에서 최소한 다음을 논문과 맞춤  
  - [ ] `mcts_branching_W`, `mcts_simulations_per_step`, `puct_variant`  
  - [ ] `broyden_max_iter`, `broyden_tol_min` / `broyden_tol_max`  
  - [ ] `tau_flops_budget`, `top_k_modules`, `soft_thought_K` / `soft_thought_d`  
  - [ ] PPO: `lr`, `batch_size`, `gamma`, `gae_lambda`, `entropy_coef`, `lora_rank`, `lora_target`  
  - [ ] `stage2_total_ppo_steps`, `stage1_max_steps`(또는 CLI `--max-steps`) — **논문 스텝 정의와 일치**시킬 것  
- [ ] **DEQ inner 경로** 논문 정렬: `CTS_DEQ_MAP_MODE` = **`parallel`** 또는 `full` (기본 `blend`는 논문 대비 경량 경로)
- [ ] Stage2 스크립트에서 **`parallel_map=True`** 등 논문 경로와 동일한 플래그 조합
- [ ] `configs/experiment_paper_protocol.yaml`의 **표1 depth 리스트** 등을 논문 표와 동일하게
- [ ] 평가 워밍업: `eval_warmup_runs` 등 프로파일·지연 비교 시 논문과 동일

### E. 재현성 (시드·결정성)

- [ ] `PYTHONHASHSEED`, `torch.manual_seed`, (필요 시) `numpy`, `random` 시드 고정
- [ ] 평가 시 **결정적 디코드**가 논문 조건이면: 샘플링 끄기·`eval_deterministic` 등 **명시적 정책** 기록
- [ ] 사용한 **커밋 해시**(이 저장소), **transformers 커밋**, **CUDA/PyTorch 버전**을 로그에 남김

### F. 학습 (논문 순서·길이)

- [ ] **Stage 1**: `scripts/run_stage1_openmath.py` — 논문과 동일 **스텝 수**(`--max-steps`), 동일 OpenMath JSONL, 필요 시 `--lora`
- [ ] 산출 체크포인트 경로 기록: `artifacts/stage1_last.pt` (또는 지정 경로)
- [ ] **Stage 2**: `scripts/run_stage2_math_ppo.py` — `--stage1-ckpt`로 Stage1 로드, **총 PPO 스텝**·배치·`parallel_map`·Critic 사용 여부 논문과 동일
- [ ] `run_paper_artifacts_pipeline.py`만 쓰지 말 것: 기본·tier는 **학습 스텝이 짧게 잡힌 경우가 많음**. 논문 동일 **전량 학습**은 직접 Stage1/2 스크립트 + YAML/CLI로 맞추는 것이 안전

### G. 평가·벤치 (표 1·2·Iso-FLOP)

- [ ] **표1 (VRAM·지연)**: `python -m cts.eval.profile_vram_latency` — 논문과 동일 `W`, depth 스윕, `approach` 정의  
- [ ] 선택: `scripts/profile_kv_measured.py` — 실측 KV (환경 허용 시)
- [ ] **표2 MATH**: `scripts/run_math500.py` — `--limit 500`, 동일 `--gemma`, 논문과 동일 **`--think-prompt` / `--chat-template`** 조합, `--out-json`으로 결과 고정
- [ ] **표2 ARC** (해당 시): `scripts/run_arc_agi_text.py` — 논문과 동일 `limit`·데이터
- [ ] **Iso-FLOP**: `report_isoflop` 등 — `flops_contract` 필드명·DEQ vs 디코드 구분 논문과 동일하게 기록
- [ ] 한 번에 돌리되 **논문 전량**이 목적이면: `scripts/run_table2_full_bench.py` 또는 런북의 명령 조합 (여전히 **학습·체크포인트**는 위 F와 일치해야 함)

### H. 산출물·로그 (비교 가능하게 남기기)

- [ ] 모든 실험에 **고정된 `artifacts/` 파일명** 또는 날짜·run id가 붙은 디렉터리
- [ ] MATH/ARC: **`--out-json`** (문항별 match, 요약 pass@1 등)
- [ ] 일괄 실행 시: `RUN_MANIFEST.json`, 환경 변수 스냅샷, `LOCAL_HARDWARE_SNAPSHOT.txt` 수준의 기록
- [ ] 논문 표의 각 셀과 **대응되는 스크립트·인자**를 README 한 줄 메모로 남기면 재현 검증이 쉬움

### I. 논문 자체 (메타)

- [ ] 논문 **PDF 부록**의 표 번호·하이퍼·시드·데이터 버전 — 저장소 요약(`doc/NeurIPS_2026_*`)과 **차이가 없는지** 최종 대조
- [ ] 논문에 **베이스라인**(Native Think, 일반 MCTS 등)이 있으면 동일 스크립트·동일 데이터로 **별도 실행**해 비교표 작성

---

## 8. A–I 충족도 비교표 (주관적 추정)

**의미:** §7 체크리스트 각 구간(A–I)을 **논문 동일 재현** 기준으로 얼마나 채웠는지에 대한 **대략적인 비율**이다. 절대적인 객관 지표가 아니며, 팀·환경마다 달라진다.

**열 정의**

| 열 | 설명 |
|----|------|
| **목표** | 논문과 **동일 조건 재현이 끝난 상태** — 정의상 각 구간 **100%** |
| **현재 저장소** | **코드·`configs/`·문서·스크립트**만으로 “준비·구현”된 비율 (계정·GPU·실행 시간은 제외) |
| **전형적 로컬** | **24GB급 GPU**, `HF_TOKEN`, 데이터 다운로드, **수일 단위** 시간을 쓸 수 있는 경우의 추정 |
| **일반 현실** | GPU/토큰/디스크/연속 실행 시간 중 **하나 이상이 부족한** 일반적인 개발 환경 추정 |

### 표 (구간별 %) — **2026-04 자동화 반영 후** 갱신

`configs/paper_parity.yaml`, `scripts/verify_repro_prereqs.py`, `scripts/log_repro_environment.py`, `CTS_GLOBAL_SEED` + `repro_seed`, `run_paper_artifacts_pipeline --tier full --config paper_parity`(학습 스텝·Broyden·parallel 플래그가 YAML을 따름), `artifacts/REPRO_ENV.json` 도입 후 **저장소가 닫을 수 있는 부분**을 상향 조정했다. **물리 GPU·HF 계정 클릭·PDF 대조**는 여전히 외부 요인이다.

| 구간 | 목표 | 저장소·자동화 (코드로 닫는 한도) | 전형적 로컬 | 일반 현실 |
|------|------|----------------------------------|-------------|-----------|
| **A** | 100% | ~78% | ~90% | ~35% |
| **B** | 100% | ~35% | ~85% | ~25% |
| **C** | 100% | ~88% | ~90% | ~55% |
| **D** | 100% | ~96% | ~88% | ~60% |
| **E** | 100% | ~88% | ~75% | ~45% |
| **F** | 100% | ~96% | ~80% | ~30% |
| **G** | 100% | ~93% | ~82% | ~40% |
| **H** | 100% | ~93% | ~88% | ~55% |
| **I** | 100% | ~58% | ~72% | ~35% |
| **9구간 단순평균** | **100%** | **~81%** | **~83%** | **~46%** |

**해석 요약**

- **저장소·자동화:** **약 81%** — 논문 PDF **표·수치 일치 증명**(I)과 **실물 GPU**(B)는 코드만으로 100%가 될 수 없다.
- **전형적 로컬:** 자원·시간이 있으면 **~83%**까지; **숫자 동일**은 여전히 **실행 결과**로만 확정.
- **일반 현실:** 제약이 많으면 **~46%** 수준.

---

## 9. 논문 정렬 실행 순서 (권장)

1. `pip install -e ".[dev]"` 및 Gemma용 `transformers` (README).
2. `python scripts/verify_repro_prereqs.py` — 디스크·데이터 경로·(선택) GPU.
3. (선택) `$env:CTS_GLOBAL_SEED = "42"` — 재현 시드.
4. `python scripts/log_repro_environment.py` — `artifacts/REPRO_ENV.json` 갱신.
5. `python scripts/download_experiment_data.py`
6. **전량·부록 길이 학습 + 평가:**  
   `python scripts/run_paper_artifacts_pipeline.py --tier full --config paper_parity`  
   (ARC는 JSONL 준비 후 `--skip-arc` 제거·`--arc-data` 지정.)

---

**참고 문서:** `configs/README.md`, `configs/experiment_paper_protocol.yaml`, `doc/COMPUTE_AND_EXPERIMENT_RUNBOOK.md`, `doc/NeurIPS_2026_CTS_paper_summary_and_implementation_plan_ko.txt`.
