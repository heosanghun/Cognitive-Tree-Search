# 컴퓨트·실험 런북 v2.0 (논문 PDF 정합)

**GPU·데이터**는 로컬에서 실행해야 합니다. 아래 절차(및 **§7 일괄 파이프라인**)로 **논문과 동일 프로토콜**을 재현할 수 있습니다. 최종 점검은 **`python scripts/verify_cts_final_goal.py`** (`--check-artifacts` 선택).

---

## 1. 사전 조건

| 항목 | 권장 |
|------|------|
| GPU VRAM | 논문 기준 **단일 RTX 4090(24GB)** — CTS는 ≤16.7 GB 사용 |
| 로컬 스냅샷 | `artifacts/LOCAL_HARDWARE_SNAPSHOT.txt` |
| 디스크 | `model.safetensors` ~16GB + Hub 캐시 |
| 소프트웨어 | `pip install -e ".[dev]"`, Gemma4용 `transformers` 개발 빌드 |
| 인증 | `HF_TOKEN` (게이트 모델), `CTS_GEMMA_MODEL_DIR` 로컬 스냅샷 |
| **FAISS** | `pip install faiss-cpu` (또는 `faiss-gpu`, FAISS Context Window용) |

### 1.1 데이터 받기 (로컬 `data/`)

```powershell
cd D:\AI\cts
pip install "datasets>=2.16"
python scripts/download_experiment_data.py
```

생성물:
- `data/math500/test.jsonl` (500문항)
- `data/openmath_instruct/train_100000.jsonl` (스트리밍 10만 행)
- `data/stage2/math_train_prompts_5000.jsonl`
- 경로: `configs/data_paths.yaml`

### 1.2 논문 실험 프로토콜 (§7.1)

| 항목 | 논문 설정 |
|------|-----------|
| Iso-FLOP 예산 | ≤ 10¹⁴ MACs |
| 브랜칭 폭 W | 3 |
| GPU | 단일 RTX 4090 |
| Vision/Audio | **오프로딩** (~0.9 GB 절약) |
| Native Think | **`enable_thinking=False`** |
| max_new_tokens | 64 (CTS 표현 병목 = K=64) |
| 시드 | 5 seeds, 95% CI |

---

## 2. 표 1 유형 (메모리·지연 vs 깊이)

### 2.1 CTS mock + 해석적 KV + (선택) 실측 KV

```powershell
$env:HF_HUB_CACHE = "D:\AI\cts\.hf_cache"
python -m cts.eval.profile_vram_latency --depths 1 5 10 15 20 35 100 --out artifacts/table1_cts_kv.csv --cuda
python scripts/profile_kv_measured.py --depths 1 5 10 15 --out artifacts/table1_kv_measured.csv
```

### 2.2 논문 표1 기대 결과

| Model / Approach | Depth 1 | Depth 15 | Depth 35 | Depth 100+ |
|------------------|---------|----------|----------|------------|
| Mamba / RWKV | 14.2 GB | 14.2 GB | 14.2 GB | 14.2 GB |
| Gemma 4 MCTS (Vanilla) | 16.5 GB | >24.0 GB (OOM) | — | — |
| Gemma 4 MCTS (+ Prefix Caching) | 16.5 GB | 18.2 GB | >24.0 GB (OOM) | — |
| **CTS-Gemma 4 E4B (Ours)** | **16.5 GB** | **16.6 GB** | **16.6 GB** | **16.7 GB** |

### 2.3 검증 포인트

- W=3, 동일 깊이 스윕, 동일 warmup (`configs/default.yaml`의 `eval_warmup_runs`)
- CTS 노드당 지연: **~25 ms** (깊이 무관)
- CSV에 M1/M2 지표 포함

---

## 3. 표 2 유형 (5종 벤치마크, Iso-FLOP)

### 3.1 MATH 500

```powershell
$env:CTS_GEMMA_MODEL_DIR = "D:\AI\cts\gemma-4-E4B-it"
python scripts/run_math500.py --data data/math500/test.jsonl --gemma --limit 500 --think-prompt --out-json artifacts/table2_math500_run.json
```

### 3.2 GSM8K (논문 표2 — 신규)

```powershell
python scripts/run_gsm8k.py --data data/gsm8k/test.jsonl --gemma --limit 1319 --out-json artifacts/table2_gsm8k_run.json
```

> **참고:** GSM8K 스크립트가 아직 없으면 `eval/gsm8k.py` 구현 후 실행. 데이터: `datasets` HF에서 `gsm8k` 다운로드.

### 3.3 ARC-AGI-Text

```powershell
python scripts/run_arc_agi_text.py --data data/arc/test.jsonl --gemma --limit 200 --out-json artifacts/table2_arc_run.json
```

### 3.4 HumanEval (논문 표2 — 신규)

```powershell
python scripts/run_humaneval.py --data data/humaneval/test.jsonl --limit 164 --out-json artifacts/table2_humaneval_run.json
```

> **참고:** HumanEval은 **로컬 오프라인 실행** (논문 §7.1). `security_eval.md` 정책 준수. 샌드박스 환경에서 코드 실행 채점.

### 3.5 논문 표2 기대 결과 (Iso-FLOP, ≤ 10¹⁴ MACs, 5 seeds, 95% CI)

| Model / Approach | MATH | GSM8K | AIME | ARC | HumanEval | Avg. MACs |
|------------------|------|-------|------|-----|-----------|-----------|
| Gemma 4 (Greedy) | 45.2 | 76.5 | 28.3 | 36.1 | 56.4 | 0.05 |
| Gemma 4 SC@14 | 59.3±0.7 | 84.2±0.5 | 34.8±0.9 | 52.4±0.8 | 65.2±0.6 | 1.0 |
| Gemma 4 Native Think | 57.0±0.6 | 82.4±0.4 | 42.5±0.9 | 50.1±0.7 | 63.3±0.5 | Internal |
| **CTS (Ours)** | **68.4±0.8** | **92.1±0.5** | **56.4±1.1** | **64.1±0.9** | **74.2±0.6** | **0.65** |

### 3.6 한 번에 5종 벤치

```powershell
python scripts/run_table2_full_bench.py
```

`artifacts/table2_full_bench_manifest.txt` 에 결과 경로 기록.

---

## 4. Stage 1 / 2 학습 (대규모 컴퓨트)

### 4.1 Stage 1 (DEQ Warm-Up — 논문 §6.1)

```powershell
python scripts/run_stage1_openmath.py
```

| 항목 | 논문 값 |
|------|---------|
| 데이터 | OpenMathInstruct-2, **10,000 examples** |
| 손실 | IFT 잔차 ∥f(z*)−z*∥²₂ |
| Gemma 4 | **동결** |
| 학습 파라미터 | LoRA r=8, α=16 + Wg (~18 MB) |
| 체크포인트 | `artifacts/stage1_last.pt` |

### 4.2 Stage 2 (PPO — 논문 §6.2)

```powershell
python scripts/run_stage2_math_ppo.py --stage1-ckpt artifacts/stage1_last.pt
```

| 항목 | 논문 값 |
|------|---------|
| 데이터 | MATH/AIME **5,000 prompts** (평가셋 분리) |
| 보상 | Eq.(5): R = 1{correct} − 0.05 · T |
| PPO | lr=3e-5, clip=0.2, γ=0.99, GAE λ=0.95 |
| Critic | lr=1e-4 |
| 선택 | `--use-critic-reward`, `--parallel-map` |
| 체크포인트 | `artifacts/stage2_meta_value.pt` |

### 4.3 한 번에 보기

```powershell
python scripts/run_full_training_stack.py     # 명령 확인
python scripts/run_full_training_stack.py --run  # 실제 실행
```

---

## 5. FAISS Latent Space Context Window (논문 §4.4)

### 5.1 설치

```powershell
pip install faiss-cpu    # GPU 환경: pip install faiss-gpu
```

### 5.2 프로파일

FAISS 인덱스의 노드당 메모리를 확인:

```powershell
python -m cts.latent.faiss_context --profile --nodes 100 200 500
```

### 5.3 기대 동작

- t ≤ 10: FAISS 비활성 (히스토리 부족)
- t > 10: Top-3 시맨틱 조상 검색 → prepended soft-prefix
- 노드당 메모리: 수 KB (mean-pooled 1D 벡터)
- p-RoPE 격리: 검색 벡터의 위치 인덱스가 활성 전이와 분리

---

## 6. 애블레이션 (논문 §7.4)

### 6.1 기존 애블레이션

```powershell
python scripts/run_ablations.py --config ablation_no_ach   # dense routing
python scripts/run_ablations.py --config ablation_static_5ht  # static νexpl
```

### 6.2 논문 신규 애블레이션

| 애블레이션 | 논문 결과 | 설정 |
|-----------|----------|------|
| w/o Latent Context Window | **−4.9%** | `faiss_enabled: false` |
| w/o ACT (νact) | MAC 과소비 | `act_enabled: false` |
| K=32 | symbolic loss 심각 | `latent_tokens: 32` |
| K=128 | Jacobian O(K²) 폭발 | `latent_tokens: 128` |
| Top-k=1 | representation collapse | `routing_top_k: 1` |
| Top-k=4+ | 무의미한 비용 증가 | `routing_top_k: 4` |

---

## 7. 일괄 실행 (표 1·2 CSV + Stage1/2 → `artifacts/`)

```powershell
cd D:\AI\cts
$env:CTS_GEMMA_MODEL_DIR = "D:\AI\cts\gemma-4-E4B-it"

# 빠른 스모크 (기본)
python scripts/run_paper_artifacts_pipeline.py --tier quick --skip-download

# 표준 실행
python scripts/run_paper_artifacts_pipeline.py --tier standard

# 논문 재현 (전체 5 seeds)
python scripts/run_paper_artifacts_pipeline.py --tier full
```

`--skip-download`: 이미 `data/`가 있을 때 Hub 스트리밍 생략.

---

## 8. RoPE 앵커 vs inner `z` (계약 완료)

- **단일 참조:** `cts/backbone/rope_contract.py`
- 앵커: `encode_context`(HF RoPE), latent 업데이트: `deq_step`에서 `z`·`context`만 사용
- FAISS soft-prefix의 p-RoPE 격리: 검색 벡터는 활성 전이의 상대 위치 인덱스와 수학적으로 분리

---

## 9. 재현성

- `PYTHONHASHSEED`, `torch.manual_seed`, `CTS_*` 환경 변수를 실험 로그에 기록
- 논문: **5 seeds, 95% CI** — `configs/default.yaml`의 `eval_seeds: 5`, `eval_ci_level: 0.95`
- `artifacts/`에 CSV·JSON 버전 관리 (git-lfs 또는 외부)

---

## 10. Broyden 수렴 통계 추적 (부록 C)

Broyden 솔버 실행 시 아래 통계를 자동 로깅:

| 지표 | 논문 기대값 |
|------|-------------|
| 수렴률 | 97.3 ± 0.4% |
| 평균 반복 수 | 11.2 ± 2.8 |
| 폴백률 | 2.7 ± 0.4% |
| 스펙트럼 반경 γ | ≈ 0.92 |

```powershell
python -m cts.deq.broyden_forward --convergence-report --out artifacts/broyden_stats.json
```

---

## 11. Vision/Audio 인코더 오프로딩 (§7.1)

텍스트 전용 추론 시 vision (~150M) + audio (~300M) 인코더를 VRAM에서 제거:

```python
from cts.model.gemma_loader import load_gemma4_e4b
model, tok = load_gemma4_e4b(
    r"D:\AI\cts\gemma-4-E4B-it",
    device_map="cuda:0",
    offload_vision_audio=True  # ~0.9 GB 절약
)
```

---

## 12. 최종 목표 검증

```powershell
python scripts/verify_cts_final_goal.py
python scripts/verify_cts_final_goal.py --check-artifacts
```

이 런북은 **실행 책임이 있는 쪽(연구실/클라우드)**에서 그대로 따라 할 수 있도록 **논문 PDF와 동일 프로토콜**을 유지합니다.
