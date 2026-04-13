# Project File Inventory (Internal Reference)

> 내부 관리용 문서. README 또는 논문에 포함하지 않음.
> 최종 업데이트: 2026-04-13

## 전체 요약

| 카테고리 | 용량 | 파일 수 | 비율 |
|----------|-----:|:-------:|:----:|
| AI 모델 가중치 (Gemma × 3 copies) | 44.76 GB | 69 | 74.9% |
| 학습 체크포인트 (Stage 1 + Stage 2) | 14.94 GB | 27 | 25.0% |
| 벤치마크 + 학습 데이터 | 25.77 MB | 10 | 0.04% |
| 문서 (논문 PDF 포함) | 19.52 MB | 18 | 0.03% |
| 소스코드 + 스크립트 + 테스트 + 설정 | 0.80 MB | 282 | 0.001% |
| **총합** | **59.77 GB** | **406** | **100%** |

> `.hf_cache/`를 정리하면 약 15GB 절약 가능 (모델 중복 저장).

---

## 1. 모델 가중치

### gemma-4-E4B/ (14.92 GB) — 기본 모델

| 파일 | 용량 | 용도 |
|------|-----:|------|
| `model.safetensors` | 14.90 GB | Gemma 4 E4B 모델 가중치 |
| `tokenizer.json` | 30.68 MB | 토크나이저 사전 |
| `README.md` | 26.1 KB | 모델 카드 |
| `config.json` | 5.0 KB | 모델 구조 설정 |
| `processor_config.json` | 1.6 KB | 프로세서 설정 |
| `.gitattributes` | 1.5 KB | Git LFS 설정 |
| `tokenizer_config.json` | 0.9 KB | 토크나이저 설정 |
| `generation_config.json` | 0.2 KB | 생성 설정 |
| `.cache/` (10개) | ~1 KB | 다운로드 메타데이터 |

> 경로: `d:\AI\cts\gemma-4-E4B\`

### gemma-4-E4B-it/ (14.92 GB) — 인스트럭션 튜닝 모델

| 파일 | 용량 | 용도 |
|------|-----:|------|
| `model.safetensors` | 14.90 GB | Gemma 4 E4B-it 모델 가중치 (대화형) |
| `tokenizer.json` | 30.68 MB | 토크나이저 |
| `chat_template.jinja` | 11.6 KB | 채팅 템플릿 |
| `README.md` | 26.1 KB | 모델 카드 |
| `config.json` | 5.0 KB | 모델 구조 설정 |
| 나머지 | ~7 KB | 설정 + 캐시 |

> 경로: `d:\AI\cts\gemma-4-E4B-it\`

### .hf_cache/ (14.92 GB) — HuggingFace 캐시

| 파일 | 용량 | 용도 |
|------|-----:|------|
| `models--google--gemma-4-E4B/.../model.safetensors` | 14.90 GB | gemma-4-E4B 캐시 복사본 |
| `models--google--gemma-4-E4B/.../tokenizer.json` | 30.68 MB | 토크나이저 캐시 |
| 데이터셋 메타 (EleutherAI, HuggingFaceH4, nvidia) | ~11 KB | 데이터셋 참조 정보 |

> 경로: `d:\AI\cts\.hf_cache\`

---

## 2. 학습 결과 / 체크포인트 — artifacts/ (14.95 GB)

| 파일 | 용량 | 용도 |
|------|-----:|------|
| `stage1_last.pt` | 14.91 GB | Stage 1 DEQ 학습 체크포인트 |
| `stage2_meta_value.pt` | 27.54 MB | Stage 2 PPO 메타 정책 체크포인트 |
| `math500_result.json` | 22.0 KB | MATH-500 평가 상세 결과 |
| `log_stage2_math_ppo.txt` | 18.3 KB | Stage 2 학습 로그 |
| `log_stage1_openmath.txt` | 4.2 KB | Stage 1 학습 로그 |
| `table2_real_benchmark_results.json` | 2.8 KB | 벤치마크 결과 |
| `table2_math500_gemma_log.txt` | 2.0 KB | MATH-500 Gemma 평가 로그 |
| `five_seed_stats.json` | 1.1 KB | 5-시드 통계 |
| `RUN_MANIFEST.json` | 1.1 KB | 실행 매니페스트 |
| `table1_cts_kv.csv` | 0.8 KB | Table 1 CTS VRAM 데이터 |
| `table1_local_run.csv` | 0.6 KB | Table 1 로컬 실행 |
| `table2_full_comparison.json` | 0.6 KB | Table 2 비교 결과 |
| `LOCAL_HARDWARE_SNAPSHOT.txt` | 0.7 KB | 하드웨어 스냅샷 |
| `REPRO_ENV.json` | 0.5 KB | 재현 환경 정보 |
| `table1_real_gemma_vram.json` | 0.5 KB | Gemma VRAM 실측 |
| `table1_kv_measured.csv` | 0.4 KB | KV 캐시 측정 |
| `log_download_experiment_data.txt` | 0.4 KB | 다운로드 로그 |
| `kv_measured_local.csv` | 0.3 KB | KV 측정 로컬 |
| `table2_paper_reproduction.json` | 0.3 KB | 논문 재현 결과 |
| `table2_math500_metrics.json` | 0.3 KB | MATH-500 메트릭 |
| `spectral_radius.json` | 0.3 KB | 스펙트럴 반경 |
| `remaining_experiments_manifest.json` | 0.2 KB | 남은 실험 목록 |
| `table2_cts_math_result.json` | 0.2 KB | CTS 수학 결과 |
| `table2_isoflop_mock.json` | 0.2 KB | Iso-FLOP 모의 결과 |
| `log_table1_kv_measured.txt` | 0.2 KB | Table 1 측정 로그 |
| `log_table1_profile_vram_latency.txt` | 0.2 KB | VRAM 프로파일 로그 |
| `_torch_test_output.txt` | 0 KB | 토치 테스트 출력 |

> 경로: `d:\AI\cts\artifacts\`

---

## 3. 벤치마크 + 학습 데이터 — data/ (25.77 MB)

| 파일 | 용량 | 용도 |
|------|-----:|------|
| `openmath_10k.jsonl` | 8.05 MB | OpenMath 학습 데이터 (루트 복사본) |
| `openmath_instruct/train_10000.jsonl` | 8.05 MB | Stage 1 학습용 (10,000문제) |
| `openmath_instruct/train_8000.jsonl` | 6.43 MB | Stage 1 학습용 (8,000문제 서브셋) |
| `stage2/math_train_prompts_5000.jsonl` | 1.48 MB | Stage 2 PPO 프롬프트 (5,000개) |
| `gsm8k/test.jsonl` | 732.7 KB | GSM8K 벤치마크 (1,319문제) |
| `math500/test.jsonl` | 436.6 KB | MATH-500 벤치마크 (500문제) |
| `arc_agi/test.jsonl` | 375.9 KB | ARC-AGI 벤치마크 |
| `humaneval/test.jsonl` | 209.5 KB | HumanEval 벤치마크 (164문제) |
| `aime/test.jsonl` | 41.4 KB | AIME 벤치마크 |
| `README.md` | 0.9 KB | 데이터 폴더 설명 |

> 경로: `d:\AI\cts\data\`

---

## 4. 문서 — doc/ (19.52 MB)

| 파일 | 용량 | 용도 |
|------|-----:|------|
| `Cognitive_Tree_Search_...(1).pdf` | 9.70 MB | 논문 PDF (최신) |
| `NeurIPS_2026_Cognitive_Tree_Search_...pdf` | 9.64 MB | 논문 PDF (이전 버전) |
| `BEGINNER_GUIDE_BENCHMARKS.md` | 28.9 KB | 초보자 벤치마크 가이드 |
| `NeurIPS_2026_CTS_DEVELOPMENT_PLAN_IMPROVED.md` | 24.5 KB | 개선 개발 계획 |
| `PAPER_CODE_VERIFICATION.md` | 19.2 KB | 논문-코드 검증 |
| `NeurIPS_2026_CTS_FULL_DEVELOPMENT_PLAN_ko.txt` | 18.5 KB | 개발 계획 (한국어) |
| `CTS_FINAL_DEVELOPMENT_PLAN.md` | 17.5 KB | 최종 개발 계획서 |
| `PAPER_VS_REPO_REPRO_CONDITIONS.md` | 15.6 KB | 논문 vs 코드 재현 조건 |
| `NeurIPS_2026_CTS_paper_summary_...ko.txt` | 14.4 KB | 논문 요약 (한국어) |
| `PRESENTATION_GUIDE.md` | 13.7 KB | 발표 가이드 |
| `PROJECT_FILE_INVENTORY.md` | — | 이 문서 (파일 인벤토리) |
| `PAPER_ALIGNMENT_PROGRESS.md` | 9.3 KB | 논문 정합 진행률 |
| `PAPER_VS_IMPROVED_PLAN_COMPARISON.md` | 9.2 KB | 논문 vs 개선안 비교 |
| `COMPUTE_AND_EXPERIMENT_RUNBOOK.md` | 9.1 KB | 실험 실행 매뉴얼 |
| `THIRD_PARTY_NOTICES.md` | 1.9 KB | 서드파티 라이선스 |
| `memory_definitions.md` | 0.8 KB | 메모리 정의 |
| `layer_mapping.md` | 0.5 KB | 레이어 매핑 |
| `stage1_objective.md` | 0.4 KB | Stage 1 목적 |
| `security_eval.md` | 0.4 KB | 보안 평가 |

> 경로: `d:\AI\cts\doc\`

---

## 5. 소스코드 — cts/ (0.33 MB)

| 모듈 | 주요 파일 | 용량 | 용도 |
|------|-----------|-----:|------|
| `deq/` | `transition.py`, `broyden_forward.py`, `gemma_latent_forward.py` | 22.0 KB | DEQ 고정점 전이 + Broyden 솔버 |
| `eval/` | `math500.py`, `gsm8k.py`, `humaneval.py`, `arc_agi_text.py` 등 | 26.9 KB | 벤치마크 평가 |
| `train/` | `stage1_openmath_train.py`, `stage2_ppo_train.py` 등 | 25.5 KB | Stage 1/2 학습 |
| `mcts/` | `episode.py`, `puct.py`, `deep_rollout.py` 등 | 18.0 KB | MCTS 트리 탐색 |
| `backbone/` | `gemma_adapter.py`, `mock_tiny.py` 등 | 11.4 KB | 모델 어댑터 |
| `latent/` | `bottleneck.py`, `faiss_context.py` | 7.9 KB | 잠복 공간 + FAISS |
| `model/` | `gemma_loader.py`, `module_partition.py` | 5.2 KB | 모델 로더 |
| `routing/` | `sparse_moe_ref.py`, `lut_mac.json` | 2.0 KB | 희소 라우팅 |
| `policy/` | `meta_policy.py` | 1.8 KB | 메타 정책 |
| `critic/` | `neuro_critic.py` | 1.5 KB | 뉴로 크리틱 |
| `rewards/` | `shaping.py` | 1.0 KB | 보상 함수 |
| `utils/` | `config.py`, `repro_seed.py` 등 | 4.2 KB | 유틸리티 |
| `types.py` | — | 2.7 KB | 타입 정의 |

> 경로: `d:\AI\cts\cts\`

---

## 6. 스크립트 — scripts/ (0.27 MB)

| 파일 | 용량 | 용도 |
|------|-----:|------|
| `run_paper_reproduction.py` | 28.9 KB | 논문 전체 재현 |
| `run_full_pipeline_final.py` | 21.4 KB | 전체 파이프라인 |
| `run_remaining_experiments.py` | 19.1 KB | 남은 실험 실행 |
| `run_paper_artifacts_pipeline.py` | 16.1 KB | 논문 산출물 |
| `run_table2_full_comparison.py` | 15.4 KB | Table 2 비교 |
| `run_gpu_experiments.py` | 13.8 KB | GPU 실험 |
| `download_all_benchmarks.py` | 10.4 KB | 벤치마크 다운로드 |
| `verify_full_pipeline.py` | 8.1 KB | 파이프라인 검증 |
| `run_full_training_and_eval.py` | 7.2 KB | 학습+평가 통합 |
| 기타 (27개) | ~55 KB | 디버그, 개별 벤치마크 등 |

> 경로: `d:\AI\cts\scripts\`

---

## 7. 테스트 — tests/ (24.2 KB 소스 + 캐시)

38개 테스트 파일 (88 test cases). 주요:

| 파일 | 용량 | 용도 |
|------|-----:|------|
| `test_broyden_convergence.py` | 2.5 KB | Broyden 수렴 |
| `test_faiss_context.py` | 2.4 KB | FAISS 컨텍스트 |
| `test_latent_projection.py` | 1.6 KB | 잠복 프로젝션 |
| `test_gemma_cts.py` | 1.5 KB | Gemma CTS 통합 |
| 기타 34개 | ~16 KB | 단위/통합 테스트 |

> 경로: `d:\AI\cts\tests\`

---

## 8. 설정 — configs/ (6.4 KB)

| 파일 | 용량 | 용도 |
|------|-----:|------|
| `default.yaml` | 2.7 KB | 기본 학습/실험 설정 |
| `paper_parity.yaml` | 1.0 KB | 논문 일치 설정 |
| `data_paths.yaml` | 0.6 KB | 데이터 경로 정의 |
| `experiment_paper_protocol.yaml` | 0.6 KB | 실험 프로토콜 |
| `ablation_no_ach.yaml` | 0.2 KB | Ablation (ACh 제거) |
| `ablation_static_5ht.yaml` | 0.1 KB | Ablation (5HT 고정) |
| `README.md` | 1.2 KB | 설정 폴더 설명 |

> 경로: `d:\AI\cts\configs\`

---

## 9. 루트 파일

| 파일 | 용량 | 용도 |
|------|-----:|------|
| `README.md` | 25.0 KB | GitHub 메인 README |
| `cursor_neurips_2026_paper_analysis_and.md` | 44.0 KB | 논문 분석 |
| `cursor_upgrade_plan_based_on_neurips_20.md` | 22.4 KB | 업그레이드 계획 |
| `LICENSE` | 10.7 KB | Apache 2.0 |
| `pyproject.toml` | 1.2 KB | Python 패키지 설정 |
| `NOTICE` | 0.2 KB | 공지사항 |
| `.gitignore` | 0.5 KB | Git 제외 목록 |
| `assets/cts_architecture.png` | 1.19 MB | 아키텍처 다이어그램 |

> 경로: `d:\AI\cts\`

---

## 10. results/ (33.0 KB)

| 파일 | 용량 | 용도 |
|------|-----:|------|
| `EXPERIMENTS.md` | 7.3 KB | 실험 결과 문서 |
| `math500_result.json` | 22.0 KB | MATH-500 결과 (복사본) |
| `RUN_MANIFEST.json` | 1.1 KB | 실행 매니페스트 |
| `table1_cts_kv.csv` | 0.8 KB | CTS VRAM 데이터 |
| `LOCAL_HARDWARE_SNAPSHOT.txt` | 0.7 KB | 하드웨어 정보 |
| `REPRO_ENV.json` | 0.5 KB | 재현 환경 |
| `table1_kv_measured.csv` | 0.4 KB | KV 측정 |
| `table2_isoflop_mock.json` | 0.2 KB | Iso-FLOP 결과 |

> 경로: `d:\AI\cts\results\`
