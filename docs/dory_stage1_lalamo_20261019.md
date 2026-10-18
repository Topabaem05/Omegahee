# Dory 4B 1단계: lalamo 구현 (2026-10-19)

## 결론

- lalamo `dory-support` 브랜치(기준 `c1e533a`, 커밋 전)에 Dory 지원을 넣었다. BF16 체크포인트가 변환되고,
  lalamo fp32 추론이 0단계 참조와 허용 오차 안에서 일치한다.
- 루프 1·2·4, `per_loop`·`last_loop`, 고정 프롬프트 6개에서 모든 비교가 통과했다(아래 표).
- 변환 결과는 물리 72층만 저장한다. 원본보다 큰 2,265,352 bytes는 학습된 스케일 벡터 1,126,400개를
  fp32로 둔 차이와 정확히 같다.
- 기존 fast 테스트는 회귀가 없다(기준 581 통과 → 587 통과, 실패 2개는 기준과 같은 gated 모델 CLI 테스트).

## 설계 (사용자 결정)

| 결정 | 선택 |
|---|---|
| 층 표현 | 새 층 타입 `NGPTAttentionLayerConfig`, `NGPTMLPLayerConfig`. `layer_configs`는 `TransformerLayerConfigBase` 레지스트리의 tagged union |
| L2 정규화 | `NormalizationConfig.reduction: NormReduction(MEAN \| SUM)`. justnorm = SUM, 가중치 없음, eps 1e-6 |
| 루프 | `TransformerConfig.recurrence: RecurrenceConfig`(start, end, num_loops, kv_cache_mode, reinjection_norm_config) |
| 스케일 | 학습된 벡터는 그대로 저장(`LearnedScale`), 상수와 변환(IDENTITY/SQUARE/ABSOLUTE)은 설정에 둔다 |

## 바뀐 파일 (lalamo)

| 파일 | 내용 |
|---|---|
| `modules/normalization.py` | `NormReduction` |
| `modules/learned_scale.py` (새) | `LearnedScaleConfig(multiplier, transform)`, `LearnedScale` |
| `modules/token_mixers/attention.py` | `query_scale_config`(RoPE 뒤 q), `gate_scale_config`, `output_norm_config` |
| `modules/mlp.py` | `DenseMLPConfig.up_scale_config`, `gate_scale_config`, `hidden_norm_config` |
| `modules/embedding.py` | `UntiedEmbeddingConfig.logit_scales_config`(어휘별) |
| `modules/transformer_layer.py` | `TransformerLayerConfigBase`(RegistryABC), `TransformerLayerBase`, `NGPTLayerActivationTrace` |
| `modules/ngpt_layer.py` (새) | nGPT 어텐션 전용·MLP 전용 층, 보간 잔차 |
| `modules/transformer.py` | `RecurrenceConfig`, `RecurrentKVCacheMode`, 실행 순번(`execution_steps`)과 상태 슬롯(`state_slots`), 선택적 `output_norm`, `standard_layers` |
| `model_import/model_configs/huggingface/dory.py` (새) | `HFDoryConfig` |
| `model_import/loaders/dory_loader.py` (새) | 가중치 로더(q·k·v·g, up·gate 융합) |
| `model_import/model_specs/nvidia.py` (새) | `NVIDIA-Nemotron-Labs-Dory1.0-4B` spec, `chat_template.jinja`, 추론 기본 켬 |
| 기존 로더·테스트 | `standard_layers`와 `output_norm is not None` 단언으로 타입 좁힘(동작 변경 없음) |
| `tests/unit/test_dory.py` (새) | 작은 무작위 모델 대 vLLM 수식 PyTorch 포트 |

`last_loop`는 KV 캐시 코드를 바꾸지 않았다. 모든 루프가 이번 forward 이전 상태를 읽고, 마지막 루프의
갱신 상태만 남긴다.

## 검증

### 단위 테스트 (`tests/unit/test_dory.py`, 6개 통과)

- 루프 1·3 × `per_loop`·`last_loop`: prefill 3토큰 → decode 1토큰 → 3토큰(슬라이딩 윈도 3 초과)에서
  로짓 일치(atol·rtol 1e-4). 체크포인트 텐서를 모두 읽었는지 확인.
- 잘못된 구현 검출: `last_loop`를 "앞 루프의 갱신 상태를 읽음"으로 바꾸면 `3-last_loop`만 실패한다(1루프는
  두 의미가 같다). 원복 후 통과.
- 루프 수와 무관하게 층 설정이 같고(가중치 복제 없음), 상태 슬롯은 `per_loop` 2+2×27, `last_loop` 4.
- 설정 JSON 왕복.

### 실제 모델 대 0단계 참조 (fp32, `bench/results/dory_lalamo_vs_ref_20261019.json`)

| 비교 | 결과 |
|---|---|
| prefill 실행 순번별 hidden(루프 1/2/4: 72/112/192단계) | 상대오차 최대 8.4e-6 |
| 마지막 로짓 최대 절대 차이 | 4.1e-5–1.6e-4 (기준 1e-3) |
| prefill top-1 일치, 첫 토큰 | 15/15 조합 100%, 첫 토큰 모두 같음 |
| decode teacher forcing, 루프 2 `per_loop` | 5/5 프롬프트 31/31, top-20 logprob 차이 ≤ 2.6e-4 |
| decode teacher forcing, 루프 2 `last_loop` | 5/5 프롬프트 31/31, ≤ 1.9e-4 |
| long 1,830토큰(슬라이딩 윈도 초과), 루프 2 | 로짓 차이 4.9e-4, top-1 1,830/1,830 |

### 변환과 실행

- `reference/lalamo_dory.py convert`(lalamo `convert`와 같은 `import_model` → `save`, 로컬 체크포인트용
  `LocalOrigin`) → `models/Dory1.0-4B-lalamo-bf16`: 363 텐서(q·k·v·g와 up·gate 융합), 정지 토큰 `[2, 11]`,
  temperature 1.0, top_p 0.95.
- `lalamo chat --temperature 0`(bf16): 한국어·산술 프롬프트에서 정상 응답. 산술 응답 앞부분이 0단계 참조
  greedy 출력과 같다.
- 정적 검사: ruff 통과, pyrefly 0 errors.

## 남은 일과 주의

- CPU에서 Pallas decode attention이 이 설정을 지원하지 않아 XLA로 대체된다는 경고가 나온다. 정확도와
  무관한 속도 경고다.
- 기존 모델의 config.json 층 항목에 `"type": "TransformerLayerConfig"`가 붙는다. uzu 파서도 같이 바꿔야
  한다(2단계).
- 루프 수는 변환 시 config.json 값(기본 2)으로 정해진다. 로드 시점 변경은 2단계 uzu에서 다룬다.
- 커밋하지 않았다. 기준 비교용 worktree `uzu-upstream/lalamo-base`(`c1e533a`)가 남아 있다.
