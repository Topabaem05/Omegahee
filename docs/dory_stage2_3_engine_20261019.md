# Dory 4B 2·3단계: Ω-Hee 엔진 CPU·Metal 지원 (2026-10-19)

## 결론

- Ω-Hee 엔진(`engine/`, uzu `6671962` 기반)이 lalamo 변환 Dory를 CPU와 Metal 두 백엔드에서 실행합니다.
- Metal 파리티: 루프 1·2·4 × `per_loop`·`last_loop` × 고정 프롬프트 5개, 1,830토큰 긴 프롬프트까지
  **31/31 통과**(0단계 fp32 참조 대비, BF16 허용 오차).
- CPU 파리티: 같은 매트릭스를 decode 8단계로 실행 중(느린 참조 백엔드). 결과는 `bench/results/dory_uzu_cpu_parity_20261019.json`.
- CLI 채팅·OpenAI 호환 서버가 Dory 채팅 인코딩(hanashi `dory`)으로 동작합니다. M5 Pro BF16 2루프 decode 약 25 tok/s.
- 엔진 단위 테스트 401개 통과(실패 0), hanashi 25개, 레거시 uzu 관련 테스트 통과.

## 엔진 변경 (`engine/crates/uzu-engine`)

| 영역 | 변경 |
|---|---|
| 설정 | `NormReduction`, `LearnedScaleConfig`, `AnyLayerConfig`(`TransformerLayerConfig`·`NGPTAttentionLayerConfig`·`NGPTMLPLayerConfig`), `RecurrenceConfig`, 선택적 `output_norm_config`, 어텐션·MLP·임베딩 스케일 필드 |
| 커널(CPU+Metal) | `Normalization`·`QKVNorm`에 sum 리덕션, `AttentionPrepare`에 RoPE 뒤 q 스케일, 새 `ScaleColumns`(채널별 곱, 행 stride·오프셋), 새 `NgptResidualMix` |
| 블록 | `ngpt_layer.rs`(하위층 1개 + 보간 잔차, 잔차 스트림을 hidden으로 전달), 어텐션 게이트 스케일·출력 정규화, MLP 채널 스케일·hidden 정규화, 어휘별 로짓 스케일 |
| Transformer | 실행 단계(층, 루프)와 상태 슬롯. `per_loop`는 루프마다 슬롯, `last_loop`는 층당 슬롯 하나(같은 suffix 위치를 루프마다 덮어씀). 재주입 `norm(state + inputs)`, 루프 끝 `norm` |
| KV | Dory 어텐션(q 스케일 있음)은 BF16 KV를 씁니다. INT8 KV prepare 경로의 q 스케일 지원은 연구 6단계로 미룹니다 |
| 도구 | `unit/session/dory_parity.rs`(`UZU_DORY_PARITY`로 로짓 덤프, prefill은 1,024토큰 단위로 분할) |

기존 모델 경로는 바꾸지 않았습니다. 표준 층은 같은 `shortcut` 융합 흐름을 그대로 쓰고, nGPT 층과 섞을 수 없습니다.
config.json 층 항목에 `"type"`이 필요하고 새 필드가 생겨 기존 변환 모델은 다시 변환해야 합니다.

## 채팅 인코딩 (hanashi)

- `configs/rendering/dory.json`: qwen3.5 렌더링 매핑 + Dory `chat_template.jinja`(빈 system 블록, `<think></think>`).
- `configs/parsing/dory.json`: qwen3.5 파싱 기반. Dory 토크나이저는 `assistant`를 `ass`+`istant` 두 토큰으로 나누므로
  role을 길이 제한 없는 greedy로 잡고 content를 `\n`에서 엽니다(llama-3 방식). qwen3.5 그대로 쓰면 `Stream desynchronization`.
- `configs/tokens/dory.json`: eos 11(`<|im_end|>`), stop `[2, 11]`.
- 변환 모델 폴더에 `encoding.json` = `{"type": "hanashi", "name": "dory"}`.

## 파리티 기준

정답은 0단계 fp32 참조, 엔진은 BF16 계산입니다(`reference/compare_uzu.py`).

- 마지막 위치 로짓 최대 절대 차이 ≤ 0.5.
- prefill: 참조 top-1/top-2 차가 0.5 nat를 넘는 위치에서 top-1이 다른 개수가, 같은 참조를 BF16으로 돌렸을 때의 개수 이하.
  짧은 프롬프트는 BF16 참조 위반이 0이라 기존 규칙과 같습니다. 1,830토큰 프롬프트는 BF16 참조 자체가 6개 위반
  (top-1 일치 96.9%), 엔진 4개(96.6%).
- decode(teacher forcing, 31단계): 같은 마진 규칙, 위반 0.

## Metal 결과 (`bench/results/dory_uzu_metal_parity_20261019.json`)

| 설정 | 마지막 로짓 최대 차이 | decode top-1 | 판정 |
|---|---|---|---|
| L1 per_loop / last_loop | 0.21–0.36 | 30–31/31 | 통과 |
| L2 per_loop / last_loop | 0.22–0.48 | 29–31/31 | 통과 |
| L4 per_loop / last_loop | 0.19–0.40 | 30–31/31 | 통과 |
| long 1,830토큰 L2 per_loop | 0.31 | 15/15 | 통과 |

## 실행 확인

```text
$ cli --model Dory1.0-4B-lalamo-bf16 -m "What is 17 * 23?"
17 * 23 = **391**     (generation 25.1 t/s, prefill 250 t/s, memory 7.5 GB)
```

`cli server`: `/v1/models`, `/v1/chat/completions` 정상, `reasoning_content` 분리.

## 다음

- CPU 매트릭스 완료 후 결과 반영.
- 4단계(성능 측정): 연구 계획 1단계 측정 기반으로 진행합니다(`docs/dory_research_log.md`).
