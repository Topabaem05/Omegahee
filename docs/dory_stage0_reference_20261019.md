# Dory 4B 0단계: BF16 원본과 참조 구현 (2026-10-19)

## 결론

- BF16 원본을 받았고, vLLM PR #60149 `dory.py`(커밋 `5507695`)를 CPU PyTorch로 옮긴 참조
  구현이 llama.cpp Q5_K와 같은 첫 토큰을 낸다.
- fp32 참조는 반복 실행에서 비트 단위로 같다(30/30).
- `per_loop` 모드에서 prefill과 토큰 단위 decode의 로짓 차이는 최대 7.5e-4다.
- 1,830토큰 프롬프트(슬라이딩 윈도 1,024 초과)에서 루프 1·2·4 모두 정답 `3743`을 낸다.
- BF16 대 fp32 차이로 다음 단계 허용 오차를 정했다(아래 "허용 오차").
- 남은 0단계 항목: Mirai에 스키마 방향 공유(초안만 작성, 게시 안 함).

## 입력

| 항목 | 값 |
|---|---|
| 저장소 | `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`, revision `b366b87f762ea334ba3eb55912031baba1515810` |
| 위치 | `/Users/topamini/code/uzu-omega/models/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16` |
| `model.safetensors` | 7,950,515,472 bytes, SHA-256 `207483664dfef54c4c4438863260d3ba423bf3d6009b43fe84264b281e9fda23`(HF LFS etag와 같음) |
| `config.json` | SHA-256 `8fcd066b…a84d`, `n_recurrent_loops: 2`, `initializer_range: 0.0177` |
| 텐서 | 507개. 물리 72층(attention 36, MLP 36), `lm_head.logit_scale [131072]` 포함 |
| 참조 코드 | `reference/vllm_pr60149_5507695/`(PR head `55076950b6c1`) |
| 업스트림 | lalamo main `c1e533a`, uzu main `6671962`(계획서 기준과 같음). 두 저장소에 Dory 이슈·PR 없음(2026-10-19 확인) |

로컬 `uzu-upstream/uzu`는 `5baa58b`이고 계획서 기준 `6671962`보다 커밋 1개 뒤다. 2단계 시작 전에 맞춘다.

## 참조 구현

- `reference/dory_ref.py`: `dory.py`를 한 줄씩 옮겼다. 병렬 linear는 행렬곱으로, `Attention`은 PR 테스트의
  `ReferenceAttention`(start_pos 기반 KV, causal GQA, sliding window)으로 바꿨다.
  dtype 변환 위치(`justnorm`의 fp32 누적, gate·suv의 fp32 곱)는 원본과 같다.
- `reference/make_prompts.py`: 채팅 템플릿을 적용한 고정 프롬프트 6개(`reference/prompts.json`).
  `hello`, `math`, `gcd`, `ko`, `think`(추론 켬), `long`(1,830토큰).
- `reference/run_matrix.sh`: 루프 1·2·4 × `per_loop`·`last_loop` × fp32·bf16, fp32 반복 실행, long 프롬프트.
  greedy 32토큰(long은 16토큰), EOS에서 멈추지 않는다.
- `reference/compare_ref.py`: 결과 요약 → `bench/results/dory_ref_stage0_20261019.json`.
- 출력(`reference/out/matrix/*.npz`, git 제외): 실행 순번별 hidden(`trace/xNNN_pPP_lL`), 루프 경계
  (`loopN_combined`, `loopN_state`), prefill·decode top-20 logprob, 마지막 로짓.

## 결과

| 검증 | 결과 |
|---|---|
| llama.cpp Q5_K와 첫 토큰 | `hello` 22177, `math` 1050, `gcd` 11745 모두 같음 |
| 반복 실행(fp32) | 30/30 비트 단위 동일(trace, 로짓, 생성 토큰) |
| 읽지 않은 텐서 | 0개(모든 실행) |
| KV 슬롯 수 | `per_loop` 36/56/96(루프 1/2/4), `last_loop` 36 |
| prefill 대 decode(`per_loop`, fp32) | 최대 로짓 차이 7.5e-4, argmax 32/32 |
| `last_loop` prefill | `per_loop`와 비트 단위 동일(한 청크 안에서는 차이가 없음) |
| `last_loop` decode | `per_loop`와 다름. 짧은 프롬프트에서 갈라지는 지점은 모두 첫 EOS 뒤 |
| 슬라이딩 윈도 경계(long) | 루프 1·2·4, 두 모드 모두 첫 토큰 `3743`(정답) |

`last_loop`는 정의상 전체 재계산과 같지 않다(재계산 대비 최대 로짓 차이 0.2–17). 정답 비교는
`per_loop`끼리, `last_loop`끼리 해야 한다.

### BF16 대 fp32 (같은 참조, 30개 조합)

| 지표 | 관측 범위 |
|---|---|
| 층 출력 상대오차(‖Δ‖/‖ref‖) | 첫 층 0.29–0.33%, 최대 2.86% (루프 수에 따라 크게 늘지 않음) |
| `assert_close(atol=1e-2, rtol=1e-2)` 위반 비율 | 0% |
| 마지막 로짓 최대 절대 차이 | 0.16–0.48 |
| prefill top-1 일치 | 91.3–100% (768 위치 중 18개 불일치) |
| 불일치 위치의 fp32 top-1/top-2 차 | 최대 0.28 nat |
| prefill per_loop 로짓 차이(bf16 자체) | 0.34–4.9 |

`ko` 루프 1의 첫 토큰은 fp32에서 top-1과 top-2의 차가 0.01 nat라 BF16에서 바뀐다. 그래서
자유 생성 greedy 일치는 정밀도가 다른 구현 사이의 통과 기준으로 쓸 수 없다.

## 허용 오차 (1단계 이후에 쓴다)

정답은 fp32 참조다. 비교 대상이 BF16이면 다음을 만족해야 한다.

- 층 출력 상대오차 ≤ 3%, `assert_close(atol=1e-2, rtol=1e-2)` 위반 ≤ 3%(lalamo tracer 기본 규칙).
- 마지막 로짓 최대 절대 차이 ≤ 0.5.
- teacher forcing에서 fp32 top-1/top-2 차가 0.5 nat보다 큰 위치는 top-1이 같아야 한다.
- 비교 대상이 fp32이면 로짓 차이 ≤ 1e-3.

## 다음

1. Mirai 스키마 공유: 초안 `docs/dory_schema_proposal_draft_20261018.md`. 게시는 사용자 승인 후.
2. 1단계(lalamo 구현): `uzu-upstream/lalamo`(`c1e533a`)에 Dory 모듈, `HFDoryConfig`, tracer를 추가한다.
   초안의 변환 시 fold(`justnorm`→RMSNorm, suv·gate·logit_scale을 가중치 행에 곱함)를 쓰면
   계획서의 "어휘별 logit_scale 벡터" 작업은 필요 없다.
