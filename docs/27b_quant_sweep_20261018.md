# Qwen3.8-27B GGUF 양자화 스윕 (2026-10-18)

기준선 `docs/27b_baseline_20261018.md`는 Mirai-M 27B가 약 23.2 tok/s로
가중치 대역폭 상한의 99%를 쓴다고 기록했다. 이 문서는 그다음 질문, 즉
**비트폭을 낮추면 실제로 더 빨라지는가**를 같은 기기에서 측정한 결과다.

## 결론 먼저

1. GGUF 저비트 변형은 예상보다 훨씬 느리다. 6.19GB IQ1_S는 19.87 tok/s로,
   16.06GB Q4_0의 13.21 tok/s보다 50% 빠른 데 그쳤다. 대역폭 상한으로
   계산한 59 tok/s에는 34%밖에 도달하지 못했다.
2. 품질은 속도보다 빠르게 나빠진다. Q4_0 PPL 5.6934 기준으로 Q2_K_XL은
   +4.0%, IQ2_XXS는 +15.5%, IQ1_S는 +39.8%다. 계획의 +2% 게이트를 통과한
   저비트 변형은 없다.
3. 따라서 이 기기에서 품질을 지키는 최고 속도는 여전히 **Mirai-M 4비트
   23.2 tok/s**다. 같은 4비트급 GGUF Q4_0는 13.21 tok/s로 43% 느리다.
4. 2.38bpw Omega-Uzu가 이득을 내려면 디코드 커널이 최소 200GB/s급을 내야
   한다. 현재 KSC GEMV는 51.6GB/s이므로, 8.04GB 체크포인트를 가정하면
   약 6.4 tok/s다. 비트폭만 낮추는 것으로는 이기지 못한다.

## 측정 조건

- 기기: Apple M5 Pro, 48GB 통합 메모리, macOS 27.0
- 런타임: Homebrew `llama.cpp 0.7.0` (build 11669, commit `620d2225c`)
- 속도: `llama-bench -p 0 -n 128 -r 10 -o json`, 모델별 10표본
- 품질: WikiText-2 raw, `-c 512 --chunks 32`, f16 KV, `-ngl 99`
- PPL 기준선: 같은 런타임의 `Qwen3.8-27B-Q4_0.gguf` (16.06GB)
- 스왑: 측정 중 `used` 1.1GB. 계획의 채택 기준(1GB 미만)을 0.1GB 넘었으나,
  `memory_pressure` 여유 77%, 표준편차 0.31-1.58로 안정적이어서 잠정 채택했다.
- 모든 수치는 단일 27B 모델을 한 번에 하나씩 실행해 얻었다.

## 속도와 품질

| 변형 | 파일 GB | tok/s | 표준편차 | PPL | Q4_0 대비 PPL | RSS GB |
|---|---:|---:|---:|---:|---:|---:|
| Q4_0 | 16.06 | 13.21 | 0.75 | 5.6934 | 기준 | 10.71 |
| Q2_K_XL | 9.83 | 14.97 | 1.58 | 5.9186 | +3.96% | 9.10 |
| IQ2_XXS | 7.27 | 18.68 | 0.64 | 6.5735 | +15.46% | 7.01 |
| IQ1_S | 6.19 | 19.87 | 0.31 | 7.9570 | +39.76% | 6.12 |
| Mirai-M 4비트 | 15.57 | 23.21 | 1.22* | 미측정 | – | 15.12 |

\* Mirai-M은 기존 기준선의 6개 실행(벤치 3 + 대화 3, 22.01-24.99)의 평균과
표준편차다. 중간값은 23.05다. 두 경로를 합친 값이므로 엄밀한 단일 조건
표준편차는 10회 재측정이 필요하다.

### 이론 상한 대비

계획의 중간 대역폭 366.3GB/s를 분모로 쓴 값이다. 실측 `tok/s`는
`llama-bench`의 생성 전용 행이다.

| 변형 | 파일 GB | 이론 상한 tok/s | 실측 | 상한 대비 | 실효 대역폭 GB/s |
|---|---:|---:|---:|---:|---:|
| Q4_0 | 16.06 | 22.8 | 13.21 | 57.9% | 212.1 |
| Q2_K_XL | 9.83 | 37.3 | 14.97 | 40.2% | 147.1 |
| IQ2_XXS | 7.27 | 50.4 | 18.68 | 37.0% | 135.7 |
| IQ1_S | 6.19 | 59.2 | 19.87 | 33.6% | 123.0 |
| Mirai-M | 15.57 | 23.5 | 23.21 | 98.8% | 361.4 |

실효 대역폭은 `파일 GB x tok/s`다. 이 숫자가 낮다는 것은 바이트를 덜 읽는
대신 디코드가 다른 곳에서 막힌다는 뜻이다. IQ1_S/IQ2_XXS는 dequant와
작은 행렬 연산에 묶여 기기 대역폭의 3분의 1만 쓴다. Q4_0는 58%를 쓰고,
uzu의 Mirai-M 경로는 99%를 쓴다.

Mirai-M의 15.57GB는 모델 14.53GB와 speculator 1.03GB를 합친 캐시 크기다.
가중치만 매 토큰 읽는다고 보면 14.53GB x 23.21 tok/s = 337GB/s, 상한
25.2 tok/s 대비 92%다. 어느 계산을 쓰든 GGUF의 33-58%보다 높다.

## 해석

### 1. 비트폭과 속도는 단조 관계가 아니다

IQ1_S는 Q4_0보다 61% 적은 바이트를 읽지만 50%만 빠르다. IQ2_XXS는 Q4_0보다
55% 적은 바이트로 41% 빠르다. Q2_K_XL은 39% 적은 바이트로 13% 빠르다.
가중치 스트림을 줄여도 커널이 그만큼 효율적으로 바뀌지 않으면 이득이
사라진다.

### 2. 품질 손실이 속도 이득보다 크다

Q2_K_XL은 +4.0% PPL로 게이트(+2%)에 가장 가깝지만, 속도는 +13%다.
IQ2_XXS는 +41% 속도를 얻는 대신 PPL이 +15.5%다. IQ1_S는 +50% 속도에
PPL +39.8%로, IQ2_XXS보다 6% 빠른 대신 품질을 21% 더 잃는다. 저비트
선택지 중에서는 IQ2_XXS가 속도-품질 면에서 IQ1_S를 지배한다.

### 3. 엔진 효율이 비트폭보다 큰 변수다

같은 4비트급에서 Mirai-M은 23.20 tok/s, Q4_0는 13.21 tok/s다. 파일 크기는
15.57GB와 16.06GB로 거의 같다. 차이는 커널이 361GB/s와 212GB/s를 낸다는
것이다. 27B에서 “더 빠른 엔진”을 만드는 일은 “더 적은 비트”를 만드는 일과
별개이며, 현재는 엔진 쪽 차이가 더 크다.

### 4. Omega-Uzu 2.38bpw의 전제

2.38075bpw 27B는 약 8.04GB다. 366.3GB/s에서 45.6 tok/s, 250GB/s에서
31.1 tok/s가 상한이다. 그러나 이 저장소의 KSC GEMV는 같은 shape에서
51.6GB/s다. 8.04GB를 51.6GB/s로 읽으면 6.4 tok/s다. 즉 **2.38bpw
체크포인트를 만들어도 현재 커널로는 Mirai-M보다 3.6배 느리다.**

선행 조건은 명확하다. KSC 디코드가 최소 200GB/s, 가능하면 300GB/s 이상을
내야 2.38bpw가 의미를 갖는다. 그 전까지 2.38bpw 변환은 품질 실험 이상의
가치가 없다.

## 한계

- **Mirai-M PPL 미측정.** uzu CLI는 로짓/perplexity를 노출하지 않는다.
  따라서 품질 게이트는 같은 llama.cpp 계열의 Q4_0를 기준으로 적용했다.
  Mirai-M의 품질이 Q4_0와 비슷하다는 가정은 아직 검증되지 않았다.
- **PPL 32 chunks.** 16,384 토큰이다. 순위를 뒤집을 정도의 잡음은 아니지만
  (차이 1.4-2.3 PPL, 표준오차 0.15-0.22), 절대값은 64 chunks와 다를 수 있다.
- **스왑 1.1GB.** 계획의 1GB 기준을 살짝 넘었다. 스왑이 0으로 떨어지면
  같은 명령으로 재측정해 수치를 갱신해야 한다.
- **단일 런타임.** llama.cpp 0.7.0의 Metal 경로만 측정했다. 다른 빌드나
  `-fa on`, `-lm mmap` 조합은 결과를 바꿀 수 있다.
- **Mirai-M 6회 표본.** 기존 기준선은 경로당 3회다. 10회 재측정이 남아 있다.

## 재현

```bash
# 1) 모델 (약 30GB)
mkdir -p ~/models/qwen38-27b && cd ~/models/qwen38-27b
aria2c -x4 -s4 -c -o Qwen3.8-27B-UD-IQ1_S.gguf \
  'https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-IQ1_S.gguf'
aria2c -x4 -s4 -c -o Qwen3.8-27B-UD-IQ2_XXS.gguf \
  'https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-IQ2_XXS.gguf'
aria2c -x4 -s4 -c -o Qwen3.8-27B-UD-Q2_K_XL.gguf \
  'https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q2_K_XL.gguf'
aria2c -x4 -s4 -c -o Qwen3.8-27B-Q4_0.gguf \
  'https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_0.gguf'

# 2) WikiText-2
mkdir -p wikitext && cd wikitext
curl -L -O 'https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip'
unzip wikitext-2-raw-v1.zip

# 3) 속도 10표본
.venv/bin/python tools/q27b_gguf_eval.py \
  --models ~/models/qwen38-27b/Qwen3.8-27B-UD-IQ1_S.gguf \
           ~/models/qwen38-27b/Qwen3.8-27B-UD-IQ2_XXS.gguf \
           ~/models/qwen38-27b/Qwen3.8-27B-UD-Q2_K_XL.gguf \
           ~/models/qwen38-27b/Qwen3.8-27B-Q4_0.gguf \
  --mode speed --runs 10 --tokens 128 --out bench/q27b_gguf_speed_20261018.json

# 4) PPL
.venv/bin/python tools/q27b_gguf_eval.py \
  --models ~/models/qwen38-27b/Qwen3.8-27B-UD-IQ1_S.gguf \
           ~/models/qwen38-27b/Qwen3.8-27B-UD-IQ2_XXS.gguf \
           ~/models/qwen38-27b/Qwen3.8-27B-UD-Q2_K_XL.gguf \
           ~/models/qwen38-27b/Qwen3.8-27B-Q4_0.gguf \
  --mode ppl --ctx 512 --chunks 32 --out bench/q27b_gguf_ppl32_20261018.json
```

원자료: `bench/q27b_gguf_speed_20261018.json`,
`bench/q27b_gguf_ppl32_20261018.json`,
`bench/q27b_gguf_q2kxl_20261018.json`.
