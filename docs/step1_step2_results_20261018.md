# 1·2단계 실측: budget 확정, KSC 품질의 전제 (2026-10-18)

앞선 budget 측정은 표준편차가 평균 차이보다 커서 판정 불가였다. 계측을 추가해
다시 측정했고, 이번에는 결론이 나왔다. 이어서 실제 9B 가중치에서 KSC 재양자화
품질을 처음으로 측정했다.

## 1단계: budget은 16이 최선이다 (예측 반증 확정)

측정 개선: budget 순서를 라운드마다 무작위화, `include_replies`로
`tokens_per_forward_pass`와 `num_decode_forward_passes` 계측, 라운드 10회 x
배치 3종 x 실행 2회 = budget당 20표본.

| budget | 평균 tok/s | 표준편차 | tokens/forward | 패스 시간 |
|---:|---:|---:|---:|---:|
| 4 | 39.07 | 2.10 | 2.31 | 59.0 ms |
| 8 | 70.10 | 4.05 | 2.45 | 35.0 ms |
| 16 | **75.07** | 5.77 | 2.69 | 35.9 ms |

표준편차가 2.1~5.8로 줄어 순서가 확정됐다. **budget 16이 가장 빠르다.**
패스 비용 모델이 예측한 "3~4가 1.77배 빠르다"는 완전히 반증됐다.

원인이 계측에 드러난다. budget 4는 `tokens/forward`가 2.31로 낮은데 패스 시간은
**59.0 ms로 더 길다**. 즉 작은 tree가 검증 패스를 싸게 만들지 않는다. 고정 비용이
지배하고, tree가 작으면 수락 길이만 잃는다. 이는 `T_verify(B) = 16.10 + 2.086B`
피팅이 이 워크로드에서 성립하지 않음을 뜻한다.

## 2단계: 실제 레이어에서 KSC는 flat 대비 이득이 없다 (L이 없을 때)

### 먼저 확인한 사실

`trymirai/Qwen3.5-9B-M`의 safetensors는 **원본 FP16이 아니라 Mirai-M 4비트로
이미 양자화된 상태**다. 구성:

- `quantized.weights`: (24576, 2048) uint8 = N x K/2 팩된 4비트
- `quantized.scales`: (128, 24576) bf16 = (K/group, N), group=32
- `quantized.zero_points`: (128, 12288) uint8 = (K/group, N/2), 2채널당 공유
- `incoherence_signs`: 입력/출력 부호

즉 "원본 FP16을 KSC로 변환"은 이 체크포인트로는 불가능하다. 재양자화만 가능하다.
`tools/mirai_raw_safetensors.py`가 bf16을 float32로 읽는 최소 파서이고,
`tools/ksc_layer_probe.py`가 디퀀타이즈와 비교를 수행한다.

### 측정

layer 0 `mlp.up_projection`(24576x4096)에서 512행, 64열 블록 단위로 비교했다.
per-block std를 스케일로 쓰는 공정한 기준선과 비교했다.

| 방식 | MSE |
|---|---:|
| flat 2-bit 반올림 (per-block scale) | 8.421846e-02 |
| KSC beam8, L=0 | 8.421846e-02 |
| KSC beam8, L=0.005 | 8.423124e-02 |
| KSC beam8, L=0.01 | 8.438184e-02 |
| KSC beam8, L=0.02 | 9.318798e-02 |

**KSC는 L=0에서 flat과 비트 단위로 같고, L을 키우면 나빠진다.** 인위적으로 넣은
L(하삼각 상수)은 실제 헤시안이 아니므로 재구성 오차만 늘린다.

### 의미

KSC-CurveTQ의 우위는 **L이 실제 캘리브레이션 헤시안에서 나올 때만** 생긴다.
지금처럼 상수 L을 쓰면 이득이 없다. 이것이 인코더의 다음 필수 작업이다.
또한 나쁜 L은 성능을 해치므로, L을 추정하지 못하면 L=0(flat)이 더 낫다.

## 남은 것

1. **실제 헤시안 추정** — 캘리브레이션 활성화로 H = X^T X를 구하고, 블록별
   Schur + LDL로 L, d를 만들어야 KSC 이득을 측정할 수 있다.
2. **품질 지표** — 지금은 재구성 MSE만 봤다. WikiText-2 PPL과 logit KL은
   전체 모델 변환 후에 가능하다.
3. **GEMV 파리티** — 미규명(별도 기록).

## 재현

```
python tools/mirai_raw_safetensors.py   # bf16 텐서 확인
python tools/ksc_layer_probe.py         # 디퀀타이즈 + KSC 비교
python tools/uzu_budget_sweep.py --cli ... --model-dir ... --task ... \
    --budgets 4,8,16 --rounds 10 --setter tools/uzu_budget_override.py --out sweep.json
```
