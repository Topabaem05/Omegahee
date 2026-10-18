# Ω-Uzu 방법론 수정: “KSC GEMV”에서 “uzu MMA 커서”로 (2026-10-18)

## 결론: 느린 결정적인 이유

느린 이유는 2비트 포맷이 아니다. **uzu는 trellis 가중치를 절대 GEMV로 보내지
않고, 항상 MMA GEMM 경로로 보낸다.** Omega는 같은 2비트 코드를 별도의 스칼라
GEMV에서 한 코드씩 풀고 FP32로 누적한다. 이 경로 차이가 10배 이상의 격차를
만든다.

코드 근거:

- `crates/uzu-engine/src/backends/metal/kernel/matmul/matmul_metal_kernel.rs`
  의 `choose_dispatch`는 `if shape.b_is_trellis { return MatmulDispatch::Gemm(...) }`
  로 끝난다. trellis는 GEMV 후보에 들어가지도 않는다.
- `gemm_trellis.metal`은 `TrellisCursor<RightFragment>`로 코드를 **int8 MMA
  fragment에 직접** 채우고 `fragment_mma(products, left_tile, right_tile)`를
  호출한다.
- `gemm/common/quantized/trellis.h`의 `decode_biased_weights`는 32비트 워드에서
  해시와 정수 필드 합으로 **4개 가중치를 한 번에** 만든다. 4엔트리 LUT도,
  코드별 분기도 없다.
- 스케일·바이어스는 `TrellisEpilogue`에서 MMA가 끝난 뒤 적용한다. 핫 루프에는
  정수 MAC만 남는다.
- Omega의 `metal/kernels/omegahee.metal`은 `ksc_gemv_coalesced`에서 16개 코드를
  직렬로 풀고, `omega_codebook` LUT를 조회하고, `partial +=` FP32 의존 사슬을
  만든다. MMA도, int8 activation도, split-K도, epilogue 스케일링도 없다.

즉 **포맷이 아니라 디코드-투-MMA 경로가 결정적**이다. 2bit 코드도 fragment로
직접 디코드하면 uzu의 대역폭을 따라갈 수 있다. 반대로 2bit 코드를 스칼라
GEMV로 풀면 아무리 로드를 응집해도 24.6GB/s 근처에 머문다.

## 현재 측정의 정정

| 항목 | 이전 문서 | 최신 파리티 통과 측정 |
|---|---:|---:|
| `ksc_gemv_k4096` (1바이트) | 51.6 GB/s | 13.2–15.0 GB/s |
| `ksc_gemv_coalesced` (uint) | – | **23.4–26.3 GB/s (중간값 24.6)** |
| uzu trellis 디코드 | 261–354 GB/s | 261–354 GB/s |

51.6GB/s는 현재 코드에서 재현되지 않는다. `docs/step3_gemv_byteorder_20261018.md`
와 `metal/build/report.json`의 최신값을 쓴다. 이 정정으로 Omega의 예상 tok/s는
이전 계산보다 절반 이하로 내려간다. Dory 4B 2 loops는 22가 아니라 약 11,
27B 2.38bpw는 6.4가 아니라 약 3.1 tok/s다.

## Uzu가 빠른 이유 (요약)

| 요소 | uzu trellis GEMM | Omega KSC GEMV |
|---|---|---|
| 디스패치 | `b_is_trellis` → 항상 `Gemm` | 독립 `ksc_gemv_*` |
| 디코드 | 해시/정수 필드로 4개/워드, fragment 직접 | 코드별 LUT + FP32 |
| 활성값 | int8 symmetric (`Int8Symmetric`) | FP32 `x` |
| 누적 | int32 MMA fragment | FP32 스칼라 `partial +=` |
| 스케일 | epilogue (`TrellisEpilogue`) | 핫 루프 안 |
| 타일 | 16×32×256 / 64×64×256, split-K | SIMD 그룹당 1행 |
| 실측 | 261–354 GB/s | 24.6 GB/s |

## 수정된 방법론: uzu-first, cursor-native

### 원칙

1. **uzu의 포맷과 커널을 대체하지 않는다.** `GemmTrellis`, `MxuMmaCore`,
   split-K, epilogue를 기준선으로 유지한다.
2. **새 포맷은 반드시 `Cursor<Fragment>` 계약을 만족한다.** 독립 GEMV를 새로
   만들지 않는다. 새 포맷이 fragment로 직접 디코드되지 않으면 채택하지 않는다.
3. **핫 루프는 정수 MAC만.** int8 코드·int8 activation·int32 accumulator를 쓰고,
   alpha/beta/scale은 epilogue로 보낸다.
4. **타일·split-K는 uzu 정책을 재사용한다.** M5 Pro 라우팅은 측정 후에만 추가한다.
5. **제어기보다 측정을 먼저 한다.** budget 3–4는 시뮬레이션 오류였고, 9B·27B
   실측 모두 16이 최선이다. 동적 제어기는 그 뒤에 검토한다.
6. **품질 게이트 없이는 2.38bpw를 채택하지 않는다.** PPL/과제 정확도가
   Mirai-M 기준 +2%를 넘으면 중단한다.

### Phase 1: `KscCursor<RightFragment>` + GEMM 라우팅

작업:

- `GemmBPrologueKind` 또는 `MatmulShape`에 `b_is_ksc`를 추가한다.
- `quantized::KscCursor<RightFragment>`를 만든다. `TrellisCursor`와 같은
  인터페이스(`load`, `advance`, fragment 좌표)를 구현한다.
- 2bit 코드 16개/uint를 정수 연산으로 int8 fragment에 채운다. 코드북은
  `OMEGA_SPAN[code] = 2*code - 3`이므로 LUT 없이 산술로 만든다.
- `GemmKsc`(또는 `GemmTrellis`의 KSC specialization)를 추가하고
  `fragment_mma`에 연결한다.
- 라우팅은 `b_is_ksc → MatmulDispatch::Gemm`으로 고정한다. GEMV 금지.
- 첫 구현은 bf16 left operand로 파리티를 맞춘 뒤 int8 activation으로 바꾼다.

합격 기준:

- CPU/Metal 파리티 ≤ 1e-3.
- 같은 shape에서 ≥150GB/s(1차), ≥200GB/s(2차).
- uzu 4bit `gemm_trellis`와 같은 프롬프트·같은 10표본으로 비교.

### Phase 2: int8 activation + epilogue

- `GemmAPrologueKind::Int8Symmetric`을 사용해 activation을 int8로 양자화한다.
- `group_sums`, `activation_scales`, `row_scales`, `scale_and_offsets`를
  `TrellisEpilogue`처럼 MMA 뒤에 적용한다.
- KSC의 `alpha`, `beta`, `scale`을 핫 루프에서 제거한다.

합격 기준:

- ≥250GB/s, 또는 uzu 4bit 디코드의 90% 이상.
- 수치 오차가 bf16 기준선 대비 허용 범위 내.

### Phase 3: M5 Pro 타일·split-K

- `Tile16x32x256_Simdgroups1x1`(M<16)과 `Tile64x64x256_Simdgroups2x2`를
  우선 시험한다.
- `select_trellis_plan`의 split-K 목표를 KSC에도 적용한다.
- M5 Pro 라우팅은 M5 Max 행을 복사해 시작하고, 10표본 중간값으로 판정한다.

합격 기준:

- 기본 라우팅 대비 중간값 개선이 표준편차보다 크다.
- 개선이 없으면 라우팅 작업을 중단한다(현재 27B는 이미 상한의 99%).

### Phase 4: 투기 디코딩은 별도 트랙

- uzu DFlash/Weaver를 유지한다. budget 8/16/32를 10표본으로 측정한다.
- 9B 실측: 4/8/16 = 39.1/70.1/75.1 tok/s, 16이 최선.
- 27B 실측: budget 16이 최선, 32 이상은 검증 패스가 비싸다.
- 27B MTP는 `--spec-type draft-mtp --spec-draft-n-max 4`에서 상대 1.47배,
  수락 길이 2.95(측정 당시 GPU 95% 점유). 깨끗한 재측정 전에는 절대값을
  인용하지 않는다.
- SparseSpec/LiLiCorr 제어기는 uzu의 측정된 수락률·패스 비용을 이길 때만 넣는다.

### Phase 5: 품질과 실제 체크포인트

- 먼저 1개 레이어를 KSC 2.38bpw로 변환해 PPL 기여를 본다.
- 그다음 27B 전체를 변환해 WikiText-2 PPL과 고정 과제 정확도를 측정한다.
- 기준선은 Mirai-M 4bit. +2%를 넘으면 2.38bpw를 중단한다.
- 헤시안 곡률 경로는 현재 L=0이 기본이고, L>0은 재구성 오차를 악화시킨다.
  품질 근거 없이 “zero-loss”를 주장하지 않는다.

## 중단할 것

- `ksc_gemv_*`를 더 튜닝하는 것. 이것은 참조 커널이지 제품 경로가 아니다.
- 51.6GB/s를 기준으로 한 모든 tok/s 계산. 최신 24.6GB/s로 다시 계산한다.
- budget 3–4 최적론. 실측은 16이다.
- MMA 포트 전의 “2.39x” 주장. 그 계수는 대역폭 포화를 전제로 한다.
- 2.38bpw 품질 미측정 상태의 zero-loss 주장.

## 다음 한 작업

**`KscCursor<RightFragment>`를 구현하고 `b_is_ksc`를 `Gemm`으로 라우팅한다.**
새 GEMV를 쓰지 않는다. 이 한 변경이 24.6GB/s를 150–250GB/s로 올릴 수 있는지
판정하는 첫 실험이며, 실패하면 2.38bpw를 접고 uzu 4bit 개선으로 돌아간다.

## 측정 계약

- 동일 프롬프트, 동일 context, GPU 유휴(`ioreg` Device Utilization <10%),
  스왑 <1GB.
- 10표본 중간값·표준편차, TTFT, 최대 RSS.
- 보고 지표: `effective_GBps = weight_bytes / time`, `tok/s`, PPL,
  수락 길이, 패스 비용.
- 원자료는 `bench/`에 JSON으로 남기고, 문서에는 재현 명령을 붙인다.

## 위험

- KSC 2bit 디코드가 fragment 채우기에서 MMA 시간보다 커지면 이득이 없다.
  이 경우 2bit를 포기하고 uzu 4bit + MTP가 더 빠르다.
- KSC 품질이 4B/27B에서 유지되지 않을 수 있다. 4B는 특히 민감하다.
- uzu upstream이 빠르게 움직인다. 커밋을 고정하고 `gemm_trellis` 계약을
  해당 커밋의 코드로 검증한다.
