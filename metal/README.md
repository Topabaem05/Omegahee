# Omegahee Metal 커널 팩 (M5 Pro 실측 계층)

> **2026-10-18 정정.** 아래 51.6GB/s는 초기 측정이다. 최신 파리티 통과
> `ksc_gemv_coalesced`는 23.4–26.3GB/s(중간값 24.6)이고, 기존 1바이트 커널은
> 13.2–15.0GB/s다. 새 방법론은 `docs/methodology_revision_20261018.md`를 따른다.

이 디렉터리는 `paper/uzu_omega_mathematical_algorithm_2026.tex` 9.1절 통합 로드맵이
요구하던 Metal/실기기 계층을 Omegahee 저장소 안에서 독립적으로 구현·측정한다.

- 대상 파일: `kernels/*.metal` (Apple Metal Shading Language)
- 하네스: `harness/main.swift` (Swift + Metal, 단일 파일 실행 파일)
- 파리티 벡터: `tools/gen_vectors.py` (Python 참조와 동일한 누산 순서로 기대값 생성)
- 실행: `bash metal/run.sh`

## 이 팩이 구현하는 계약

1. **KSC-CurveTQ 런타임 디코드** (`ksc_decode_dot`)
   - 계약 출처: `docs/uzu_metal_kernel_mapping.md` §1, uzu `gemm_trellis` 의
     `trellis_symbol` / `trellis_step` 상태 기계.
   - 2비트 네이티브 기저 코드를 트렐리스 심볼로 읽고, 128-엔트리 상수 양자화 LUT 대신
     `codebook[code]` 를 곱한다. `rotation_applied = false` — 핫 루프에 하다마드
     역회전이 없다. 14비트 시작 상태 `s0*` 는 코덱 프롤로그에서 소비되며 디코드
     루프에 들어오지 않는다.
2. **AWSRC 시드 잔차 에필로그** (`awsrc_dot`)
   - 계약 출처: 같은 문서의 `splitmix_sign(seed, col) * rsqrt(block)` 인라인 누산기.
   - 시드는 레지스터 안에서 재생성된다. 공유 LUT나 사이드카 기저 행렬을 읽지 않는다.
     읽는 바이트는 `seeds`(`M` × 8B)와 `gammas`(rows × `M` × 4B)뿐이다.
3. **ReadKV 비트 평면 프리픽스 로더** (`readkv_prefix_load`)
   - 계약 출처: 문서 §3의 "load only planes `0 .. r_c^K-1`".
   - 8개 평면을 32채널/워드로 전치 저장하고, 채널별 읽기 예산 `r_c ∈ {0,2,4,8}` 만큼만
     평면을 읽는다. 읽지 않은 평면은 0으로 기여한다.
4. **LiLiCorr 단일 패스 상관** (`lilicorr_score`)
   - 계약 출처: 문서 §2의 `S_t[k,k'] = tau * cos(U_out[t,k], U_in[t+1,k'])`.
5. **달성 대역폭 프로브** (`stream_probe`)

   - 512 MiB float4 스트림을 4패스 돌려 기기 대역폭을 측정한다. 실행별 편차가 크다.
6. **SIMD 협력 KSC 디코드** (`ksc_decode_simd`)

   - KSC 코드 스트림이 랜덤 액세스가 되는 성질을 검증한다: 레인 l이 니블 l로 열 2l, 2l+1을 담당한다.
     직렬 트렐리스 워크 없이 디코드된다. 허용 오차 2e-4.
7. **DRAM 스케일 디코드** (`ksc_stream_simd`)

   - 16 MiB 코드 버퍼를 한 번 스트림해 캐시 아티펙트를 제거한다.
8. **uzu shape GEMV** (`ksc_gemv_k4096`)

   - N=12288, K=4096으로 uzu와 같은 형태를 맞춘다. 이 커널이 처음으로
     동일 조건 비교를 가능하게 한다.
9. **처리량 프로브** (`ksc_throughput`)

   - 스칼라 디코드를 반복 실행해 가중치 바이트/초를 측정한다.

## 파리티 방법

`tools/gen_vectors.py` 가 Python으로 입력과 기대값을 만들고, Swift 하네스가 같은
바이트를 Metal 커널에 넣어 최대 절대 오차를 비교한다. 기대값은 커널과 **동일한 누산
순서**로 계산한다. 따라서 남는 차이는 float64(파이썬) → float32(셰이더) 반올림뿐이다.

판정 허용 오차는 커널별로 다르다.

| 커널 | 허용 오차 | 이유 |
|---|---|---|
| `ksc_decode_dot` | 5e-4 | 64항 float32 누산 |
| `ksc_decode_simd` | 2e-4 | 레인 병렬 누산으로 합산 순서가 달라짐 |
| `awsrc_dot` | 5e-4 | 시드별 정수 해시는 정확, 누산만 float32 |
| `readkv_prefix_load` | 0 | 정수 비트 연산만 |
| `lilicorr_score` | 1e-4 | 코사인 정규화 float32 |

## 실행

```bash
bash metal/run.sh
```

산출물은 `metal/build/report.json` 이다. 이 파일은

- 커널 4종의 파리티 판정과 최대 오차,
- `stream_probe` 로 측정한 M5 Pro 대역폭(GB/s),
- `ksc_decode_dot` 가중치 스트리밍 처리량(GB/s),
- 측정 대역폭에 대한 roofline 함의(예: 2.38 bpw 9B 체크포인트의 단일 패스 TPS 상한)

를 담는다.

## 한계

- 이 팩은 uzu 저장소를 수정하지 않는다. uzu 크레이트에 이식하려면
  `crates/uzu-engine/src/backends/metal/kernel/quant_matmul/` 에 대응 커널을 추가해야
  한다. 현재 uzu에는 `gemm_trellis` 만 있고 AWSRC·ReadKV·LiLiCorr 커널은 없다
  (2026-10-17, uzu `5baa58b7` 확인).
- 측정은 합성 입력 기준이다. 실제 Mirai-M 체크포인트 추론이 아니다.
- 대역폭은 커널 런치 오버헤드를 포함한 벤치 값이다. 커널별 세부 튜닝은 하지 않았다.

## 2026-10-17 M5 Pro 실측 (Apple M5 Pro, 48GB, macOS 27.0, Xcode 27.1)

커널 4종 파리티는 모두 통과했다.

| 커널 | 최대 절대 오차 | 허용 오차 | 판정 |
|---|---:|---:|---|
| `ksc_decode_dot` (64행) | 1.91e-06 | 5e-4 | PASS |
| `awsrc_dot` (64행, 시드 4) | 7.45e-08 | 5e-4 | PASS |
| `readkv_prefix_load` (32채널) | 불일치 0건 | 0건 | PASS |
| `lilicorr_score` (3쌍 x 6x6) | 1.19e-07 | 1e-4 | PASS |

대역폭(`stream_probe`, 512 MiB private 버퍼, 4 패스, 샘플 15회):

| 지표 | 값 |
|---|---:|
| 최고 | 267.8--272.5 GB/s |
| 중간값 | 152.4--261.0 GB/s |

최고값은 Apple M5 Pro 공개 대역폭(약 273 GB/s)과 일치한다. 중간값의 큰 편차는
이 기기가 동시에 다른 작업을 수행하는 상태에서 측정했기 때문이다. 재현 시
유휴 상태에서 실행하면 중간값도 최고값에 가까워진다.

roofline 대입: paper가 투영한 `Qwen3.5-9B` 2.38075 bpw 체크포인트의 가중치 바이트는
2.68 GB다. 측정 최고 대역폭 272 GB/s를 그대로 나누면 단일 패스 상한은 약
102 tok/s다. paper 표의 `Omega-Uzu (투기 OFF, KSC)` 64.4 tok/s는 이 상한의 63%로,
공개된 uzu 0.6.2 대역폭 효율 91.4%보다 낮다. 즉 paper 수치는 달성 불가가 아니라
미검증 여유를 남긴다. 실제 커널 구현이 그 효율을 얼마나 회복하는지는 이 팩이
측정하지 않는다.

## 2026-10-17 추가 측정: uzu shape 비교

커널 3종을 추가해 같은 shape에서 uzu와 비교했다.

| 커널 | Shape | 가중치 처리량 |
|---|---|---|
| `ksc_throughput` (스칼라) | K=64 | 25.8–38.7 GB/s |
| `ksc_gemv_k4096` (SIMD 그룹당 1행) | N=12288, K=4096 | **51.6 GB/s 최고, 47.5 중간값** |
| uzu 디코드 실효 (실제 9B 모델) | – | **261–354 GB/s** |

경격은 5.1–6.9배이고, 원인은 산술 집약도다. KSC GEMV는 코드 바이트당 약 2.7 MAC이고
측정 피크의 13%를 쓴다. uzu의 타일드 trellis GEMM은 한 번의 로드로 MMA 타일을
채워 69–93%에 도달한다. 자세한 분석은 `docs/performance_comparison.md`에 있다.

## 관련 문서

- `docs/plan_completion.md` - 남은 통합 항목의 상태
- `docs/uzu_metal_kernel_mapping.md` - 인용한 계약 원문
- `metal/build/report.json` - 위 수치를 생성한 최신 실행 보고서
