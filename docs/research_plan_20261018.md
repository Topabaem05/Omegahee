# Uzu-first 연구 계획 (2026-10-18 reset)

## 목표

`trymirai/uzu`를 이 Mac(Apple M5 Pro, 48GB)에서 **안정적으로 측정하고,
측정된 병목만 개선**한다. 새 포맷을 따로 만드는 것이 목표가 아니다.

Omega-Uzu의 2.38bpw/KSC GEMV는 main에서 제거하고 아카이브했다. 다시
쓰려면 uzu의 `GemmTrellis`/MMA 경로에 `Cursor<Fragment>`로 들어와야 한다.

## 시작 사실

| 사실 | 근거 |
|---|---|
| uzu 9B 디코드 50–68 tok/s | 이전 실측(재측정 필요) |
| uzu 27B 디코드 약 23 tok/s | 이전 실측(재측정 필요) |
| uzu trellis는 항상 GEMM/MMA로 라우팅 | `matmul_metal_kernel.rs` |
| KSC 독립 GEMV 최신 24.6GB/s | 파리티 통과 측정 |
| 27B는 대역폭 상한의 약 99% | 이전 계산(재측정 필요) |
| 9B 투기 경로는 109–119GB/s | 이전 실측(재측정 필요) |
| budget 16이 9B/27B 모두 최선 | 이전 실측, 새 환경에서 재확인 |
| MTP 1.47x | GPU 95% 점유 조건, 폐기 후 재측정 |

## 원칙

1. **uzu-first.** `gemm_trellis`, `MxuMmaCore`, split-K, epilogue를 기준선으로
   유지한다.
2. **청정 게이트.** `tools/clean_env_check.py`가 0으로 끝나지 않은 측정은
   기록하지 않는다.
3. **한 번에 한 변수.** GPU, 스왑, 모델, 컨텍스트, 라우팅을 동시에 바꾸지 않는다.
4. **cursor-native.** 새 양자화 포맷은 `Cursor<Fragment>`로 MMA fragment에
   직접 디코드되어야 한다. 독립 GEMV는 만들지 않는다.
5. **측정 우선.** 시뮬레이션·논문 계수보다 10표본 실측을 먼저 본다.
6. **품질 게이트.** PPL/과제 정확도가 기준선 +2%를 넘으면 중단한다.

## Phase 0: 안정화 (선행)

산출물: `docs/uzu_stabilization_20261018.md`, `bench/results/baseline_9b.json`,
`baseline_27b.json`.

- [x] uzu `5baa58b` / `cli 0.6.9` / binary hash 고정.
- [x] 38GB의 이전 GGUF를 외장으로 이동해 내부 여유 확보.
- [ ] GPU 유휴·스왑 <1GB·디스크 >20GB 확인. (현재 GPU 96%, 스왑 6.8GB로 차단)
- [x] 9B 모델 다운로드. 스모크 테스트는 게이트 통과 후.
- [x] 9B 10표본 기준선: 중간값 80.97 tok/s, 표준편차 10.65, 6.22GB
  (`bench/results/baseline_9b_20261018.json`). 잔여 스왑 때문에 잠정값.
- [x] 27B 모델 다운로드(현재 외장 아카이브). 10표본 기준선 때 내부로 복원.

게이트: 10표본 중간값·표준편차가 있고, 스왑 <1GB, GPU <10%인가?
아니면 Phase 1로 가지 않는다.

## Phase 1: uzu 병목 프로파일링

질문: 9B/27B에서 시간이 **디코드 대역폭, 투기 검증, KV, CPU/그래프 분할** 중
어디로 가는가?

- [ ] 9B spec off/on: tok/s, tokens/forward, pass time, effective GB/s.
- [ ] 9B budget 8/16/32: 같은 조건 10표본.
- [ ] 27B spec off/on: 같은 조건 10표본.
- [ ] 27B budget 8/16/32: 같은 조건 10표본.
- [ ] `ioreg` GPU utilization과 `vm.swapusage`를 실행마다 기록.
- [ ] decode가 상한의 90% 이상인지, 투기 경로가 대역폭을 비는지 판정.

게이트: 측정 구성요소로 전체 시간의 90%를 설명할 수 있는가?
설명 못 하면 최적화하지 않는다.

## Phase 2: 저위험 개선

### 2A. M5 Pro 라우팅

조건: Phase 1에서 decode가 상한의 90% 미만일 때만.

- [ ] uzu 포크에 `M5Pro` 행 추가, M5 Max 타일로 시작.
- [ ] `Tile16x32x256`, `Tile64x64x256`, `Tile128x128x256` 10표본.
- [ ] `qmv` 경로와 GEMM 경로를 분리해 측정.

게이트: 중간값 개선이 표준편차보다 클 때만 채택. 27B가 이미 99%면 중단.

### 2B. 투기 디코딩

- [ ] uzu DFlash/Weaver의 budget 8/16/32를 청정 조건에서 재측정.
- [ ] 27B MTP(`draft-mtp`, n_max 1–4)를 청정 조건에서 재측정.
- [ ] 수락 길이와 pass cost를 함께 기록.

게이트: 중간값 +10% 이상이고 품질이 유지될 때만 채택.

### 2C. 메모리·컨텍스트

- [ ] uzu의 실제 컨텍스트/KV 기본값을 확인한다.
- [ ] 4k/32k/128k에서 메모리와 tok/s를 측정한다.
- [ ] 262k 기본값이 있는 경로는 명시적으로 낮춘다.

## Phase 3: 구조 개선

### 3A. Cursor-native 2bit

- [ ] `KscCursor<RightFragment>` 구현.
- [ ] `b_is_ksc`를 `MatmulDispatch::Gemm`으로 라우팅.
- [ ] int8 activation + int32 accumulator + epilogue 스케일.
- [ ] 파리티 ≤1e-3, 같은 shape에서 1차 ≥150GB/s, 2차 ≥200GB/s.

실패 시: 2.38bpw를 중단하고 uzu 4bit 개선으로 돌아간다.

### 3B. ReadKV/OptR/AATC

Phase 1에서 KV 또는 타일 병목이 확인될 때만 진행한다. 4k 디코드 속도
개선을 기대하지 않는다.

## Phase 4: 품질과 최종 보고

- [ ] 1개 레이어 KSC 변환, PPL 기여 확인.
- [ ] 27B 전체 변환, WikiText-2 PPL + 고정 과제.
- [ ] Mirai-M 4bit 대비 +2% 게이트.
- [ ] 최종 보고서: tok/s, 표준편차, 메모리, 품질, 재현 명령.

## Track B: Dory 1.0-4B 참조 트랙

ALoDLM은 uzu 비호환·저속으로 판정해 삭제했다. 현재 4B 참조 모델은
`nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF` Q5_K_M(3.673GB)이다.

- [x] Dory 지원 llama.cpp 포크(`brianchmielNV/llama.cpp`, revision
  `98a74c97…`) 빌드. SHA 검증 완료.
- [x] 1/2/4 loops 기준선(128토큰, 3표본): 20.0 / 8.9 / 10.3 tok/s.
  2 loops 값은 재현되지 않음(새 측정 14.7). 인용하지 않는다.
- [x] `--override-kv "dory.num_loops=int:N"`로 loop 수를 바꾼다
  (`tools/dory_bench.py`).
- [x] BF16 스케일 텐서의 CPU 배치를 고쳐 그래프 분할 8,085 → 14.
  디코드 1/2/4 loops = **61.7 / 42.2 / 25.1 tok/s**(수정 전 19.3 / 14.7 / 8.95),
  greedy 파리티 9/9. 잠정값(스왑·디스크 게이트 실패, GPU 유휴 시작).
  `docs/dory_speed_20261018.md`, `patches/dory-llama-cpy-placement.patch`.
- [ ] 청정 게이트 통과 후 10표본 재측정.
- [ ] L2 norm 단일 커널화(남은 간격: 4 loops 커널 상한 약 36.5 tok/s의 69%).
- [x] GGUF 매니페스트: 46 fields, 507 tensors.
- [x] Omega-Uzu `DoryModel` adapter와 1/2/4-loop parity. 4토큰 생성이
  llama.cpp와 일치한다(`Hello! How can`).
- [ ] KV 캐시/streaming generation.
- [ ] Q5_K Dory의 긴 문맥/속도 기준선.
- [ ] KSC 2.38bpw는 **조건부 spike**로만: Dory 4B 2-loop에서 >150GB/s와
  PPL +2%를 통과할 때만 채택. 현재 24.6GB/s로는 기본 경로가 아니다.

Dory는 uzu 모델이 아니라 별도 참조 트랙이다. uzu의 테스트 모델은 계속
Qwen3.5-9B-M을 쓴다.

## 저장소 구조

```text
docs/research_plan_20261018.md
docs/uzu_stabilization_20261018.md
tools/clean_env_check.py
bench/results/          # 새 측정만
paper/                  # 역사적 수학 명세, 현재 수치 아님
```

## 마일스톤

| 단계 | 완료 조건 |
|---|---|
| M0 | 청정 9B/27B 기준선 10표본 |
| M1 | 병목 구성요소 90% 설명 |
| M2 | 저위험 개선 중간값 +10% 또는 개선 없음 판정 |
| M3 | cursor-native 2bit 200GB/s 또는 중단 판정 |
| M4 | 품질 게이트와 최종 보고 |

## 위험

- 다른 MPS 학습이 GPU를 점유하면 모든 수치가 무효다. 게이트로 막는다.
- 27B는 이미 대역폭 상한에 가까워 커널 튜닝 이득이 작다.
- 2bit는 fragment 디코드 비용이 MMA 시간을 넘으면 이득이 없다.
- uzu upstream이 움직인다. 커밋을 고정하고 계약을 코드로 검증한다.
