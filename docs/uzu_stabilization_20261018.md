# Uzu 안정화 기준선 (2026-10-18)

## 목적

이전 Omega-Uzu 실험은 GPU 경합, 스왑, 기본 262k KV, 오래된 수치가 섞여
있었다. 이 문서는 **다시 측정해도 되는 환경**을 고정하고, 그 전에는 어떤
성능 수치도 채택하지 않기 위한 기준선이다.

## 고정 자산

| 항목 | 값 |
|---|---|
| uzu 소스 | `5baa58b` (2026-10-17) |
| 릴리스 바이너리 | `cli 0.6.9` |
| 바이너리 SHA-256 | `00e3365fe560915decbb47182e1699c6e6f7861537ed278a993b840864189f47` |
| 기기 | Apple M5 Pro, 48GB, macOS 27.0 |
| 기준 모델 | `trymirai/Qwen3.5-9B-M` (5,189,121,136 B) |
| 대형 모델 | `trymirai/Qwen3.8-27B-M` (14.53GB + speculator 1.03GB) |

uzu 포크는 `5baa58b`에서 수정하지 않는다. 소스 변경은 Phase 2 이후에만
하고, 그때도 새 커밋을 고정한다.

## 오염 감사: 왜 이전 수치를 버리는가

| 이전 수치 | 문제 |
|---|---|
| 27B 23.2 tok/s | 모델 캐시가 삭제됐고, 현재 환경과 같은 조건이 아님 |
| GGUF 13.2–19.9 tok/s | 측정 중 다른 MPS 작업과 시스템 상태가 섞임 |
| MTP 1.47x | 측정 당시 다른 YOLO 학습이 GPU를 95% 점유 |
| KSC 51.6GB/s | 현재 코드에서 재현되지 않음. 최신 파리티 통과는 24.6GB/s |
| budget 3–4 최적 | 실측에서 16이 최선으로 반증됨 |

특히 `llama-cli`의 기본 262,144 컨텍스트는 KV 캐시 16GB를 잡아 RSS를
23.42GB까지 올렸다. 이는 uzu의 문제가 아니라 llama.cpp 기본값 문제였고,
uzu 기준선과 섞으면 안 된다.

결론: **이전 문서의 모든 tok/s는 “당시 환경의 관측”일 뿐, 새 기준선이 아니다.**
아카이브 태그 `archive/omega-pre-reset-20261018`에 보존한다.

## 현재 환경 상태

2026-10-18 정리 시점:

- GPU `Device Utilization` **95%** (다른 YOLO 학습)
- 스왑 **6.9GB** 사용
- 내부 디스크 여유 **11GB** (정리 전)
- 27B GGUF 38GB를 외장 `/Volumes/LinuxData/Models/qwen38-27b-archive`로 이동
- `~/.cache/mirai` 비어 있음

따라서 현재는 **측정 차단** 상태다. 다른 학습을 강제로 종료하지 않는다.
GPU가 유휴가 되고 스왑이 1GB 미만으로 내려간 뒤 Phase 0을 시작한다.

## 청정 측정 게이트

측정 전 반드시 실행한다.

```bash
python tools/clean_env_check.py
```

통과 조건:

- GPU utilization ≤ 10%
- swap used ≤ 1,024MB
- 내부 디스크 여유 ≥ 20GB
- `train_yolo_seg`, `train_dense_mps`, `fuse_yolo_unet`, `jadx`,
  `llama-cli`, `llama-server`, `mirai` 프로세스가 없음

하나라도 어긋나면 exit 1이며, 그 실행의 tok/s는 기록하지 않는다.
게이트를 통과한 뒤에도 10표본의 표준편차가 중간값의 10%를 넘으면
재측정한다.

## 기준선 실행 순서

1. `python tools/clean_env_check.py` 통과.
2. 9B 모델 다운로드 및 스모크 테스트(성능 아님).
3. 같은 task JSON으로 10표본 측정. 원자료는 `bench/results/`에 JSON.
4. 27B 모델 다운로드 및 스모크 테스트.
5. 27B 10표본 측정. spec off/on과 budget 8/16/32를 별도 라운드로.
6. 결과에 uzu commit, binary hash, 모델 크기, GPU/스왑, 명령을 붙인다.

## 현재 결론

Uzu 자체는 `5baa58b` / `cli 0.6.9`로 고정됐다. 하지만 **성능 기준선은 아직
없다.** GPU 경합이 끝난 뒤 위 순서로 새로 측정한다. 그 전에는 이전 23.2,
19.9, 1.47x, 51.6을 인용하지 않는다.
