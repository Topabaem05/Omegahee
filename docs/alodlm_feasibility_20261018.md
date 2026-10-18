# ALoDLM-8B 확인 결과와 uzu 호환성 (2026-10-18)

## 확인한 사실

- 저장소: `amazon/ALoDLM-8B`, 공개, revision `2c4623d9…`, 라이선스 CC BY-NC 4.0.
- 가중치: BF16 safetensors 4샤드, 총 **16,381,470,720바이트(16.38GB)**.
- 추가 파일: `exit_gate.pt`, `alodlm_config.json`, tokenizer, chat template.
- config: `Qwen3ForCausalLM`으로 표기되지만 실제로는 ALoDLM 래퍼다.
  - 36 layers, hidden 4096, 32 heads, 8 KV heads, context 40960, BF16.
  - `loop_start=10`, `loop_end=26`, `max_depth=4`, `mask_token_id=151665`,
    attention backend `sdpa`.

## 핵심: uzu에 바로 넣을 수 없다

ALoDLM은 일반 autoregressive 모델이 아니다. **masked diffusion + adaptive
recurrent refinement + learned exit gate**로 디코딩한다. 같은 Qwen3 백본을
쓰지만, 표준 `Qwen3ForCausalLM` 경로로 읽으면 ALoDLM의 반복·게이트·병렬
커밋이 전부 빠진다. 모델 카드도 “표준 Transformers generation pipeline은
ALoDLM의 recurrent decoding과 exit gate를 생략한다”고 명시한다.

uzu는 Mirai-M 4bit 체크포인트를 로드하는 autoregressive 엔진이다. BF16
safetensors와 ALoDLM 커스텀 디코더를 지원하지 않는다. uzu에 넣고 Qwen3
백본으로 돌리면 **ALoDLM이 아니라 일반 Qwen3-8B AR 근사**가 되고, 출력은
모델의 실제 성능이 아니다. 따라서 ALoDLM-8B를 uzu의 테스트 모델로 쓰면
측정 자체가 무효다.

## 공식 런타임

| 경로 | 지원 | Mac M5 Pro |
|---|---|---|
| optimized nano-vLLM | CUDA + CUDAGraph | 불가 |
| portable PyTorch SDPA | CUDA BF16, CPU FP32 | CPU만 가능 |
| MPS/Metal | 없음 | 불가 |

portable decoder의 `--device` 선택은 `cpu`와 `cuda`뿐이다. MPS 경로가 없다.
따라서 이 Mac에서는 **CPU FP32**로만 돌릴 수 있다. BF16 16.38GB가 FP32에서
약 32.8GB가 되고, KV/workspace를 더하면 48GB에서 매우 빠듯하다. 8B 모델은
CPU에서도 느리고, diffusion window/commit 방식이라 AR tok/s와 직접 비교할
수 없다.

계산 구조는 36개 기본 레이어 중 10–26층을 최대 4회 반복한다. depth별
레이어 등가 수는 1회 36, 2회 52, 4회 84다. 같은 가중치를 반복해서 읽으므로
CPU 메모리 대역폭이 지배한다. 정확한 tok/s는 청정 게이트를 통과한 뒤
실측해야 한다. 1.7B(3.4GB BF16, 6.8GB FP32)가 CPU 참조 실험에 더 현실적이다.

## 수정된 트랙

- **Track A — uzu:** `Qwen3.5-9B-M`(5.6GB)으로 uzu 기준선과 커널 개선.
  27B는 외장에 아카이브했고, 필요할 때 내부로 복원한다.
- **Track B — ALoDLM:** 공식 portable decoder를 별도 Python 3.12 환경에서
  CPU FP32로 실행하거나, 별도 Metal/MPS 포트를 연구한다. uzu와 같은 표에
  넣지 않는다.

ALoDLM을 uzu 안에서 돌리려면 masked diffusion, recurrent core, exit gate,
parallel commitment를 uzu에 새로 구현해야 한다. 이는 모델 교체가 아니라
**새 엔진 프로젝트**다. uzu의 `gemm_trellis`/MMA 경로를 재사용할 수는 있지만,
디코딩 루프와 KV/마스크 관리가 전혀 다르다.

## 설치 상태

- 가중치: `/Users/topamini/models/alodlm/ALoDLM-8B`에 설치 완료(16.4GB).
- 코드: `amazon-science/ALoDLM`을 별도 경로에 클론.
- 런타임: `/Users/topamini/models/alodlm/venv312`에 Python 3.12.14,
  PyTorch 2.8.0, Transformers 4.56.1, `alodlm 0.1.0` 설치 완료.
  PyTorch 수준에서는 MPS가 available이지만, 공식 CLI의 `--device`는
  `cpu`와 `cuda`만 허용한다. MPS 실행은 별도 러너/패치가 필요하다.
- 실행 게이트: GPU/메모리 청정 조건과 별개로, CPU FP32 33GB가 들어갈 수
  있는지 확인해야 한다.
