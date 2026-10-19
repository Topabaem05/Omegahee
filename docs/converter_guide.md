# Ω-Hee 모델 변환 가이드

새 모델을 Ω-Hee(uzu 기반 엔진)에서 돌리는 절차입니다. Dory 4B 지원을 실제 예로 듭니다.
저장소 구조: `converter/`(lalamo 기반 변환기), `engine/`(uzu 기반 엔진), `reference/`(참조 구현과 비교
도구), `tools/`(변환·측정 도구).

## 1. 이미 지원되는 모델 변환

1. Hugging Face 체크포인트를 내려받습니다.
   ```bash
   hf download <repo> --local-dir ~/models/<name>
   ```
2. 로컬 체크포인트를 변환합니다. 원본 폴더는 바꾸지 않습니다. 설정 덮어쓰기는 임시 복사본에만 적용됩니다.
   ```bash
   cd converter && uv run python ../tools/omega_convert.py \
       --spec <spec 이름 또는 repo> --checkout ~/models/<name> --out ~/models/<name>-omega \
       --encoding '{"type": "hanashi", "name": "<preset>"}'
   ```
   - spec 목록: `cd converter && uv run lalamo list-models`
   - `--override key=value`: `config.json` 값을 바꿉니다(예: `n_recurrent_loops=4`).
   - 출력 폴더의 `conversion.json`에 spec, 덮어쓰기, 입력·출력 SHA-256이 남습니다.
3. 실행합니다.
   ```bash
   cd engine && cargo run --release -p cli -- --model ~/models/<name>-omega -m "Hello"
   ```
   `UZU_BACKEND=cpu`를 붙이면 CPU 백엔드로 돕니다(느림, 검증용).

`encoding.json`은 엔진의 채팅 인코딩(hanashi 프리셋)입니다. 변환기(lalamo)는 이 파일을 만들지 않으므로
`--encoding`으로 넣습니다. 프리셋 목록은 `engine/crates/legacy/hanashi/configs/configs.json`에 있습니다.

## 2. 새 아키텍처 추가

정답 기준을 한 칸씩 내려가며 검증합니다: 참조 구현 → 변환기(JAX) → 엔진 CPU → 엔진 Metal.

### 2.1 참조 구현과 고정 출력

- Hugging Face `transformers` 구현이 있으면 변환기 tracer(`converter/tests/tracer/`)를 그대로 씁니다.
- 없으면(Dory처럼) vLLM 등의 구현을 순수 PyTorch로 옮깁니다. 예: `reference/dory_ref.py`.
  - fp32·bf16 두 번 돌려 허용 오차를 정합니다(`reference/compare_ref.py`, 기록 `docs/dory_stage0_reference_20261019.md`).
  - 고정 프롬프트(`reference/make_prompts.py`)의 실행 단계별 활성값·로짓을 저장합니다.

### 2.2 변환기 (`converter/lalamo`)

1. **HF 설정 클래스**: `model_import/model_configs/huggingface/<model>.py`에 `HuggingFaceLMConfig` 하위
   클래스를 만들고 `to_decoder_config()`를 구현합니다. `__init__.py` 두 곳에 등록합니다.
2. **모델 spec**: `model_import/model_specs/<vendor>.py`에 `LanguageModelSpec`을 두고
   `model_specs/__init__.py`의 `ALL_MODEL_LISTS`와 `tests/model_test_tiers.toml`에 넣습니다.
   채팅 템플릿이 `tokenizer_config.json`에 없으면 `ConfigMap(chat_template=FileSpec("chat_template.jinja"))`.
3. **가중치 로더**: 이름 규칙이 HF 표준과 다르면 `_load_weights`를 재정의하고 `model_import/loaders/<model>_loader.py`를
   만듭니다. 모든 체크포인트 텐서를 읽었는지 단위 테스트에서 확인합니다(`tests/unit/test_dory.py`의 `RecordingMapping`).
4. **모듈**: 먼저 기존 블록으로 표현합니다. Dory 때 추가된 재사용 블록:

   | 블록 | 용도 |
   |---|---|
   | `NormalizationConfig.reduction = SUM` | 가중치 없는 L2 정규화 |
   | `LearnedScaleConfig(multiplier, transform)` | 학습된 채널별 스케일(원본 값 저장, 상수·변환은 설정) |
   | `AttentionConfig.query_scale_config / gate_scale_config / output_norm_config` | RoPE 뒤 q 스케일, 스케일된 게이트, 게이트 뒤 정규화 |
   | `DenseMLPConfig.up_scale_config / gate_scale_config / hidden_norm_config` | MLP 채널 스케일, down 전 정규화 |
   | `UntiedEmbeddingConfig.logit_scales_config` | 어휘별 로짓 스케일 |
   | `NGPTAttentionLayerConfig`, `NGPTMLPLayerConfig` | 한 하위층 + nGPT 보간 잔차 |
   | `TransformerConfig.recurrence` | 공유 층 구간 반복(루프), `per_loop`/`last_loop` KV |
   | `TransformerConfig.output_norm_config = None` | 마지막 정규화 없음 |

   층 설정은 `TransformerLayerConfigBase` 레지스트리라 config.json 층마다 `"type"`이 붙습니다.
5. **검증**: 작은 무작위 모델로 참조 수식과 비교하는 단위 테스트(prefill·decode·KV 모드), 그리고 실제 가중치로
   참조와 비교(`reference/compare_lalamo.py`). 잘못된 구현을 일부러 넣어 테스트가 실패하는지도 봅니다.
6. `uv run pytest -m fast -n=4`, `uv run ruff check`, `uv run pyrefly check`.

### 2.3 엔진 (`engine/crates/uzu-engine`)

1. **설정 구조체**: 변환기 dataclass와 필드 단위로 1:1입니다(`src/config/`). `uzu_config`는 알 수 없는 필드를 거부하고
   `Option`도 명시적 `null`을 요구합니다. 변환기에 필드를 더하면 여기도 반드시 더합니다. 다형 설정은
   `uzu_config_abstract`(예: `AnyLayerConfig`, `AnyTokenMixerConfig`).
2. **블록**: `src/encodable_block/`. 기존 층 흐름(잔차를 다음 정규화 커널에 융합하는 `shortcut`)에 맞지 않으면
   별도 블록을 둡니다(예: `ngpt_layer.rs`). 공통 경로는 바꾸지 않는 것을 원칙으로 합니다.
3. **커널**: 같은 이름으로 `src/backends/metal/kernel/<op>/<op>.metal`(DSL `common/dsl.h`)과
   `src/backends/cpu/kernel/<op>/<op>.rs`(`#[kernel]`)를 둡니다. 두 시그니처가 다르면 빌드가 실패합니다.
   CPU 모듈은 `src/backends/cpu/kernel/mod.rs`에 등록합니다. 단위 테스트는 `unit/backends/common/kernel/`에서
   `for_each_backend!`로 두 백엔드를 호스트 계산과 비교합니다(예: `ngpt_test.rs`).
4. **가중치 이름**: 변환 결과 `model.safetensors`의 경로(`decoder.transformer.layers.{i}.mixer...`)를 그대로 읽습니다.
   로드가 끝나면 `assert_all_tensors_read()`가 남은 텐서를 거부합니다.
5. **파리티**: `unit/session/dory_parity.rs`(`UZU_DORY_PARITY` 환경 변수)로 토큰 ID → 로짓을 덤프하고
   `reference/compare_uzu.py`로 참조와 비교합니다. 엔진 계산은 BF16이므로 BF16 허용 오차를 씁니다.
6. **채팅 인코딩**: hanashi에 프리셋을 추가합니다(`configs/rendering/<name>.json`에 채팅 템플릿,
   `configs/tokens/<name>.json`, `configs/configs.json`, `HanashiConfig` enum). 파싱 규칙은 비슷한 모델 것을 재사용합니다.

## 3. Dory 예시 명령

```bash
# 변환 (2 loops, per_loop 기본값)
cd converter && uv run python ../tools/omega_convert.py --spec NVIDIA-Nemotron-Labs-Dory1.0-4B \
    --checkout ~/code/uzu-omega/models/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 \
    --out ~/code/uzu-omega/models/Dory1.0-4B-lalamo-bf16 --encoding '{"type": "hanashi", "name": "dory"}'

# 참조 대 변환기, 참조 대 엔진
cd converter && uv run python ../reference/compare_lalamo.py --model <checkout> \
    --prompts ../reference/prompts.json --ref-dir ../reference/out/matrix --out result.json --long
cd reference && ./run_uzu_matrix.sh metal result.json 31 16
```

## 4. 주의

- 원본 체크포인트는 수정하지 않습니다. 루프 수 같은 설정은 `--override`나 엔진 로드 시점에서 바꿉니다.
- 엔진의 기존 모델 파일(층에 `"type"`이 없거나 새 필드가 없는 config.json)은 이 브랜치의 엔진에서 다시 변환해야 합니다.
- 양자화(`IntSpec`, `MLXSpec`) 변환은 lalamo의 기존 경로를 따릅니다(5절).
- `reference/compare_uzu.py --dump-dir`에는 절대 경로를 씁니다. 덤프는 `engine/`에서 도는 cargo test가 씁니다.

## 5. 양자화 변환

`tools/omega_convert.py`의 `--linear-spec`(transformer 선형층)과 `--readout-spec`(출력 임베딩)에 lalamo 양자화 spec을
JSON으로 줍니다. 엔진 Metal GEMV(decode)가 지원하는 형식만 쓰십시오. 지원하지 않는 형식은 느린 경로로 떨어지거나
실행되지 않습니다.

| spec | 엔진 형식 | decode GEMV | 비고 |
|---|---|---|---|
| `{"type": "MLXSpec", "bits": 4, "group_size": 64}` | scale + bias(MLX) | 있음(batch 1, batch 2 입력 2행 타일) | Dory INT4 기본 |
| `{"type": "MLXSpec", "bits": 4, "group_size": 32}` | scale + bias | 있음(batch 1) | 메타데이터가 늘어 g64보다 약간 느림 |
| `{"type": "MLXSpec", "bits": 8, ...}` | scale + bias 8비트 | 없음 | 쓰지 마십시오 |
| `{"type": "IntSpec", "bits": 8, "group_size": 64, "is_symmetric": true}` | 대칭 8비트 | 있음 | 8비트 readout |
| group 128 | — | 없음(GEMV는 group 32, 64만) | |

예(Dory INT4, readout INT4):

```bash
cd converter && uv run python ../tools/omega_convert.py --spec NVIDIA-Nemotron-Labs-Dory1.0-4B \
    --checkout ~/code/uzu-omega/models/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 \
    --out ~/code/uzu-omega/models/Dory1.0-4B-mlx4g64-r4 --encoding '{"type": "hanashi", "name": "dory"}' \
    --linear-spec '{"type": "MLXSpec", "bits": 4, "group_size": 64}' \
    --readout-spec '{"type": "MLXSpec", "bits": 4, "group_size": 64}'
```

양자화 모델은 참조와 BF16 허용 오차로 비교할 수 없습니다. 품질 묶음(6절)으로 BF16 변환 결과와 비교합니다.

## 6. 측정

| 도구 | 용도 |
|---|---|
| `tools/omega_bench.py` | decode·prefill tok/s(benchmarks-uzu, GPU 60°C 이하 대기, 워밍업 제외 중앙값). 결과는 `bench/results/research_runs.jsonl` |
| `tools/omega_eval.py` | 품질 묶음(GSM8K, HumanEval, MMLU-Pro, 16K 검색). 결과는 `bench/eval_runs/<label>`, `bench/results/eval_runs.jsonl` |
| `tools/dory_bench_requests.py` | 고정 요청 묶음(128, 2K, 16K) 생성 |
| `unit/backends/metal/kernel/matmul/dory_bench.rs` | decode 모양 GEMV 마이크로벤치와 타일 비교(`cargo bench -p uzu-engine --lib -- "DoryLayers"`). 모양 표를 새 모델 것으로 바꿔 씁니다 |

- 이론 상한은 토큰당 가중치 바이트(루프 수만큼 반복되는 층 포함)와 KV 바이트를 함께 셉니다. sliding window 층의 KV는
  window 길이까지만 셉니다. `omega_bench.py`의 `upper`는 가중치만 셉니다.
- 같은 GPU를 쓰는 작업(eval 서버 등)이 돌면 속도 수치를 믿을 수 없습니다. 측정할 때는 그 작업을 멈춥니다(eval
  클라이언트에 `kill -STOP`, 끝나면 `kill -CONT`; greedy라 점수는 바뀌지 않습니다).
- 커널 종류별 시간은 해당 커널을 빼고(출력은 버림) decode 시간을 다시 재는 ablation으로 나눕니다
  (`docs/dory_research_log.md`의 ablation 절). GPU trace는 커널별 시간을 대부분 나누지 못했습니다.

## 7. 루프 wavefront decode

반복 층 구간(`TransformerConfig.recurrence`)이 있는 모델은 decode에서 루프 wavefront를 쓸 수 있습니다
(`encodable_block/transformer/wavefront.rs`). 한 단계에서 토큰 t의 2번째 루프와 토큰 t+1의 1번째 루프를 반복 층에서
batch 2로 실행하고, t+1은 1루프 조기 종료 출력으로 추측합니다.

- 조건: nGPT 층, 루프 2회, `per_loop` KV, greedy, grammar와 speculator 없음. 켜기: `OMEGA_WAVEFRONT=1`.
- 새 모델에 쓰기 전 확인:
  1. 1루프 조기 종료와 2루프 출력의 다음 토큰 일치율(생성 문장 기준). Dory는 92.5%였습니다. 낮으면 이득이 없습니다.
  2. teacher-forced 덤프(`unit/session/dory_wavefront.rs`의 `teacher_forced_dump_dir`)를 `compare_uzu.py --skip-run
     --dump-dir`로 참조와 비교하고, 순차 decode 덤프와 logits 차이가 BF16 잡음 수준인지 봅니다.
  3. 두 턴 대화 상태 검사(`dory_wavefront_turns`).
  4. 근사 변경이므로 품질 묶음으로 판정합니다.
- 양자화 모델은 batch 2 GEMV 비용이 이득을 좌우합니다. 형식별 batch 2 타일을 `dory_bench.rs` 타일 비교로 확인합니다.
