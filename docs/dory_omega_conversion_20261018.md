# Dory 1.0-4B → Omega-Uzu 변환 계약 (2026-10-18)

## 결론

Dory를 Omega-Uzu에서 쓰려면 **가중치 포맷 변환만으로는 부족하다.** Omega-Uzu에
Dory forward(nGPT L2 norm, SWA/full attention, loop 재주입, residual alpha/gate,
last-loop KV)를 먼저 구현해야 한다. 그 전에 KSC로 인코딩한 파일은 로드해도
실행할 수 없다.

공식 Dory 포크는 빌드·검증했고, 이 모델의 정확한 텐서 계약과 기준 속도를
확보했다. Omega-Uzu 변환은 다음 순서로 진행한다.

## 확인된 GGUF 계약

- 파일: `NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf`
- 크기: 3,673,288,864 bytes
- SHA-256: `96080be49ffd67b916018b7bdef8c484762576fb52780e10ab98ac4978a660e0`
- HF revision: `0963c8000791e77b2b1b49aa387d92a6da9988d6`
- architecture: `dory`
- physical layers: 72 = 16 input + 40 recurrent + 16 output
- virtual layers: `16 + 40*loops + 16` (4 loops 기본 = 192, 2 loops = 112)
- `n_embd=2560`, `n_ff=7168`, `n_head=16`, `n_head_kv=4`, head_dim 256
- `n_swa=1024`, context 262144
- rope: `theta_1=1e7`, `theta_2=1e4`, profile 72개(값 1/2)
- `init_std=0.0177`, `alpha_init=0.05`, `gate_init=50.596443`
- layer pattern: `S-S-S-*-S-S-S-*-...` (S=SWA, `*`=full attention, `-`=FFN-only)
- 46 metadata fields, 507 tensors
- Q5_K weights type 13, token embedding/output BF16 type 30

주의: 모델 카드 설명의 8/20/8은 이 GGUF의 실제 메타데이터(16/40/16)와 다르다.
변환은 **GGUF 메타데이터**를 정본으로 삼는다.

## 텐서 계약

공통:

- `token_embd.weight`
- `output.weight`
- `output_scale.weight`

각 physical layer `p` (0..71):

- `blk.p.residual_alpha.weight`
- attention layer:
  - `blk.p.attn_q.weight`, `attn_k.weight`, `attn_v.weight`, `attn_output.weight`
  - `blk.p.attn_gate.weight`
  - `blk.p.attn_qk_scale.weight`, `attn_gate_scale.weight`
- FFN-only layer:
  - `blk.p.ffn_gate.weight`, `ffn_up.weight`, `ffn_down.weight`
  - `blk.p.ffn_gate_scale.weight`, `ffn_up_scale.weight`
- Q5_K 텐서마다 `input_scale`/`scale` sidecar가 붙는다.

`blk.*`는 virtual layer 이름으로 반복되지만 실제 물리 가중치는 72개다.
변환기는 virtual→physical 매핑을 먼저 만들고 중복 저장을 피해야 한다.

## Forward 계약 (dory.cpp 기준)

`norm(x) = rms_norm(x, eps/n) / sqrt(n)` (unit L2, eps=1e-6).

```text
for virtual layer il:
    p = physical[il]
    if il == n_input: input_h = cur
    if il is start of a recurrent loop: cur = norm(cur + input_h)
    residual = cur

    if pattern[p] == '-':
        g = ffn_gate(cur) * (suv_gate * sqrt(n_embd))
        u = ffn_up(cur) * (suv_up * sqrt(n_embd))
        cur = norm(silu(g) * u)
        cur = ffn_down(cur)
    else:
        q,k,v = qkv(cur)
        if profile[p] != 0: q,k = rope(q,k, theta_profile)
        q = norm(q) * sq
        k = norm(k)
        gate = sigmoid(attn_gate(cur) * gate_scale)
        cur = attention(q,k,v, cache_layer[il])
        cur = norm(cur * gate)
        cur = attn_output(cur)

    a = norm(residual)
    b = norm(cur)
    alpha = abs(residual_alpha * alpha_init / init_std)
    cur = norm(a + (b - a) * alpha)
    if il is end of a recurrent loop: cur = norm(cur)

logits = output(cur) * (output_scale / init_std)
```

`recurrent_kv_cache_mode`는 `per_loop` 또는 `last_loop`이다. last_loop는
recurrent layer의 cache slot을 물리 레이어로 공유해 KV를 한 벌만 유지한다.

## 변환 단계

1. **Reference 고정** (완료)
   - Dory 지원 llama.cpp `98a74c9` 빌드.
   - SHA 검증, 텐서 매니페스트 추출.
   - 128토큰, 3표본 기준선: 1 loop 20.0, 2 loops 8.9, 4 loops 10.3 tok/s.
   - 4 loops가 2 loops보다 높은 값은 출력 길이/캐시 효과가 섞인 신호이므로
     절대 속도 결론으로 쓰지 않는다.
2. **Dory adapter** (다음 코드 작업)
   - Omega-Uzu에 `DoryModel`/`DoryLayer`를 추가한다.
   - Q5_K GGUF를 dequantize하고 `scale`/`input_scale`을 적용한다.
   - nGPT norm, SWA/full attention, loop 재주입, residual alpha/gate를 구현한다.
   - 한 토큰 logits 또는 greedy 토큰이 llama.cpp와 일치하는지 parity를 잡는다.
3. **KSC 인코딩**
   - adapter parity 후에만 가중치를 2.38075bpw KSC로 인코딩한다.
   - 독립 GEMV를 만들지 않는다. uzu `GemmTrellis`처럼 MMA fragment로 직접
     디코드되는 `Cursor<Fragment>`를 사용한다.
   - KSC GEMV 24.6GB/s 기준으로는 Dory 4B가 2 loops 10–11 tok/s 수준이라
     공식 포크보다 느리다. 150–200GB/s 이상이 되기 전에는 채택하지 않는다.
4. **Metal 포팅**
   - Python reference parity 후 Rust/Metal로 옮긴다.
   - nGPT/SWA/loop KV를 한 그래프로 융합한다.

## 현재 차단점

- Omega-Uzu에는 Dory forward가 없다. 아카이브된 ALoDLM은 단순 RMSNorm/표준
  attention이라 Dory의 nGPT/SWA/gate를 표현하지 못한다.
- Dory 가중치는 Q5_K + sidecar scale이라 단순 `safetensors` 변환이 아니다.
- KSC 인코더는 행렬을 인코딩할 수 있지만, Dory 런타임과 연결되지 않았다.
- 따라서 “변환된 체크포인트”만 만드는 것은 사용 가능한 결과가 아니다.

## 다음 작업

`DoryModel`의 Python reference를 별도 브랜치에서 구현하고, 1 loop·한 토큰
parity를 먼저 통과시킨다. 그 뒤 2/4 loops와 KSC를 붙인다.

원자료:

- `bench/results/dory_manifest_20261018.json` — 46 fields, 507 tensors
- `bench/results/dory_4b_loops_20261018.json` — 1/2/4 loop 기준선
- `tools/dory_gguf_manifest.py` — 매니페스트 추출기
