## 새 아이디어 2. 데이터 병렬 replica 간 공유 시드 stochastic rounding: 부호 편향이 있을 때만 생기는 1/P 평균화 붕괴의 진단과 측정

### 가설

**원래 가설에서 버린 것.** 원래 T01의 두 주장, "공유 RNG state면 SR 오차가 rank 간 ρ=1/2로 결합된다"와 "stratified dither면 O(1/P²)"는 그대로는 성립하지 않는다.
- micro-batch 재사용은 일어나지 않는다. TE는 quantize 호출마다 Philox offset을 전진시킨다.
- stratified 이득은 최대 2배이다.
- FP4 SR은 sign-magnitude 방식(|x|를 0 쪽 또는 먼 쪽으로 반올림)이다. 그래서 같은 uniform을 공유해도 **gradient 부호가 rank 간에 독립이고 대칭이면 오차 상관은 0이다** (수치 확인 Part B). "공유 시드 = 완전 결합"은 틀렸다.

**좁힌 가설 (반증 가능).** 기본 설정의 Megatron-LM + Transformer Engine(TE) NVFP4 학습을 가정한다.
- 전제: gradient operand(dY)의 SR 난수가 모든 DP rank에서 같은 원소 위치에 같은 값으로 배정된다. 이것은 소스 코드를 읽어서 얻은 결론이며 아직 실행으로 확인하지 않았다.
- 결합 계수: 원소 위치 (t,i)에서 rank 간 SR 오차 상관은 ρ_ti ≈ ½·(E[sign dY_ti])²이다. uniform residue를 가정한 값이며, NVFP4의 비균일 residue에서는 상한이 약 0.35이다.
- 평균된 Wgrad의 SR 잡음: 입력 채널 k에 대해 Var ≈ T·v·[(μ_k²+σ_k²)/P + (1−1/P)·ρ_i·μ_k²]이다. 따라서 **부호가 편향된 출력 feature i와 평균이 0이 아닌 입력 채널 k가 만나는 칸에서만** 잡음이 P에 대해 plateau를 이룬다. post-SwiGLU FC2 입력이나 massive-activation 채널이 그런 k의 예이다.
- 교차점: plateau가 1/P 항을 넘는 지점은 P* ≈ (μ_k²+σ_k²)/(ρ_i μ_k²)이다.
- 실패 기준 1 (해롭지 않은 버그): 실제 1B급 모델의 모든 linear layer에서 측정한 max_{i,k} ρ_i·μ_k²/(μ_k²+σ_k²)가 1/1024보다 작으면, P≤1024에서는 공유 시드가 무해한 버그이다.
- 실패 기준 2 (전제 붕괴): 2-GPU 실측에서 SR 난수가 rank 간에 비트 단위로 일치하지 않으면 진단 자체가 무너진다.

### 문헌상 공백과 선행 연구 위치

- [Correlated quantization for distributed mean estimation and optimization 2203.04925](https://arxiv.org/abs/2203.04925)
  - l.492-503: "Ui = πi/n + γi", Qi(xi)=1{πi/n+γi<xi}. l.155: "Correlated quantization uses shared randomness to introduce correlation".
  - **rank-stratified dither 해법은 이미 출판되어 있다** [V]. 다만 [0,1] 입력 vector를 직접 평균하며, sign-magnitude FP SR이나 GEMM 후 all-reduce는 다루지 않는다.
- [Correlated Quantization for Faster Nonconvex Distributed Optimization 2401.05518](https://arxiv.org/abs/2401.05518)
  - l.1882: "the permutations require no extra communications, since they can be seeded" [V].
- [Training with Fewer Bits ... Stochastic Rounding 2511.00874](https://arxiv.org/abs/2511.00874)
  - l.578-584: "stochastic rounding is performed independently on each number", "SR variance ... diminishes as the mini-batch size b increases" (Lemma 3, 1/b).
  - 이 아이디어가 겨냥하는 바로 그 독립 가정이다 [V].
- [Pretraining LLMs with NVFP4 2509.25149](https://arxiv.org/abs/2509.25149)
  - l.571: "Use stochastic rounding for gradients and round-to-nearest-even for weights and activations".
  - App. E.4.3 l.1709-1713: RHT sign을 "seed per instance"로 바꿔도 "no improvements from inducing randomness at every transform instance".
  - 상관된 randomness가 무해할 수 있다는 반대 증거이다. 단, SR이 아니라 RHT 결과이다 [V].
- [Quartet II 2601.22813](https://arxiv.org/abs/2601.22813)
  - App. A l.1544-1545: "we re-randomize rotations per-tensor per-micro-batch". micro-batch 재사용 주장을 반박하며, DP rank는 논의하지 않는다 [V].
- [Low-Discrepancy Dither for Quantized Recurrent State Caches 2609.39185](https://arxiv.org/abs/2609.39185)
  - 시간 합산 축을 따라 dither 상관 구조를 설계한다 (Proposition 1, Weyl). l.1209 부근: "a dither correlated across entries"가 해롭다. 원리는 같고 축만 다르다 [V].
- [What is New in Stochastic Rounding (survey) 2603.06060](https://arxiv.org/abs/2603.06060)
  - l.328-340: NVIDIA 특허는 LFSR seed를 layer/block 위치로 정한다. 이렇게 하면 replica 간에 randomness가 공유될 수 있다. 상관 분석은 없다 [V].
- [Format-Aware Fusion for Fast FP4 Pretraining 2610.00053](https://arxiv.org/abs/2610.00053)
  - l.2070-2071: worker별 SR seed와 counter offset을 checkpoint한다. 시간 축 재사용만 다루고, rank 간 동일성은 다루지 않는다 [V].

**남는 공백 (이번 검색, arXiv 쿼리 약 27회와 WebSearch 4회에서 발견되지 않음):**
1. 기본 Megatron-LM + TE NVFP4에서 SR Philox state가 DP rank 간에 동일하다는 진단 [S, 소스 읽기].
   - Megatron은 `data_parallel_random_init=False`가 기본값이며, `seed += 10*dp_rank`는 이 옵션이 켜져 있을 때만 적용된다.
   - TE의 quantizer.cpp l.2656-2666과 cast.cpp l.173-185는 `getDefaultCUDAGenerator()`를 사용한다.
2. sign-magnitude SR에서 공유 난수의 효과가 (E[sign])²로 제곱 억제된다는 관찰과, 이것이 GEMM→all-reduce 경로에서 μ_k²와 곱해진다는 channel × feature 분해 [S].
3. stratified 이득도 같은 (E[sign])² 인자에 묶여, 부호에 편향이 없는 gradient에서는 이득이 1배라는 정정 [S].

### 수학 (증명 가능한 것과 추측)

**명제 1 (교과서 수준, 초등).** 가정: residue f_r ~ iid U(0,1), e_r = 1[u_r<f_r] − f_r.
- 독립 u: Var(ē) = 1/(6P).
- 공유 u: Cov = E[min(f_r,f_s)] − E[f_r f_s] = 1/12이므로 Var(ē) = 1/12 + 1/(12P).
- stratified (u_r=(π_r+γ_r)/P): Var(ē) = 1/(12P) + 1/(12P²).
- 원래 주장한 O(1/P²)는 거짓이다. 이 값은 residue가 rank 간에 같거나 집중된 경우, 즉 Suresh et al.의 small-deviation regime에서만 성립한다.
- 증명은 직접 계산이며, 새로운 결과가 아니라 DME 문헌의 특수한 경우이다.

**명제 2 (sign-magnitude 감쇠, 새 관찰이지만 초등).**
- 가정: FP SR은 e = s·δ·(1[u<f] − f)이다. 여기서 s는 부호, δ는 grid 간격, f는 크기 residue이다.
- PTX `cvt.rs`가 "toward zero / away from zero"를 가수 carry로 고른다는 문서 기술에 근거한다 [S, 웹 문서].
- 공유 u에서는 E_u[e_r e_s] = s_r s_s δ_r δ_s (min(f_r,f_s) − f_r f_s)이다.
- 부호가 크기와 독립이고 rank 간에 독립이면 Cov = (E s)²·E[δδ(min−ff)]이다. uniform f에서는 ρ = ½(E s)²이다.
- stratified의 음의 상관도 같은 인자 (E s)²에 비례한다.
- floor 기반 SR(정수 grid, f = x − ⌊x⌋)에서는 부호와 무관하게 ρ = ½이다. 따라서 INT형 gradient 압축이라면 원래 결론이 그대로 성립한다.

**명제 3 (Wgrad 전이).**
- 가정: rank r은 E_r[i,k] = Σ_t e^r_{t,i} x^r_{t,k}를 계산한다. 데이터는 rank 간에 독립이며, 계산 편의상 dY와 x도 독립으로 둔다.
- 결과: Var(mean_r E_r[i,k]) = T·v_i·[(μ_k²+σ_k²)/P + (1−1/P)·ρ_i·μ_k²].
- 고정 sign RHT(token 축 16점, 모든 rank에서 동일)를 쓰더라도, mean 에너지 Σ_j E[x̃_j]² = Σ_j μ²이 보존되므로 결합은 사라지지 않는다. 오히려 평균이 일부 위치에 집중되어 ρ가 볼록하게 커질 수 있다 [S, 수치로만 확인].

**추측 (미검증).**
- 실제 LLM의 dY는 출력 feature 단위의 부호 편향 E[sign dY_{:,i}]이 작다.
- 다만 BOS·attention-sink 위치나 FC2 출력의 일부 feature에서는 위치에 고정된 편향이 있을 수 있다.
- 이 편향과 massive-activation 채널(μ_k ≫ σ_k)이 겹치는 칸이 실제로 존재하는지가 이 아이디어의 사활을 가른다.

### 판정 실험 (Step 0-1 약 20 GPU-시간, Step 2 조건부 약 1-2 GPU-주)

- **Step 0: 전제 실측 (2×B200, 1시간 미만).**
  - Megatron-LM과 TE NVFP4 기본 recipe를 DP=2로 실행한다.
  - (a) 두 rank에 동일한 tensor를 SR quantize해서 출력이 비트 단위로 같은지 본다.
  - (b) 서로 다른 데이터에서 층별로 FC2·QKV·out-proj의 dY SR 오차 상관을 잰다.
  - `--data-parallel-random-init`를 켠 경우를 대조군으로 둔다.
  - (a)가 불일치하면 즉시 종료한다.
- **Step 1: channel × feature 통계 (단일 노드, 약 20 GPU-시간).**
  - 공개 1B·7B checkpoint와 자체 350M의 여러 시점에서 층별 ρ_i = ½(E[sign dY_{t,i}])², μ_k²/(μ_k²+σ_k²), 예측 P*_{ik}를 잰다.
  - DP는 micro-batch로 에뮬레이션한다. P개 micro-batch에 공유/독립/stratified uniform을 강제 배정하면 수학적으로 DP와 같다.
  - P ∈ {8, 64, 256, 1024}에서 Wgrad SR 잡음을 channel-resolved로 측정한다.
  - **성공:** 어떤 layer의 (i,k) 칸에서든 P=1024에서 공유/독립 잡음비가 2 이상이고, 그 비가 명제 3의 예측과 ±30% 안에서 맞는다.
  - **실패:** 모든 칸에서 비가 1.1 미만이다. 이 경우 "무해한 버그" 노트로 종료한다.
- **Step 2: 학습 (Step 1 성공 시에만).**
  - 350M·1B, NVFP4 backward, 에뮬레이션 DP 256·1024 (전역 batch 고정).
  - arm {공유, 독립, stratified} × 3 seed, 20B token.
  - 지표: 최종·후반부 loss gap, downstream 5종, Wgrad 잡음 중 sampling 잡음 비율.
  - **성공:** 공유와 독립 사이의 loss gap이 seed std의 2배를 넘는다.
  - **실패:** gap이 seed std 이내이다.
  - stratified는 독립 대비 이득을 따로 보고한다. 예상 이득은 최대 2배이며 실제로는 거의 1배이다.

### 수치 확인

스크립트는 `scripts/new_ideas/idea02_check.py`이다. CPU에서 약 40초가 걸리며 seed는 고정되어 있다.

- **Part A (명제 1).** P·Var를 이론값과 비교한다.

  | 방식 | P=8 (MC / 이론) | P=64 (MC / 이론) | P=512 (MC / 이론) |
  |---|---|---|---|
  | 독립 | 0.1678 / 0.1667 | 0.1659 / 0.1667 | 0.1667 / 0.1667 |
  | 공유 | 0.7493 / 0.7500 | 5.4200 / 5.4167 | 42.5728 / 42.7500 |
  | stratified | 0.0850 / 0.0938 | 0.0833 / 0.0846 | 0.0840 / 0.0835 |

  - 명제 1이 확인된다. stratified는 O(1/P²)가 아니라 1/(12P)에 머문다.
- **Part B (명제 2).** P=2에서 rank 간 원소별 오차 상관 ρ를 잰다.
  - floor-SR, 공유 u: ρ = +0.5000.
  - **NVFP4 sign-magnitude, 평균 0인 Gaussian gradient, 공유 u: ρ = +0.0004.** 원래 주장한 "완전 결합"을 반박한다.
  - gradient 평균/표준편차 비 m_g에 따른 공유 u 상관 (괄호 안은 ½(2Φ(m_g)−1)² 예측):

    | m_g | 0.25 | 0.5 | 1 | 2 | 모두 양수 |
    |---|---|---|---|---|---|
    | 공유 ρ | 0.0218 (0.0195) | 0.0771 (0.0733) | 0.2193 (0.2330) | 0.3683 (0.4555) | 0.3536 (0.5) |

  - stratified의 ρ도 −0.0009에서 −0.2620으로 같은 경향을 따른다.
- **Part C (명제 3).** 공유 u의 Wgrad 잡음 penalty(공유/독립)를 P=256에서 잰다.
  - (m_g=0, μ=3): 1.32
  - (m_g=0.5, μ=3): **14.2**
  - (m_g=2, μ=3): **96.8**
  - μ=0인 칸: 0.96-1.11
  - plateau 비 Var(256)/Var(16) (순수 1/P = 0.0625): 공유 u에서 (0.5, 3) 0.45, (2, 3) 0.72, 나머지 칸 약 0.06. 고정 RHT를 써도 0.68, 0.75로, RHT는 결합을 없애지 못한다.
  - stratified의 독립 대비 이득은 0.84-1.42이며 MC 잡음이 약 15%이다.
- **결론.** 원래 명제는 floor-SR에서만 맞는다. FP4 sign-magnitude SR에서는 **부호 편향이 없으면 효과도 없다.** 가설은 이번 결과에 맞춰 약화했다.

### 평가

- **신규성: 2/5 (낮음-보통).**
  - kill test는 3/5로 평가했다. 그러나 명제 2 때문에 효과가 (E sign)² × μ² 칸으로 제한되어 한 단계 낮췄다.
  - 진단(기본 설정에서 SR 난수가 rank 간에 동일)과 sign-magnitude 감쇠 분석은 이번 검색 범위에서 출판된 것이 없다.
  - 해법(per-rank seed offset, correlated quantization)은 이미 출판되어 있거나 한 줄 수정으로 끝난다.
- **main-track 가능성: 낮음 (10% 미만).**
  - 대규모 DP에서 loss gap이 나와야 하는데, 그러려면 큰 부호 편향과 massive channel이 겹쳐야 하며 사전적으로 그럴듯하지 않다. NVIDIA RHT seed ablation(2509.25149 E.4.3)도 무해 쪽을 가리킨다.
  - 현실적 결과물은 workshop 노트나 TE/Megatron 이슈 보고이다.
- **가장 큰 위험.**
  1. Step 0에서 rank 간 난수가 실제로는 정렬되지 않는 경우 (kernel launch 수 차이, fork context).
  2. 실제 dY의 부호 편향이 너무 작아, 측정 가능한 칸이 P*>1024에만 존재하는 경우.
  3. TE가 이미 rank offset을 넣었거나 조용히 수정해 전제가 사라지는 경우.
- **선점 위험.** NVIDIA TE·Megatron 유지보수자가 한 줄 수정으로 문제와 논문 가치를 함께 없앨 수 있다. SR 분산의 batch scaling을 이미 다루는 Quartet II·2511.00874 저자군도 독립 가정을 점검하다 같은 관찰에 도달할 수 있다.
