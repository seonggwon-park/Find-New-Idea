## 새 아이디어 7. 저비트 키 프록시 기반 top-k 선택의 레짐 의존 사후 질량 랭킹: 크로스오버 분석과 비가우시안 경험적 베이즈 사전

### 가설

**설정.** 사후(post-hoc) 저비트 키 프록시로 토큰이나 페이지를 고르는 경우만 다룬다. 대상은 1–2 bit RaBitQ나 sign sketch, 저비트 Quest bound, 양자화 K 위의 H2O 축출, recompute 선택이다. 이 경우 프록시 점수 ŝ_i = s_i + δ_i의 토큰별 잡음 σ_i를 저장된 scale과 norm에서 계산할 수 있다. QAT로 학습한 indexer(DSA 계열)는 제외한다. 그 점수는 학습된 ranker 자체이고, 정의된 s_i를 잡음 섞어 추정한 값이 아니기 때문이다.

**주장.** 질의마다 경험적 베이즈(EB) 사전을 적합하고 사후 기대 질량 E[e^{s_i} | ŝ_i]로 순위를 매기면 점추정 ŝ 랭킹보다 낫다.
- 보정의 부호는 레짐에 따라 바뀐다. 가우시안 사전에서 잡음 큰 토큰은 크로스오버 ŝ = μ + τ²/2 위에서는 강등되고 아래에서는 승격된다.
- 따라서 이미 출판된 고정 보정 ŝ − σ²/2(Jensen debias)를 랭커로 그대로 쓰면, 예산 경계가 크로스오버 아래에 있는 레짐(날카로운 head, 큰 예산)에서는 오히려 해롭다.
- logit 상단 꼬리가 두꺼우면 가우시안 사전이 과수축한다. 이때는 NPMLE 같은 비가우시안 사전이 필요하다.

**정량 예측.** 수치 확인 결과에 맞춰 원래 주장보다 약하게 잡는다.
- 이득이 의미 있는 크기(질량 ≥0.3pp, 또는 gap 회복률 ≥0.3)가 되는 것은 σ²/τ² ≳ 0.2이고 예산이 ≤5%일 때다.
- σ²/τ² ≤ 0.05이거나 FP8·INT8 수준(≈1e-3)이면 이득은 0이다. 이것을 음성 대조로 쓴다.

**실패 기준.** 아래 셋 중 하나면 가설을 기각한다.
1. 1-bit 프록시, 1–5% 예산에서 posterior 랭킹이 point-estimate recall과 exact recall 사이 gap의 30% 미만만 회복하고, RULER나 LongBench 이득도 1점 미만이다.
2. 크로스오버 아래 레짐에서 EB가 Jensen-debias 랭커보다 낫지 않다.
3. 이득이 σ_i의 분산(dispersion)과 상관이 없다.

### 문헌상 공백과 선행 연구 위치

**선택 규칙과 보정 항.**
- [Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution 2510.00636](https://arxiv.org/abs/2510.00636): Eq. 7(l.325-343)이 z_i = exp(μ_qᵀk_i/√d + k_iᵀΣ_q k_i/(2d))를 "moment-generating function of a Gaussian"으로 유도하고, 이 값으로 KV를 순위 매겨 축출한다 [V]. 원 가설의 정리(top-k by m+v/2)와 순위 규칙이 같다. 불확실성의 원천만 다르다(미래 query vs 양자화 잡음). 수축(shrinkage)이나 사전 적합은 없다.
- [Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in Video Diffusion 2605.26266](https://arxiv.org/abs/2605.26266): l.286-300에서 키 양자화 잡음의 "Jensen bias" E[Ẑ_S] ≥ Z_S를 보이고, l.382-393에서 b_i ≈ σ_i²/2 보정을 CGF로 정당화한다 [V]. dense attention용이며 top-k 선택이나 수축은 다루지 않는다.

**통계학의 표준 결과.**
- [Making the cut: improved ranking and selection for large-scale inference 1312.5776](https://arxiv.org/abs/1312.5776): 초록(l.13)에 "local maximum likelihood (ML) approaches tend to favor units with high measurement error"라고 쓰고, EB 랭킹으로 이를 고친다 [V]. 원래의 "winner's curse" 주장은 여기서 이미 교과서적 결과다.
- [Invidious Comparisons: Ranking and Selection as Compound Decisions 2012.12550](https://arxiv.org/abs/2012.12550): 이질적 정밀도 아래 선택을 compound decision으로 놓고 Kiefer–Wolfowitz NPMLE 사전을 쓴다(l.17, l.222-224) [V]. 비가우시안 사전이라는 우리 쪽 요소도 통계학에는 이미 있다.

**top-k 및 KV 선택 쪽 문헌.**
- [UNIQUE: Universal Top-k Sparse Attention for Training-free Inference and Sparsity-aware Training 2605.27740](https://arxiv.org/abs/2605.27740): page score = q·mean + λ‖q‖·std (l.205, l.340) [V]. 휴리스틱 분산 보너스이고, 양자화 잡음과는 무관하다.
- [RaBitQCache: Rotated Binary Quantization for KVCache in Long Context LLM Inference 2606.31519](https://arxiv.org/abs/2606.31519): unbiased estimator를 강조하고(l.28, l.113) 점추정으로 순위와 threshold를 정한다 [V]. 선택 편향 분석은 없다.
- [Fathom 2609.17652](https://arxiv.org/abs/2609.17652)는 분산을 bit-plane 깊이에만 쓰고 추정치로 순위를 정한다 [V, note 기준]. [SAKI 2608.03228](https://arxiv.org/abs/2608.03228) Prop. 2(l.279)는 margin 조건(2ε < γ_k) 아래 top-k 정확성만 보인다 [V, note 기준]. [Self-Indexing KVCache 2603.14224](https://arxiv.org/abs/2603.14224), [AMEND 2609.09823](https://arxiv.org/abs/2609.09823), [Runtime-Certified Bounded-Error Quantized Attention 2605.20868](https://arxiv.org/abs/2605.20868), [WitCert 2607.28699](https://arxiv.org/abs/2607.28699)는 오차 상계·인증·margin audit이며 posterior/수축 랭킹은 없다 [S, note 요약 기준].
- [Spend Bytes on Breadth: Precision-Count Trade-offs for Decode-Time KV Compression in Long Chain-of-Thought Reasoning 2610.05685](https://arxiv.org/abs/2610.05685): 10% 보존 시 oracle도 미래 attention 질량의 약 13%를 잃는다(l.26-27, l.436-437). 저자들은 이를 "a floor that no scorer can remove"(l.434)라고 부른다 [V]. 이것은 선택 오류가 섞이지 않은 oracle 하한이므로 "선택 오류가 곧 질량 손실"이라는 근거로 쓸 수 없다.
- [DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression 2609.19969](https://arxiv.org/abs/2609.19969): l.1002에서 indexer의 FP4 query와 key에 QAT를 쓴다고 밝힌다 [V]. DSA 계열을 이 모델의 적용 대상에서 빼는 근거다. V3.2의 FP8 indexer 주장은 2차 블로그로만 확인된다 [S].

**남은 공백.** 다음 조합은 이번 검색(쿼리 약 23회: kill test 20회와 본 절 작성 중 arXiv 3회)에서 발견되지 않았다.
- 질의별로 적합한 EB 사전(가우시안 또는 NPMLE) 아래 E[e^{s}|ŝ]로 순위를 매긴다.
- 대상은 사후 양자화 키 프록시의 top-k 선택이다.
- 크로스오버 ŝ = μ + τ²/2를 기준으로 예산과 head 날카로움에 따른 부호 반전을 분석한다.

본 절 작성 중 추가 검색에서 "posterior bias"라는 표현을 쓰는 pre-hoc sparsity 논문(Gao et al., 2026)이 걸렸다. 초록을 보면 선택 시점을 앞당기는 방법이고 EB 랭킹과는 무관하다 [S, 초록만 확인].

### 수학 (증명 가능한 것과 추측)

**명제 1 (교과서, 자명).**
- 가정: s_i ~ N(μ, τ²), ŝ_i | s_i ~ N(s_i, σ_i²)이고 토큰끼리 독립이다.
- 사후분포: s_i | ŝ_i ~ N(m_i, v_i). 여기서 m_i = (τ²ŝ_i + σ_i²μ)/(τ²+σ_i²), v_i = τ²σ_i²/(τ²+σ_i²)이다.
- 결론: 기대 질량 E Σ_{i∈S} e^{s_i}를 최대화하는 크기 k 집합은 m_i + v_i/2 기준 top-k다. 기대값의 선형성과 lognormal 평균에서 바로 나온다.
- 이 순위 규칙은 2510.00636 Eq. 7과 같은 형태다. 따름정리 E[s−ŝ | ŝ] = −σ²(ŝ−μ)/(τ²+σ²)는 Tweedie 수축이다. **새로운 것 없음.**

**명제 2 (크로스오버; 대수적으로는 자명하지만 이 문맥에서 지적된 적은 없음).**
- (m_i + v_i/2) − ŝ_i = σ_i²/(τ²+σ_i²) · (μ − ŝ_i + τ²/2).
- 따라서 잡음이 큰 토큰은 ŝ_i > μ + τ²/2이면 강등되고 그 아래면 승격된다.
- 예산 분위 z_k에서 경계는 대략 μ + τ z_k에 있다. 그래서 z_k > τ/2이면(작은 예산이거나 평평한 head) winner's curse 방향이 되고, z_k < τ/2이면(날카로운 head이거나 큰 예산) 반대 방향이 된다.
- 따름정리: 고정 보정 ŝ − σ²/2(2605.26266의 dense 보정을 랭커로 전용한 것)는 경계가 크로스오버 아래에 있을 때 부호가 틀린다. **새로운 관찰은 이 부분이다.** 수학적 깊이는 얕다.

**추측 1.**
- 꼬리가 두꺼운 logit 분포(sink나 heavy hitter)에서는 가우시안 EB가 상단 꼬리를 과수축해 naive보다 나빠질 수 있다.
- NPMLE 사전은 이를 피하며, 이때 크로스오버는 점 하나가 아니라 ŝ에 대한 함수 log E[e^s|ŝ] − ŝ의 영점 집합이 된다.
- 증명은 없고 합성 실험으로만 지지된다.

**추측 2.**
- RaBitQ나 Hadamard처럼 공유 회전을 쓰면 토큰 간 오차가 상관되어 독립 가정이 깨진다.
- 이 상관이 이득을 얼마나 줄이는지는 분석하지 않았다.

### 판정 실험 (3–5 GPU-days)

**설정.**
- 모델: Llama-3.1-8B, Qwen3-8B.
- 과제: RULER와 LongBench, 32–128K.
- 선택 키:
  - RaBitQ 1-bit와 sign sketch: σ_i는 1-bit 추정량 분산식에서 얻는다.
  - 2-bit per-token INT: σ_i² ≈ ‖q‖²Δ_i²/12.
  - 2-bit KIVI per-channel.
  - 음성 대조로 FP8과 INT8.
- 선택 방식: Quest식 page top-k와 token top-k, 예산 1/2/5/10%. 추가로 양자화 K 위 H2O 축출.

**비교 arm.**
- (a) ŝ
- (b) ŝ − σ²/2 (Jensen debias 랭커)
- (c) ŝ + σ²/2 (flat prior)
- (d) 가우시안 EB m + v/2
- (e) NPMLE EB log E[e^s|ŝ]
- (f) exact top-k

**측정.**
- 지표: 포착한 exact 질량, oracle recall, 과제 정확도.
- 층과 head별로 σ²/τ²와 "경계 − 크로스오버" 값을 기록하고, 이득과의 상관을 본다.

**성공 기준.** 다음을 모두 만족해야 한다.
- 1-bit, 1–5% 예산에서 (e)가 (a) 대비 gap 회복률 ≥0.3, 과제 정확도 +1점 이상.
- 크로스오버 아래 레짐(경계 < μ + τ²/2)의 head에서 (e)가 (b)보다 낫다.
- FP8과 INT8에서 이득이 0이다.

**실패.** 위 조건 중 하나라도 어기거나, 이득이 σ 분산과 무상관이면 실패다.

### 수치 확인

**스크립트.** `scripts/new_ideas/idea07_check.py`. numpy, seed 20261011, 실행 56초.
- N = 4096, σ_i는 lognormal(disp 0.5), 질의 150개(NPMLE는 30개).
- 표기: Δ는 naive 대비 질량 변화, g는 gap 회복률.

**C1 크로스오버.** 수치 계산과 닫힌형이 모든 점에서 일치한다. τ = 3일 때 ŝ = 4.5에서 보정이 +0.0000이 되고, 7.0에서는 σ² = 4이면 −0.7692다. **지지.**

**C2 음성 대조.** σ²/τ² = 0.001에서 모든 arm의 Δ는 |Δ| ≤ 0.001pp다. 등분산 대조(disp = 0)에서도 gEB − naive = 0.000이다. **지지.**

**C3 방향.** τ = 3에서 "경계 − (μ+τ²/2)"와 "naive 선택 평균 σ − gEB 선택 평균 σ"는 1%: +3.24 / +0.427, 10%: −0.31 / −0.023, 20%: −1.78 / −0.089. 8개 설정 모두 부호가 예측과 일치했다. **지지.**

**C4 크기.** 원래 예측보다 약하다.
- σ²/τ² = 0.05에서 gEB의 Δ는 최대 +0.139pp(τ = 3, 1%)이고 g는 ≤ 0.14다. "0.05 이상이면 non-negligible"이라는 원래 문턱은 **기각**한다.
- g ≥ 0.3을 넘는 설정은 다음 넷뿐이다. 모두 1% 예산이다. 10% 예산에서 g는 ≤ 0.16이다.
  - τ = 1, r = 0.2: g = 0.377
  - τ = 1, r = 1.0: g = 0.479
  - τ = 3, r = 1.0: g = 0.351 (Δ = +15.2pp)
  - heavy 사전, r = 1.0, npEB: g = 0.320
- gEB가 recall을 오히려 낮추는 경우도 있다(τ = 3, r = 1, 20%: 0.591 vs naive 0.614). 질량 최적과 recall 최적은 다르다.

**Jensen-debias 랭커와의 비교 (핵심 위협).**
- 작은 예산에서는 ŝ − σ²/2가 gEB 이득의 대부분을 이미 얻는다: τ = 3, r = 1, 1%에서 +14.69 vs +15.25pp; τ = 1, r = 1, 5%에서 +2.10 vs +2.51pp.
- 경계가 크로스오버 아래인 레짐에서는 (b)가 크게 해롭고 gEB는 손해가 없다: τ = 3, r = 1, 10%에서 −9.38 vs +0.08pp; τ = 3, r = 0.2, 10%에서 −1.54 vs −0.02pp.
- flat prior (c)는 일관되게 파국적이다(최대 −46.7pp).

**C5 꼬리가 두꺼운 경우.** 사전은 bulk N(0,1)에 2% spike N(5, 1.5²)를 섞었다.
- 가우시안 EB가 naive보다 나쁘다: r = 0.2, 1%에서 −2.10pp, r = 1, 1%에서 −2.21pp.
- NPMLE는 r = 0.2, 1%에서 −0.016pp(본전), r = 1, 1%에서 +3.10pp(g = 0.32)다.
- 가우시안 데이터에서는 NPMLE가 gEB와 거의 같다(+2.44 vs +2.47pp). **추측 1을 지지.**

**결론.**
- 크로스오버, 음성 대조, 비가우시안 사전의 필요성은 지지된다.
- "이득이 크다"는 주장은 σ²/τ² ≳ 0.2이면서 예산이 ≤5%인 경우로 좁혀야 한다.
- 크로스오버 위 레짐에서 EB가 출판된 Jensen 보정을 넘어서 더하는 몫은 0.4–0.6pp 수준이다.

### 평가

**신규성: 2/5 (낮음).**
- 순위 규칙은 2510.00636, σ²/2 보정은 2605.26266, EB 랭킹과 NPMLE는 1312.5776과 2012.12550에 이미 있다.
- 남는 기여는 세 가지다. (i) 크로스오버와, 그로부터 고정 Jensen 보정이 레짐에 따라 해롭다는 진단. (ii) 사후 저비트 프록시 선택에 NPMLE EB를 적용한 것. (iii) 1–2 bit에서 실제로 얻는 이득의 측정.

**메인 트랙 가능성: 낮음 (대략 10–15%) [S].**
- 이론은 두 줄 결합이라 이론 기여로는 통과하지 못한다.
- 합성 결과에서 큰 이득은 1% 예산과 σ²/τ² ≈ 1이라는 극단에 몰려 있다. 실제 1-bit RaBitQ의 σ²/τ²가 이 범위인지는 측정되지 않았다 [S].
- 10% 예산에서는 이득이 거의 없고, oracle 하한(약 13% 손실) 자체가 이득의 상한을 누른다.
- 현실적인 경로는 둘이다. 크로스오버 진단과 1-bit 엔드투엔드 +1점을 모두 보이면 workshop이나 short paper. 아니면 RaBitQCache나 Self-Indexing 계열 논문의 한 절(ablation)로 흡수되는 것이다.

**가장 큰 위험.**
1. 실제 모델의 σ²/τ²가 0.05 미만이면 이득이 0이다.
2. 공유 회전에서 오는 토큰 간 오차 상관을 아직 분석하지 않았다.
3. 실제 head들이 대부분 크로스오버 근처에 있으면 Jensen 랭커와 EB의 차이가 사라진다.
4. 질량 이득이 RULER 정확도로 이어지지 않을 수 있다. recall이 오히려 떨어지는 사례가 있다.

**선점할 수 있는 쪽.**
- Expected Attention 저자들(Devoto, Jeblick, Jegou): MGF 점수에 양자화 분산 항만 더하면 된다.
- Quantized Keys Steal Attention 저자들(Tuncer et al.): Jensen 보정을 top-k 선택으로 확장하기만 하면 된다.
- RaBitQCache나 Fathom 팀: 이미 분산 추정치를 가지고 있다.

셋 모두 한 번의 후속 작업으로 이 아이디어를 흡수할 수 있는 위치에 있다.
