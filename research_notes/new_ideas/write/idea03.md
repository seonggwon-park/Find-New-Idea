## 새 아이디어 3. Haar 회전 GPTQ와 WaterSIC 사이 최악 간극의 정확한 점근식과 damping 상한

### 가설

고율(high-rate), entropy-coded 모델에서, 스펙트럼이 λ_min/λ_max ≥ ε를 만족하는 경우 전체에 대해 Haar 회전 GPTQ와 WaterSIC의 rate 간극 sup은 다음과 같다.

**sup gap(ε) = ½log₂(1/ε) − log₂ln(1/ε) + ½log₂(2/e²) + o(1) bit**

- gap의 정의는 ½log₂(AM(D)/GM(λ))이고, D는 QΛQᵀ의 LDL pivot이다.
- sup은 two-point 스펙트럼 {ε, 1}에서 달성되며, 이때 ε 쪽 질량은 f* = 1 − 2/ln(1/ε)이다.
- 따름정리 1: PCA 기저(Specht ratio)와 비교한 회전 이득(rotation dividend)은 정확히 ½log₂ln(1/ε) + O(1) bit이다.
- 따름정리 2: damping δ가 있을 때 기준이 되는 값은 ε_eff = (λ_min+δ)/(λ_max+δ)이다.

반증 기준은 세 가지다.
- (i) n ≥ 400에서 exact Haar pivot으로 구한 two-point sup이 ε ∈ [1e-6, 1e-3] 구간에서 극한식과 0.1 bit 넘게 어긋난다.
- (ii) two-point sup보다 0.05 bit 넘게 큰 일반 스펙트럼이 발견된다.
- (iii) 실제 층에서 예측 gap과 실측 gap의 Spearman 상관이 0.7 미만이다. 이 경우에도 이론은 남지만 진단 도구로서의 가치는 없다.

원안에서 철회한 내용은 다음과 같다.
- log-log 항의 계수 ½를 1로 고친다.
- ε = 1e-3에서의 1.23 bit는 최대화가 덜 된 값이었다. 실제 값은 1.38이다.
- "massive channel이 gap을 키운다"는 주장은 철회한다(수치 확인 F 참조).

### 문헌상 공백과 선행 연구 위치

- [High-Rate Quantized Matrix Multiplication II 2605.13768](https://arxiv.org/abs/2605.13768) (O&P, v2 2026-06-08)
  - l.1594-1597에서 "worst possible gap (over possible spectra λj ≥ ϵ) between GPTQ with rotation and WaterSIC"를 열린 문제로 제시한다 [V].
  - Eq. 33(l.1551-1559)은 pivot을 e_k/e_{k−1}로 근사한다. 기댓값은 footnote 3에 따라 정확하지만 concentration은 증명되지 않았다. AM에서 HM으로 이어지는 ramp는 서술만 되어 있다 [V].
  - Llama-3-8B에서는 "within 0.1 bit"를 보고한다(l.32). singular Σ_X 층은 "we skip"이라고 하여 다루지 않는다(Fig. 5, l.1438) [V].
- [QuIP 2307.13304](https://arxiv.org/abs/2307.13304)
  - Lemma 2(l.303-309)는 tr(D) ≤ (μ²/n)(tr H^{1/2})²이다. Haar를 쓰면 μ = polylog(n)이다(l.412, l.468) [V].
  - 이를 합치면 gap ≤ ½log₂μ² + ½log₂(AM(√λ)²/GM(λ))가 바로 나온다. 둘째 항은 이미 올바른 rate를 가진다 [S, 본 계산].
  - 따라서 "unbounded"라는 결론 자체는 새롭지 않다. O&P도 이 Lemma를 인용한다(l.1484) [V].
- [CurveTQ 2610.09212](https://arxiv.org/abs/2610.09212)
  - Prop. 2(l.1320): native-weighted 방식과 rotated 방식 사이에 어느 한쪽이 항상 우월하다는 dominance가 없다 [V].
  - App. A.3: 2×2 예에서 0.10 bit [V].
  - ramp는 "checked numerically, not proved"라고만 되어 있다(l.1399) [V].
- [WaterSIC 2603.04956](https://arxiv.org/abs/2603.04956)
  - 회전이 없을 때 gap이 무한히 커질 수 있음을 보였다(AM-GM, l.619) [V].
  - 실험 설정은 δ = 1e-4에 dead-feature erasure를 더한 것이다(l.1033-1040) [V].
- [Hegland 2011.14626](https://arxiv.org/abs/2011.14626): volume-sampling 소거 오차가 elementary symmetric 다항식의 비가 된다는 결과로, 증명 도구 후보다 [V].
- [WaterKron 2609.14706](https://arxiv.org/abs/2609.14706): 최악 gap 분석이 없다 [V].
- [서베이 2608.25188](https://arxiv.org/abs/2608.25188): l.4248-4250에서 이 문제를 "open"이라고 다시 확인한다 [V].

**남은 공백.** 이번 검색(arXiv API 쿼리 약 30회, 웹 3회, 2026-08 이후 포함)에서는 다음 세 가지가 발견되지 않았다.
- 회전 gap sup의 정확한 상수
- μ-free 상한
- damping을 ε_eff로 연결하는 분석

### 수학 (증명 가능한 것과 추측)

1. **교과서적 결과**
   - Newton 부등식에 따라 r_k = E_k/E_{k−1}은 단조 감소하고, GM(r) = GM(λ)이다.
   - [ε,1]에서 AM/GM의 최대는 Specht ratio다. ½log₂S = ½log₂(1/ε) − ½log₂ln(1/ε) − ½log₂e + o(1)이다.
2. **새 결과, 증명 가능성 높음 (하한)**
   - Eq. 33의 n→∞ 극한에서 two-point 스펙트럼은 r(t) = (1−f−t)/(1−t) (t < 1−f)를 준다.
   - 따라서 AM(r) → (1−f) + f ln f + O(ε ln(1/ε))이고, 이는 x = 1−f일 때 약 x²/2이다.
   - 이를 최적화하면 x* = 2/ln(1/ε)이고 상수는 −0.943이다.
   - Haar에 대해 엄밀하게 하려면, two-point 스펙트럼이 ε·I + (1−ε)P 꼴이므로 사영(projection)의 free compression(Jacobi 법칙) 극한이나 log-det concentration을 쓰면 된다 [S].
3. **추측 A (μ-free 기댓값형 Lemma 2)**
   - 주장: (1/n)Σ_k E_k/E_{k−1} ≤ AM(√λ)².
   - 이것이 성립하면 QuIP의 ½log₂μ² slack이 사라지고, sup gap ≤ ½log₂(1/ε) − log₂ln(1/ε) + (1 − log₂e) + o(1)을 얻는다. rate는 확정되고, 상수 차이 ½ bit만 남는다.
   - n = 2이면 차이가 √(ab)(√(ab)/(a+b) − ½) ≤ 0이므로 성립한다 [S].
4. **추측 B (two-point 극값성)**
   - 극한 범함수를 Schur-concavity로 다루면 정확한 상수가 확정된다.
5. **따름정리**
   - dividend는 ½log₂ln(1/ε) + c이고, c → +0.221이다.
   - damping은 ε를 ε_eff로 바꾸는 것뿐이다. 다만 damping은 목적함수 자체를 바꾸므로, 이 상한은 damped H에 대한 gap이다.

### 판정 실험 (CPU + 약 2 GPU-day)

- **모델**: Llama-3-8B, Qwen3-8B의 모든 linear 층(224개).
- **Hessian**: Wikitext-2에서 m ∈ {32, 512} 시퀀스로 구한다. 여기에 token-starved 조건(128/512 토큰)과 O&P가 건너뛴 singular 층을 더한다.
- **비교군(arm)**: GPTQ+Haar/RHT, WaterSIC, PCA 기저 GPTQ를 같은 damped H(δ ∈ {1e-4, 1e-2, 1e-1}·mean) 위에서 비교한다. 지표는 같은 distortion에서의 entropy-coded rate다.
- **측정**
  - 실측 gap
  - Eq. 33 예측 gap: log 영역 DP로 O(n²)이므로 n = 14336도 CPU에서 계산된다.
  - ε_eff
- **합성 단계**: two-point 스펙트럼, n ∈ {400, 1000}, Haar 100회.
- **성공 기준**
  - Spearman ≥ 0.7
  - 예측과 실측의 차이 ≤ 0.05 bit
  - 1% damping에서 일부 층이 0.3 bit를 넘음
- **실패 기준**
  - δ ≥ 1e-2인 모든 층에서 gap < 0.3 bit이면 실무적 의미가 없다.
  - 반증 기준 (i)이나 (ii)가 나오면 가설을 기각한다.

### 수치 확인

`scripts/new_ideas/idea03_check.py` (numpy, seed 고정, 24초)

- **A. Eq. 33 대 Haar MC** (n = 200)
  - ε = 1e-1 / 1e-2 / 1e-3 / 1e-4 / 1e-6에서 Eq. 33은 0.045 / 0.466 / 1.381 / 2.568 / 5.289, Haar MC는 0.048 / 0.472 / 1.384 / 2.567 / 5.284다.
  - 차이는 0.006 bit 이내다.
- **B. two-point sup**
  - Eq. 33(n = 1000)은 1.375 / 2.566 / 5.277 / 8.175다(ε = 1e-3 / 1e-4 / 1e-6 / 1e-8). 선도항 극한 1.332 / 2.556 / 5.272 / 8.169와 0.04 bit 이내로 맞는다.
  - 잔차는 −0.862 / −0.885 / −0.906 / −0.915로, −0.943을 향해 느리게 수렴한다.
  - 1−f*는 0.308 / 0.227 / 0.149 / 0.111이고, 예측값 2/ln(1/ε)는 0.290 / 0.217 / 0.145 / 0.109다.
  - ε = 1e-2에서는 극한(0.309)과 n = 1000 값(0.464)이 어긋난다. 따라서 점근식은 ε ≤ 1e-4에서만 정량적으로 쓸 만하다.
- **C. dividend − ½log₂ln(1/ε)**
  - ε = 1e-3에서 1e-20까지 0.145 / 0.164 / 0.184 / 0.194 / 0.203 / 0.211로, 이론값 +0.221로 수렴한다.
- **D. QuIP(μ = 1) sup − 회전 sup**
  - 0.536 / 0.495 / 0.471 / 0.474 / 0.482 / 0.489이고, 극한은 0.500이다.
  - 무작위 스펙트럼 1500개에서 max log₂(AM(D_Eq33)/AM(√λ)²) = +0.0000이었다(등호는 평탄한 스펙트럼).
  - 소규모 Haar MC에서 max(gap − QuIP) = −0.016이었다.
  - 추측 A의 수치적 근거일 뿐, 증명은 아니다.
- **E. 국소 탐색** (n = 40, ε = 1e-4)
  - 탐색 최댓값은 2.588로 two-point 2.630보다 작다. 끝점은 모두 {ε, 1}에 모였다.
  - 추측 B에 대한 약한 지지다.
- **F. 합성 near-singular 공분산** (n = 1024, 1% damping)
  - massive channel이 없을 때 gap은 0.743(128 토큰, Haar MC 0.766) / 0.497(512) / 0.186(16384)이다.
  - massive channel이 있을 때(λ_max/mean ≈ 940)는 0.095 / 0.072 / 0.057이다. δ가 mean 기준이라 bulk에 비해 커지기 때문이다.
  - ε_eff가 1e-5에서 1e-4 사이일 때 최악 상한은 2.5-3.9 bit로, 실제 gap보다 3-40배 느슨하다.
- **판정**
  - 지지: 점근식, dividend, ½ bit 차이, token 부족 시 gap 증가.
  - 반박: massive channel이 gap을 키운다는 주장.
  - 확인된 한계: damping 상한은 상한일 뿐이고 실제 gap을 예측하지 못한다.

### 평가

- **신규성: 3/5 (보통).** 명시된 열린 문제에 답하는 연구다. 그러나 "unbounded"와 rate 자체는 QuIP Lemma 2와 Specht ratio에서 거의 바로 나온다. 실제로 남는 기여는 다음 세 가지다.
  - μ-free 상한(추측 A)
  - 정확한 상수 −0.943
  - log-log dividend
- **본 트랙 가능성: 낮다.** 좁은 이론 노트로 읽힐 가능성이 크다. 또 결과가 entropy-coded 고율 rate에 관한 것이라 고정 비트폭 커널에는 그대로 이어지지 않는다. ISIT/TMLR이나 더 큰 논문의 한 절이 현실적이다.
- **위험**
  - (1) 추측 A와 B의 증명이 막히면 정확한 상수는 수치 결과로만 남는다.
  - (2) F에서 보듯 실제 층에서는 damping과 massive channel 때문에 gap이 0.1 bit 수준일 수 있고, 그러면 실무적 동기가 약하다.
  - (3) Eq. 33의 concentration을 엄밀하게 다루려면 free probability 기술이 필요하다.
- **선점 위험**: 가장 높은 쪽은 문제를 낸 O&P 본인들이다(Eq. 33과 footnote 3를 이미 보유). 그다음은 CurveTQ와 WaterKron 저자들이다.
- **실행 판단**: 착수한다면 추측 A(일반 n)를 먼저 검증해야 한다. 이것이 가장 깔끔한 새 lemma이다.
