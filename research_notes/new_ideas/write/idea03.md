## 새 아이디어 3. Haar 회전 GPTQ와 WaterSIC 사이 최악 간극의 정확한 점근식과 damping 상한

### 가설

고율(high-rate)·entropy-coded 모델에서, 입력 공분산 스펙트럼이 λ_min/λ_max ≥ ε 인 모든 경우에 대해 Haar 회전 GPTQ와 WaterSIC의 rate 간극의 상한(sup)은 다음과 같다.

  sup gap(ε) = ½log₂(1/ε) − log₂ln(1/ε) + ½log₂(2/e²) + o(1) bit

- 여기서 gap = ½log₂(AM(D)/GM(λ))이고, D는 QΛQᵀ의 LDL pivot이다.
- 최댓값은 two-point 스펙트럼 {ε, 1}이 달성하며, ε 쪽 질량 비율은 f* = 1 − 2/ln(1/ε)이다.
- 따름정리 1 (rotation dividend): PCA 기저(Specht ratio)의 최악 간극과 비교하면, 회전이 줄여 주는 양은 정확히 ½log₂ln(1/ε) + O(1) bit이고 그보다 크지 않다.
- 따름정리 2 (damping): damping δ가 있을 때 결정 변수는 ε_eff = (λ_min+δ)/(λ_max+δ)이다. 원안의 δ/mean(λ)는 틀린 정규화였다.

**반증 기준**
- (i) n ≥ 400에서 exact Haar pivot으로 계산한 two-point sup이 ε ∈ [1e-6, 1e-3] 구간에서 위 극한식(유한 ε 보정 포함)과 0.1 bit 넘게 어긋나는 경우.
- (ii) 일반 스펙트럼 탐색에서 two-point sup보다 0.05 bit 넘게 큰 스펙트럼이 나오는 경우.
- (iii) 실제 LLM 층에서 Eq. 33 예측 gap과 실측 gap의 Spearman 상관이 0.7 미만인 경우. 이때 이론은 남지만 진단 도구로서의 가치는 사라진다.

원안에서 철회하거나 수정한 부분은 다음과 같다.
- "−½log₂log(1/ε)" 항은 틀렸다. 정확한 계수는 1이다.
- ε = 1e-3에서의 1.23 bit는 최대화가 덜 된 값이다. 실제 값은 1.38이다.
- "massive channel이 있으면 gap이 커진다"는 합성 실험에서 반대로 나왔다(수치 확인 F 참조).

### 문헌상 공백과 선행 연구 위치

- [High-Rate Quantized Matrix Multiplication II 2605.13768](https://arxiv.org/abs/2605.13768) (Ordentlich & Polyanskiy, v2 2026-06-08)
  - l.1594-1597에서 "worst possible gap (over possible spectra λj ≥ ϵ) between GPTQ with rotation and WaterSIC"를 열린 문제로 명시한다 [V].
  - Eq. 33(l.1551-1559)은 pivot을 elementary symmetric 평균의 비 e_k/e_{k−1}로 근사한다. 기댓값 자체는 footnote 3의 Cauchy-Binet로 정확하지만, 비율의 concentration은 증명되지 않았다 [V].
  - pivot이 AM에서 HM으로 매끄럽게 감소한다는 서술이 있다 [V].
  - Llama-3-8B에서 측정한 간극은 "within 0.1 bit"이다(l.32) [V].
  - singular Σ_X 층은 "we skip"이라고 해서 측정에서 제외했다(Fig. 5, l.1438) [V].
  - l.1484에서 QuIP Lemma 2를 인용한다 [V].
- [QuIP 2307.13304](https://arxiv.org/abs/2307.13304)
  - Lemma 2(l.303-309)는 μ-incoherent H에 대해 tr(D) ≤ (μ²/n)(tr H^{1/2})²를 보인다 [V].
  - Haar 켤레(conjugation)는 μ = polylog(n)을 준다(l.412, l.468) [V].
  - 이 둘을 결합하면 상한 gap ≤ ½log₂μ² + ½log₂(AM(√λ)²/GM(λ))가 바로 나온다. 둘째 항의 [ε,1] 위 sup은 이미 올바른 rate ½log₂(1/ε) − log₂ln(1/ε)을 가진다 [S, 본 계산]. 따라서 "unbounded"라는 결론 자체는 새롭지 않다.
- [CurveTQ 2610.09212](https://arxiv.org/abs/2610.09212) (2026-10-06)
  - Prop. 2(l.1320): native-weighted 방식과 rotated-Euclidean 방식 중 어느 쪽도 다른 쪽을 지배하지 않는다 [V].
  - App. A.3: 2×2 예에서 0.10 bit [V].
  - AM-to-HM ramp는 "checked numerically, not proved"라고 적혀 있다(l.1399) [V].
- [WaterSIC 2603.04956](https://arxiv.org/abs/2603.04956)
  - 회전 없는 GPTQ의 간극이 무한함을 AM-GM 항으로 보인다(l.619) [V].
  - WaterSIC는 회전 불변이다.
  - 실험에서는 dead-feature erasure와 δ = 1e-4를 쓴다(l.1033-1040). 즉 기준이 되는 공분산도 정규화되어 있다 [V].
- [Hegland 2011.14626](https://arxiv.org/abs/2011.14626): volume-sampling 소거 오차의 기댓값이 elementary symmetric 다항식의 비라는 결과다. 증명 도구 후보이다 [V].
- [WaterKron 2609.14706](https://arxiv.org/abs/2609.14706), [변환 서베이 2608.25188](https://arxiv.org/abs/2608.25188)
  - 둘 다 최악 간극 분석이 없다.
  - 서베이 l.4248-4250은 이 문제가 "open"이라고 재확인한다 [V].

**남은 공백.** 이번 검색(arXiv API 쿼리 약 30회, 웹 검색 3회. 2026-08 이후 쿼리 포함)에서는 다음 세 가지가 발견되지 않았다.
- 회전 GPTQ 간극의 정확한 sup 상수
- 그 상수를 μ와 무관하게(μ-free) 만드는 상한
- damping을 ε_eff로 연결하는 분석

### 수학 (증명 가능한 것과 추측)

1. **교과서적 결과**
   - Newton 부등식에 의해 Eq. 33 pivot r_k = E_k/E_{k−1}은 단조 감소한다. 또 GM(r) = GM(λ)이다.
   - [ε,1] 위에서 AM/GM의 최댓값은 Specht ratio이고, ½log₂S = ½log₂(1/ε) − ½log₂ln(1/ε) − ½log₂e + o(1)이다.
2. **새 결과, 증명 가능성 높음 (하한)**
   - Eq. 33의 n→∞ 극한에서 two-point 스펙트럼의 pivot은 t < 1−f 구간에서 r(t) = (1−f−t)/(1−t)이다. 따라서 AM(r) → (1−f) + f ln f + O(ε ln(1/ε))이다.
   - x = 1−f로 두면 AM ≈ x²/2이다. 이를 최적화하면 x* = 2/ln(1/ε)이고, 가설의 상수 ½log₂(2/e²) ≈ −0.943이 나온다.
   - 이 결과를 Haar pivot에 대해 엄밀하게 만들려면 두 가지 경로가 있다.
     - (a) 사영(projection)의 free compression(Jacobi/Wachter 법칙) 극한을 이용하는 경로.
     - (b) Haar compression의 log-determinant concentration을 이용하는 경로.
   - two-point 스펙트럼은 ε·I + (1−ε)P 꼴이라서 (a)가 비교적 직접 적용된다 [S].
3. **추측 A (μ-free 기댓값 수준의 Lemma 2)**
   - 추측: 모든 λ > 0에 대해 (1/n)Σ_k E_k/E_{k−1} ≤ (AM(√λ))²이다.
   - 이것이 증명되면 QuIP의 ½log₂μ² 여유항이 사라지고, sup gap ≤ ½log₂(1/ε) − log₂ln(1/ε) + (1 − log₂e) + o(1)이 된다.
   - 즉 rate(log-log 계수 1 포함)는 확정되고, 상수만 ½ bit 차이로 남는다.
   - n = 2에서는 직접 계산으로 성립한다. 이 경우 차이는 √(ab)(√(ab)/(a+b) − ½) ≤ 0이다 [S].
4. **추측 B (two-point 극값성)**: Eq. 33 극한 범함수가 [ε,1] 위에서 two-point 스펙트럼으로 최대화된다. Schur-concavity를 이용한 경로를 예상한다. 이것이 맞으면 정확한 상수가 확정된다.
5. **따름정리 (자명)**
   - 위 결과와 Specht ratio를 합치면 rotation dividend는 ½log₂ln(1/ε) + c이고, c → −½log₂e − ½log₂(2/e²) = +0.221이다.
   - damping 상한은 ε를 ε_eff로 치환하면 바로 나온다. 다만 damping은 목적함수 자체를 바꾸므로, 상한은 "damped H 기준 간극"에 대한 것이다.

### 판정 실험 (CPU + 약 2 GPU-day)

- **대상**
  - Llama-3-8B와 Qwen3-8B의 모든 linear 층(224개).
  - Hessian은 Wikitext-2 기준 m ∈ {32, 512} 시퀀스로 수집한다.
  - 추가로 token-starved 조건(128 / 512 토큰)과, O&P가 건너뛴 singular 층을 포함한다.
- **비교군(arm)**: 모두 같은 damped H 위에서, 같은 distortion tr(ΔW Σ ΔWᵀ) 조건으로 entropy-coded rate를 비교한다.
  - GPTQ + Haar 회전 / RHT
  - WaterSIC
  - PCA 기저 GPTQ
  - damping δ ∈ {1e-4, 1e-2, 1e-1}·mean(λ)
- **측정 항목**
  - 실측 gap
  - Eq. 33 예측 gap. 로그 영역 DP로 계산하며, n = 14336에서도 O(n²)로 CPU에서 가능하다.
  - ε_eff
- **합성 단계**: two-point 스펙트럼에 대해 n ∈ {400, 1000}, Haar 100회로 exact pivot을 구해 sup을 확인한다.
- **성공 기준**
  - 층별 Spearman ≥ 0.7.
  - Eq. 33과 실측의 차이 ≤ 0.05 bit.
  - 1% damping에서 일부 층(token-starved 층)이 0.3 bit를 넘는다.
- **실패 기준**
  - δ ≥ 1e-2인 모든 층·조건에서 gap이 0.3 bit 미만이면, 실무적 의미가 없는 것으로 판정한다. 정리 자체는 유지된다.
  - 합성 단계에서 반증 기준 (i) 또는 (ii)에 해당하면 가설을 기각한다.

### 수치 확인

스크립트: `scripts/new_ideas/idea03_check.py` (numpy, seed 고정, 24초)

- **A. Eq. 33과 Haar MC의 일치 (n = 200, 30회)**: 아래 다섯 경우 모두 오차 0.006 bit 이내였다.

  | ε | Eq. 33 | Haar MC |
  |---|---|---|
  | 1e-1 | 0.045 | 0.048 |
  | 1e-2 | 0.466 | 0.472 |
  | 1e-3 | 1.381 | 1.384 |
  | 1e-4 | 2.568 | 2.567 |
  | 1e-6 | 5.289 | 5.284 |

- **B. sup gap과 점근식의 비교**
  - Eq. 33(n = 1000)의 sup은 1.375 / 2.566 / 5.277 / 8.175 bit이다(ε = 1e-3 / 1e-4 / 1e-6 / 1e-8).
  - 이 값들은 선도항 극한 1.332 / 2.556 / 5.272 / 8.169와 0.04 bit 이내로 일치한다.
  - 잔차 [극한 − (½log₂(1/ε) − log₂ln(1/ε))]는 −0.862 / −0.885 / −0.906 / −0.915로, 상수 −0.943을 향해 느리게 수렴한다.
  - 1−f*는 0.308 / 0.227 / 0.149 / 0.111이고, 예측값 2/ln(1/ε)는 0.290 / 0.217 / 0.145 / 0.109이다.
  - 주의: ε = 1e-2에서는 O(ε ln(1/ε)) 보정이 커서 선도항 극한(0.309)과 n = 1000 값(0.464)이 어긋난다. 따라서 점근식은 ε ≤ 1e-4에서만 정량적으로 쓸 만하다.
- **C. rotation dividend**
  - dividend − ½log₂ln(1/ε)은 0.145 / 0.164 / 0.184 / 0.194 / 0.203 / 0.211이다(ε = 1e-3에서 1e-20까지).
  - 이론값 +0.221로 수렴하므로 따름정리를 지지한다.
- **D. QuIP 범함수(μ = 1)와 비교한 sup 차이**
  - ε별 값은 0.536 / 0.495 / 0.471 / 0.474 / 0.482 / 0.489이고, 해석적 극한은 0.500이다. "약 ½ bit 아래"라는 주장을 지지한다.
  - 무작위 스펙트럼 1500개에서 max log₂(AM(D_Eq33)/AM(√λ)²) = +0.0000이었다. 등호는 평탄한 스펙트럼에서만 성립한다.
  - 소규모 Haar MC에서 max(gap − QuIP) = −0.016 bit였다.
  - 이 결과는 추측 A의 수치적 근거이지 증명이 아니다.
- **E. 극값성 탐색 (n = 40, ε = 1e-4)**
  - 국소 탐색으로 찾은 최댓값은 2.588로, two-point 최댓값 2.630을 넘지 못했다.
  - 탐색 끝점에서 값은 전부 {ε, 1}에 몰렸다. 약한 지지 근거이다.
- **F. 합성 near-singular 공분산 (n = 1024, 1% damping)**
  - massive channel이 없는 경우 gap은 0.743(m = 128 토큰, Haar MC 0.766) / 0.497(m = 512) / 0.186(m = 16384)이다.
  - massive channel(λ_max/mean ≈ 940)이 있는 경우 gap은 0.095 / 0.072 / 0.057이다. mean으로 정규화한 δ가 bulk 대비 커지기 때문이다.
  - ε_eff는 1.1e-5에서 1e-4이고, 이로부터 나오는 최악 상한은 2.5-3.9 bit이다. 실제 gap보다 3-40배 느슨하다.
- **판정**
  - 점근식, dividend, ½ bit 차이는 모두 지지된다.
  - "token 부족 시 수십 분의 1 bit 이상"이라는 주장은 지지된다.
  - "massive channel이 gap을 키운다"는 주장은 반박된다.
  - "damping 상한이 실제 gap을 예측한다"는 주장은 지지되지 않는다. 상한일 뿐이다.

### 평가

- **신규성: 3/5 (보통).** 명시된 열린 문제에 답하는 것은 분명하다. 그러나 "unbounded"라는 부분과 rate는 QuIP Lemma 2와 Specht ratio로 거의 즉시 나온다. 남는 기여는 다음 세 가지다.
  - μ-free 상한(추측 A)
  - 정확한 상수 −0.943
  - log-log dividend
- **본 트랙 가능성: 낮다.** NeurIPS/ICLR 리뷰어는 이를 좁은 이론 노트로 볼 가능성이 크다. 결과가 entropy-coded 고율 rate에 관한 것이어서 고정 비트폭 커널에는 그대로 이어지지 않는다. ISIT나 TMLR, 또는 더 큰 논문의 한 절로 들어가는 편이 현실적이다.
- **가장 큰 위험**
  - (1) 추측 A와 B의 증명이 막힐 수 있다. 특히 Schur-concavity 경로가 막히면 상수는 수치 결과로만 남는다.
  - (2) 합성 실험 F가 보여주듯, 실제 LLM 층에서는 damping과 massive channel 때문에 gap이 0.1 bit 수준에 머물러 실무적 동기가 약할 수 있다.
  - (3) Eq. 33의 concentration을 엄밀히 증명하는 데 free probability 기술이 필요하다.
- **선점(scoop) 위험**
  - 가장 높은 쪽: 문제를 낸 Ordentlich & Polyanskiy 본인들이다. Eq. 33과 footnote 3을 이미 갖고 있다.
  - 그다음: CurveTQ 저자들(ramp 성질을 수치로 확인함)과 WaterKron 저자.
- **실행 판단**: 착수한다면 정리 증명보다 추측 A를 먼저 확인해야 한다. 이것이 n = 2를 넘어 일반적으로 증명되면, 그 자체가 가장 깔끔한 새 lemma이다.
