## 아이디어 3. 블록 스케일 마이크로포맷 왜곡 이론의 확장: PoT 스케일 라운딩 계열과 편향(gain)

### 가설
Gaussian 블록의 최대값 밀도 × E2M1 구간 적분으로 기대 MSE를 계산하는 틀은 이미 있다(Fasoli). 남은 가설은 OCP v1 스케일 규칙의 학습 문제가 MSE에서 오지 않는다는 것이다. OCP와 올림의 Gaussian MSE는 사실상 같다. 원인은 최대 원소의 체계적 축소(편향)의 승법 누적이다. 이 틀을 PoT 라운딩 계열과 gain으로 확장하면 Mishra 레시피, UFP4 shrinkage, Li의 미유도 교차항을 한 틀에서 설명한다.

### 문헌상 공백
- **해석 이론은 이미 있다**("전부 경험적"은 틀렸다).
  - Fasoli et al.: 정확한 블록 최대값 밀도 eq.(5)에 E2M1 Voronoi 구간 MSE eq.(2)–(4)를 결합한다. χ²≈2·10⁻⁹이고 UE4M3 스케일에서 4·10⁻⁸(App. F)이다. UE4M3 subnormal에 의한 블록 크기 역전을 보이고 UE5M3를 제안한다 ([Is Finer Better? 2601.19026, ICLR 2026](https://arxiv.org/abs/2601.19026) [V]).
  - Soloveychik et al.: Gumbel(Lemma 2), PoT 올림 BFP 상한(Prop. 2), 4비트 최적 n=64를 준다. 정수 가수만 다룬다 ([2210.05470](https://arxiv.org/abs/2210.05470) [V]).
  - Li/Wu/Shen: UE8M0 올림 γ=2^U 모델(eq.9), E[γ]=1/ln2, RMSE 0.57. cos(e_scale,e_grid)≈−0.66은 "we do not derive in closed form"이라고 명시한다(Remark 3.3) ([2605.20402](https://arxiv.org/abs/2605.20402) [V]).
  - Chen et al.: Theorem 1–2가 UE8M0(ρ=1.5 고정)과 E4M3 스케일의 QSNR 닫힌 형태다 ([INT v.s. FP 2510.25602](https://arxiv.org/abs/2510.25602) [V]).
  - MXAttention Theorem 1: E8M0 최적 경계 7.25는 log2 M의 분포와 무관하다. Table 1은 OCP=e_floor(8)로 공식화한다 ([2607.24377](https://arxiv.org/abs/2607.24377) [V]).
  - Egiazarian et al.도 Sec. 3.1에 해석적 MSE 경계(Lemma 1–2)를 둔다. 회전 후 kurtosis는 0.02–0.05다 ([2509.23202, ICLR 2026](https://arxiv.org/abs/2509.23202) [V]).
- **경험적 보고(정정).**
  - Meta(ICML 2026, PTQ): OAS+MBS로 MXFP4를 61.25에서 66.50으로 올린다. NVFP4는 67.02다 ([2603.08713](https://arxiv.org/abs/2603.08713) [V]).
  - NVIDIA의 "MXFP4 2.5% vs NVFP4 1.5%"는 8B/1T에서 BF16 대비 **학습 손실** 상대오차다(Fig. 6a). 텐서 오차가 아니다 ([2509.25149](https://arxiv.org/abs/2509.25149) [V]).
  - Mishra et al.: OCP v1(⌊log2 amax⌋−emax, 사실상 내림)은 843M(Fig. 5)에서 "training issues"로 조기 중단됐다(v1 초록은 "divergence"). ceil+E4M3는 8B/15T에서 val-ppl 차이 <0.50%다(fake-quant). MXFP8 결과이며, 원인은 저자가 "We hypothesize"라고 쓴다 ([2506.08027](https://arxiv.org/abs/2506.08027) [V]).
- **포맷 제안(정정).**
  - UE5M3는 Hu et al. [2509.17791](https://arxiv.org/abs/2509.17791) [V]가 도입했고, Fasoli가 MSE 분석으로 제안했다. [2609.02846](https://arxiv.org/abs/2609.02846) [V]는 단일 seed 에뮬레이션 후속 실증이다.
  - Four-over-Six는 E4M3 스케일을 유지하는 블록 스케일 **선택** 알고리즘이다. 사전학습에서는 MAE로 선택한다 ([2512.02010](https://arxiv.org/abs/2512.02010) [V]).
  - HiFloat4 포맷은 [2602.11287](https://arxiv.org/abs/2602.11287) [S]가 제안했고, [2604.08826](https://arxiv.org/abs/2604.08826) [V]는 사전학습 연구다.
- **UFP4.** 국소 균일 밀도의 bin 수준 닫힌 형태 Eq.(2) (2q_i−q_{i−1}−q_{i+1})/4(q=2에서 −0.125)와 승법 누적 Eq.(3)–(5)가 있다. 분포 수준 식, 블록·스케일 의존, α 측정은 없다. RHT 증폭의 근거는 ΔSQNR(MSE형)뿐이다 ([2606.20381](https://arxiv.org/abs/2606.20381) [V]).
- Sun et al.(ICML 2025)은 (N, D, E, M, B)에 대한 경험적 적합식이다 ([2501.02423](https://arxiv.org/abs/2501.02423) [V]).
- **남는 공백(좁다).**
  - (i) OCP, ceil, UOS, E4M3, UE5M3를 **편향(gain)·최대 원소 편향**까지 비교한 결과가 없다.
  - (ii) cos≈−0.66 유도가 없다.
  - (iii) 밀도 기울기 shrinkage의 분포 수준 크기와 깊이 판정식이 없다.
  - (iv) Gumbel 점근식은 편의 이상이 아니다.

### 수학
블록 iid N(0,σ²), M=max|x_i|. 정확 밀도는 Fasoli eq.(5)를 쓰고, 점근은 E[M]≈σ√(2 ln 2B)다. 원소는 E2M1 {0,.5,1,1.5,2,3,4,6}에서 RTN하고 6에서 포화한다.

- **E4M3 스케일**: s_q/s∈1±2⁻⁴다. UE4M3 subnormal에서는 블록 크기 역전이 생긴다 ([2605.08565](https://arxiv.org/abs/2605.08565) [V]). "32 vs 16 → E[M] 비 1.10" 단조성은 범위 항을 넣어야 성립한다.
- **UE8M0 올림(ceil(log2(M/6)))**: γ=2^U에서 균일 근사 E[γ²]=E[4^U]=3/(2 ln 2)≈2.16이다. Li 모델의 2차 모멘트이며, E2M1 비균일성 때문에 실제 비는 이보다 작다.
- **OCP v1(⌊log2 M⌋−2, 정규화 최대 [4,8))** ([2310.10537](https://arxiv.org/abs/2310.10537) [V]): 올림과 다른 곳은 클리핑 대역 M/X∈(6,8)뿐이다. 해당 블록 비율은 Gaussian B=32에서 29.6%, MXFP8에서 16.9%다.
- **MSE/σ²**(2×10⁵ 블록): NVFP4형 0.0092, 올림 0.0133, OCP 0.0132, UOS 0.0125, 정확 B=16 0.0089. 이전 판의 "내림 0.0414"는 항상 클리핑하는 순수 floor(log2(M/6)) 규칙의 값이며 OCP가 아니다. "MSE 3.1배"와 "RMS 비 2.12"는 철회한다.
- **차이는 편향에 있다**: MXFP8에서 OCP/올림의 상대 MSE 비는 1.22배이지만, 최대 원소 편향은 −1.28% vs −0.08%이고 gain은 0.9972 vs 0.9993이다. MXFP4 B=32의 gain은 올림 0.990, OCP 0.980, UOS 0.988, NVFP4 0.996이다.
- **Shrinkage 분해**: UFP4 구간 편향은 전역 균일 밀도에서 합이 0이다(0.25·(−0.125)+0.75·(−0.125)+1.5·(−0.25)+1·(+0.5)=0). 순 편향은 밀도 기울기, deadzone, 클리핑에서 나온다. Gaussian B=16의 구간별 오차는 q=2 −0.112, q=4 −0.164, q=6 +0.545다. 순 1−α≈0.0033은 거의 상쇄된 잔차이고 부록 C의 −0.66%와 같은 계산이다(판정 재계산은 −0.70%, α 0.9959 [S]).
- **판정식**: 중점 경계는 MSE 최적이므로 편향은 그 대가다. (L·bias)²가 추가 분산보다 크면 비편향 라운딩이 유리하다. RMSNorm이 균일 감쇠를 상쇄하는지는 미확인이다 [S].

증명 가능한 것: e_floor/e_ceil 계열의 MSE·gain·최대 원소 편향을 1차원 적분으로 정확히 계산하는 것. H2: cos≈−0.66은 올림 조건 γ≥1에서 오는 B 무관 상수로 유도된다. 추측: H1(편향 누적이 불안정의 원인), RHT 후 Gaussian성. OffQ는 Hadamard가 고노름 토큰을 토큰별 상수 오프셋으로 바꿀 뿐임을 보인다(PTQ) ([OffQ 2606.07116](https://arxiv.org/abs/2606.07116) [V]).

### 판정 실험(CPU 수치 + 수일)
1. Fasoli 공개 코드로 즉시 재현한다. 그 위에 OCP/ceil/UOS/E4M3/UE5M3 × B × Gaussian·Laplace·t3의 MSE, gain, 최대 원소 편향, 클리핑 비율, cos(e_scale,e_grid)를 계산한다.
2. Llama-3.1-8B·Qwen3-8B의 RHT 후 블록에서 같은 표를 만든다. 기준점은 Egiazarian Fig. 3과 Fasoli Fig. 2이며, NVIDIA 손실 비율과는 비교하지 않는다.
3. 300M MXFP8/MXFP4 사전학습을 4개 arm으로 돌린다: (a) 올림, (b) 실제 OCP, (c) OCP + 블록별 gain 보정(MSE 유지, 편향 제거), (d) UOS. 클리핑 비율을 기록한다.
실패 기준: (c)도 불안정하면 H1을 기각한다. 실 데이터의 gain 순위가 해석식과 다르면 Gaussian 틀은 무용하다.

### 평가
novelty 낮음~중하. 핵심인 닫힌 형태 기대 왜곡은 Fasoli(2601.19026)가 출판했다. Soloveychik, Chen, Li, MXAttention이 나머지를 나눠 선점했다. 남는 기여는 편향 축의 확장이다. 메인트랙 가치는 실험 3이 H1을 확인하는지에 달렸다. 확인되지 않으면 부록 A로 내리는 것이 맞다.

---
