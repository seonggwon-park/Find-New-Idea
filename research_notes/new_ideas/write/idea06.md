## 새 아이디어 6. 오차 피드백을 반영한 WUSH: 주변(marginal) 블록 대신 Schur 보수(pivot) 블록으로 만든 AbsMax 최적 블록 변환

### 가설

이 가설은 kill test를 통과한 좁은 형태만 다룬다.

**(H1) weight-only.** 설정은 다음과 같다.
- per-group AbsMax 양자화 (INT4, MXFP4/NVFP4)
- group 크기 = block 크기인 GPTQ식 block 오차 피드백(block SIC)

이 설정에서 WUSH Algorithm 2가 변환을 만들 때 쓰는 입력 `MX(i) ← diagonal blocks of H`(주변 블록 H_gg)를 pivot 블록 D_g = (L_gg L_ggᵀ)^{-1} = H_gg − H_{g,>g} H_{>g,>g}^{-1} H_{>g,g}로 바꾼다(pivot-WUSH). 바꾸면 layer proxy tr(ΔWᵀHΔW)가 WUSH(+GPTQ)보다 낮아진다. 개선 폭은 층별 inter-group 상관, 즉 ‖H_gg − D_g‖/‖H_gg‖가 클수록 커진다.

**(H2) W4A4.** 단순 치환은 W4A4에서 오히려 나쁠 것으로 예상한다(아래 수치 확인 참조). 그래서 활성화 오차는 주변 Σ_gg로, 가중치 오차는 D_g로 가중하는 비대칭 목적함수를 따로 풀어야 한다.

**실패 기준.**
- (H1): Llama-3-8B와 Qwen3-8B에서 q/k/v, gate/up 층의 proxy 개선이 WUSH+GPTQ 대비 평균 2% 미만이다.
- 또는 개선량과 층별 gap 지표의 Spearman 상관이 0.3 미만이다.
- 또는 MXFP4/INT4 g32 weight-only에서 Wiki2 PPL 변화가 seed 간 편차 안에 들어간다.

셋 중 하나라도 해당하면 H1은 기각한다. (H2)에서 비대칭 해가 WUSH-marginal보다 낫지 않으면 W4A4 확장은 포기한다.

### 문헌상 공백과 선행 연구 위치

- [WUSH: Near-Optimal Adaptive Transforms for LLM Quantization 2512.00956](https://arxiv.org/abs/2512.00956) (v2 2026-05-30, ICML 2026). Algorithm 2 "GPTQ case"(p.3)는 다음과 같다. [V]
  - `MX(i) ← diagonal blocks of H`로 변환을 만든다.
  - `LL⊤ ← Cholesky(H^{-1})`로 이미 pivot (L_ii L_iiᵀ)^{-1} = D_i를 계산한다. 이를 intra-block GPTQ Hessian H(i) = T(L_ii L_iiᵀ)^{-1}Tᵀ로 쓴다.
  - MW(i)는 피드백으로 갱신된 가중치의 2차 모멘트이므로, 변환은 처리 순서와 이미 결합되어 있다.
  - 그러므로 제안은 **이미 공개된 알고리즘에서 이미 계산된 양으로 한 줄을 바꾸는 것**이다.
  - Theorem 4.1(최적 변환)의 증명은 확률적 unbiased 양자화 모델 위에서만 이루어진다(Eq. 12–14). 논문 스스로 이 모델을 "tractable surrogate"라 부른다.
  - 모든 GPTQ 결과는 W4A4다. 초록 기준 GPTQ 개선은 MXFP4에서 Hadamard 대비 +0.7점이다.
- [SchurQuant 2608.15567](https://arxiv.org/abs/2608.15567) (2026-08-16). 이 논문은 "group을 raw block G_cc로 평가하지만 suffix가 오차를 흡수할 수 있으니, Schur complement가 정확한 곡률"이라고 말한다(본문 69–70행). 같은 통찰을 scale/zero-point refit과 integer code 선택에 적용했다. 변환에는 적용하지 않았다. [V]
- [CurveTQ 2610.09212](https://arxiv.org/abs/2610.09212). Viterbi metric에 e⊤He = Σ_b η_bᵀ D_b η_b를 쓰며(284–285행), Eq. 2에서는 diag(D_b)만 사용한다. 회전/변환은 쓰지 않는다. [V]
- [Ordentlich–Polyanskiy 2605.13768](https://arxiv.org/abs/2605.13768). 이 논문의 결과는 세 가지다. [V]
  - GPTQ = SIC다(Lemma 1).
  - WaterSIC 격차는 pivot들의 AM/GM이다.
  - 회전은 pivot의 AM을 GM 쪽으로 당긴다.
  - 단, 스칼라·엔트로피 부호화 설정이며 AbsMax가 아니다.
- [변환 서베이 2608.25188](https://arxiv.org/abs/2608.25188). 다음 두 문장이 공백을 정의한다. [V]
  - Thm 4.9 서술(2095–2104행): WUSH는 "unbiased noise"를 가정하고 GPTQ로는 "advantage transfers empirically"라고만 한다.
  - §13 open problem 1(6949–6957행): "jointly optimal (T, Q) pair" 이론이 없다.
- [Chen et al. 2507.18553](https://arxiv.org/abs/2507.18553). GPTQ = Babai이며, basis reduction은 future work로 남겼다. [V]
- 함께 확인했으나 겹치지 않은 것: WUSH-KV 2609.38121, Cherniuk 2609.11687, JARQ 2609.38599, OJBKQ 2602.08376, HeRo-Q, MR-GPTQ(block Hadamard, 웹으로만 확인). [V/S]
- 인접 고전 분야: MIMO의 GMD/UCD precoder + SIC 순서(V-BLAST)는 "변환과 순서의 공동 선택"이라는 틀 자체의 선행 연구다. [S]

**남은 것.** 이번 검색(arXiv 쿼리 17회 + 웹 검색 2회, 2026-10-11 재검색 1회)에서 발견되지 않은 것은 다음 넷이다.
- (a) block SIC 하에서 AbsMax 왜곡이 Σ_g E‖δ_g‖²_{D_g}로 분해된다는 사실을 WUSH 변환 최적화와 연결한 명제
- (b) D_g로 만든 WUSH가 weight-only 확률 모델 안에서 최적이라는 증명
- (c) W4A4 비대칭 bilateral 최적해
- (d) 순서와 T의 공동 선택, 그리고 개선량과 inter-group 상관의 연결에 대한 실측

vLLM, torchao, llm-compressor에도 구현이 없다(웹 확인). [S]

### 수학 (증명 가능한 것과 추측)

**설정.** 기호는 다음과 같다.
- W ∈ R^{din×dout}, H = Σ, loss = tr(ΔWᵀΣΔW)
- group g는 연속한 d개 입력 채널이고, 처리 순서는 1..G다.
- group g를 변환 T_g 아래에서 양자화하면 원좌표 오차는 δ_g = T_gᵀ q(T_g^{-ᵀ} w_g) − w_g다.
- 피드백은 W_{>g} ← W_{>g} − Σ_{>g,>g}^{-1} Σ_{>g,g} δ_g다.

**Lemma A (교과서).** 최종 loss는 정확히 Σ_g tr(δ_gᵀ D_g δ_g)다.
- block LDL 분해의 대수적 항등식이며, 근사가 아니다.
- QuIP/LDLQ와 O–P Lemma 1의 블록판이다.
- 수치로도 상대오차 1e-16 수준에서 확인했다(아래). 새롭지 않다.

**Lemma B (block 합동 동변성, 교과서).** block-diagonal T에 대해 변환 좌표의 pivot은 T_g D_g T_gᵀ다. WUSH Alg. 2의 H(i)가 바로 이 양이다. [V]

**Proposition C (새로움, 증명은 짧음).** 가정은 다음과 같다.
1. WUSH의 확률 모델: unbiased AbsMax 잡음, 열들을 i.i.d. 표본으로 취급한다.
2. group 내부 피드백이 없다(block 단위 SIC).
3. 고해상도 근사: δ_g가 후속 group의 2차 모멘트 M_W(h>g)를 바꾸는 효과를 무시한다.

결론: 각 단계의 기대 왜곡은 WUSH Eq. 13의 가중치 쪽 항에서 X′X′ᵀ = M_X를 D_g로 바꾼 것과 같다. 따라서 Theorem 4.1을 그대로 적용하면, D_g의 Cholesky로 만든 WUSH가 FP에서는 정확히 최적이다. INT에서는 Theorem 4.1이 주는 오차 인수 범위(Gaussian/Laplacian에서 d·o(1), 일반 분포에서 d배) 안에서 최적이다.

이 명제의 기여는 최적성 주장의 범위를 RTN에서 "AbsMax + block 피드백"으로 넓히는 데까지다. 가정 3을 빼면 단계별(greedy) 최적일 뿐 전역 최적은 아니다.

**비대칭 W4A4 (미해결, 추측).** 목적함수는 다음과 같다.

E‖C_gᵀ T_W^{-1} ε(T_W w)‖²(가중치 항, C_g C_gᵀ = D_g) + E‖W′ᵀ T_X^{-1} ε(T_X x)‖²(활성화 항, x의 분포는 주변 Σ_gg)

두 항이 서로 다른 행렬(D_g와 Σ_gg)을 쓰므로, WUSH의 "두 항 각각의 최적해가 T_W = T_X^{-ᵀ}로 일치한다"는 성질이 깨진다. 단일 SVD 닫힌 형태는 성립하지 않는다.

추측: 최적해는 D_g와 Σ_gg 사이의 일반화 고유문제(generalized eigenproblem) 해에 Hadamard를 붙인 형태다. 증명은 없다.

**순서 (추측).** D_g는 순서에 의존한다. 다음 두 기준은 휴리스틱이며 최적성을 주장하지 않는다.
- det(D_g) 기준 greedy 순서
- 변환 후 group 내 pivot의 AM/GM 기준

### 판정 실험 (약 4 GPU-일, H100 1장 기준)

**모델.** Llama-3-8B, Qwen3-8B. 보조로 WUSH 원 논문 설정인 Llama-3.1-8B-Instruct를 쓴다.

**형식.**
- weight-only: MXFP4(g32), NVFP4(g16), INT4 g32
- W4A4: MXFP4

**Arms.**
- GPTQ + {none, block Hadamard(MR-GPTQ식), WUSH-marginal(원 Alg. 2), WUSH-pivot}
- 순서 ablation: 자연 순서 / act-order / det(D_g) greedy. 이로써 "순서 기여"와 "변환 기여"를 2×2로 분리한다.
- W4A4에서는 위 arm에 비대칭 해 후보를 추가한다.

**측정.**
- 층별 proxy tr(ΔWᵀHΔW)
- 층별 gap 지표 ‖H_gg − D_g‖_F/‖H_gg‖_F
- Wiki2/C4 PPL, lm-eval 평균
- seed 3개

**성공 기준.**
- q/k/v, gate/up에서 proxy가 WUSH 대비 5% 이상 낮다.
- 개선량과 gap 지표의 Spearman 상관이 0.5 이상이다.
- MXFP4 weight-only에서 WUSH→BF16 PPL 격차의 20% 이상을 회수한다.

**실패 기준.** 가설 절의 기준과 같다.
- proxy 개선이 2% 미만이다.
- 상관이 0.3 미만이다.
- PPL 변화가 노이즈 수준이다.

**사전 점검(1시간).** 먼저 실제 층의 gap 지표 분포를 잰다. 대부분 층에서 0.3 미만이면 아래 수치 확인 기준으로 기대 이득이 2% 미만이다. 이 경우 실험 전에 접는다.

### 수치 확인

**스크립트.** `scripts/new_ideas/idea06_check.py`, CPU 29초, seed 0–2.

**합성 설정.**
- din = 512 (32×16 group), dout = 256
- Σ = (1−ρ)·비등방 block-diagonal + ρ·공유 rank-8 인자
- 가중치는 t(4) 분포에 채널 스케일을 곱했다.
- 실측 LLM 값이 아니다.

**Lemma A 확인.**
- full loss = 9.868934e+02, pivot 합 = 9.868934e+02, 상대차 1.15e-16 (blk)
- gptq 변형: 7.710276e+02 = 7.710276e+02, 상대차 1.47e-16

**weight-only 결과.** 값은 tr(W⊤ΣW)로 정규화한 proxy이며, 마지막 값이 pivot-WUSH의 marginal-WUSH 대비 개선율이다.

| 형식 | 변형 | ρ | gap | Had | WUSH-marg | WUSH-piv | 개선 |
|---|---|---|---|---|---|---|---|
| INT4 | blk | 0.0 | 0.000 | 7.970e-3 | 5.590e-3 | 5.590e-3 | +0.00% |
| INT4 | blk | 0.3 | 0.476 | 5.696e-3 | 4.197e-3 | 4.011e-3 | +4.48% |
| INT4 | blk | 0.6 | 0.823 | 3.426e-3 | 2.761e-3 | 2.367e-3 | +14.23% |
| INT4 | blk | 0.9 | 0.976 | 1.143e-3 | 1.039e-3 | 6.996e-4 | +32.70% |

같은 ρ = 0.0/0.3/0.6/0.9에서 다른 변형의 개선율:
- INT4, Alg. 2식 intra-block GPTQ: +0.00 / +1.60 / +8.46 / +25.00%
- FP4 E2M1, blk: +0.00 / +3.91 / +16.10 / +35.94%
- FP4 E2M1, gptq: +0.00 / +1.55 / +10.80 / +28.51%

ρ = 0.9인 INT4 gptq에서는 WUSH-marginal(6.809e-4)이 Hadamard(6.553e-4)보다도 나빴다. 주변 블록으로 만든 변환이 피드백 아래에서 해가 될 수 있다는 뜻이다.

**W4A4 결과.** 가중치는 blk-SIC, 활성화는 RTN이며 같은 변환 쌍을 썼다. 값은 출력 상대 MSE 기준 pivot-WUSH의 marginal-WUSH 대비 개선율이다.

| 형식 | ρ = 0.0 | ρ = 0.6 | ρ = 0.9 |
|---|---|---|---|
| INT4 | +0.00% | −19.82% | −89.71% |
| FP4 | +0.00% | −15.65% | −68.56% |

**해석.**
- H1의 방향은 지지된다. 이득이 0에서 단조 증가하고 gap 지표와 함께 커진다.
- 실제 피드백 구조(intra-block GPTQ)에서는 이득이 약 3분의 1에서 절반가량 줄어든다.
- 단순 치환은 W4A4에서 크게 나빠진다. kill test의 지적("W4A4에서 verbatim 전이 불가")이 수치로 확인되었고, H2는 반드시 별도 해가 필요하다.
- 이득의 크기는 전적으로 합성 ρ에 달려 있다. 실제 LLM 층의 gap이 이 범위 어디에 있는지는 모른다. 결론적으로 원 가설의 "verbatim 전이"는 weight-only로만 약화해 유지한다.

### 평가

**신규성: 3/5 (보통).**
- weight-only 부분은 WUSH Alg. 2의 한 줄 수정이다. 그 줄에 들어갈 양(D_i)을 Alg. 2가 이미 계산하고 있다.
- SchurQuant가 같은 "주변 대신 Schur 곡률" 원리를 scale/code 선택에 이미 적용했다.
- 남은 신규성은 세 가지뿐이다.
  - Proposition C의 명시와 증명
  - W4A4 비대칭 최적해
  - 순서–변환 공동 선택

**메인트랙 가능성: 낮음.**
- weight-only 결과만으로는 리뷰어가 "incremental refinement"로 볼 것이 거의 확실하다. WUSH 자체의 GPTQ 이득이 Hadamard 대비 +0.7점에 불과해, 남은 개선 여지가 작다.
- 비대칭 W4A4 해가 닫힌 형태나 증명 가능한 bound로 나오고, 실측에서 WUSH를 넘어야 한다. 그러면 서베이 open problem 1(공동 최적 (T, Q))에 대한 부분 답이 되어 중간 정도 가능성이 생긴다.
- 현재 W4A4에 대해서는 추측만 있고, 합성 실험은 단순판이 실패함을 보였다.

**최대 위험.**
1. 실제 LLM 층에서 gap 지표가 작아, 이득이 2% 미만으로 사라질 수 있다.
2. Proposition C의 가정 2(group 내부 피드백 없음)는 WUSH Alg. 2의 실제 구조가 아니다. 그 구조에서는 정리가 성립하지 않고 경험적 이득만 남는다.
3. 4-bit에서는 고해상도·unbiased 가정이 약하다.
4. W4A4 비대칭 문제가 닫힌 형태를 갖지 않을 수 있다.

**선점 위험.** 다음 순서로 높다.
- WUSH 저자군(IST Austria, Alistarh 그룹): 코드에 D_i가 이미 있고 WUSH-KV로 후속 연구를 이어 가는 중이다. 가장 유력하다.
- SchurQuant 저자들
- O–P(pivot과 회전의 관계를 이미 분석)
- 이 셋 모두 몇 주 안에 같은 수정을 시도할 수 있는 위치다.
