## 아이디어 8. 추론 모델 PTQ 캘리브레이션의 도메인 효과 분해: 엔트로피 프로필 대 Hessian 부분공간 정렬

### 가설
generic에서 자가생성 CoT로 캘리브레이션을 바꿀 때 GPTQ가 얻는 큰 이득에는 두 후보 원인이 있다. (i) 활성 Hessian 부분공간 정렬과 (ii) 토큰 엔트로피 프로필이다. 등방 로짓 섭동에서 토큰 end KL은 (σ²/2)(1 − ‖p_t‖²)이다. (ii)가 맞다면 forward 한 번으로 얻는 w_t = 1 − ‖p_t‖₂²를 Hessian에 넣은 H_w = Σ_t w_t x_tx_tᵀ만으로 generic 데이터에서도 이득의 상당 부분을 재현하고 CoT 길이 증가를 줄일 수 있어야 한다. 이 가중은 GuidedQuant 토큰 가중을 backward 없이, 채널 등방으로 근사한 것이다. 초안의 전제 "C4는 저엔트로피, 추론 트레이스는 고엔트로피"는 실측과 반대일 수 있어 0단계에서 먼저 잰다.

### 문헌상 공백
- **도메인 효과**
  - GPTQ에서만 나타난다. R1-Distill-Qwen-1.5B를 WikiText2 대신 NuminaMath로 캘리브레이션하면 평균 +9.8%다 ([2504.04823, COLM 2025](https://arxiv.org/abs/2504.04823) [V]).
  - Qwen3-4B-Thinking INT3 GPTQ g128을 자가생성 OpenR1-Math로 캘리브레이션하면 정확도 28%→65%, CoT 30k→12k다(Sec. 7.2, Fig. 5). 다만 저자들은 "캘리브레이션 최적화는 두 축 모두 QAT보다 엄격히 아래"라고 결론짓는다 ([2606.25519](https://arxiv.org/abs/2606.25519) [V]).
- **경쟁 가설과 반대 증거**
  - LAQuant는 도메인 효과를 부분공간 정렬로 설명한다(Sec. 4.1 Eq. (2), principal-angle cosine c̄_k).
  - 같은 논문의 측정이 이 아이디어에 불리하다. E2E 메트릭은 30배 이상 비등방이고 "no scalar reweighting can repair it"이다(App. C Table 6). softmax-Fisher 감독은 top-1 일치율을 오히려 떨어뜨린다(88.44% vs 93.95%). AIME25 트레이스의 약 56%는 H < 0.1이다 ([LAQuant 2605.08755](https://arxiv.org/abs/2605.08755) [V]).
  - 다른 측정에서도 Qwen3 CoT의 50.64%가 H < 10⁻²였다 ([2506.01939, NeurIPS 2025](https://arxiv.org/abs/2506.01939) [V]).
- **토큰 가중 Hessian은 선점됐다**
  - 초안은 GuidedQuant가 "토큰 믹스를 바꾸지 않는다"고 했으나 틀렸다. GuidedQuant의 H_k = Xᵀ Diag(mean_{j∈J_k}(∂ℓ/∂z_j)²) X(Eq. (4), Sec. 3.2)는 토큰 × 출력 채널 그룹을 가중하는 block-diagonal 근사다. 비용은 gradient caching 7B 0.3h(1×A100), 70B 2.7h(6×A100, Table 9)이고, 여기에 Hessian caching 7B 0.3h가 더해진다 ([GuidedQuant 2505.07004, ICML 2025](https://arxiv.org/abs/2505.07004) [V]).
  - RSQ는 H = 2XR²Xᵀ를 휴리스틱 가중으로 쓰고 r_min = 0.01 floor를 둔다 ([RSQ 2503.01820, TMLR 2026](https://arxiv.org/abs/2503.01820) [V]).
  - OAC Eq. (11)의 x π(1−π) xᵀ는 1 − ‖p‖²의 이진형이다 ([2405.15025](https://arxiv.org/abs/2405.15025) [V]). BRECQ Sec. 3.3은 대각 FIM 가중을 쓴다 ([2102.05426, ICLR 2021](https://arxiv.org/abs/2102.05426) [V]).
  - DPQ의 entropy-only 표본 선택은 arm (e)의 표본 수준판이다 ([DPQ 2608.21019, EMNLP Findings 2026](https://arxiv.org/abs/2608.21019) [V]).
  - LRM 프루닝에도 토큰 가중 공분산이 있다 ([OBC-Prune 2609.17890](https://arxiv.org/abs/2609.17890) [S]).
- **2606.00206의 범위**
  - KL(FP‖Q)과 BF16 Shannon 엔트로피의 Spearman ρ = 0.92(Fig. 4c), 그리고 "실패의 최대 52%"(139/265, GPT-5 judge)는 모두 한 설정의 결과다. 그 설정은 R1-Distill-Qwen-1.5B, AWQ 3-bit, Pile 캘리브레이션, MATH-500이다.
  - 같은 논문의 GPTQ-3bit는 이미 Numina 자가생성 트레이스로 캘리브레이션돼 있다(71.6%/6.8K 토큰 vs AWQ 47.0%/23.4K). 이 비교는 알고리즘 차이와 교란된다.
  - ρ는 보조정리 자체가 예측하는 값이라 캘리브레이션 결함의 증거가 아니다. 이 논문이 낸 처방은 디코딩 측 페널티뿐이다 ([Lotfi et al. 2606.00206](https://arxiv.org/abs/2606.00206) [V]).
- **저엔트로피 쪽 실패**: NVFP4는 저엔트로피 기호 토큰에서 오표집을 늘린다 ([ReSET 2606.13233](https://arxiv.org/abs/2606.13233) [V]). w_t → 0인 곳이 보호되지 않는다는 뜻이다.
- **남는 공백**: 정확히 1 − ‖p‖²를 GPTQ 토큰 가중으로 쓴 논문은 없다. 도메인, 엔트로피, 부분공간 정렬을 인과적으로 분리한 실험도 없다.

### 수학
**보조정리 8.1 (표준 결과의 trace 형태).** J_t = diag(p_t) − p_tp_tᵀ이고, δ_t ~ (0, σ²I)이면 E[KL] ≈ (σ²/2)tr(J_t) = (σ²/2)(1 − ‖p_t‖₂²)이다. 1 − ‖p‖²는 Gini/Tsallis-2이며 1 − e^{−H₂}로 collision entropy의 단조함수다. 2차 전개 자체는 이미 알려져 있다(BRECQ, OAC Eq. (11), LAQuant Prop. 1 M_E2E = JᵀH_KL J, Lemma 1 σ_max(H_KL) ≤ 2 max p_i(1−p_i) ≤ 1/2). 새로운 것은 이 trace를 토큰 가중으로 쓰는 점뿐이다.

**층 전달.** 층 출력 섭동이 Jacobian G_t로 로짓에 전달되면 KL_t ≈ ½ε_tᵀG_tᵀJ_tG_tε_t이다. on-policy 트레이스에서는 GuidedQuant의 토큰별 Σ_j g_tj² 기댓값이 tr(G_tᵀJ_tG_t)와 같다. 따라서 GuidedQuant-on-traces가 이상형이고, w_t는 G_t를 버린 근사다. 이 근사가 기대는 G_t 동질성 가정은 LAQuant Table 6과 충돌한다.

**따름정리 8.2.** 최종 은닉에서 tr(G_tᵀJ_tG_t) = Σ_i p_i‖u_i‖² − ‖Σ_i p_i u_i‖²이다(u_i는 언임베딩 행, 최종 norm 제외). 이 값도 forward만으로 정확히 계산된다.

**IS 해석.** 캘리브레이션과 배포에서 w_t 분포가 다르면 H_w는 토큰 수준 공변량 이동(아이디어 5의 (4))이 된다. 이동의 방향은 아직 측정되지 않았다.

증명 가능한 것은 8.1과 8.2다. G_t 동질성과 가중만으로 도메인 효과를 재현하는지는 추측이다.

### 판정 실험(약 3–4 GPU·일 [S])
- **0단계**: C4, WikiText, 트레이스에서 w_t 분포와 c̄_k를 재고, KL을 1−‖p‖²와 H에 대해 함수형으로 비교한다.
- **모델**: R1-Distill-Qwen-7B, Qwen3-8B(thinking). GPTQ/GPTAQ W4/W3 g128로 알고리즘을 고정한다.
- **기본 arm**: (a) C4, (b) 자가생성 CoT, (c) CoT + w_t, (d) C4 + w_t, (e) C4 w_t 상위 분위.
- **추가 arm**: 같은 트레이스 위 (f) GuidedQuant g=1·4, (g) RSQ-AttnCon, (h) DPQ entropy-only, (i) CoT+generic 혼합, (j) 8.2 가중.
- **2×2 분리**: c̄_k를 맞춰 재표집한 C4 × 엔트로피를 평탄화한 트레이스.
- **가중 floor**: w_t ← max(ε, w_t^γ).
- **지표**: AIME24/25, MATH-500, GPQA, LiveCodeBench, CoT 길이, overthinking 비율(2606.00206의 GPT-5 judge 프로토콜, 정의 명시), 저엔트로피 기호 토큰의 꼬리 질량, K/V RMSE. 3 seed.
- **성공 기준**: (d)가 (a)–(b) 격차의 절반 이상을 회복하고 c̄_k를 통제한 뒤에도 그 회복이 남는다. (c) 또는 (j)는 (f)와 동등해야 한다.
- **기각 조건**: (d)≈(a)이거나, c̄_k를 맞춘 C4가 (b)를 재현한다.

### 평가
**novelty: 낮음–중.** 방법은 GuidedQuant, RSQ, OAC/BRECQ 계열의 forward-only, 채널 등방 특수형이다. 보조정리 8.1은 교과서 수준이다. 남는 기여는 둘이다. (i) backward 없는 closed-form 가중(8.1, 8.2), (ii) 도메인 효과의 인과 분해.

**리스크.**
- 트레이스가 뾰족하다는 실측이 이미 있다.
- LAQuant의 Fisher 희석 결과와 ReSET의 저엔트로피 실패가 가설에 불리하다.
- 2606.25519는 QAT가 캘리브레이션 최적화보다 우월하다고 결론지었다.
- GuidedQuant가 7B에서 약 0.6h라 비용 이점이 작다.
- AIME는 분산이 커서 seed가 중요하다.

**main track: 낮음–중.** 2×2 실험이 엔트로피 프로필의 독립 효과를 보이면 기작 논문이 된다. 그렇지 않으면 workshop 수준이다. 0단계에서 트레이스 w_t가 C4보다 낮고 c̄_k가 효과를 설명하면 부록 A로 내린다.

---
