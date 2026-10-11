## 아이디어 4. FP4 사전학습의 LR-decay 말기 격차: 분산 바닥과 편향 바닥의 서명 분리, 그리고 말기 정밀도 전환의 조건

### 가설
저정밀 학습의 손실 격차는 두 바닥의 합이다. (A) 비편향 SR 기울기 노이즈의 정지분포 초과손실 ≈ (η/4)tr(C_q)는 η에 비례해 decay와 함께 사라진다. (B) forward RTN 가중치 오차 b(w), |b| ≤ Δ/2 의 ½E[bᵀHb] ≈ (Δ²/24)tr(H)는 고정된 H에서 η와 무관하다. 이 구조 자체는 장난감 모델에서 이미 증명되어 있다(아래). 주장은 이것이 블록 스케일 FP4 LLM 사전학습의 decay 말기 격차를 정량적으로 설명한다는 것이다. decay 중 tr(H)가 커지므로(측정된 사실) (B)도 커진다. 그리고 (B)는 다시 둘로 나뉜다. (B0) 고정 w에서 forward만 BF16으로 평가하면 즉시 사라지는 평가 편향, (B1) η>0 학습이 있어야 회복되는 재최적화 성분이다. NVIDIA의 10T 늦은 전환이 덜 회복된 것은 B1을 시사한다. Nemotron 3 Ultra의 발산은 바닥이 아닌 불안정성 문제로 분류된다.

### 문헌상 공백
- **두 바닥은 알려져 있다.** ECO Thm 3.8과 §3.4 식 (18): master weight를 쓰는 SGDM에서 lim_{η→0} E[(∇f(x̂))²] = L²σ²이다(1D 2차식). master weight가 없으면 ∝1/η이다(식 19) ([2601.22101](https://arxiv.org/abs/2601.22101) [V]). 손실로 바꾸면 ½Lσ²이고, σ²=Δ²/12를 넣어 차원별로 더하면 (Δ²/24)tr(H)이다(이 보고서의 환산). Ichikawa et al. Prop. VI.1은 η에 비례하는 SGD 바닥을, Thm VI.3(informal)은 ε*_g = ε_g^(0) + σ_ψ²Δ²p(1−p) + o(η)를 준다(STE 고차원 선형회귀, 셀 내 위치 p 의존) ([2510.10693](https://arxiv.org/abs/2510.10693) [V]). Chmiel et al. §4.1은 E[ΔL] = −η‖∇L‖² + ½η²(∇LᵀH∇L + σ_q²tr(H))와 √3 SNR 전환 기준을 준다. App. B.2는 편향 rounding의 η-무관 잔여 손실 μ_ε²/(2λ)을 준다. §5 QAF(backward만 BF16, LR rewarm)는 7B/1T 격차를 완전히 닫았다 ([FP4 All the Way 2505.19115, NeurIPS 2025](https://arxiv.org/abs/2505.19115) [V]). 인접: SWALP noise ball ([1904.11943, ICML 2019](https://arxiv.org/abs/1904.11943) [V]), Quartet의 forward=eff_N/backward=eff_D 스케일링 법칙 ([2505.14669](https://arxiv.org/abs/2505.14669) [S]), 저장 가중치 SR의 η-무관 페널티 ([Ozkara et al. 2502.20566, AISTATS 2025](https://arxiv.org/abs/2502.20566) [S]).
- NVIDIA NVFP4 12B/10T(기술 보고서, 정리 없음) ([2509.25149](https://arxiv.org/abs/2509.25149) [V]).
  - FP8 대비 상대 검증 손실 오차가 안정 구간에서 1% 미만이었다가 decay에서 1.5% 남짓으로 벌어진다(Sec. 3, Fig. 2).
  - 전환은 BF16으로만 실험했고, MXFP8은 가능성만 언급했다.
  - 8.2T(decay 시작 8T 직후)에 전환한 결과(App. D, Fig. 7): forward만 바꾸면 1.5%→0.5%, forward+backward는 FP8과 같아진다. backward만 바꾸면 개선이 없다. 이 결과는 Chmiel과 충돌한다.
  - 10T 전환은 "presumably because the learning rate is too low" 완전 회복에 실패했다.
  - 텐서 측 ablation은 있다. 그러나 decay 중 격차가 벌어지는 *기작*(편향 대 분산, tr(H))을 가른 ablation은 없다.
- 각 그룹이 서로 다른 병목을 주장한다. 같은 LR 지점에서 이들을 분리한 실험은 없다.
  - Quartet II: backward 원소별 FP4 SR 분산(Table 1 ×1e-3: SR 23.5 vs RTN 9.0)을 병목으로 보고, 확률성을 FP8 micro-scale로 옮긴 MS-EDEN(9.4)을 제안한다 ([2601.22813, ICML 2026](https://arxiv.org/abs/2601.22813) [V]).
  - Dong et al.: hot channel이 병목이다. GLA-1.3B/60B, cosine 단일 런이며 상관 수준 증거이다 ([2602.02047](https://arxiv.org/abs/2602.02047) [V]).
  - TetraJet-v2: bin 경계 가중치 진동이 병목이다 ([2510.27527, ICML 2026](https://arxiv.org/abs/2510.27527) [V]).
  - UFP4: E2M1 shrinkage bias가 병목이다. bin 조건부 식 Eq. 2를 주며, FP32 단일 스케일 실험이다 ([2606.20381](https://arxiv.org/abs/2606.20381) [V]).
  - Full-Stack FP4: optimizer state·attention까지 NVFP4 적용 범위를 넓혔다. 격차 감소는 projection의 LoRA-SVD에서 온다(1.40%→0.61%) ([2607.04422](https://arxiv.org/abs/2607.04422) [V]).
- Nemotron 3 Ultra: 5T/10T/16T에서 전 텐서를 BF16으로 바꾼 분기가 고정밀 손실을 "substantially recovers"했다. 상대 격차는 처음 5B 토큰에서 0.27/0.28/0.25%, 74B 토큰 뒤 0.33/0.34/0.03%이다. 발산(§2.7)은 BF16 분기에서도 재현되었다 ([2606.15007](https://arxiv.org/abs/2606.15007) [V]). NVFP4 논문과 모순이 아니며, 즉시 회복분과 재최적화분을 가를 데이터이다.
- Catalan-Tatjer, Ajroldi, Geiping: 최대 32B(OLMo2, 4–6T) / 15T(Apertus-8B) 궤적을 분석했다 ([2510.06213, ICLR 2026](https://arxiv.org/abs/2510.06213) [V]). PTQ(GPTQ)만 다루며, Taylor 식과 정량 적합은 없다.
  - "once learning rates decay, validation loss and quantization error diverge"를 보였다(WSD 기준).
  - peak LR·스케줄·weight decay·LAWA 개입 실험을 수행했다.
  - 160M에서 decay 중 모든 고유값과 trace가 오른다(Fig. 9).
  - 큰 peak LR이 tr(H)와 PTQ 오차를 함께 낮춘다(Fig. 26).
- 반론과 교란: forward 전환이 B를 즉시 없애지 못할 수 있다는 shadow weight 대 endpoint 관점 ([2609.21834](https://arxiv.org/abs/2609.21834) [S]), INT4 PTQ 붕괴가 LR decay가 아니라 FP32 수렴 시점에 시작된다는 보고 ([2604.15167](https://arxiv.org/abs/2604.15167) [S]), BF16 master weight의 update swamping ([2010.06192](https://arxiv.org/abs/2010.06192) [S]).
- Edge of stability: full-batch GD에서 λ_max ≈ 2/η가 경험적으로 성립한다 ([Cohen et al. 2103.00065, ICLR 2021](https://arxiv.org/abs/2103.00065) [V]). Adam에서는 λ₁(P⁻¹H)가 38/η에서 평형을 이룬다 ([2207.14484](https://arxiv.org/abs/2207.14484) [V]). EoS는 top 고유값만 제약하므로 tr(H) 증가를 설명하지 못한다. 동기로만 쓴다.

### 수학
2차 근사 L(w) ≈ L* + ½(w−w*)ᵀH(w−w*). SGD(학습률 η, 노이즈 공분산 C + C_q)의 정지분포에서 E[L] − L* ≈ (η/4)tr(C + C_q)이다(작은 η, 노이즈와 곡률이 독립이라는 표준 근사). forward 양자화 w_q = w + b(w)에서 셀 내 오프셋이 균일하면 E[b] = 0, E[b²] = Δ²/12이고 추가 초과손실은 ≈ (Δ²/24)tr(H)이다. 이는 ECO 식 (18)과 Ichikawa의 p(1−p) 바닥을 셀 내에서 평균한 것과 같다. 상수 1/24는 셀 내 위치와 그리드가 모두 균일할 때만 성립한다. E2M1에서는 bin마다 Δ가 다르고 E[b] = b̄ ≠ 0이다(UFP4 Eq. 2, 아이디어 3). 그래서 1차 항 ∇Lᵀb̄가 추가된다. 또 absmax 스케일에서 Δ_g는 가중치 범위에 따라 η, λ와 함께 변한다(Catalan-Tatjer Sec. 6.1). 따라서 측정된 E[Δ_g²]를 쓴다.

예측 1: 격차(η) = A·η + B0 + B1, B0 = (1/24)Σ_g E[Δ_g²]tr(H_g(η)) + 1차 편향항. B의 η 의존은 H, Δ_g를 통해서만 온다.
예측 2(조건부): forward만 BF16으로 바꾸면 B0가 즉시 사라진다. B1은 남은 ∫η dt에 비례해 회복된다. backward 전환의 무효과는 backward 양자화기가 비편향일 때만 성립한다. 편향 μ_ε가 있으면 μ_ε²/(2λ) 바닥이 생겨 backward 전환도 효과가 있다. 이 조건으로 NVIDIA와 Chmiel의 충돌을 설명하는 것이 핵심이다.
예측 3: B ∝ 측정된 E[Δ_block²]. 블록 16→8은 Δ²를 1/4로 만들지 않는다(극값 통계, 아이디어 3). 8은 시뮬레이션으로만 가능하다. 균일 그리드(E1M2, b̄≈0)는 1차 항 대조군이다.
예측 4: 최종 η가 다른 두 WSD 런에서 격차의 η-절편이 B이다. 단 수렴 상태를 통제해야 한다.
예측 5: (Δ'²/24)tr(H_final) ≪ (η_final/4)tr(C)이면 MXFP8 전환으로 충분하고, 아니면 BF16이 필요하다. 전환 시점은 남은 ∫η dt가 B1 회복에 충분한지로 정한다. 이 기준을 Chmiel의 √3 SNR 기준과 비교한다.

가중치 진동(TetraJet)은 b(w)의 동적 표현이다. bin 경계 근처의 master weight는 매 스텝 forward 오차의 부호를 바꾼다. 이는 η로 줄지 않는 Δ/2 크기 노이즈, 즉 (B)의 분산 성분이다.

알려진 것: 2차·선형 모델에서 두 바닥의 분리와 η 의존성(ECO 식 18, Ichikawa Thm VI.3, Chmiel App. B.2). 이 아이디어의 이론 몫은 새 정리가 아니다. 그것을 블록 스케일 FP4 forward, tr(H(η_t)), B0/B1로 확장한 진단 모델이다. 기존 측정: decay 중 tr(H) 증가(BF16 160M). 추측: Adam·transformer에서의 근사 정확도, FP4 런의 tr(H), 1차 편향항의 크기, B1의 ∫η 비례.

### 판정 실험(수십 GPU·일)
125M–1B Llama형 모델, WSD 스케줄, NVFP4(또는 시뮬레이션), FP32 master weight(swamping 배제). decay 구간 여러 지점에서 다음을 잰다.
(i) BF16 대비 격차.
(ii) FP4 런 위의 Hutchinson tr(H_g), λ_max, E[Δ_g²].
(iii) B0와 B1: 영-스텝 forward 전환으로 B0을, k 스텝 BF16 학습으로 B1을 잰다.
(iv) 같은 LR 지점 ablation: forward만 BF16, backward만 BF16(SR 대 RTN backward), 블록 8 시뮬레이션, E1M2, SR forward.
(v) 최종 η 두 값.
(vi) decay 대신 LAWA.
격차 ≈ aη + c·ΣE[Δ_g²]tr(H_g,t)를 회귀해 적합도를 보고, c가 1/24에 가까운지 확인한다. 실패 기준은 두 가지다. B0 예측이 영-스텝 측정과 2배 이상 어긋나거나, backward 편향 유무가 backward 전환 효과를 예측하지 못하면 모델을 기각한다.

### 평가
novelty 중. 2-바닥 이론은 ECO·Ichikawa·Chmiel이, decay 중 tr(H) 증가는 Catalan-Tatjer가 선점했다. 신규 정리는 없다. 남는 신규성은 네 가지다.
1. NVFP4 사전학습 *중* 격차를 aη + cΔ²tr(H)로 적합하는 정량 진단.
2. 같은 LR 지점의 matched ablation과 FP4 런 위의 tr(H) 측정.
3. B0/B1 분해.
4. backward 편향 조건으로 NVIDIA와 Chmiel의 충돌을 해소하고, 전환 기준을 √3 SNR의 대안으로 제시하는 것. 출하된 레시피(Megatron Bridge 등)는 일부 레이어를 런 전체에서 BF16으로 유지할 뿐이고, 말기 전환 기준은 NVFP4 App. D의 경험칙뿐이다 [S].

메인 트랙 가능성은 중이다. 이론 논문이 아니라 진단 실험 논문이므로 결과의 선명도가 운명을 정한다. 리스크는 두 가지다. ≤1B 실험은 10T 현상의 대리로 약하다. 또 c가 1/24에서 크게 벗어나거나 B1이 지배적이면 정성 진단만 남는다.

---
