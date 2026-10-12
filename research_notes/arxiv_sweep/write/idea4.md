## 아이디어 4. FP4 사전학습의 LR-decay 말기 격차: 분산 바닥과 편향 바닥의 서명 분리, 그리고 말기 정밀도 전환의 조건

### 가설
저정밀 학습의 손실 격차는 두 바닥의 합이다. (A) 비편향 SR 기울기 노이즈가 만드는 초과손실 ≈ (η/4)tr(C_q)는 η에 비례하며, decay와 함께 사라진다. (B) forward RTN 가중치 오차가 만드는 ½E[bᵀHb] ≈ (Δ²/24)tr(H)는 고정된 H에서 η와 무관하다. 이 구조는 장난감 모델에서 이미 증명되었다(아래). 주장은 이것이 FP4 LLM의 decay 말기 격차를 정량 설명한다는 것이다. decay 중 tr(H)가 오르므로(측정된 사실) (B)도 커진다. (B)는 다시 둘로 나뉜다. (B0)는 고정 w에서 forward만 BF16으로 바꾸면 즉시 사라지는 평가 편향, (B1)은 η>0 학습으로만 회복되는 재최적화 성분이다. NVIDIA의 10T 늦은 전환이 덜 회복된 것은 B1이 있다는 신호이다. Nemotron 3 Ultra의 발산은 바닥 문제가 아니라 불안정성 문제이다.

### 문헌상 공백
- **두 바닥 구조는 알려져 있다.**
  - ECO §3.4 식 (18): master weight가 있는 SGDM에서 lim_{η→0} E[(∇f(x̂))²] = L²σ²이다(1D 2차). master weight가 없으면 오차는 ∝1/η이다 ([2601.22101](https://arxiv.org/abs/2601.22101) [V]). 이를 손실로 환산하고 σ²=Δ²/12를 넣으면 (Δ²/24)tr(H)이다(이 보고서의 환산).
  - Ichikawa et al.: Prop. VI.1의 SGD 바닥은 η에 비례하고, Thm VI.3은 η와 무관한 바닥 σ_ψ²Δ²p(1−p)을 준다 ([2510.10693](https://arxiv.org/abs/2510.10693) [V]).
  - Chmiel et al.: §4.1은 ½η²σ_q²tr(H) 항과 √3 SNR 전환 기준을 준다. App. B.2는 편향 rounding의 η-무관 바닥 μ_ε²/(2λ)을 준다. §5 QAF(backward만 BF16, LR rewarm)는 7B/1T 격차를 완전히 닫았다 ([FP4 All the Way 2505.19115, NeurIPS 2025](https://arxiv.org/abs/2505.19115) [V]).
  - 인접 결과: SWALP noise ball ([1904.11943, ICML 2019](https://arxiv.org/abs/1904.11943) [V]), Quartet의 forward=eff_N / backward=eff_D ([2505.14669](https://arxiv.org/abs/2505.14669) [S]).
- **NVIDIA NVFP4 12B/10T** ([2509.25149](https://arxiv.org/abs/2509.25149) [V]).
  - FP8 대비 상대 검증 손실 오차가 안정 구간 1% 미만에서 decay 구간 1.5% 남짓으로 벌어진다(Fig. 2).
  - 정밀도 전환은 BF16만 실험했고, MXFP8은 언급만 했다.
  - App. D: 8.2T에서 forward만 전환하면 1.5%→0.5%로 줄고, backward만 전환하면 개선이 없다(Chmiel과 충돌). 10T 전환은 "presumably" LR이 낮아서 덜 회복된다.
  - decay 기작(편향 대 분산, tr(H))을 분리한 ablation은 없다.
- **같은 LR 지점에서 분리된 적 없는 병목 주장들.**
  - Quartet II: backward 원소별 SR 분산(SR 23.5 vs RTN 9.0 ×1e-3)을 FP8 micro-scale SR(MS-EDEN 9.4)로 줄인다 ([2601.22813, ICML 2026](https://arxiv.org/abs/2601.22813) [V]).
  - Dong et al.: hot channel. GLA-1.3B, cosine, 상관 증거이다 ([2602.02047](https://arxiv.org/abs/2602.02047) [V]).
  - TetraJet-v2: bin 경계 진동 ([2510.27527, ICML 2026](https://arxiv.org/abs/2510.27527) [V]).
  - UFP4: shrinkage bias. bin 조건부 Eq. 2, FP32 스케일 ([2606.20381](https://arxiv.org/abs/2606.20381) [V]).
  - Full-Stack FP4: optimizer와 attention까지 범위를 넓혔고, 격차 감소는 projection LoRA-SVD에서 나온다(1.40%→0.61%) ([2607.04422](https://arxiv.org/abs/2607.04422) [V]).
- **Nemotron 3 Ultra** ([2606.15007](https://arxiv.org/abs/2606.15007) [V]).
  - 전 텐서 BF16 분기가 손실을 "substantially recovers"했다.
  - 격차는 첫 5B 토큰 0.27/0.28/0.25%에서 74B 후 0.33/0.34/0.03%이다.
  - 발산은 BF16 분기에서도 재현되었다. NVFP4 논문과 모순이 아니며, 즉시 회복분과 재최적화분을 가를 데이터이다.
- **Catalan-Tatjer, Ajroldi, Geiping** ([2510.06213, ICLR 2026](https://arxiv.org/abs/2510.06213) [V]).
  - 최대 32B(OLMo2) / 15T(Apertus-8B) 궤적에서 "once learning rates decay, validation loss and quantization error diverge"(WSD).
  - LR·스케줄·weight decay·LAWA 개입을 수행했다.
  - 160M에서 decay 중 모든 고유값과 trace가 상승한다(Fig. 9). 큰 peak LR은 tr(H)와 PTQ 오차를 함께 낮춘다(Fig. 26).
  - PTQ만 다루고 정량 적합은 없다.
- **반론과 교란:** shadow weight 대 endpoint ([2609.21834](https://arxiv.org/abs/2609.21834) [S]), FP32 수렴 시점에 시작되는 INT4 PTQ 붕괴 ([2604.15167](https://arxiv.org/abs/2604.15167) [S]), BF16 master weight의 update swamping ([2010.06192](https://arxiv.org/abs/2010.06192) [S]).
- **EoS.** GD에서 λ_max ≈ 2/η([Cohen et al. 2103.00065, ICLR 2021](https://arxiv.org/abs/2103.00065) [V]), Adam에서 λ₁(P⁻¹H) ≈ 38/η([2207.14484](https://arxiv.org/abs/2207.14484) [V]). 둘 다 top 고유값만 제약하므로 tr(H) 상승을 설명하지 못한다. 동기로만 쓴다.

### 수학
2차 근사 L(w) ≈ L* + ½(w−w*)ᵀH(w−w*)를 쓴다. SGD(학습률 η, 노이즈 공분산 C + C_q)의 정지분포에서 E[L] − L* ≈ (η/4)tr(C + C_q)이다. forward 양자화 w_q = w + b(w)에서 셀 내 오프셋이 균일하면 E[b²] = Δ²/12이고, 초과손실은 ≈ (Δ²/24)tr(H)이다. (Ichikawa p(1−p)의 셀 내 평균; 위치와 그리드가 균일할 때만). E2M1에서는 b̄ ≠ 0이어서(UFP4 Eq. 2, 아이디어 3) 1차 항 ∇Lᵀb̄가 더해진다. absmax 스케일의 Δ_g는 η와 λ에 따라 변하므로(Catalan-Tatjer Sec. 6.1) 측정된 E[Δ_g²]를 쓴다.

예측 1: 격차(η) = A·η + B0 + B1, B0 = (1/24)Σ_g E[Δ_g²]tr(H_g(η)) + 1차 편향항.
예측 2(조건부): forward 전환은 B0를 즉시 없앤다. B1은 남은 ∫η dt에 비례해서만 회복된다. backward 전환이 무효인 것은 backward 양자화기가 비편향일 때뿐이고, 편향이 있으면 μ_ε²/(2λ) 바닥이 생긴다. 이것이 NVIDIA와 Chmiel의 충돌에 대한 설명 후보이다.
예측 3: B는 측정된 E[Δ_block²]에 비례한다. 블록 16→8이 Δ²를 1/4로 만들지는 않는다(극값 통계). 대조군은 E1M2(b̄≈0)이다.
예측 4: 최종 η가 다른 두 WSD 런의 η-절편이 B이다. 수렴 상태를 통제해야 한다.
예측 5: (Δ'²/24)tr(H_final) ≪ (η_final/4)tr(C)이면 MXFP8로 충분하다. 전환 시점은 남은 ∫η dt로 정한다. 이 기준을 √3 SNR 기준과 비교한다.

가중치 진동(TetraJet)은 b(w)의 동적 표현이다. bin 경계 근처 master weight가 매 스텝 오차 부호를 바꿔 η로 줄지 않는 Δ/2 노이즈를 낸다. 즉 (B)의 분산 성분이다.

알려진 것: 2차·선형 모델에서 두 바닥의 분리(ECO 식 18, Ichikawa Thm VI.3, Chmiel App. B.2). 따라서 이론 몫은 새 정리가 아니라 블록 스케일 FP4, tr(H(η_t)), B0/B1로 확장한 진단 모델이다. 추측: Adam·transformer에서 근사의 정확도, FP4 런의 tr(H), 1차 항 크기, B1의 ∫η 비례.

### 판정 실험(수십 GPU·일)
설정: 125M–1B Llama형, WSD, NVFP4(또는 시뮬레이션), FP32 master weight.

decay 구간 여러 지점에서 다음을 측정한다.
- (i) BF16 대비 격차.
- (ii) FP4 런의 Hutchinson tr(H_g), λ_max, E[Δ_g²].
- (iii) 영-스텝 forward 전환으로 B0를, k 스텝 BF16 학습으로 B1을 잰다.
- (iv) 같은 LR 지점 ablation: forward만 BF16, backward만 BF16(SR 대 RTN), 블록 8, E1M2, SR forward.
- (v) 최종 η 두 값.
- (vi) decay 대신 LAWA.

분석: 격차 ≈ aη + c·ΣE[Δ_g²]tr(H_g,t)로 회귀해 적합도를 보고, c가 1/24에 가까운지 본다.

실패 기준: B0 예측이 영-스텝 측정과 2배 넘게 어긋나거나, backward 편향 유무가 backward 전환 효과를 예측하지 못하면 모델을 기각한다.

### 평가
novelty는 중이다. 2-바닥 이론은 ECO·Ichikawa·Chmiel이, decay 중 tr(H) 상승은 Catalan-Tatjer가 이미 보였다. 신규 정리는 없다.

남는 신규성은 다음과 같다.
- FP4 사전학습 *중* 격차의 aη + cΔ²tr(H) 적합.
- matched ablation과 FP4 런 위의 tr(H) 측정.
- B0/B1 분해.
- NVIDIA–Chmiel 충돌을 backward 편향 조건으로 해소하는 전환 기준. 기존 기준은 NVFP4 App. D의 경험칙뿐이다 [S].

메인 트랙 가능성은 중이다(진단 실험 논문). 리스크: ≤1B 실험은 10T 현상의 대리로 약하다. c가 1/24에서 크게 벗어나거나 B1이 지배적이면 정성 진단만 남는다.

---
