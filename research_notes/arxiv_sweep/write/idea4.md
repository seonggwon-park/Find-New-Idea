## 아이디어 4. FP4 사전학습의 LR-decay 말기 격차: 분산 바닥과 편향 바닥의 서명 분리, 그리고 말기 정밀도 전환의 조건

### 가설
저정밀 학습의 손실 격차는 두 바닥의 합이다. (A) 비편향 SR 기울기 노이즈의 정지분포 초과손실 ≈ (η/4)tr(C_q)는 η에 비례해 decay와 함께 사라진다. (B) forward RTN 가중치 오차 b(w)가 만드는 ½E[bᵀHb] ≈ (Δ²/24)tr(H)는 고정된 H에서 η와 무관하다. 이 구조는 장난감 모델에서 이미 증명되어 있다(아래). 이 아이디어의 주장은 그 구조가 블록 스케일 FP4 LLM의 decay 말기 격차를 정량적으로 설명한다는 것이다. decay 중 tr(H)가 커지므로(측정된 사실) (B)도 커진다. (B)는 다시 둘로 나뉜다. (B0)는 고정 w에서 forward만 BF16으로 평가하면 즉시 사라지는 평가 편향이다. (B1)은 η>0 학습이 있어야 회복되는 재최적화 성분이다. NVIDIA의 10T 늦은 전환이 덜 회복된 것은 B1이 있다는 신호이다. Nemotron 3 Ultra의 발산은 바닥의 문제가 아니라 불안정성의 문제로 분류된다.

### 문헌상 공백
- **두 바닥 구조는 이미 알려져 있다.**
  - ECO §3.4 식 (18): master weight를 쓰는 SGDM에서 lim_{η→0} E[(∇f(x̂))²] = L²σ²이다(1D 2차식). master weight가 없으면 바닥은 ∝1/η이다 ([2601.22101](https://arxiv.org/abs/2601.22101) [V]). 이 결과를 손실로 바꾸면 ½Lσ²이고, σ²=Δ²/12를 넣으면 (Δ²/24)tr(H)이 된다(이 보고서의 환산).
  - Ichikawa et al. Thm VI.3: ε*_g = ε_g^(0) + σ_ψ²Δ²p(1−p) + o(η)이다. Prop. VI.1의 SGD 바닥은 η에 비례한다 ([2510.10693](https://arxiv.org/abs/2510.10693) [V]).
  - Chmiel et al. §4.1: E[ΔL]에 ½η²σ_q²tr(H) 항이 있고, √3 SNR 전환 기준을 제시한다. App. B.2: 편향 rounding은 η와 무관한 바닥 μ_ε²/(2λ)을 남긴다. §5 QAF에서는 backward만 BF16으로 바꾸고 LR을 rewarm해 7B/1T 격차를 완전히 닫았다 ([FP4 All the Way 2505.19115, NeurIPS 2025](https://arxiv.org/abs/2505.19115) [V]).
  - 인접 연구: SWALP noise ball ([1904.11943, ICML 2019](https://arxiv.org/abs/1904.11943) [V]), Quartet의 forward=eff_N / backward=eff_D 법칙 ([2505.14669](https://arxiv.org/abs/2505.14669) [S]).
- NVIDIA NVFP4 12B/10T ([2509.25149](https://arxiv.org/abs/2509.25149) [V]). 정리가 없는 기술 보고서이다.
  - FP8 대비 상대 검증 손실 오차는 안정 구간에서 1% 미만이다가, decay 구간에서 1.5% 남짓으로 벌어진다(Fig. 2).
  - 전환 실험은 BF16으로만 했다. MXFP8 전환은 가능성만 언급했다.
  - App. D: 8.2T에서 forward만 전환하면 1.5%→0.5%가 되고, backward만 전환하면 개선이 없다. 이 결과는 Chmiel과 충돌한다.
  - 10T 전환이 덜 회복된 이유를 "presumably" LR이 너무 낮아서라고 추정한다.
  - 텐서 측 ablation은 있다. 없는 것은 decay 중 격차가 벌어지는 기작(편향 대 분산, tr(H))의 ablation이다.
- 2025–26년 그룹들은 서로 다른 병목을 주장한다. 같은 LR 지점에서 이들을 분리한 실험은 없다.
  - Quartet II: backward 원소별 FP4 SR 분산(SR 23.5 vs RTN 9.0 ×1e-3)이 병목이며, 확률성을 FP8 micro-scale로 옮겨(MS-EDEN 9.4) 줄인다 ([2601.22813, ICML 2026](https://arxiv.org/abs/2601.22813) [V]).
  - Dong et al.: hot channel이 병목이다. 근거는 GLA-1.3B, cosine 단일 런에서의 상관 증거이다 ([2602.02047](https://arxiv.org/abs/2602.02047) [V]).
  - TetraJet-v2: bin 경계에서의 가중치 진동이 병목이다 ([2510.27527, ICML 2026](https://arxiv.org/abs/2510.27527) [V]).
  - UFP4: shrinkage bias가 병목이다. bin 조건부 Eq. 2를 쓰며 FP32 스케일로 실험했다 ([2606.20381](https://arxiv.org/abs/2606.20381) [V]).
  - Full-Stack FP4: optimizer와 attention까지 양자화 범위를 넓혔고, 격차 감소는 projection LoRA-SVD에서 나온다(1.40%→0.61%) ([2607.04422](https://arxiv.org/abs/2607.04422) [V]).
- Nemotron 3 Ultra ([2606.15007](https://arxiv.org/abs/2606.15007) [V]).
  - 전 텐서 BF16 분기는 손실을 "substantially recovers"했다.
  - 상대 격차는 첫 5B 토큰 동안 0.27/0.28/0.25%였고, 74B 토큰 후 0.33/0.34/0.03%였다.
  - 발산은 BF16 분기에서도 재현되었다.
  - 따라서 NVFP4 논문과 모순되지 않는다. 오히려 즉시 회복분과 재최적화분을 가를 데이터이다.
- Catalan-Tatjer, Ajroldi, Geiping ([2510.06213, ICLR 2026](https://arxiv.org/abs/2510.06213) [V]).
  - 최대 32B(OLMo2, 4–6T) / 15T(Apertus-8B) 궤적에서 "once learning rates decay, validation loss and quantization error diverge"를 보였다(WSD 기준).
  - peak LR·스케줄·weight decay·LAWA 개입을 수행했다.
  - 160M 모델에서 decay 중 모든 고유값과 trace의 상승을 측정했다(Fig. 9). 큰 peak LR은 tr(H)와 PTQ 오차를 함께 낮춘다(Fig. 26).
  - PTQ만 다루며, Taylor 식이나 정량 적합은 없다.
- 반론과 교란:
  - shadow weight 대 endpoint 관점 ([2609.21834](https://arxiv.org/abs/2609.21834) [S]).
  - INT4 PTQ 붕괴는 FP32 수렴 시점에 시작된다 ([2604.15167](https://arxiv.org/abs/2604.15167) [S]).
  - BF16 master weight에서는 update swamping이 생긴다 ([2010.06192](https://arxiv.org/abs/2010.06192) [S]).
- Edge of stability:
  - full-batch GD에서 λ_max ≈ 2/η이다([Cohen et al. 2103.00065, ICLR 2021](https://arxiv.org/abs/2103.00065) [V]).
  - Adam에서는 λ₁(P⁻¹H) ≈ 38/η이다([2207.14484](https://arxiv.org/abs/2207.14484) [V]).
  - 두 결과 모두 top 고유값만 제약하므로 tr(H) 상승을 설명하지 못한다. 이 아이디어에서는 동기로만 쓴다.

### 수학
2차 근사 L(w) ≈ L* + ½(w−w*)ᵀH(w−w*)를 쓴다. SGD(학습률 η, 노이즈 공분산 C + C_q)의 정지분포에서 E[L] − L* ≈ (η/4)tr(C + C_q)이다(작은 η, 표준 근사). forward 양자화 w_q = w + b(w)에서 셀 내 오프셋이 균일하면 E[b²] = Δ²/12이고, 초과손실은 ≈ (Δ²/24)tr(H)이다. 이는 Ichikawa의 p(1−p) 바닥을 셀 내에서 평균한 값과 같다. 1/24은 셀 내 위치와 그리드가 모두 균일할 때만 성립한다. E2M1에서는 E[b] = b̄ ≠ 0이므로(UFP4 Eq. 2, 아이디어 3) 1차 항 ∇Lᵀb̄가 더해진다. 또 absmax 스케일에서 Δ_g는 η와 λ에 따라 변하므로(Catalan-Tatjer Sec. 6.1) 측정된 E[Δ_g²]를 쓴다.

예측 1: 격차(η) = A·η + B0 + B1, B0 = (1/24)Σ_g E[Δ_g²]tr(H_g(η)) + 1차 편향항이다. B의 η 의존은 H와 Δ_g를 통해서만 들어온다.

예측 2(조건부): forward만 BF16으로 바꾸면 B0는 즉시 사라지고, B1은 남은 ∫η dt에 비례해 회복된다. backward 전환이 효과가 없다는 결론은 backward 양자화기가 비편향일 때만 성립한다. 편향 μ_ε가 있으면 μ_ε²/(2λ) 바닥이 생기므로 backward 전환도 효과가 있다. 이것이 NVIDIA와 Chmiel의 결과가 충돌하는 이유에 대한 설명 후보이다.

예측 3: B ∝ 측정된 E[Δ_block²]이다. 블록을 16에서 8로 줄여도 Δ²는 1/4이 되지 않는다(극값 통계). 블록 8은 시뮬레이션으로 확인한다. E1M2(b̄≈0)를 대조군으로 둔다.

예측 4: 최종 η가 다른 두 WSD 런에서 격차의 η-절편이 B이다. 두 런의 수렴 상태는 통제해야 한다.

예측 5: (Δ'²/24)tr(H_final) ≪ (η_final/4)tr(C)이면 MXFP8 전환으로 충분하고, 아니면 BF16이 필요하다. 전환 시점은 남은 ∫η dt가 B1 회복에 충분한지로 정한다. 이 기준은 √3 SNR 기준과 비교한다.

가중치 진동(TetraJet)은 b(w)의 동적 표현이다. bin 경계 근처의 master weight는 forward 오차의 부호를 매 스텝 바꾸고, 그 결과 η로 줄지 않는 Δ/2 크기 노이즈를 주입한다. 이것은 (B)의 분산 성분이다.

이미 알려진 것은 2차·선형 모델에서 두 바닥의 분리와 η 의존성이다(ECO 식 18, Ichikawa Thm VI.3, Chmiel App. B.2). 따라서 이 아이디어의 이론적 몫은 새 정리가 아니라, 블록 스케일 FP4 forward, tr(H(η_t)), B0/B1로 확장한 진단 모델이다. 추측으로 남는 것은 Adam·transformer에서의 근사 정확도, FP4 런에서의 tr(H), 1차 편향항의 크기, B1의 ∫η 비례이다.

### 판정 실험(수십 GPU·일)
설정은 125M–1B Llama형 모델, WSD 스케줄, NVFP4(또는 시뮬레이션), FP32 master weight이다. decay 구간 여러 지점에서 다음을 측정한다.
- (i) BF16 대비 격차.
- (ii) FP4 런 위의 Hutchinson tr(H_g), λ_max, E[Δ_g²].
- (iii) 영-스텝 forward 전환으로 B0를, k 스텝 BF16 학습으로 B1을 잰다.
- (iv) 같은 LR 지점에서의 ablation: forward만 BF16, backward만 BF16(SR 대 RTN backward), 블록 8, E1M2, SR forward.
- (v) 최종 η 두 값.
- (vi) decay 대신 LAWA.

회귀식 격차 ≈ aη + c·ΣE[Δ_g²]tr(H_g,t)의 적합도를 보고, c가 1/24에 가까운지 확인한다. 실패 기준: B0 예측과 영-스텝 측정이 2배 넘게 어긋나거나, backward 편향의 유무로 backward 전환 효과를 예측하지 못하면 모델을 기각한다.

### 평가
novelty는 중이다. 2-바닥 이론은 ECO·Ichikawa·Chmiel이, decay 중 tr(H) 상승은 Catalan-Tatjer가 이미 보였다. 이 아이디어에 신규 정리는 없다. 남는 신규성은 다음과 같다.
- FP4 사전학습 *중* 격차를 aη + cΔ²tr(H)로 적합하는 정량 진단.
- matched ablation과 FP4 런 위의 tr(H) 측정.
- B0/B1 분해.
- NVIDIA–Chmiel 충돌을 backward 편향으로 해소하는 전환 기준. 기존 말기 전환 기준은 NVFP4 App. D의 경험칙뿐이다 [S].

메인 트랙 가능성도 중이다. 진단 실험 논문이므로 결과가 얼마나 선명한지가 평가를 좌우한다. 리스크는 둘이다. ≤1B 실험은 10T 현상의 대리로 약하다. c가 1/24에서 크게 벗어나거나 B1이 지배적이면 정성 진단만 남는다.

---
