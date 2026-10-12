# 아이디어 4 판정 — FP4 사전학습 LR-decay 말기 격차: 분산 바닥(∝η) 대 편향 바닥(Δ²tr(H)/24)

작성일 2026-10-11. 입력: 아이디어 본문(보고서 191–222행), discover/idea4_citations.md, discover/idea4_keywords.md.
원문 전문을 직접 확인한 논문: 2505.19115, 2510.10693, 2601.22101, 2510.06213, 2509.25149, 2606.15007, 1904.11943(추가).
추가 검색: arXiv 3건(low-precision SGD noise ball / 2026-06 이후 LR·이론 / NVFP4 scaling study 제목), WebSearch 3건(Megatron NVFP4 전환 레시피, 2026 이론, ICML 2026 NVFP4 scaling study).
논문 텍스트는 신뢰하지 않는 데이터로 취급했고, 인용만 했다.

## 판정 요약

- novelty: **medium (이론은 선점, 진단 실험만 신규)**. 표 문구: 중간(이론 선점: 2601.22101·2510.10693·2505.19115, 진단 실험만 신규)
- 주된 선행: ECO(2601.22101) §3.4 식 (18)과 Thm 3.8, Ichikawa et al.(2510.10693) Prop. VI.1과 Thm VI.3, Chmiel et al.(2505.19115) §4.1과 App. B.2, Catalan-Tatjer et al.(2510.06213) §6.2와 Fig. 9/26.
- 이유: 아이디어가 "증명 가능한 것"으로 내세운 2차 모델의 두 바닥 분리는 이미 출판되어 있다. η에 비례해 사라지는 SGD 노이즈 항과, η와 무관하게 Δ²에 비례하는 가중치 양자화 바닥이 그것이다. ECO는 master weight를 쓰는 SGDM에서 lim_{η→0} E[(∇f(x̂))²] = L²σ²를 1차원 2차식으로 보였다. 손실로 바꾸면 ½Lσ²이고, σ²=Δ²/12를 넣고 차원별로 더하면 바로 Δ²tr(H)/24이다. Ichikawa는 고차원 선형회귀(STE)에서 ε* = ε^(0) + σ_ψ²Δ²p(1−p) + o(η)을 보였다. Chmiel은 SR 분산 항 ½η²σ_q²tr(H)와, 편향 rounding이 η와 무관한 잔여 손실 μ²/(2λ)을 남긴다는 결과를 이미 썼다. "decay 중 tr(H)가 커진다"도 Catalan-Tatjer Fig. 9가 이미 측정한 사실이다(160M BF16). 추측이 아니다.
- 남는 신규성: (1) FP4/NVFP4 LLM 사전학습 *중* 격차를 gap ≈ aη + cΔ²tr(H_t)로 적합하고 c≈1/24을 검증하는 일. (2) 같은 LR 지점에서 forward만/backward만/그룹/SR-forward를 바꾸는 matched ablation과, FP4 런 위에서 tr(H)를 측정하는 일. 둘 다 아무도 하지 않았다. (3) NVIDIA(forward 전환 효과)와 Chmiel(backward 전환 효과) 사이의 충돌을 조건부로 해소하는 일. (4) 전환 포맷 기준 (Δ'²/24)tr(H) 대 (η/4)tr(C). 기존 전환 기준은 Chmiel의 √3 SNR 하나뿐이다.
- 시스템 측면: Megatron Bridge와 NVIDIA 블로그는 일부 레이어를 런 전체에서 BF16으로 유지하는 방식(F0L4, first_last_layers_bf16)만 문서화한다. 말기 정밀도 전환을 자동화하는 기준이 출하된 사례는 찾지 못했다. 전환 권고는 NVFP4 논문 App. D의 경험칙("LR decay 시작 직전에 전환")뿐이다.

## 읽은 논문과 겹침

| 논문 | 겹치는 것(정확한 위치) | 남는 신규성 |
|---|---|---|
| 2601.22101 ECO (Nikdan, Zandieh, Alistarh, Mirrokni; 2026-01) | Thm 3.8: σ²_quant = 4η²β²L²G²/(1−β)² + 4L²σ²/(1−β²)이며 "As η → 0 … noise floor persists". §3.4 식 (18): master weight를 쓰는 SGDM에서 lim_{η→0} L_MW = L²σ², "even with master weights, since the final solution must lie on the quantization grid, a noise floor of L²σ² is unavoidable". 식 (19): master weight가 없으면 ∝1/η. → **편향 바닥의 η-독립성과 Δ² 비례가 1D로 증명되어 있다** (preempts_part) | 손실 단위의 tr(H) 형태, forward 활성 양자화, LLM decay 실험 |
| 2510.10693 Ichikawa, Kashiwamura, Sakata (2025-10, stat.ML) | Prop. VI.1: q*에 η σ_ψ²(ρ+σ²)(σ_ψ²+λ) 항이 있어 SGD 바닥이 η에 비례한다. 안정 조건은 0<η<2(σ_ψ²+λ)/σ_ψ⁴. Thm VI.3(informal): ε*_g = ε_g^(0) + σ_ψ²Δ²p(1−p) + o(η)로, η와 무관한 Δ² 바닥이다. → **두 바닥 구조의 고차원 선형모델 버전** (preempts_part) | transformer/Adam, tr(H) 성장, FP4 블록 스케일, decay 실험. 주의: 바닥이 셀 내 위치 p에 의존하므로 1/12 평균은 가중치가 많고 위치가 고르게 퍼질 때에만 맞다 |
| 2505.19115 FP4 All the Way (Chmiel, Fishman, Banner, Soudry; NeurIPS 2025) | §4.1: E[ΔL] = −η‖∇L‖² + ½η²(∇LᵀH∇L + σ_q²tr(H)). 임계 SNR은 ‖∇L‖/√(3d) 수준이다. 404행: "deterministic rounding introduces a persistent bias, leading to an irreducible error floor". App. B.2: L∞ = μ_ε²/(2λ)이며 η와 무관하다. §5/Fig. 6b: QAF는 "forward pass kept in FP4, while the backward pass is executed in BF16"이고 "completely close the gap"한다. LR은 reset 후 40 iter warmup과 cosine을 쓰고, peak는 FP4 마지막 LR이다(Table 5 캡션). → 분산 항, 편향→바닥 이분법, 전환 기준을 선점하고 **예측 2와 정면 충돌**한다 (preempts_part + contradicts_premise) | 편향 바닥을 forward 가중치 오차로 tr(H)와 연결하는 것, 그리고 NVIDIA와의 충돌 해소 |
| 2510.06213 Catalan-Tatjer, Ajroldi, Geiping (ICLR 2026) | §6.2와 Fig. 9(160M/100BT): λ_max가 "consistent rapid surge whenever the learning rate decays"하고 "in the decay phase all eigenvalues increase"하며, trace는 "remaining stable under a constant learning rate, and rising abruptly as it decays"한다. 가설: "sharper region … more sensitive to perturbations such as quantization". Fig. 26: peak LR이 클수록 sharpness와 trace가 작고 PTQ 오차도 작다. LAWA는 더 넓은 최소점을 준다(Fig. 7). 초록: "once learning rates decay, validation loss and quantization error diverge". → **tr(H) 증가는 측정된 사실**이다 (preempts_part, PTQ 측) | 학습 중 저정밀 격차, Δ²tr(H)/24 공식(이 논문에는 Taylor 식이 없다), FP4 런 위의 tr(H) |
| 2509.25149 NVIDIA NVFP4 (2025-09) | 285행: "widens to slightly above 1.5% as the learning rate is decayed". App. D와 Fig. 7: 8.2T에서 forward만 전환하면 1.5%→0.5%이고, backward만 전환하면 "no such improvement"이다. 10T 전환은 "fails to fully recover, presumably because the learning rate is too low for the weight updates". 권고는 "switching … shortly before the onset of learning rate decay"이다. (supports_premise, 단 아래 전제 오류 참고) | 원인 메커니즘 분해 |
| 2606.15007 Nemotron 3 Ultra (NVIDIA, 2026-06) | 478–490행: 5T/10T/16T에서 전 텐서를 BF16으로 바꾼 분기의 상대 격차가 처음 5B 토큰 평균 0.27/0.28/0.25%였고, 74B 토큰 뒤 0.33/0.34/0.03%였다. "switching all tensors to BF16 substantially recovers the high-precision loss". "did not resolve the training divergence (§2.7)". → **NVFP4 논문과 모순이 아니다** (supports_premise; 아이디어 문구가 틀렸다) | 즉시 회복분(첫 5B)과 재최적화분(74B)의 분리 |
| 1904.11943 SWALP (Yang et al., ICML 2019), 추가 | Thm 1–3: 저정밀 SGD는 δ로 정해지는 noise ball로 수렴한다("asymptotic accuracy floor"). 누산기를 매 스텝 양자화하면 O(δ), 평균하면 O(δ²)이다. → 고전적인 "양자화 바닥" 선행 (adjacent_only, master weight 없음) | master weight를 쓰는 forward 양자화, LLM |

discover 노트에서 확인했지만 이번에 재독하지 않은 것도 그대로 인용해야 한다. Quartet 2505.14669(forward→eff_N, backward→eff_D)는 정적 스케일링 법칙 수준의 forward=용량/backward=노이즈 이분법이다. Ozkara 2502.20566 Thm 1/3은 master weight 없는 SR/RTN에서 η→0일 때 편향 항이 오히려 폭발함을 보인다. 2609.21834(shadow weight 대 endpoint)와 2604.15167(INT4 붕괴는 LR decay가 아니라 FP32 수렴 시점에 시작)은 반론이다.

## 전제 오류

1. **"Nemotron 3 Ultra … NVFP4 논문과 모순"은 틀렸다.** 2606.15007 482행은 "switching all tensors to BF16 substantially recovers the high-precision loss"라고 쓴다. 해결되지 않은 것은 §2.7의 *발산*뿐이다(490행). 격차가 0.27→0.33%로 커진 것은 NVFP4 런 대비 BF16 분기가 계속 앞서간다는 뜻, 즉 회복이 계속된다는 뜻이다. 실패가 아니다. 고칠 문구: "Nemotron 3 Ultra에서 BF16 전환은 손실 격차를 회복했지만 발산은 고치지 못했다. 발산은 바닥이 아니라 불안정성 문제이다."
2. **"NVIDIA … 원인 ablation은 없다"는 과장이다.** 2509.25149 Fig. 7에는 forward만/backward만/전체 전환을 비교한 텐서 측 ablation이 있다. 없는 것은 메커니즘(편향 대 분산, tr(H)) ablation과 같은 LR 지점 비교이다. 또 NVIDIA 스스로 10T 늦은 전환이 "learning rate is too low for the weight updates" 때문에 덜 회복된다고 썼다. 이는 B를 "고정 w에서의 평가 편향이라 forward 전환으로 즉시 사라지는 항"으로 보는 아이디어의 그림과 어긋난다. 재최적화 성분이 있다.
3. **"tr(H)가 decay 중 커진다"를 추측으로 분류한 것은 낡았다.** Catalan-Tatjer Fig. 9가 BF16 160M에서 이미 측정했다. 같은 논문 Fig. 26은 큰 peak LR이 tr(H)를 작게 만든다고 보이므로, B는 H(η)를 통해 η에 의존한다. "B는 η와 무관"은 고정된 H에서만 성립한다고 명시해야 한다.
4. **EoS 근거가 약하다.** λ_max ≈ 2/η는 full-batch GD에 대한 결과이다(Adam은 전처리된 sharpness를 따른다). Catalan-Tatjer는 decay 중 상위 고유값만이 아니라 *모든* 고유값이 오른다고 보고하는데, EoS는 top 고유값만 제약하므로 tr(H) 증가를 설명하지 못한다. EoS는 동기로만 남기고, 메커니즘으로 내세우지 말아야 한다.
5. **예측 2는 이미 경험적으로 반박 사례가 있다.** Chmiel(2505.19115 §5, Fig. 6b)에서는 backward만 BF16으로 바꾼 QAF가 7B/1T 격차를 완전히 닫았다. LR은 rewarm했고 peak는 마지막 FP4 LR이었다. NVIDIA는 그 반대를 보고했다. "backward만 바꾸면 B는 그대로"는 backward 양자화기가 비편향일 때에만 성립한다. Chmiel App. B.2에 따르면 편향 μ_ε가 있으면 backward 쪽에서도 η와 무관한 바닥 μ²/(2λ)이 생긴다.
6. **"증명 가능한 것: 2차 모델에서의 두 바닥 분리"는 신규 기여가 아니다.** ECO 식 (18), Ichikawa Thm VI.3, Chmiel App. B.2가 이미 보였다. 아이디어의 이론 절은 "알려진 결과의 LLM/FP4 일반화"로 낮춰 써야 한다.
7. **예측 3(그룹 16→8이면 Δ²가 1/4)은 블록 스케일 포맷에서 틀린다.** Δ ∝ 블록 최댓값/그리드 최댓값이고, 블록을 반으로 줄여도 기대 최댓값은 조금만 준다(극값 통계, 아이디어 3). NVFP4의 블록 16은 하드웨어 고정이므로 8은 시뮬레이션해야 한다. "B ∝ 측정된 E[Δ_block²]"로 바꿔야 한다.
8. 상수 1/24는 셀 내 균일 오프셋과 균일 그리드를 가정한다. E2M1에서는 bin마다 Δ가 다르고, 평균 오프셋 b̄≠0이다(UFP4 2606.20381). Ichikawa의 p(1−p) 의존성을 보면, 셀 내 위치 분포가 균일하지 않을 때 상수가 달라진다.

## 수정 요구

- "문헌상 공백"에 다음을 추가한다. ECO 2601.22101 Thm 3.8과 식 (18)(η→0 바닥 L²σ²), Ichikawa 2510.10693 Prop. VI.1과 Thm VI.3, Chmiel 2505.19115 §4.1·App. B.2·QAF, Quartet 2505.14669 Eq. 1, SWALP 1904.11943. 이 논문들을 "두 바닥 구조는 알려져 있다"의 근거로 놓는다.
- "수학" 절의 "증명 가능한 것"을 철회한다. 기여는 "알려진 2-바닥 정리를 블록 스케일 FP4 forward 양자화와 tr(H(η_t))로 확장한 진단 모델 + 정량 검증"으로 다시 쓴다.
- Catalan-Tatjer Fig. 9/26을 "추측" 목록에서 빼서 "기존 측정"으로 옮긴다. 미측정으로 남는 것은 FP4 런 위의 tr(H)이다.
- Nemotron 3 Ultra 문장을 고친다(전제 오류 1).
- 예측 2를 조건부로 바꾼다. "backward 양자화기가 비편향이면 backward 전환 효과 없음. RTN/편향 스케일이면 μ²/(2λ) 바닥이 있어 backward 전환도 효과 있음." 이 문장으로 NVIDIA와 Chmiel의 충돌을 설명하는 것을 핵심 실험으로 둔다.
- B를 둘로 나눈다. (B0) 고정 w에서 forward만 BF16으로 평가할 때 즉시 사라지는 평가 편향, (B1) η>0 학습이 있어야 회복되는 재최적화 성분(NVIDIA 10T, Nemotron 첫 5B 대 74B, 2609.21834). 예측 4(η-절편)는 수렴 상태를 통제해야 한다(2604.15167).
- 예측 3을 E[Δ_block²] 기준으로 바꾼다. 균일 그리드(UFP4/E1M2)에서는 1차 항 b̄≈0인 대조군을 둔다.
- 예측 5를 Chmiel의 √3 SNR 기준과 직접 비교한다.
- 실험 조건에 FP32 master weight를 명시한다. BF16 master weight에서는 update swamping이라는 세 번째 바닥(η↓일 때 커짐; 2502.20566, 2010.06192)이 섞인다.
- 평가 문구 "novelty 중상"을 "중(이론 선점, 진단·조정 실험 신규)"로 낮춘다.

## 새 가설

1. **영-스텝 forward 전환 검사로 B0와 B1을 분리한다.** decay 구간 FP4 체크포인트의 master weight를 학습 없이 BF16 forward로 평가하면 B0을 얻는다. 같은 지점에서 k 스텝 BF16 학습을 하면 B1을 얻는다. 예측: B0 ≈ (1/24)ΣΔ_g²tr(H_g)이고, B1은 남은 ∫η dt에 비례해서만 회복된다. 근거: NVIDIA App. D의 10T 부분 회복, Nemotron 3 Ultra의 0.27%(처음 5B)→0.33%(74B), ECO 식 (18)(고정 w의 바닥). 파이프라인 단계: 사전학습 정밀도 스케줄. 전환 시점 규칙은 "남은 LR 적분 ≥ B1 회복에 필요한 양".
2. **backward 편향 바닥으로 NVIDIA와 Chmiel의 충돌을 설명한다.** Chmiel은 RTN 가중치를 forward/backward에 쓰고 SR 기울기를 쓰며 LR을 rewarm했다. NVIDIA는 2D 가중치 스케일, SR 기울기, RHT를 썼다. backward 전환의 이득은 backward 경로 양자화기의 평균 편향 μ_ε²/(2λ)에 비례할 것이다. 측정량은 기울기 편향의 per-tensor 추정값이다(Quartet II의 스케일 SR 편향 포함). 근거: Chmiel App. B.2, NVFP4 Fig. 7, Quartet 2505.14669의 "data efficiency is linked to the bias in the gradient estimator". 파이프라인 단계: 학습 레시피(backward 양자화기 선택).
3. **FP4 런에서는 LR decay 대신 가중치 평균(LAWA/EMA)이 B를 줄인다.** decay는 tr(H)를 키워 B를 키운다. LAWA는 tr(H)가 낮은 넓은 최소점을 준다(Catalan-Tatjer Fig. 7/26). 그렇다면 FP4 사전학습 말기에 decay를 줄이고 평균을 쓰면 A·η 감소의 대부분을 얻으면서도 B 증가를 피할 수 있다. 그 결과 배포 시 PTQ 강건성도 함께 오를 것이다. 근거: 2510.06213 §6.2, 2610.05329(lag–noise 평균화), SWALP 1904.11943(평균으로 noise ball 축소). 파이프라인 단계: 사전학습 스케줄 → PTQ 배포.

## 생략한 것
- Quartet 2505.14669, 2609.21834, 2604.15167, 2502.20566 원문은 이번 판정에서 다시 읽지 않았다. discover 노트에서 확인된 인용을 그대로 썼다.
- ICML 2026 "What Matters for NVFP4 Training? A Scaling Study …"(icml.cc/virtual/2026/75179)는 arXiv ID를 찾지 못해 원문을 확인하지 못했다. 초록상으로는 레시피 ablation이며 LR-decay 분해는 없는 것으로 보인다.
