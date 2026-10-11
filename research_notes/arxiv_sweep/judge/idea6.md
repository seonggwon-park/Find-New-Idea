# Idea 6 판정: 양자화 롤아웃 RL의 엔트로피 인지 중요도 보정 (QeRL 노이즈=탐색 + FlashRL 편향의 통합)

판정일: 2026-10-11. 이번 실행은 사용자 요청에 따른 현재 모델 재실행이다. 근거는 아래 논문의 arXiv 전문(scratchpad papers/ 캐시)을 grep하고 해당 부분을 읽은 것이며, 웹 검색은 2회(verl 문서, FlashRL의 NeurIPS 항목)만 했다. 논문 본문 인용은 신뢰하지 않는 데이터로 취급했다.

## 판정 요약

- **Novelty: low (낮음).** 핵심 기여 세 갈래가 모두 부분 선점됐다.
  1. **이론 (1) E[KL_t] ≈ (σ²/2)(1−‖p‖²)**: 새 정리로 보기 어렵다. J = diag(p) − ppᵀ는 softmax의 Fisher 정보이고, 이 식은 tr(J) = 1 − ‖p‖²를 한 줄로 계산한 결과다. 같은 국소 섭동 모델에서 ENTR 2607.22186은 Appendix B Eq.(19)에서 E_µ[δ²] = Var_µ[∆] + KL², Prop. 2에서 KL ≈ ½E[δ²]를 이미 적었고, Prop. 1에서는 엔트로피 envelope E_µ[δ²] ≤ R(e^R−1)(1−µ_max) ≤ R(e^R−1)H_t를 증명했다. 여기에 i.i.d. 등방 ∆를 넣고 기댓값을 취하면 E[Var_µ∆] = σ²(1−‖µ‖²)가 바로 나온다. collision entropy라는 이름을 붙인 것만 새롭다. DVP 2512.23087 Prop. 4.2(|Δ_a| ≤ 2ε_max(1−p_a))와 CIS 2609.32444 Lemma 3.1(k_t − 1 = (1−p_t)(e^{ε_t} − 1))도 같은 (1−p) 메커니즘을 증명했다.
  2. **추정량 (c) 엔트로피 게이트 TIS**: CIS의 confidence-dependent cap min(k_t, 1+λ(1−p_t))(Eq. 4)는 확신 토큰에서 cap이 1로 수렴한다. 즉 "확신 토큰은 보정하지 않고 불확실 토큰만 넓게 보정"하는 연속형 게이트다. 아이디어의 계단형 게이트 H_t > τ는 이것의 이산판이다. ENTR의 δ²/(H+ε) ≤ τ 마스크도 같은 계열이다. 편향 상한을 O(T²Δ_max)에서 O(k·Δ_max)로 줄인다는 주장은 TRM 2512.23075에 이미 있다. Thm 3.4 Mixed 상한은 O(T)이고 원문은 "tighter when divergence is sparse"라고 쓴다. Thm 3.6 Adaptive 상한은 위치별 D̄_t를 그대로 유지하며, 원문은 "concentrated at a few tokens … D̄_t ≪ ε at most positions"라고 쓴다.
  3. **통합 (ii) QeRL 노이즈 = FlashRL 편향**: AIS 2605.13907 §1이 서술 수준에서 이미 선점했다. 원문은 "quantization mismatch operates as a double-edged sword: early in training, it serves as a stochastically driven exploration bonus … as training converges, the same noise devolves into a source of instability"이다. MXFP4 분해 2605.20402 App. D.4, Eq.(27)은 노이즈→온도 메커니즘을 식으로 적었다. 로짓에 η ~ N(0, σ_η²I)가 더해질 때 T_eff ≈ √(1 + 2σ_η²/Var(Δℓ)) > 1이므로 H[π_Q] > H[π]라는 식이다. 게다가 이 통합은 전제부터 틀렸다(아래 전제 오류 P1).
- **남는 것**: (a) FP8/INT8/NVFP4 롤아웃에서 CIS·ENTR·TRIAGE·Score Centering을 TIS와 무보정 대조군과 함께 비교한 연구는 아직 없다. (b) 양자화 롤아웃에서 토큰 단위 KL과 엔트로피의 결합 분포(PTQ 추론에서는 Lotfi ρ=0.92)를 RL 학습 중에 측정한 연구도 없다. (c) "로짓 규모 증가 × (1−‖p‖²) 감소 → KL이 비단조"라는 예측은 검증되지 않았다. 다만 AIS는 단조 증가를 관찰했으므로 이 예측을 반박하는 쪽의 증거가 있다. 결국 이 아이디어는 측정 연구와 기존 보정 방법의 양자화 환경 벤치마크로 재포지셔닝해야 한다.
- **시스템 측면**: verl Rollout Correction은 token/sequence/geometric 수준의 truncate와 mask, 그리고 KL(π_rollout‖π_old) 모니터링을 제공한다. 엔트로피 기반 마스크는 문서에서 찾지 못했다(웹 검색 1회 기준). 엔트로피 게이트를 시스템으로 구현한 사례는 확인되지 않았지만, CIS와 ENTR은 코드 수준에서 간단히 구현된다.

## 읽은 논문과 겹침

| 논문 | 겹침 분류 | 겹치는 것 | 남는 것 |
|---|---|---|---|
| CIS 2609.32444 (2026-09-26) | preempts_part (추정량) | Lemma 3.1 k_t = p_t + (1−p_t)e^{ε_t}(l.222-235); ε_t는 확신도와 거의 무관(Spearman −0.028, Fig. 4, l.428); E[e^{2ε}] = 9.2×10⁶ 대 가중 2차 모멘트 22.9(l.1892-1893); cap min(k_t, 1+λ(1−p_t))(Eq. 4, l.435)은 확신 토큰에서 보정이 꺼지는 연속 게이트이고, 원문은 "the factor 1−p_t in the cap is not a design choice"라고 쓴다. Thm 3.2/3.3은 MSE 상한. bf16 엔진 불일치, MoE 3종, Table 1. | 양자화 롤아웃 실험이 없고 엔트로피/KL 프레이밍이 없다. 탐색과의 연결도 없다. 다만 본문에 "Policy entropy" 곡선(l.2799)은 있다. |
| ENTR 2607.22186 (v2 2026-10-08) | preempts_part (이론과 마스크) + contradicts_premise | Prop. 1 엔트로피 envelope; Eq.(19) E[δ²] = Var_µ[∆] + KL²; Prop. 2 KL ≈ g·E[δ²]; 마스크 δ²/(H+ε) ≤ τ(Eq. 23-24). 저엔트로피 "concave arc"에서는 E[δ²|q] ∝ (1−q)/q → ∞(Eq. 7, l.463-499). | 비동기 staleness와 엔진 불일치만 다루고 양자화는 다루지 않는다. 등방 기댓값(collision entropy) 형태도 없다. |
| AIS 2605.13907 (2026-05-13) | preempts_part (통합 서사) + contradicts_premise (비단조) | FP8 롤아웃과 BF16 학습기 조합에서 "double-edged sword"(l.85-91); 원문은 "D_KL … grows monotonically during training as the policy sharpens"(l.244). ESS는 0.95 초과에서 0.70 미만으로 떨어진다(l.347). TIS는 AIME25 26.81→34.23으로 여전히 BF16보다 2.47 낮고, AIS는 43.33(Qwen3-8B). BF16 초과 성능은 "exploration noise" 덕으로 돌리고 정확한 규명은 "future work"로 남겼다(l.897). | 토큰 단위 위치(어느 토큰에서 일어나는가)와 엔트로피 기반 해석은 없다. 보정은 배치 단위 α 혼합이다. |
| TRM 2512.23075 (v5 2026-06-26) | preempts_part (상한) | 고전적 상한 O(T²), Pinsker-Marginal O(T^{3/2}), Mixed O(T)이며 Mixed는 "tighter when divergence is sparse"(Thm 3.4, l.387). Adaptive 상한은 D̄_t를 그대로 유지(Thm 3.6, l.479-481). | 엔트로피를 D̄_t의 사전 예측자로 쓰는 시도는 없다. |
| 2605.20402 MXFP4 분해 (2026-05-19) | preempts_part (노이즈→엔트로피 식) | CLT로 η ~ N(0,σ_η²I)를 얻고 T_eff ≈ √(1+2σ_η²/Var(Δℓ)) > 1, H[π_Q] > H[π](Eq. 27, l.1666-1678). dense 모델에서는 TIS=2가 "effectively zeroing out the learning signal"이 되어 TIS=5가 필요하다(l.867-870). | 토큰 단위 KL 집중과 IS 게이팅은 다루지 않는다. 주의: 등방 로짓 노이즈가 개별 표본의 엔트로피를 항상 올린다는 보장은 없다. 이 식은 온도 근사에 기댄다. |
| TRIAGE 2610.07043 (2026-10-05) | adjacent_only / contradicts_premise (집중 축) | 학습기와 롤아웃 모두 NVFP4 W4A4. 해로운 불일치는 방향성 영역 𝒜⁻(A<0, δ<0)에 있다. 상위 5% 세그먼트(64토큰)에 amplifying-tail 토큰의 약 1/4이 있으며, 이는 균일 배치일 때의 약 5배다(l.505-509). pre-terminal 단계에서 tail-heavy 세그먼트를 가진 응답의 97.2%(4B)/79.9%(30B)는 응답 평균으로 보면 여전히 \|δ̄\| < 0.05다(l.527-530). reward, entropy, 평균 \|logprob gap\|을 함께 추적한다(l.919). 처리량은 BF16 대비 2.3배. | 엔트로피는 기준으로 쓰지 않는다. 주의: discover 노트가 적은 "97.2%/79.9%가 상위 5% 세그먼트"는 오독이다. 정정한 수치는 위와 같다. |
| (추가 검색) Full-pipeline FP8 RL 2609.22870 (2026-09-19, ByteDance Seed) | contradicts_premise | 원문은 "FP8-Rollout-BF16-Train settings remain stable"(TIS 기본, l.288-291)이다. full-pipeline FP8에서는 학습 중반 엔트로피가 급등하는데, 이를 이끄는 것은 엔트로피 > 1.0인 깨진(garbled) 응답이고 그중 양의 advantage는 6.64%뿐이다(l.299-306). lower-bound over-clip 토큰의 약 90%가 이런 고엔트로피 응답에서 나온다(l.395). 비율의 노이즈 증폭은 1.7-2.9배다. | 고엔트로피는 "탐색 fork"가 아니라 붕괴의 증상일 수 있다. 엔트로피 게이트가 이런 토큰을 오히려 더 크게 보정·가중할 위험이 있다. |
| Score Centering 2609.20807 (2026-09-17) | contradicts_premise (원인) | 불안정의 원인을 분산이 아니라 "drift"(지속적으로 누적되는 편향)로 본다. 양자화 sampler에서 SC 단독이 IS 계열 이상의 성능을 낸다(l.9-12, l.154). | 엔트로피 축은 없다. |
| QeRL 2510.11696 | contradicts_premise (통합 전제) | NVFP4 가중치를 롤아웃과 prefill(로짓 계산) 모두에 쓰고 LoRA로 역전파한다(l.226-227). FlashRL의 8비트/16비트 불일치와 이중 모델 문제를 피하려는 설계다(l.210-213). 엔트로피 증가 = 탐색(l.218-219). | 롤아웃과 학습 사이에 정밀도 불일치가 거의 없다. 따라서 QeRL의 "노이즈"는 off-policy 편향과 같은 양이 아니다. |
| Lotfi 2606.00206, DVP 2512.23087 | supports_premise / preempts_part | Lotfi: 위치별 KL(BF16‖AWQ-3bit)과 엔트로피의 Spearman ρ = 0.92(Fig. 4c, MATH-500). DVP: Prop. 4.2 \|Δ_a\| ≤ 2ε_max(1−p_a); Mode[Δ'_a] ≈ σ²(1−p_a)(1−p'_a). | RL 학습 중 측정은 없다. |

검색 메모: 추가 arXiv 쿼리 4개를 돌렸다. 그중 엔트로피×양자화×롤아웃 조합은 2026-06 이후 0건이었고(엔트로피 게이트를 양자화 롤아웃에 적용한 논문은 없음), mismatch 최신 쿼리에서 2609.22870이 새로 나왔다. 웹 검색: verl 문서에서 엔트로피 마스크를 찾지 못했다. FlashRL 논문은 NeurIPS 2025 virtual 항목 124259/126728("On the Rollout-Training Mismatch in Modern RL Systems")과 Notion 블로그로만 존재하고 arXiv에는 없다. 저자는 Feng Yao, Liyuan Liu 외이며, 워크숍 항목일 가능성이 있으나 확인하지 못했다.

## 전제 오류

- **P1 (핵심)**: "(ii) QeRL의 '노이즈가 엔트로피를 올려 탐색' 현상은 같은 토큰에서 일어나는 같은 현상의 다른 면"이라는 전제는 틀렸다. QeRL은 같은 NVFP4 모델로 롤아웃과 로짓 계산을 모두 하므로 μ ≈ π이고, 정밀도 불일치 기인 KL(μ‖π)은 거의 0이다(2510.11696 l.210-227). QeRL의 양은 KL(π_Q‖π_BF16)이다. 이는 학습에 쓰이지 않는 참조 정책과의 차이이고, 편향이 아니라 정책 자체의 섭동(parameter noise)이다. 수정: "같은 로짓 섭동 δ의 1차 효과(엔트로피 상승)와 2차 효과(off-policy KL)"로 다시 쓰고, QeRL은 on-policy 섭동, FlashRL/AIS는 off-policy 불일치로 구분해야 한다.
- **P2**: 수학 (3)은 KL의 비단조 궤적을 AIS의 관찰로 소개하지만, AIS는 "grows monotonically"라고 적었다(2605.13907 l.244, ESS 0.95 초과 → 0.70 미만). 같은 문서의 "문헌상 공백" 문단은 단조 증가로 올바르게 적고 있어 내부 모순이다. 수정: 비단조성은 이 아이디어 자신의 예측이라고 명시하고, AIS 데이터는 반대 증거로 밝힌다.
- **P3**: "엔트로피, KL(μ‖π), 보상을 한 런에서 함께 보고한 논문이 없다"는 낡은 전제다. TRIAGE 2610.07043은 NVFP4 RL에서 reward, entropy, 평균 \|logprob gap\|을 함께 추적한다(l.919, Fig. 4). AIS는 KL/ESS와 정확도를, 2609.22870은 엔트로피·clip fraction·정확도를 보고한다. 수정: 공백을 "토큰 단위 KL–엔트로피 결합 분포(어느 토큰에서 불일치가 생기는가)를 양자화 롤아웃 RL 학습 중에 측정한 연구는 없다"로 좁힌다.
- **P4**: "분야가 매우 빠르다(2026년에만 5편)"는 수가 낡았다. 2026년 양자화·저정밀 롤아웃 RL 논문만 10편이 넘는다: Jet-RL 2601.14243, FP8-RL 2601.18150, QuRL 2602.13953, QaRL 2604.07853, AIS 2605.13907, MXFP4 분해 2605.20402, QUADS 2607.15810, ACRL 2607.24062, HiFloat4 2607.26515, Full-pipeline FP8 2609.22870, Score Centering 2609.20807, TRIAGE 2610.07043, TRACE 2610.07767.
- **P5**: "고엔트로피 = 유익한 fork"라는 암묵 전제에도 반례가 있다. full-pipeline FP8에서 엔트로피 급등은 깨진 응답이 이끌고, 그 응답의 93% 이상은 음의 advantage를 받는다(2609.22870 l.299-306). 저엔트로피 위치의 희귀 표본은 E[δ²|q] ∝ (1−q)/q로 비율 분산을 지배한다(ENTR Eq. 7, CIS §3.3). 기대 KL 논증(Pinsker/TV)은 이 꼬리를 다루지 않는다.
- **P6**: 편향을 O(T²Δ_max)에서 O(k·Δ_max)로 줄인다는 주장은 비교 대상이 맞지 않는다. O(T²)는 시퀀스 surrogate의 trust-region 상한이다(TRM). 저엔트로피 토큰에서 보정을 끄는 일은 편향을 줄이지 않는다. 그 토큰들의 TV가 작다는 가정을 통해 잔여 편향이 작아질 뿐이며, 이는 TRM Adaptive 상한의 특수한 경우다.
- **P7**: FlashRL을 "NeurIPS 2025 [S]"로 단정한 부분은 확인되지 않았다. NeurIPS virtual 항목은 있지만 arXiv 논문은 없고, 본 트랙인지 워크숍인지도 확인하지 못했다.

## 수정 요구

1. novelty를 "중상"에서 "낮음"으로 내린다. 기여를 (a) 측정 연구(양자화 롤아웃 RL에서 토큰 KL–엔트로피–advantage 부호의 결합 분포)와 (b) 기존 보정 방법(CIS, ENTR, TRIAGE, Score Centering, IcePop/KPop, AIS)의 FP8/INT8/NVFP4 롤아웃 비교로 재포지셔닝한다.
2. 수학 (1)은 "정리"가 아니라 Fisher 2차 근사의 따름정리로 표기한다. ENTR Eq.(19)와 Prop. 1·2, DVP Prop. 4.2, CIS Lemma 3.1을 인용한다. 등방·i.i.d. δ 가정이 양자화에서 깨지는지(δ ∝ ‖z‖, outlier 채널)를 검증 대상으로 둔다.
3. 수학 (2)의 상한은 TRM Thm 3.4/3.6의 특수한 경우로 재서술한다. "엔트로피는 D̄_t의 사전 예측자"라는 것이 이 아이디어의 기여다.
4. 암 (c)는 CIS(1+λ(1−p_t))의 계단형 근사로 명시하고, CIS와 ENTR을 필수 비교군에 넣는다. TRIAGE의 방향성(A<0, δ<0) 게이트와 Score Centering도 추가한다.
5. 통합 (ii)는 P1에 맞춰 다시 쓰고, AIS의 "double-edged sword" 서술과 MXFP4 Eq.(27)을 선행 연구로 인용한다.
6. 수학 (3)은 "AIS의 관찰"이 아니라 "AIS가 관찰한 단조 증가와 대조되는 예측"으로 고친다. 꺾이는 시점을 정량적으로 예측해야 반증할 수 있다.
7. 판정 실험에 진단 두 가지를 추가한다. 첫째, KL 질량 상위 토큰이 고엔트로피 fork인지, 깨진 응답의 토큰인지, 아니면 저엔트로피의 희귀 표본인지 3분류한다. 둘째, CIS ε_t의 확신도 불변성(ρ ≈ −0.03)이 양자화 롤아웃에서도 유지되는지 확인한다.
8. P3, P4, P7에 맞춰 사실 서술을 정정한다.

## 새 가설

- **H1 (롤아웃 보정, 양자화 특이적 cap)**: 양자화 롤아웃에서는 CIS가 bf16 엔진 불일치에서 관찰한 "ε_t가 확신도와 무관"(ρ = −0.028)이 깨지고, ε_t의 분산이 로짓 규모 ‖z‖와 함께 커진다는 가설이다. 그렇다면 최적 cap은 1+λ(1−p_t)가 아니라 1+λ(1−p_t)·g(‖z_t‖) 꼴이 된다. 근거는 CIS Fig. 4(엔진 불일치), MXFP4 분해의 상대 grid noise, 아이디어의 δ ∝ ‖z‖이다. 단계: RL 롤아웃 보정.
- **H2 (롤아웃 보정, 엔트로피×방향 결합 게이트)**: 해로운 토큰은 "고엔트로피이면서 A<0, δ<0"인 교집합에 몰려 있고, 엔트로피 단독이나 방향 단독 게이트보다 결합 게이트가 같은 마스킹 비율에서 붕괴를 더 늦춘다는 가설이다. 근거는 TRIAGE의 방향 비대칭 ρ_asym 1.02→2.32와 2609.22870의 "lower-bound over-clip의 약 90%가 고엔트로피 깨진 응답에서 나오고 그 응답의 93% 이상이 음의 advantage"이다. 단계: RL 롤아웃 보정.
- **H3 (롤아웃 엔진 PTQ 캘리브레이션)**: 롤아웃 양자화기의 스케일과 캘리브레이션을 고엔트로피 위치의 KL에 가중해 맞추면(아이디어 8과 연결), 같은 비트폭에서 ESS가 덜 감소하고 TIS cap 의존도(C=2 대 5)가 줄어든다는 가설이다. 근거는 Lotfi의 위치별 KL–엔트로피 ρ = 0.92와 MXFP4 분해의 "TIS=2가 dense에서 학습 신호를 0으로 만든다"이다. 단계: PTQ 캘리브레이션(롤아웃 엔진).

## 건너뛴 것
ESPO 2512.00499, ACRL 2607.24062, KPop/IcePop의 원문 전문은 이번에 다시 읽지 않았다(discover 노트의 요약에 의존). Score Centering의 "FP8에서 무보정 58%" 수치는 grep으로 재확인하지 못했다.
