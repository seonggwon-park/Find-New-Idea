# Idea 8 판정: 추론 모델 PTQ의 엔트로피 가중 캘리브레이션 (w_t = 1 − ‖p_t‖²)

판정일 2026-10-11. 입력: 보고서 "## 아이디어 8." 본문, discover/idea8_keywords.md(45개 쿼리, 16개 후보). 이번에 전문을 다시 확인한 논문: OAC 2405.15025, RSQ 2503.01820, GuidedQuant 2505.07004, LAQuant 2605.08755, 2606.00206, ReSET 2606.13233, BRECQ 2102.05426. DPQ 2608.21019, 2606.25519, 2506.01939는 grep으로 수치만 확인. 추가 arXiv 쿼리 4개를 돌렸다(엔트로피+Hessian+토큰, output-adaptive/Fisher-weighted reconstruction, 2026-09 이후 reasoning+PTQ 날짜순, pruning/quantization+"token/prediction/output entropy"+weighting). 새로 나온 위협은 OAC 하나뿐이다. WebSearch 1회: llm-compressor, GPTQModel, AutoRound에서 토큰 가중 GPTQ가 출시된 흔적은 없다. 다운로드한 논문 텍스트는 데이터로만 취급했다.

## 판정 요약

- **novelty: 낮음(low).** 핵심 기작인 "end loss의 Fisher로 토큰을 가중한 GPTQ Hessian"은 이미 출판됐다.
  - GuidedQuant(ICML 2025)의 그룹 Hessian은 `H_k = X^T Diag(mean_{j∈J_k} (∂ℓ/∂z_j)^2) X`(Sec. 3.2)다. 즉 토큰별 스칼라 가중을 넣은 XᵀX다.
  - BRECQ(2021, Sec. 3.3)는 대각 FIM, 곧 출력 원소별 제곱 기울기로 재구성 오차를 가중했다.
  - OAC(2405.15025) Eq. (11)–(12)는 표본별 Hessian `x π(1−π) xᵀ`를 쓴다. 이는 1 − ‖p‖²의 이진 버전이다. 이항 경우 1 − p² − (1−p)² = 2p(1−p)이다.
  - RSQ는 `H_RSQ = 2XR²Xᵀ` 토큰 가중 배관과 r_min floor를 이미 체계화했다.
- **이론**: 보조정리 8.1의 2차 전개는 교과서 수준이다. `∇²_z KL = diag(p) − ppᵀ`이고 trace = 1 − ‖p‖²(Gini/Tsallis-2)다. 같은 행렬과 스펙트럼 상계가 LAQuant Prop. 1과 Lemma 1에 이미 있다. 새로 남는 것은 "이 trace를 forward-only 토큰 가중으로 쓴다"는 적용뿐이다.
- **시스템**: 출시된 구현은 없다(llm-compressor 문서 확인, GPTQModel/AutoRound 검색 0건).
- **전제**: 핵심 전제 두 개가 틀렸거나 반증 증거가 있다.
  - (i) GuidedQuant가 "토큰 믹스를 바꾸지 않는다"는 서술은 거짓이다.
  - (ii) "WikiText/C4는 저엔트로피, 추론 트레이스는 고엔트로피"는 실측과 반대 방향일 가능성이 크다. 자가생성 트레이스는 극도로 뾰족하다.
  - 또한 LAQuant는 softmax-Fisher 인자가 감독 신호를 희석해 해롭다는 통제 실험을 제시한다(top-1 일치율 88.44% vs 93.95%).
  - 도메인 효과의 기작 설명(Hessian 부분공간 정렬)도 LAQuant가 이미 제안했다.
- **남는 기여**:
  - 1 − ‖p‖²라는 특정 forward-only 가중을 GPTQ Hessian에 넣은 논문은 없다.
  - "도메인/어휘 효과 vs 엔트로피 프로필 효과"를 인과적으로 분리한 실험도 없다.
  - 다만 둘 다 GuidedQuant(7B 기준 0.3 h, 1×A100)와 LAQuant의 부분공간 가설을 이겨야 의미가 있다.
- **주요 선점**: GuidedQuant 2505.07004(방법), LAQuant 2605.08755(이론, 도메인 효과 설명, 반증 방향). 보조: RSQ 2503.01820, OAC 2405.15025, BRECQ 2102.05426.

## 읽은 논문과 겹침

| 논문 | 겹치는 것 (정확한 위치) | 남는 새로움 | 분류 |
|---|---|---|---|
| GuidedQuant 2505.07004 (ICML 2025) | Eq. (4) `‖(∂ℓ/∂Z) ⊙ (XW − XŴ)‖²_F`. Sec. 3.2에서 그룹 평균한 `H_k = Xᵀ Diag(mean_j (∂ℓ/∂z_j)²) X`, 즉 토큰별 가중 XᵀX다. 경험적 Fisher이며 한 번의 backward로 계산한다. 비용(App. Table 8/9): Llama-2-7B는 0.3 h(1×A100), Llama-2-70B는 2.7 h(6×A100). 캘리브레이션은 RedPajama 1024×4096. | 추론 모델 평가 없음. 자가생성 트레이스 위에서 쓰면 라벨이 on-policy 표본이므로 경험적 Fisher의 기대값이 참 Fisher와 같다. 따라서 GuidedQuant-on-traces가 아이디어의 "이상형"이고, w_t는 G_t를 버린 더 거친 근사다. | preempts_core |
| LAQuant 2605.08755 (2026-05) | Prop. 1 `M_E2E = Jᵀ H_KL J`, `H_KL = diag(p⋆) − p⋆p⋆ᵀ`. Lemma 1 `σ_max(H_KL) ≤ 2 max p⋆_i(1−p⋆_i) ≤ 1/2`. Sec. 4.1 Eq. (2) 도메인 효과 = 활성 Hessian 부분공간 정렬(top-k principal-angle cos c̄_k, Fig. 3b, Table 9). GPTQ Qwen3-4B W3G128: generic→R1 트레이스로 AIME25 +38.45 pp. Qwen3-1.7B: 0.00%→22.55%. App. C Table 6: 비주도 방향 응답도 중앙값 0.025–0.037. "no scalar reweighting can repair it". 통제 E2E vs LW top-1 일치율 88.44% vs 93.95%. Table 8: H<0.1인 토큰이 약 56%이고, LA−LW 격차는 H∈[2,5)에서 +2.50 pp. 부록: AIME25 트레이스 top-1 확률 평균 0.877, 중앙값 0.996, 53.4%가 max p>0.99. | trace 항등식 자체와 토큰 가중으로의 사용. 엔트로피 프로필과 부분공간 정렬을 분리하는 실험. | preempts_part + contradicts_premise |
| RSQ 2503.01820 (TMLR 2026) | Sec. 4.2 `H_RSQ = 2XR²Xᵀ`. Sec. 4.3 "we avoid using any global information, such as model gradients". AttnCon 등 7개 휴리스틱. r_min=0.01일 때 최적(Wiki ppl 9.028). | 엔트로피/Fisher 가중, 추론 모델, CoT 길이는 다루지 않음. | preempts_part (배관) |
| OAC 2405.15025 | Sec. 4.4.1 Eq. (11) 로지스틱 Hessian `x π(1−π) xᵀ` → Eq. (12) Fisher identity → Eq. (13)–(14) 행별 output-adaptive Hessian(경험적 Fisher, 표본=시퀀스 단위 G[i]). | 다범주 trace 1 − ‖p‖²를 토큰 대각 가중으로 쓰는 형태는 없음. OAC는 G[i]가 토큰 합이라 토큰 간 교차항을 포함한다. | preempts_part (이론) |
| BRECQ 2102.05426 (ICLR 2021) | Sec. 3.3: "diagonal FIM to replace the pre-activation Hessian"이며, 대각은 제곱 기울기다. 블록 출력 원소별 가중 재구성. | LLM, 추론, forward-only 근사는 없음. | preempts_part (이론) |
| 2606.00206 (Meta FAIR) | Fig. 4c: 위치별 KL(BF16 vs AWQ 3-bit, MATH-500) 대 BF16 엔트로피, Spearman ρ=0.92. Fig. 7: 고엔트로피 위치에서 thinking 토큰이 2–4배 더 자주 top-20에 듦(τ=80번째 백분위). 실패의 최대 52%. App. A.1: GPTQ는 이미 NuminaMath 자가생성 트레이스 128×2048로 캘리브레이션했다. Sec. 7은 λ를 엔트로피로 적응시키는 디코딩 측 확장만 언급. | 캘리브레이션 측 가중은 없음. ρ=0.92는 등방 노이즈 하의 보조정리가 그대로 예측하는 상관이라, 캘리브레이션이 고엔트로피를 과소적합한다는 증거는 아니다. | supports_premise (부분) |
| ReSET 2606.13233 | Sec. 3.1: "quantization increases incorrect sampling at low-entropy symbolic tokens". 오류는 top-1 flip이 아니라 비-top-1 꼬리 질량이 표집되는 데서 온다. Sec. 3.2: τ0≈0.6 = 상위 20% 엔트로피. 저엔트로피 토큰의 온도를 낮추는 것이 이득이다. | 디코딩 측. | contradicts_premise (부분) |
| DPQ 2608.21019 (EMNLP-F 2026) | 표본 수준 doubt b(x)=1−p(1)−p(2)와 entropy-only 변형 H(p0(·|x))로 상위 표본을 고르고 generic anchor를 섞은 뒤 "unchanged GPTQ". | 토큰 가중 없음. 추론 트레이스 없음. 아이디어의 arm (e)는 이 레시피를 이식한 것이다. | preempts_part (arm e) |
| 2606.25519 | Sec. 7.2 / Fig. 5: Wikitext→Math-CoT로 28%→65%, CoT 30k→12k. QAT만 "Pareto-dominant", 캘리브레이션 최적화는 "strictly below QAT on both axes". | 기작 분석 없음. | supports_premise |
| 2506.01939 (NeurIPS 2025) | Qwen3 CoT 토큰의 50.64%가 엔트로피 <10⁻², 80번째 백분위 = 0.672. | 양자화와 무관. 엔트로피 분포의 근거로만 씀. | contradicts_premise (분포 서술) |

이전 sweep의 노트(교차 확인하지 않음): OBC-Prune 2609.17890(LRM 프루닝에서 `C̃ = Σ w_t x_t x_tᵀ`, ε=0.1 floor), FAIR-Calib 2606.06547(dLLM, sharpness/negative-entropy 게이트를 쓴 위치 가중 MSE, Lemma 4.6 KL ≤ ¼‖Δz‖²), TARQ 2605.27808("implicitly weighting positions by their empirical frequency"; Fisher 기반 가중이 이상적이지만 라벨과 backward가 필요하다고 서술), TASA 2607.00908(과제 데이터만 쓰면 Hessian 꼬리 고유값이 억눌려 성능이 떨어질 수 있음), REAL-Q 2609.00049(GuidedQuant를 "per-token saliency scores … used to weight the GPTQ Hessian"으로 기술).

## 전제 오류

1. **"GuidedQuant는 출력 채널을 가중하지만 토큰 믹스는 바꾸지 않는다"는 거짓이다.**
   - GuidedQuant Sec. 3.2: `H_k = Xᵀ Diag(mean_{j∈J_k}(∂ℓ/∂z_j)²) X`. 토큰별 가중이 그룹 평균 뒤에도 살아남는다.
   - 수정: "GuidedQuant는 end-loss 기울기 제곱으로 (토큰 × 출력 채널 그룹)을 가중한다. w_t는 그 forward-only, 채널 등방, 기대 Fisher 근사다."
   - 비용 표기 "6–7×A100"은 "6×A100"(App. Table 8, Llama-2-70B 2.7 h)으로 고친다.
2. **"엔트로피 가중 캘리브레이션 토큰을 다룬 논문은 없었다"는 과장이다.**
   - 토큰 가중 Hessian: RSQ.
   - 엔트로피 기반 캘리브레이션 표본 선택: DPQ의 entropy-only 변형.
   - 위치별 sharpness/negative-entropy 가중 캘리브레이션: FAIR-Calib(dLLM).
   - LRM 캘리브레이션 토큰 가중: OBC-Prune(프루닝).
   - 정확히 1 − ‖p‖²를 GPTQ 토큰 가중으로 쓴 논문만 없다.
3. **"WikiText/C4는 저엔트로피 연속 토큰이 대부분, 추론 트레이스는 평균 w_t가 높고 꼬리가 두껍다"는 방향이 반대일 가능성이 크다.**
   - 자가생성 트레이스는 뾰족하다. LAQuant 부록: Qwen3-1.7B AIME25 트레이스에서 top-1 평균 0.877, 중앙값 0.996, 53.4%가 >0.99, 약 56%가 H<0.1. 2506.01939: 50.64%가 H<0.01.
   - teacher-forced generic text는 7–8B 모델에서 ppl이 약 6–10(평균 NLL 약 2 nats)이라 토큰당 엔트로피가 대체로 더 높다. WikiText 쪽 수치는 이번에 직접 확인하지 못한 추정이다.
   - 실측 전에는 이 전제를 쓰면 안 된다. 반대로 나오면 "도메인 효과 = 엔트로피 프로필" 가설은 arm (d)를 돌리기 전에 이미 약해진다.
4. **"무엇이 효과를 내는지 분리되지 않았다"는 일부 낡았다.**
   - LAQuant Sec. 4.1이 활성 Hessian 부분공간 정렬로 설명하고 진단 지표(c̄_k)까지 제시했다. 인과 분리는 아니지만 경쟁 가설로 명시해야 한다.
5. **보조정리 8.1을 "증명 가능한 새 수학"으로 내세우는 것은 부적절하다.**
   - 2차 전개는 BRECQ(2021), OAC Eq. (11), LAQuant Prop. 1/Lemma 1, FAIR-Calib Lemma 4.6에 있다. trace 항등식은 한 줄짜리다.
   - "collision entropy의 단조함수"는 맞다(−log‖p‖²). 다만 가중 자체는 Rényi-2가 아니라 Tsallis-2/Gini다.
6. **G_t의 토큰 간 등방·동질 가정은 LAQuant 측정과 충돌한다.**
   - Table 6: E2E 메트릭이 30–40배 비등방이다. 통제 E2E(softmax-Fisher 감독)가 LW보다 top-1 일치율이 낮다(88.44% vs 93.95%). 이는 Fisher 인자를 곱하는 것이 해로울 수 있다는 직접 증거다.
7. **"비용 0의 처방"은 이점이 작다.**
   - GuidedQuant는 7B에서 0.3 h(1×A100)다. on-policy 트레이스에서는 GuidedQuant가 이론상 더 정확한 상위 호환이다.
8. **2606.00206 인용 보완.**
   - ρ=0.92는 AWQ 3-bit(Pile 캘리브레이션) 기준이다. 이 논문의 GPTQ 기준선은 이미 Numina 트레이스로 캘리브레이션돼 있다.
   - KL–엔트로피 상관은 등방 노이즈 하의 보조정리가 그대로 예측하므로, 캘리브레이션 결함의 증거로 쓰면 순환논법이다.
9. **ReSET 반례.**
   - NVFP4 실패가 저엔트로피 기호 토큰의 꼬리 질량 누출에서도 생긴다. w_t → 0인 곳이 보호되지 않는다.

확인된 수치(유지 가능): 28%→65%, 30k→12k(2606.25519 Sec. 7.2), 52%(2606.00206), Liu et al. COLM 2025, GuidedQuant 7B 0.3 h(1×A100).

## 수정 요구

- **novelty 문단 재작성.** "GuidedQuant(토큰×채널 경험적 Fisher 가중), RSQ(토큰 가중 Hessian, r_min floor), OAC/BRECQ(Fisher 기반 출력 가중)의 forward-only·채널 등방 특수형"으로 위치를 다시 잡는다. 기여는 둘로 한정한다.
  - (i) 1 − ‖p‖²라는 무비용 closed-form 가중
  - (ii) 도메인 대 엔트로피 프로필의 인과 분리
- **보조정리 8.1의 지위 변경.** "새 정리"에서 "표준 결과(LAQuant Prop. 1, OAC Eq. 11)의 trace 형태"로 낮추고 인용을 붙인다.
- **경쟁 가설 명시.** LAQuant의 Hessian 부분공간 정렬을 경쟁 가설로 두고 2×2 설계를 추가한다(부분공간 정렬 c̄_k × 엔트로피 프로필). 예: C4에서 c̄_k가 트레이스와 맞도록 재표집한 arm, 트레이스에서 엔트로피를 평탄화한 arm.
- **실측 먼저.** 전제 3(코퍼스별 w_t 분포: C4, WikiText, 자가생성 트레이스)을 실험 0단계로 측정하고 보고한다.
- **기준선 추가.** 같은 자가생성 트레이스 위에서 GuidedQuant(g=1,4), RSQ-AttnCon, DPQ-entropy-only(표본 수준)를 넣는다. 성공 기준에 "GuidedQuant 대비 동등 이상, 비용 1/N"을 추가한다.
- **가중 floor/temper 필수화.** w_t ← max(ε, w_t^γ)로 두고, RSQ r_min=0.01과 OBC-Prune ε=0.1을 참조한다. 평가 지표에 다음을 추가한다.
  - 저엔트로피 기호 토큰의 꼬리 질량과 오표집률(ReSET)
  - K/V 재구성 RMSE(LAQuant: RMSE와 Pass@1의 상관 r≈−0.85)
- **혼합 캘리브레이션 arm 추가.** TASA가 보인 과제 데이터만 쓸 때의 열화를 통제하기 위해 CoT+generic 혼합 arm을 넣는다.
- **평가 문구 수정.** "novelty 중"을 "novelty 낮음–중(방법은 선점된 계열의 최저비용 근사, 기여는 기작 분리 실험)"으로 바꾼다. 2606.25519의 "캘리브레이션 최적화는 QAT보다 엄격히 아래" 결론을 리스크로 명시한다.

## 새 가설

1. **언임베딩 분산 가중(forward-only, G_t 일부 복원).**
   - 최종 층에서는 tr(Gᵀ J G)가 정확히 `Σ_i p_i‖u_i‖² − ‖Σ_i p_i u_i‖²`(u_i = 언임베딩 행)이다. 즉 p 아래에서 언임베딩 벡터의 분산이다.
   - 이 값은 1 − ‖p‖²보다 G_t 이질성을 반영하면서도 forward-only다.
   - 가설: 이 가중이 1 − ‖p‖²와 GuidedQuant 사이의 격차 대부분을 메운다.
   - 근거: LAQuant Table 6의 비등방성, GuidedQuant의 Fisher 구조. 단계: GPTQ Hessian 가중.
2. **엔트로피 프로필이 아니라 꼬리 누출이 결정한다.**
   - 2차 Fisher 가중은 저엔트로피 토큰을 무시한다. 그러나 추론 실패의 상당 부분은 저엔트로피 기호 토큰에서 비-top-1 질량이 표집되는 데서 온다(ReSET Sec. 3.1).
   - 가설: w_t = (1 − ‖p‖²) + λ·(기대 1 − q_top 증가) 형태의 "표집 오류 가중"이 순수 엔트로피 가중보다 AIME 정확도와 CoT 길이를 함께 개선한다.
   - 근거: ReSET, LAQuant Table 8(이득이 고엔트로피에 집중되지만 저엔트로피가 토큰의 56%). 단계: 캘리브레이션 가중 + 평가 지표.
3. **도메인 효과 = 부분공간 정렬 > 엔트로피.**
   - 활성 부분공간 overlap c̄_k를 맞춘 C4 하위표본은 엔트로피 프로필과 무관하게 Math-CoT 캘리브레이션 이득의 대부분을 재현하고, 엔트로피 가중은 그 위에 소폭만 더한다.
   - 반대 결과가 나오면 아이디어 8의 핵심이 산다.
   - 근거: LAQuant Sec. 4.1/Fig. 3b, 2504.04823(도메인 효과가 GPTQ에서만 나타나고 AWQ/FlatQuant에서는 없음). 단계: 캘리브레이션 데이터 선택.

## 생략한 것
- OBC-Prune, FAIR-Calib, TARQ, TASA, REAL-Q는 이전 sweep 노트에 의존했고 이번에 다시 grep하지 않았다.
- WikiText에서 추론 모델의 토큰 엔트로피를 직접 보고한 논문은 찾지 못했다(전제 3의 generic 쪽 수치는 추정).
- WebSearch는 1회만 사용했다.
