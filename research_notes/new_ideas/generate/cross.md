# 교차 파이프라인·인접 분야 렌즈: 선점되지 않은 새 아이디어 10개 (2026-10-11)

범위: 단계 사이의 상호작용(사전학습 레시피 → PTQ, 양자화 → RL/보상모델/탐지·워터마크/평가 통계, 장문맥 → KV 비트)과 다른 분야에서 가져온 도구(Wiener/디노이징, 디더·노이즈 셰이핑, 극값 이론, conformal risk control, probit/IRT, 연상 기억 모델).
제외: 보고서 0절 표와 0.4절의 선점 목록(softmax −σ²/2, 재귀 상태 ΣΔ/EF, 블록 포맷 EVT 왜곡, FP4 LR-decay 바닥, 캘리브레이션 표본복잡도, 양자화 RL IS, 양자화 셀 정리, 엔트로피/Fisher 캘리브레이션, pre/post-RoPE, MLA latent)과 그 얇은 변형.

근거 표기:
- [V] 이번 세션에서 arXiv 전문을 grep으로 확인
- [abs] 초록만 확인
- [notes] `research_notes/arxiv_sweep/*`에서 가져온 것(이전 세션의 전문 확인 노트)이며 이번에 다시 확인하지 않음

선행 연구 선별: 아이디어마다 `scripts/arxiv_search.py` 쿼리를 4–6개 돌렸다. 그중 2개 이상은 `--sort date --from 20250601`, 1개 이상은 인접 분야 어휘를 썼다. 아이디어 B, J, R은 WebSearch로 한 번 더 확인했다. 쿼리 원문은 각 절 끝에 있다.

탈락한 후보:

| 후보 | 탈락 이유(선행 연구) |
|---|---|
| 안전 방향(refusal direction) 보존 PTQ | 2606.09864(KV 양자화 정렬 붕괴, 안전 부분공간 취약성과 완화책), 2606.10154, 2511.07842, 2601.12033, Q-resafe 2506.20251 |
| MoE route-flip 이론 | 2608.11212, 2606.05688, EAC-MoE 2508.01625 |
| 양자화 추론 길이 팽창 | 2606.00206, 2606.25519, 2609.39801 |
| looped/재귀 모델 고정점 민감도 | LoopQ 2605.16343, 2609.30820, Loopy 2610.05265, 2607.16237 |
| decode 단계 KV 오차 누적 | KVarN 2606.03458 |
| 암기 추출 | 2607.25451 |

---

## A1. 층간 오차 상쇄(counteraction)의 기원: 잔차 블록은 학습 중 Wiener 디노이저가 된다 → 오차가 놓이는 부분공간이 PTQ 강건성을 결정

- **단계**: 사전학습 레시피 → PTQ(이론 + 방법)
- **가설**
  - 2609.11716이 보고한 ⟨Δh^(ℓ−1), Δu^(ℓ)⟩ < 0(counteraction)은 블록 Jacobian의 대칭부가 은닉상태 공분산의 저분산(off-manifold) 방향에서 음의 정부호이기 때문에 생긴다.
  - 따라서 상쇄는 오차 에너지가 은닉 공분산의 하위 고유방향에 있을 때만 일어난다. 상위 PC 방향(이상치·massive 채널) 오차는 상쇄되지 않는다.
  - 학습 중 잡음(중첩 간섭, SGD 잡음, 주입 잡음)의 크기 τ²가 상쇄 강도를 정한다.
- **공백 근거**
  - 2609.11716 [V]
    - 초록: "This counteracting residual interaction develops during pretraining". 기작은 제시하지 않는다.
    - l.1395–1397 한계 절: 집계 통계만 다룬다.
    - l.1789: "Clarifying this mechanism may inspire future PTQ methods".
  - 2609.32060 [abs]: 1차 이론에서 G²이 깊이에 선형으로 자란다(상쇄 없는 랜덤워크형). 이것과 2609.11716의 느린 성장 사이의 긴장을 설명하는 이론이 없다.
- **수학 핵심(증명 가능, 선형-가우시안 모형)**
  - 가정
    - 블록 입력 h = x + n, x ~ N(0, Σ), n ~ N(0, τ²I)
    - 블록 f는 깨끗한 x를 복원하도록 제곱오차로 학습된다(잔차 h + f(h)).
  - 명제: 최적 선형 블록은 h + f(h) = Σ(Σ+τ²I)^{-1}h이다(Wiener). 따라서
    - J_f = −τ²(Σ+τ²I)^{-1} ≺ 0
    - 입력 오차 Δh의 고유방향 i 에너지 비율을 w_i라 하면 상쇄 계수는 κ := −⟨Δh, J_f Δh⟩/‖Δh‖² = Σ_i w_i τ²/(λ_i+τ²)
    - 블록 통과 후 방향 i의 오차는 λ_i/(λ_i+τ²)배로 줄어든다.
  - 따름정리: 깊이 L 뒤의 오차 에너지는 Σ_i w_i Π_ℓ g_{ℓ,i}² 꼴이다. 오차가 상위 PC에 정렬되면(λ_i ≫ τ²) 상쇄가 0에 가깝고, 2609.32060식 선형 성장으로 돌아간다.
  - 비선형·학습 동역학 전체에 대해서는 증명할 수 없다. 이 모형은 국소 선형화 근사다.
- **방법**: Wiener-가중 GPTQ.
  - 블록 출력 오차 지표를 G_down = Π(다음 블록들의 Wiener 이득)로 가중한다. G_down은 캘리브레이션 은닉 공분산에서 닫힌형으로 추정한다.
  - 회전·라운딩이 오차를 저분산 방향으로 보내도록 유도한다.
- **판정 실험(≈1.5 GPU·주)**
  1. Qwen3-8B, OLMo-2-1B(중간 체크포인트 포함) W4/W3 GPTQ에서 κ를 은닉 PC 대역(상위 1%, 1–10%, 나머지)별로 측정한다.
  2. 125M 모델을 5B 토큰으로 2개 arm 사전학습한다: 은닉 잡음 주입 τ² ∈ {0, 0.05·RMS²}. W4 RTN 후 κ와 ΔPPL을 측정한다.
  3. Llama-3.1-8B W3에서 Wiener-가중 GPTQ와 GPTQ, QEP를 비교한다.
- **실패 기준(하나라도 해당하면 기각)**
  - 하위 대역 κ와 상위 대역 κ의 비가 1.5 미만
  - 잡음 주입 arm의 κ 변화가 시드 잡음 이내
  - Wiener-가중 GPTQ의 W3 PPL 개선이 0.05 미만
- **최근접 선행 연구와 차이**
  - 2609.11716: 현상과 항등식만 있다(Prop. 1). 기작·예측·방법은 없다.
  - 2610.07522 [notes]: 국소 잡음 주입 = 크기 감소 ridge. 층간 상쇄 이론은 아니다.
  - QEP·MPAR: 종단 목적은 같은 방향이지만 상쇄의 부분공간 구조를 쓰지 않는다.
  - Patrawala et al. 2025: FP 잔차 기여들이 서로 반대 방향이라는 관찰이다.
- **쿼리**
  - D1 `abs:quantization AND abs:error AND abs:(cancellation OR "self-correct" OR "self-correction" OR attenuation) AND abs:layers AND abs:(LLM OR transformer)` (date ≥20250601)
  - D2 `abs:(quantization OR quantized) AND abs:(denoising OR denoise) AND abs:residual AND abs:(manifold OR "off-manifold")`
  - D3 `abs:"post-training quantization" AND abs:(downstream OR "next layer" OR "subsequent layers") AND abs:(Jacobian OR contraction OR propagation)` (date)
  - D4(인접) `abs:transformer AND abs:"noise robustness" AND abs:(emerges OR emergence) AND abs:(residual OR layers) AND abs:perturbation`

## A2. 어텐션을 V 재구성 필터로 보기: 일관(coherent) 양자화 오차, 서브트랙티브 디더, 토큰 축 노이즈 셰이핑

- **단계**: 서빙(KV) × 신호처리(디더·ΣΔ)
- **가설**
  - V 캐시 오차가 토큰 사이에서 상관되어 있으면 어텐션 가중평균이 오차를 줄이지 못한다. 상관은 상수에 가까운 채널, deadzone(|v| < Δ/2가 0으로 반올림됨), 그룹 공유 스케일에서 생긴다.
  - (토큰, 채널) 해시 시드로 서브트랙티브 디더를 넣으면(저장 0바이트) 오차가 1/N_eff로 평균된다.
  - 토큰 축 1차 ΣΔ는 carry가 머리당 O(d) 하나뿐이다. 2609.39185가 SSM 상태에 대해 지적한 "저장 2배" 반론이 성립하지 않는다.
- **공백 근거**
  - WitCert 2607.28699 [notes]: Lemma 2에서 서브트랙티브 디더를 인증(sub-Gaussian proxy) 용도로만 쓴다. 평균화 이득은 다루지 않는다.
  - 2605.13810 [abs]: 디더 RHT의 비편향·MSE만 다룬다.
  - 2606.02288 [notes]: 키에 거의 상수인 bias 벡터가 있다. 이것이 일관 오차의 원천이다.
  - KV 노이즈 셰이핑 쿼리 2건 모두 0건이었다.
- **수학 핵심(증명 가능)**
  - 출력 오차 e_o = Σ_t p_t e_t이므로 E‖e_o‖² = Σ_{t,s} p_t p_s E⟨e_t, e_s⟩.
  - (a) 서브트랙티브 디더: 오차는 신호와 독립이고 U(±Δ/2) iid다. 따라서 E‖e_o‖² = (dΔ²/12)·Σ_t p_t² = dΔ²/(12 N_eff).
  - (b) RTN, deadzone 상수 채널 v: 오차 −v가 모든 토큰에 공통이므로 |e_o| = |v|·Σp_t = |v|, N_eff와 무관하다. 디더가 이기는 조건은 N_eff > Δ²/(12v²)다.
  - (c) 토큰 축 ΣΔ(u_t = v_t + ε_{t−1}, ŷ_t = Q(u_t)): e_t = ε_{t−1} − ε_t이므로 e_o = Σ_t ε_t(p_{t+1} − p_t)이고 E‖e_o‖² = (dΔ²/12)Σ_t(p_{t+1} − p_t)².
    - 매끄러운 어텐션에서는 디더보다 작다.
    - 뾰족한 어텐션에서는 최대 2배 나쁘다.
  - 헤드별 선택 규칙: Σ(Δp)²와 Σp²를 비교한다.
  - K에는 적용하지 않는다. 공통 모드 로짓 이동은 softmax 불변이며 SageAttention의 K-smoothing이 이미 쓴다.
- **판정 실험(≈1 GPU·주)**
  - 모델: Llama-3.1-8B, Qwen2.5-7B
  - 설정: V INT2/INT3(per-token, per-channel g=128) × {RTN, 디더, 토큰-ΣΔ, 헤드별 선택}. K는 KIVI-2로 고정한다.
  - 측정: 어텐션 출력 MSE를 헤드별로 측정하고 N_eff·TV 예측과의 상관을 본다. 그 위에 LongBench, RULER, GSM8K-CoT decode를 측정한다.
  - 실패 기준(둘 중 하나라도 해당하면 기각)
    - 디더/ΣΔ의 출력 MSE 이득이 예측식과 상관 ρ < 0.5
    - 과제 지표 개선이 RTN 대비 1pt 미만
- **최근접 선행 연구와 차이**
  - WitCert: 디더를 인증 용도로만 쓴다.
  - 2609.39185: 재귀 상태의 시간 평균이다. 저장 2배 문제가 있다.
  - KVarN: 분산 정규화이며 상관 구조는 다루지 않는다.
  - SageAttention: K 공통 모드만 다룬다.
- **쿼리**
  - W1 `abs:"KV cache" AND abs:(dither OR dithering OR "stochastic rounding")`
  - W2 `abs:(quantization OR quantized) AND abs:attention AND abs:(averaging OR "error averaging") AND abs:(correlated OR coherent OR independent) AND abs:value` (date)
  - W3 `abs:"value cache" AND abs:quantization AND abs:(bias OR correlated)` (date)
  - W4(인접) `abs:dither AND abs:(decorrelate OR decorrelation OR "signal-independent") AND abs:(neural OR transformer OR attention)`
  - W5 `abs:"KV cache" AND abs:("noise shaping" OR "sigma-delta" OR "error diffusion" OR "error feedback")`
  - W6 `... "error feedback" ... token ... attention` (date)

## A3. 장문맥 검색의 극값 법칙: KV 비트 요구량은 문맥 길이에 대해 ½·log₂ ln N으로 자란다

- **단계**: 장문맥 × KV 양자화 × 통계(EVT)
- **가설**: needle 검색(argmax 또는 top-k 판독) 실패의 기준은 평균 MSE가 아니다. 기준은 극값이다. N개 방해 키의 잡음 섞인 로짓 최대값이 needle 로짓을 넘으면 실패한다.
- **공백 근거**
  - WitCert 2607.28699 [notes]: raw fp8 KV에서도 hard RULER 점수가 22.8이고, 게이팅으로 79.7까지 회복한다. fp8에서 생기는 이 큰 붕괴는 길이 의존 이론 없이 설명되지 않는다.
  - Block-GTQ 2606.24033 Lemma 2 [notes]: 로짓 오차 상한에 길이 항이 없다.
  - N1–N5 쿼리에서 길이-비트 법칙은 찾지 못했다.
- **수학 핵심(가정 하 증명 가능)**
  - 가정
    - 방해 로짓 s_j iid N(μ, s²)
    - 양자화 로짓 잡음 ε_j = q·e_j iid N(0, σ²), σ² = Σ_c q_c² Δ_c²/12
    - needle 로짓 s*
  - 결과(Gumbel 극값)
    - 성공 확률 → Φ((s* − μ)/√(s²+σ²) − √(2 ln N)) (선두 차수)
    - 실패 시작 길이 ln N*(σ) = (s*−μ)²/(2(s²+σ²))
    - σ ∝ 2^{−b}를 넣으면 ln N*(b) = A/(1 + c·4^{−b}). 1비트를 늘리면 N*가 거의 N*^{(1+c4^{−b})/(1+c4^{−b−1})}이 된다.
  - softmax 질량(합) 판독의 −σ²/2 편향은 선점(2605.26266)이므로 쓰지 않는다. 이 아이디어는 max/순위 판독만 다룬다.
  - 상관된 방해 키에 대해서는 유효 표본수 N_eff로 대체한다(가정으로 명시).
- **방법**: 목표 문맥 길이를 받는 retrieval-head 비트 할당. b_h = ½log₂(c_h/(A_h/ln N − 1))이고, 짧은 문맥에서 (A_h, c_h)를 적합한다.
- **판정 실험(≈1 GPU·주)**
  - Llama-3.1-8B, Qwen2.5-7B-1M; 합성 needle과 RULER NIAH; 4K–128K; K 2/3/4/8bit와 fp8
  - 4K–16K에서 적합한 법칙으로 32K–128K의 실패 시작 길이를 예측한다.
  - 실패 기준(둘 중 하나라도 해당하면 기각)
    - 예측 N*가 2배 넘게 빗나감
    - 고정 비트에서 정확도가 ln N에 대해 단조 감소하지 않음
- **최근접 선행 연구와 차이**
  - Block-GTQ: RoPE 대역 할당이며 길이 무관이다.
  - WitCert: 최악 TV 상한만 있다.
  - DuoAttention·RazorAttention: retrieval head를 full cache로 둔다. 비트 법칙은 없다.
  - SSMax 2501.19399: 로짓을 log n으로 스케일하면 잡음도 같이 스케일되므로 SNR 이득이 없다는 예측을 부산물로 낸다.
- **쿼리**
  - N1 `abs:"KV cache" AND abs:quantization AND abs:("context length" OR "long context" OR long-context) AND abs:(scaling OR law OR logarithmic OR "extreme value")` (date)
  - N2 `abs:(quantization OR quantized) AND abs:(needle OR retrieval) AND abs:(margin OR distractors) AND abs:attention`
  - N3(인접 EVT) `abs:attention AND abs:(dilution OR fading OR "scalable softmax" OR "log n") AND abs:(length OR distractors) AND abs:(noise OR precision OR quantization)` (date ≥20250101)
  - N4 `abs:("attention temperature" OR "scalable-softmax" OR "SSMax" OR "logit scaling") AND abs:(quantization OR quantized OR "low-bit")`
  - N5 `... "KV cache" ... (retrieval OR needle) ... (length OR "128K" OR "long context") ... degradation` (date)

## A4. 양자화된 생성기는 zero-shot 탐지를 회피하고 워터마크 탐지는 강화한다: 엔트로피·KL 항등식

- **단계**: 서빙 양자화 → AI 텍스트 탐지·워터마크(안전/출처)
- **가설**
  - 탐지기의 채점 모델은 FP다. 생성은 양자화 모델이 한다.
  - 이때 Fast-DetectGPT 통계량의 기대 이동은 ΔH − KL로 정확히 정해진다. 양자화가 엔트로피를 올리면 탐지율이 떨어진다.
  - 같은 ΔH가 KGW류 워터마크의 녹색 토큰 기대 비율 하한(spike entropy)을 올린다. 따라서 두 탐지 체계는 반대 방향으로 움직인다.
- **공백 근거**
  - B1–B6 쿼리와 WebSearch에서 양자화 생성기의 탐지율을 직접 측정한 연구를 찾지 못했다. WebSearch 결과에서 가장 가까운 언급은 2605.14240(패러프레이즈 공격 연구)의 부수적 언급이다. 이 논문은 Falcon-7B를 양자화했을 때 약 2% 열화를 보고하고 "unexpected"라고 적었다. 이 주장은 WebSearch 결과 요약을 거친 것이며 원문으로 확인하지 않았다.
  - 2606.00206 [notes]: 위치별 KL(FP‖Q)과 FP 엔트로피의 Spearman ρ = 0.92. 엔트로피가 높은 위치에서 KL이 크다.
- **수학 핵심(정확한 항등식, 증명 가능)**
  - 토큰 t에서 채점 모델은 p, 실제 생성 분포는 q̃(양자화 모델에 디코딩을 적용한 것)다.
  - E_{x~q̃}[log p(x)] − E_{x̃~p}[log p(x̃)] = H(p) − H(q̃) − KL(q̃‖p)
  - 따라서 Fast-DetectGPT 분자의 기대 이동은 Σ_t[H(p_t) − H(q̃_t) − KL(q̃_t‖p_t)]이다. Binoculars도 같은 꼴(교차 perplexity 비)이다.
  - KGW: 녹색 리스트 편향 δ에서 기대 녹색 비율 하한은 생성 분포의 spike entropy에 대해 단조 증가한다(Kirchenbauer et al. 2301.10226의 정리). 따라서 H(q̃) > H(p)이면 z-점수가 오른다.
- **방법**: precision-marginalized 채점. 배포 비트 분포 π(b)에 대해 μ̃를 E_b[−H(p_b)]로 보정한다. 탐지기 쪽 보정식을 닫힌형으로 준다.
- **판정 실험(≈1 GPU·주)**
  - 생성기: Llama-3.1-8B-Instruct, Qwen2.5-7B를 BF16, W8, W4(AWQ/GPTQ), W3, KV4로 돌린다. 프롬프트 2k개(XSum, WritingPrompts, PubMedQA).
  - 탐지기: Fast-DetectGPT, Binoculars(FP 채점), KGW(γ = 0.25, δ = 2).
  - 측정: 토큰별 ΔH와 KL을 측정해 AUROC 이동을 예측하고 실측과 비교한다.
  - 실패 기준(둘 중 하나라도 해당하면 기각)
    - 예측한 통계량 이동과 실측 이동의 상관 < 0.7
    - W4에서 AUROC 이동이 0.01 미만
- **최근접 선행 연구와 차이**
  - Fast-DetectGPT 2310.05130, Binoculars 2401.12070: 생성기와 채점기의 정밀도 불일치를 다루지 않는다.
  - 모델 대체 감사 2504.04715, IRIS 2607.20860: 출력 분포로 양자화 대체를 탐지하는 것이 목적이다. 텍스트 출처 탐지가 아니다.
- **쿼리**
  - B1 `abs:quantization AND abs:detection AND (abs:"machine-generated" OR abs:"AI-generated" OR abs:DetectGPT OR abs:Binoculars)`
  - B2 `abs:quantized AND (abs:"machine-generated text" OR abs:"AI-generated text" OR abs:"zero-shot detection")` (date)
  - B3 `abs:(quantization OR quantized OR compressed) AND abs:watermark AND abs:(LLM OR "language model")` (date)
  - B4(인접) `abs:detector AND abs:"source model" AND abs:mismatch AND abs:(entropy OR KL) AND abs:text`
  - B5 `... (detectability OR detectors) AND (evade OR evasion OR robustness) AND text`
  - B6 `ti:(detection OR detector OR detecting) AND abs:(quantized OR quantization) AND abs:(generated OR LLM-generated)` (date ≥20250101)
  - WebSearch 1회

## A5. 양자화된 보상모델의 Best-of-N: 기준은 RewardBench 정확도가 아니라 프롬프트 내 상관 ρ_within과 오차의 꼬리 지수

- **단계**: 양자화 → 정렬(RM, BoN, RLHF)
- **가설**
  - 양자화 RM의 BoN 후회(regret)는 (1 − ρ_within)σ_r√(2 ln N)으로 자란다. ρ_within은 같은 프롬프트의 N개 on-policy 응답 안에서 잰 진짜 보상과 양자화 보상 사이의 상관이다.
  - 쌍 정확도(RewardBench)는 여백(margin)이 큰 쌍으로 측정하므로 ρ_within을 과대평가한다.
  - 양자화 오차가 활성 이상치에서 오면 꼬리가 두껍다. 이 경우 큰 N에서 gold 보상이 오히려 감소한다(catastrophic Goodhart).
- **공백 근거**
  - J1–J5 쿼리와 WebSearch 모두에서 양자화 RM의 BoN 또는 과최적화 연구가 0건이었다.
  - Kwa et al. 2407.14503(꼬리가 두꺼운 오차에서 최적화 이득이 0으로 감)과 Gao et al. 2210.10760(RM 과최적화 법칙)은 오차의 원천이 양자화인 경우를 다루지 않는다.
- **수학 핵심(증명 가능)**
  - (i) 가우시안, ε ⊥ r: E[r(x̂_N)] − E[r] = ρσ_r E[Z_(N)], ρ = σ_r/√(σ_r² + σ_ε²) (회귀 Goodhart). 분산은 모두 프롬프트 내 분산이다.
  - (ii) 쌍 정확도 = Φ(m/√2σ_ε)는 여백 분포에 의존한다. 프롬프트 내 보상 간격이 작으면 같은 σ_ε에서도 ρ_within이 작다.
  - (iii) ε의 꼬리가 r보다 두꺼우면 E[r(x̂_N)] → E[r] (N → ∞). Kwa et al.의 정리를 적용한 것이다.
- **판정 실험(≈1 GPU·주)**
  - RM: Skywork-Reward-V2-Llama-3.1-8B 계열을 BF16, W8A8, W4(GPTQ), W3로 둔다.
  - 정책: Llama-3.1-8B-Instruct, 프롬프트 500개, N ≤ 256
  - gold 기준: BF16 RM과 더 큰 gold RM
  - 측정: ρ_within, ε의 Hill 꼬리 지수, BoN 곡선, RewardBench 정확도
  - 실패 기준(둘 중 하나라도 해당하면 기각)
    - 후회가 (1 − ρ_within)√(2lnN) 예측에서 30% 넘게 벗어남
    - RewardBench 정확도가 BoN 후회를 ρ_within만큼 잘 예측함
- **최근접 선행 연구와 차이**
  - 2511.15694: RL 정책의 양자화를 다룬다(RM 아님).
  - 2603.05739: BoN 준최적성 일반론이다.
  - 2608.30051: PRM 과최적화를 다룬다(양자화 아님).
- **쿼리**
  - J1 `abs:(quantized OR quantization) AND abs:"reward model" AND abs:(best-of-n OR "best-of-N" OR overoptimization OR hacking)`
  - J2 `abs:(quantized OR quantization OR "low-bit") AND abs:("reward model" OR verifier OR "process reward") AND abs:(LLM OR RLHF)` (date)
  - J3(인접) `abs:Goodhart AND abs:(quantization OR compressed OR compression OR distillation) AND abs:reward`
  - J4 `abs:"reward model" AND abs:(compression OR quantized OR pruned OR "low precision") AND abs:(robustness OR fidelity)` (date)
  - J5 `abs:("reward model" OR "reward models") AND abs:(quantization OR quantized OR "4-bit" OR "8-bit")`
  - WebSearch 1회

## A6. KL 앵커의 양자화: 양자화된 reference/teacher는 길이에 비례하는 잡음 보상 shaping이다

- **단계**: QAT/PTQ → RL(KL 정규화 RLHF, GRPO-KL, on-policy distillation)
- **가설**
  - 메모리를 아끼려고 reference 모델이나 distillation teacher를 저비트로 두면, 목적함수는 FP 앵커 목적함수에 보상 βη를 더한 것과 정확히 같다. η = Σ_t[log π_ref^q − log π_ref]이다.
  - η의 표준편차는 √T로 자란다. 따라서 긴 CoT에서는 W4 앵커가 이진 보상과 같은 크기의 잡음을 주입하고, FP8 앵커는 무해하다.
  - η의 평균 부분은 양자화가 평탄화한 토큰 쪽으로 엔트로피를 올린다.
- **공백 근거**
  - FP8 RL 스택 2601.18150, 2609.22870 [V]: 롤아웃·학습 불일치만 다룬다. 2609.22870 l.2507: "We do not apply any KL divergence loss between the actor and reference models".
  - 2512.21852(KL 추정량 비교): 정확한 reference를 가정한다.
  - R1–R5 쿼리와 WebSearch: 양자화 reference 연구 0건.
- **수학 핵심(정확한 항등식과 따름정리)**
  - KL(π‖π_ref^q) = KL(π‖π_ref) + E_π[log π_ref − log π_ref^q]
  - 따라서 max E_π[r] − βKL(π‖π_ref^q) ≡ max E_π[r + βη] − βKL(π‖π_ref)이고, 최적해는 π* ∝ π_ref^q e^{r/β}다.
  - 따름정리 1: 토큰별 η_t가 약한 의존이면 Var(βη) ≈ β²T·Var(η_t). 그룹 상대 advantage에서 신호 대 잡음비는 SNR = σ_r/(β√T sd(η_t))다.
  - 따름정리 2: on-policy distillation(보상 log π_T^q − log π_S)에서도 같은 η가 더해진다.
  - k3 추정량(loss로 넣는 형태)의 기울기 편향은 2512.21852의 분석과 결합해 따로 유도해야 한다. 이 부분은 미증명이다.
- **판정 실험(≈1–1.5 GPU·주)**
  - Qwen3-1.7B GRPO, MATH/GSM8K, β ∈ {0.01, 0.04}
  - reference: BF16, FP8, W4-RTN, W4-GPTQ; 시드 3개, 200 step
  - 측정: 정확도, 엔트로피, BF16 ref 기준 KL. 사전 측정한 sd(η_t)로 SNR 붕괴 지점을 예측한다.
  - 추가 arm: W4 teacher on-policy distillation
  - 실패 기준(둘 중 하나라도 해당하면 기각)
    - 예측 SNR < 1인 설정에서 BF16 ref 대비 정확도 또는 엔트로피 차이가 시드 잡음 이내
    - 예측 SNR > 5인 설정(FP8)에서 차이가 남
- **최근접 선행 연구와 차이**
  - AIS 2605.13907, CIS 2609.32444, QUADS 2607.15810: 롤아웃 불일치의 IS를 다룬다. 앵커 양자화가 아니다.
  - 2608.01743: KL 정규화 설계를 다룬다(양자화 아님).
- **쿼리**
  - R1 `abs:"reference model" AND abs:(quantized OR quantization OR FP8 OR "low-precision") AND abs:(KL OR RLHF OR GRPO OR PPO)`
  - R2 `... ("KL penalty" OR "KL regularization" OR "KL divergence") ... reference` (date)
  - R3(인접) `abs:"KL estimator" AND abs:(bias OR biased) AND abs:(reference OR prior) AND abs:(policy OR RL)` (date)
  - R4 `abs:(memory-efficient OR "memory efficient") AND abs:RLHF AND abs:(quantize OR quantized OR quantization) AND abs:("reference" OR "critic" OR "reward model")`
  - R5 `abs:("reference policy" OR "reference model") AND abs:(precision OR quantization OR "8-bit") AND abs:("entropy" OR drift) AND abs:(alignment OR RL)` (date)
  - WebSearch 1회

## A7. 분포 무관 충실도 보장을 갖는 정밀도 게이팅: 최악 경우 인증서 대신 conformal risk control

- **단계**: 서빙 × 통계
- **가설**
  - 저비트 패스의 값싼 점수 S(x)로 요청 단위 또는 토큰 이후 고정밀로 승격한다. S(x)는 토큰 여백, 엔트로피, WitCert 밴드별 잔차 노름 Σ_b‖q_b‖w_{t,b} 등이다.
  - 이렇게 하면 "FP 답과 달라진 응답 비율 ≤ α"를 유한 표본으로 보장하면서 승격률을 최소화할 수 있다.
  - 1–2bit에서 WitCert식 최악 경우 상한은 무의미해지지만 CRC 보장은 유지된다.
- **공백 근거**
  - WitCert 2607.28699 [notes]: Tier-A 상한이 1–2bit에서 수 자릿수 무의미(vacuous)하다. 확률적 인증은 비적응적 쿼리에서만 성립한다고 저자가 명시한다.
  - 2606.01850 [abs]: 압축 모델의 conformal coverage 보존을 벤치마크한다. 라우팅이나 보장은 아니다.
  - H1–H4 쿼리: 양자화 cascade의 위험 통제 연구 0건.
- **수학 핵심**
  - CRC(Angelopoulos et al. 2208.02814): 손실 L_i(λ) = 1{불일치 ∧ 미승격}이 λ에 대해 단조이면 λ̂ = inf{λ: (n/(n+1))R̂_n(λ) + 1/(n+1) ≤ α}가 E[L_{n+1}(λ̂)] ≤ α를 보장한다(교환가능성 가정).
  - 이 연구에서 새로 증명할 부분:
    - (a) 토큰 이후 승격(prefix를 저비트로 생성한 뒤 FP로 이어감)에서도 손실이 단조임을 보이는 보조정리. 승격 시점이 λ에 대해 단조여야 한다는 조건이 필요하다.
    - (b) 정밀도 혼합 분포에서 가중 conformal로 공변량 이동을 보정한다.
  - 정리 자체는 차용이다. 기여는 교차 적용, 손실 설계, 점수 설계다. main-track 위험 요소로 명시한다.
- **판정 실험(≈1 GPU·주)**
  - Llama-3.1-8B: W4A4KV4 → BF16 승격
  - 과제: GSM8K, MATH, RULER, MT-Bench 판정
  - 점수 5종: 여백, 엔트로피, WitCert 잔차, KL probe, 학습형
  - 측정: α ∈ {0.01, 0.05}에서 승격률. WitCert 게이팅과 동일 위험에서 비교한다.
  - 실패 기준(둘 중 하나라도 해당하면 기각)
    - 실측 위반률 > α + 0.01
    - 최선 점수의 승격률이 무작위 승격의 0.8배 이상(이득 없음)
- **최근접 선행 연구와 차이**
  - WitCert: 결정적 상한 게이팅이다.
  - VeriCache 2605.17613: 무손실 검증으로 매 토큰 FP를 확인한다. 비용 구조가 다르다.
  - 2605.20868: 런타임 인증 어텐션이다.
  - FrugalGPT식 cascade: 보장이 없다.
- **쿼리**
  - H1 `abs:(conformal OR "risk control" OR "learn then test") AND abs:(quantized OR quantization OR precision) AND abs:(cascade OR routing OR deferral) AND abs:(LLM OR "language model")` (date)
  - H2 `abs:(quantized OR quantization) AND abs:(cascade OR escalation OR fallback) AND abs:(full-precision OR "full precision") AND abs:(LLM OR "language model")` (date)
  - H3(인접 통계) `abs:"conformal risk control" AND abs:(compression OR "model cascade" OR "efficient inference")`
  - H4 `abs:(mixed-precision OR "dynamic precision") AND abs:"per-request" AND abs:(guarantee OR certified)`

## A8. 문항·토큰 수준 probit 손상 법칙: "기존 실패 증폭, 새 실패 없음"(2607.27275)과 "저엔트로피 토큰 집중"(ReQAT)의 모순 해소

- **단계**: 양자화 → 평가 과학, 에이전트, 추론(통계/IRT 수입)
- **가설**
  - 양자화는 문항 i의 잠재 여백 μ_i를 δ_i ~ N(−b, σ²)만큼 이동시킨다. δ_i는 μ_i와 독립이다(등분산).
  - 그러면 FP 통과율 p_i와 양자화 통과율 p'_i는 probit 공간에서 Φ^{-1}(p'_i) = Φ^{-1}(p_i) + δ_i/s를 만족한다.
  - 이 등분산 법칙은 2607.27275의 관찰(기존 실패 최대 2.5배 증폭, 새 실패 0.18%)을 재현한다.
  - 같은 법칙은 고확신(저엔트로피) 토큰이 가장 안전하다고 예측한다. 따라서 ReQAT의 "FP4 실패는 저엔트로피 토큰에 집중"이 사실이면 잡음이 이분산이어야 한다(숫자·연산자 토큰에서 σ_i가 크다). 이 법칙은 그 이분산성을 직접 검정하는 도구가 된다.
- **공백 근거**
  - 2607.27275 [abs]: 위 수치를 보고했다. 기작 모형은 없다.
  - ReQAT 2606.15682 [abs, ICML 2026]: 저엔트로피 토큰 집중을 주장한다.
  - P1–P4 쿼리: IRT/probit 연구 0건.
- **수학 핵심(증명 가능)**
  - p_i = Φ(μ_i/s), p'_i = Φ((μ_i + δ_i)/s). 이로부터:
    - (a) 기대 통과율 E[p'_i | p_i] = Φ((Φ^{-1}(p_i)·s − b)/√(s² + σ²))
    - (b) FP에서 p_i ≥ 1 − ε인 문항의 신규 실패 비율에 대한 닫힌형 상한
    - (c) maj@k 정확도는 1{μ_i + δ_i > 0}의 함수다. 따라서 maj@k 손실 = P(부호 반전)이고 p ≈ 0.5인 문항에 집중된다.
    - (d) 토큰 수준에서도 같은 식이 성립한다. 이분산 검정은 probit 잔차 분산이 토큰 클래스나 엔트로피 bin에 따라 달라지는지 보는 우도비 검정이다.
- **판정 실험(≈1 GPU·주)**
  - 모델: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct
  - 과제: GSM8K, MATH500, τ²-bench 일부; 문항당 32 샘플
  - 설정: BF16, W8, W4(GPTQ/AWQ), W3, NVFP4 W4A4
  - 방법: 무작위 효과 probit 모형으로 (b, σ/s)를 최대우도 적합한 뒤, 보류한 비트의 문항별 통과율을 예측한다.
  - 실패 기준(둘 중 하나라도 해당하면 기각)
    - Brier 개선이 "p' = p − 상수" 기준선 대비 10% 미만
    - 등분산이 p-bin에서 기각되는데(χ² p < 0.01) 이분산 확장도 ReQAT 패턴을 재현하지 못함
- **최근접 선행 연구와 차이**
  - "Accuracy is not all you need" flips 2407.09141: 반전을 집계만 한다.
  - 2609.18005 [abs]: 측정 도구를 제안한다.
  - 2607.27275, ReQAT: 관찰만 있고 공통 모형이 없다.
- **쿼리**
  - P1 `abs:(quantization OR quantized) AND abs:(item-level OR per-item OR "per-sample" OR flips) AND abs:(pass OR accuracy) AND abs:(probit OR "item response" OR difficulty)` (date)
  - P2 `abs:"item response theory" AND abs:(quantization OR compression OR quantized)`
  - P3 `abs:(quantization OR quantized) AND abs:(flips OR "answer flips" OR "per-question") AND abs:(LLM OR "language model") AND abs:(difficulty OR "pass rate" OR uncertainty)` (date)
  - P4(인접 심리측정) `abs:(probit OR "latent trait" OR psychometric) AND abs:(LLM OR "language model") AND abs:(perturbation OR compression OR "model variants")`

## A9. 학습 정밀도의 강건화는 포맷 사이에서 전이되는가: 오차 공분산 정렬 정리

- **단계**: 사전학습 포맷(FP8/NVFP4/MXFP4/INT-QAT) → 다른 포맷의 PTQ
- **가설**
  - 저정밀 학습의 "robustification"(Kumar et al.)은 학습 포맷과 배포 포맷의 오차 공분산이 정렬된 만큼만 전이된다.
  - FP 계열 학습(상대 오차 ∝ |w|)은 INT4 g128 PTQ(절대 오차 균일)에 대한 강건화를 거의 주지 않는다. 반대 방향도 마찬가지다.
  - 실무 사례: MXFP4 네이티브 gpt-oss를 GGUF/INT4로 변환하는 경우, FP8로 학습한 DeepSeek-V3를 INT4로 변환하는 경우.
- **공백 근거**
  - Kumar et al. 2411.04330 [notes]:
    - l.1073: "We assume training precision is strictly greater than inference precision, and define degradation as identically zero if they are equal".
    - integer-type만 다룬다.
    - robustification 효과(l.1064–1072)는 적합식이며 정리가 아니다.
  - I1–I5 쿼리: 포맷 간 전이 연구 0건.
- **수학 핵심(국소 2차 근사 하에서 증명 가능)**
  - 학습 forward 잡음 ε_T의 공분산을 Σ_T(w)라 하자. 정상점은 L̃(w) = L(w) + ½tr(HΣ_T(w))의 최소점 근방이다.
  - 배포 포맷 P의 손실 증가는 δ_P ≈ ½tr(HΣ_P)다.
  - 대각 근사에서 학습이 줄이는 양은 Σ_i H_ii σ_{T,i}²다. 따라서 전이 계수는 τ(T→P) = Corr_H(σ_T², σ_P²)에 비례한다(H-가중 프로파일 상관).
  - 이 계수는 포맷 쌍마다 닫힌형으로 계산된다.
    - INT: Δ_g²/12
    - FP: |w|²·2^{−2m}·c
    - NVFP4/MXFP4는 스케일 효과 포함
  - 비국소(basin 이동) 효과는 다루지 못한다.
- **판정 실험(≈2 GPU·주, 범위 상한)**
  - 125M/350M 모델을 10B 토큰으로 학습한다: BF16, NVFP4-fwd 에뮬레이션, INT4-g128 fake-quant fwd 3개 arm
  - 각 arm을 INT4 g128, INT3, NVFP4, MXFP4로 PTQ한 4×3 전이 행렬을 측정하고 예측 τ 행렬과 비교한다.
  - 실패 기준(둘 중 하나라도 해당하면 기각)
    - 전이 행렬이 거의 균일함(행 내 변동 < 20%)
    - 예측과의 순위 상관 < 0.5
- **최근접 선행 연구와 차이**
  - Kumar 2411.04330: 같은 integer-type만 다룬다.
  - 2606.09012 [notes]: QAT basin 이론이며 포맷 간 전이는 없다.
  - Matryoshka/any-precision: 같은 계열 내부의 비트 중첩이다.
- **쿼리**
  - I1 `abs:(FP4 OR FP8 OR "low-precision" OR "low precision") AND abs:(pretraining OR "pre-training" OR trained) AND abs:"post-training quantization" AND abs:(robust OR robustness OR friendly)` (date)
  - I2 `abs:(quantization OR quantized) AND abs:"training format" AND abs:(transfer OR mismatch OR deployment)`
  - I3 `abs:"natively trained" AND abs:(FP4 OR NVFP4 OR MXFP4 OR FP8) AND abs:quantization` (date)
  - I4(인접) `abs:"noise injection" AND abs:training AND abs:(covariance OR anisotropic) AND abs:(quantization OR "weight perturbation") AND abs:robustness`
  - I5 `abs:(QAT OR "quantization-aware") AND abs:(format OR formats) AND abs:(mismatch OR cross-format OR "different format" OR transfer) AND abs:(LLM OR "language model")` (date)

## A10. 사실 빈도와 양자화 취약성: "더 학습할수록 PTQ에 약해진다"(Ouyang/Kumar)와 여백 직관의 조성 효과에 의한 화해

- **단계**: 사전학습 데이터 통계 → PTQ(연상 기억 모델 수입)
- **가설**
  - 학습 데이터에 자주 나온 사실일수록 고정 체크포인트에서 여백이 커서 양자화에 강하다.
  - 그런데도 학습이 진행될수록 집계 QiD가 커지는 것(Ouyang 2411.17691, Kumar 2411.04330)은 조성(composition) 효과다. 간섭을 정밀하게 상쇄해야 겨우 기억되는 희귀 사실이 새로 기억 집합에 들어오기 때문이다.
  - 따라서 QiD(t) = Σ_f n_mem(f, t)·P(flip | f, t)에서 빈도 bin 내부의 P(flip)는 학습 시간에 대해 감소하거나 평탄하다. 증가분은 n_mem의 희귀 bin에서 온다.
- **공백 근거**
  - Ouyang 2411.17691 [notes]: "undertrained" 해석은 정보병목 비유에 기댄 것이고 검증은 없다.
  - Kumar App. H [notes]: "speculative"라고 명시한다.
  - Q1, Q2, Q5 쿼리: LLM에서 사실 빈도를 축으로 PTQ 취약성을 본 연구 0건. 비전에서는 Hooker 2010.03058, 1911.05248이 압축이 롱테일 표본을 불균형하게 해친다고 보고했다.
- **수학 핵심(장난감 모형에서 증명 가능)**
  - 모형: 연상 기억 W = Σ_x α_x(t) e_{y(x)}u_x^T. u_x는 거의 직교하는 임베딩(d < 사실 수), α_x(t)는 교차엔트로피 GD로 학습되며 빈도 f_x에 따라 증가한다.
  - 사실 x의 여백은 m_x = α_x − 간섭 I_x다. 양자화 잡음 ΔW(항목별 분산 σ²)는 로짓 잡음 σ‖u_x‖√V_eff를 준다.
  - 명제:
    - (a) 고정 t에서 P(flip | x) = Φ(−m_x/σ_eff)는 f_x에 대해 감소한다.
    - (b) 용량 한계에서 새로 기억되는 사실의 여백은 0⁺에서 시작한다. 따라서 집계 QiD는 t에 대해 증가할 수 있다.
  - 실제 LLM에 대한 일반 정리는 아니다. 실증으로 판정한다.
- **판정 실험(≈1 GPU·주)**
  - 모델: Pythia-1.4B/2.8B deduped 중간 체크포인트 8개(Pile에서 infini-gram으로 빈도 계산), OLMo-2-1B 체크포인트(Dolma)
  - 사실 집합: PopQA와 LAMA T-REx
  - 설정: W3/W4 RTN·GPTQ
  - 측정: 빈도 bin × 체크포인트별로 FP에서 맞힌 사실의 반전율을 재고, 조성 분해를 한다.
  - 실패 기준(둘 중 하나라도 해당하면 기각)
    - 고정 체크포인트에서 반전율이 빈도에 대해 감소하지 않음(Spearman > −0.2)
    - QiD 증가의 50% 미만만 조성 항으로 설명됨
- **최근접 선행 연구와 차이**
  - Hooker et al.: 비전 롱테일을 다루며 학습 시간 축과 조성 분해가 없다.
  - Ouyang/Kumar: 집계 법칙만 있다.
  - PopQA(Mallen et al. 2212.10511): 인기도와 정확도의 관계를 보며 양자화는 다루지 않는다.
  - 2608.11786(언어 조건부 역양자화): 언어 축이며 빈도 이론은 없다.
- **쿼리**
  - Q1 `abs:(quantization OR quantized OR compression) AND abs:(long-tail OR "long tail" OR rare OR frequency) AND abs:(knowledge OR facts OR entities) AND abs:(LLM OR "language model")` (date)
  - Q2 `abs:(quantization OR quantized) AND abs:("pretraining data" OR "pre-training data" OR "training data frequency" OR "infini-gram") AND abs:(knowledge OR recall)`
  - Q3(인접) `abs:"associative memory" AND abs:(frequency OR Zipf) AND abs:(noise OR quantization OR precision) AND abs:(transformer OR "language model")`
  - Q4 `abs:(quantization OR "low-bit") AND abs:(multilingual OR "low-resource") AND abs:(degradation OR disproportionate) AND abs:(frequency OR data)` (date)
  - Q5 `abs:(quantization OR pruning OR compression) AND abs:(popularity OR PopQA OR "entity popularity" OR "less popular")`

---

## 냉정한 총평

- **main-track 가능성이 비교적 높은 것**
  - A1: 기작, 정리, 방법을 모두 갖춘다. 단 선형 모형이 실제를 설명하지 못할 위험이 있다.
  - A2: 정리가 깨끗하고 판정 실험이 싸다. 단 실제 V 오차가 이미 충분히 비상관이면 이득이 작다.
  - A6: 항등식은 자명하지만, 실무 설정(FP8/W4 앵커)에서 결정적 실패를 보이면 영향이 크다. 단 DAPO류 KL 없는 레시피가 늘고 있어 적용 범위가 좁다.
- **중간**
  - A3: 극값 법칙. iid 가정이 약점이다.
  - A5: RM의 BoN.
  - A8: probit 법칙. 평가 과학 성격이다.
  - A10: 조성 분해. 데이터 빈도 계산의 비용과 잡음이 크다.
- **약함·위험**
  - A4: 항등식이 단순하다. 기여는 측정과 보정이다.
  - A7: 정리가 차용이다.
  - A9: 국소 2차 근사가 QAT basin 이동을 놓칠 수 있다.
- **공통 위험**: 이 분야의 선점 속도는 몇 달 단위다(0.4절). 위 쿼리는 2026-10-11 시점의 arXiv API 결과이며, Semantic Scholar 인용 이웃은 확인하지 못했다(429).
