## 아이디어 7. 양자화 셀 정리의 두 영역(FP base 비편향 대 on-grid base 삭제)과 비밀 시드 subtractive dither 방어

### 가설
후학습 델타 D(|d| ≪ Δ)를 재양자화한 D̃ = Q(W+D) − Q(W)는 기저 W에 따라 두 영역으로 갈린다.
- **(i) FP base를 직접 양자화하는 경우.** 셀 내 오프셋 u가 균일하다고 보면 D̃는 비편향이고 분산은 Δ|d| − d²이다.
- **(ii) on-grid base(u≡0)인 경우.** QLoRA int4와 NVFP4 체크포인트가 여기에 속한다. |d|<Δ/2이면 D̃는 결정론적으로 삭제된다(편향 −d).

LoRA 머지 실패 수치는 (ii)에서 나왔다. unlearning 역전은 (i)에서 일어난다. 그런데 (i)에서는 1차 효과가 기대값으로 보존되므로, 등방 SNR ‖D‖_F²/(Δ‖D‖_1) ≪ 1만으로는 역전을 설명할 수 없다. 설명 후보는 matched-input SNR_x와 곡률 항이다.

비밀 시드 subtractive dither는 원리적 무력화가 아니다. 공격자가 노리는 대상을 결정론적 실현값에서 노이즈 분포로 바꾸도록 강제할 뿐이다.

### 문헌상 공백
- **정리 7.1의 확률 부분은 새 결과가 아니다.**
  - 교차 확률 |d|/Δ, 비편향성, 분산 Δ²p(1−p)는 SR 항등식이다 ([1502.02551](https://arxiv.org/abs/1502.02551), [2207.10321](https://arxiv.org/abs/2207.10321) [S]).
  - 균일 위상에서 교차 확률을 거쳐 기대 이동에 이르는 구조는 [2609.21834](https://arxiv.org/abs/2609.21834)의 Model 2·Corollary 6(App. D.3)에 이미 있다 [V]. 같은 논문은 2-bit에서 위상 균일성이 깨진다고 보고한다(Fig. 2).
  - 남는 기여는 이 확률을 SNR로 집계하는 것과 그 예측을 검정하는 것이다.
- **머지(영역 ii).**
  - [2609.18063](https://arxiv.org/abs/2609.18063)(Lin et al. 2026, preprint) [V] §3.3은 int4 base 위 LoRA 델타의 retention을 보고한다. attention 34%, dense 2%, logits 수준 18%이며, 단일 프롬프트에서 두 projection만 측정한 값이다.
  - Scale-QLoRA [2609.04526](https://arxiv.org/abs/2609.04526) [V]는 cos(Q(W+Δ)−W, Δ)=0.017을 보고하고 이를 "노이즈가 아니다"라고 명시한다. retention은 RTN 15%, GPTQ 94%다.
  - 정리 7.1은 기울기 ≈1을 예측하므로 머지 실패는 노이즈가 아니라 편향이다.
  - 정확한 머지 방법은 이미 있다. QA-LoRA는 그룹 상수 델타를 zero-point에 흡수하므로 구성상 오차가 0이다 ([2309.14717, ICLR 2024](https://arxiv.org/abs/2309.14717) [V]). LoTA-QAF ([2505.18724](https://arxiv.org/abs/2505.18724) [V]), IntLoRA ([2410.21759](https://arxiv.org/abs/2410.21759) [S]), Scale-QLoRA도 있다.
  - 비제약 LoRA를 머지한 뒤 PTQ할 때의 오차 상한은 문헌에 없다.
- **unlearning(영역 i).**
  - [Zhang et al. 2410.16454, ICLR 2025](https://arxiv.org/abs/2410.16454) [V]: 효용 제약 GA/NPO 변형은 MUSE(LLaMA-2 7B/ICLM-7B)에서 잊어야 할 지식을 FP에서 평균 21% 유지하고, 4-bit RTN 후에는 83% 유지한다. 이 집계의 정의는 논문에 없다. 8-bit에서는 FP와 비슷하다. 정리는 없고, Sec 5 Eq.(3)의 비형식 논증("must exceed the quantization step size ∆")만 있다. |Δw|/Δ와 코드 변경률은 측정하지 않았다. SURE(모듈 단위 saliency + 큰 LR)는 VerbMem을 완화하지만, KnowMem에서는 결과가 섞여 있다(NPO_KLR 34.2→36.9, Table 3).
  - 후속 연구 중 셀 논증형: QUAIL([2601.15538](https://arxiv.org/abs/2601.15538) [V])은 Δ/2 논증을 쓰고 버킷 overlap이 99.9%를 넘는다고 보고한다. MANSU([2605.15138](https://arxiv.org/abs/2605.15138) [V])는 Lemma 1 "Any update |∆θ_i| ≥ w_i changes the quantized value"와 magnitude floor를 제시한다.
  - 후속 연구 중 경쟁 설명: DurableUn([2605.02196](https://arxiv.org/abs/2605.02196) [V])은 Prop. 1을 제시하고, INT4에서 최대 22배 복귀하지만 merged NF4에서는 복귀하지 않는다고 보고한다. [2609.27355, COLM 2026](https://arxiv.org/abs/2609.27355) [V]는 Thm 1에서 곡률비로 설명한다.
  - 이 문헌들에는 비편향·분산·SNR 형태의 진술이 없다. 정리 7.1은 FP base에서 "must exceed ∆"와 MANSU의 "silently erased"를 확률적으로 교정한다. 각 원소는 확률 |d|/Δ로 살아남고, 기대값은 보존된다.
  - 일반화에 대한 반례도 있다. AWQ INT4에서 8개 모델 중 7개의 ASR이 FP16 대비 1.6pp 이내다([2606.29581](https://arxiv.org/abs/2606.29581) [S]). LoRA unlearning은 4-bit에서 더 잘 살아남는다([2602.13151](https://arxiv.org/abs/2602.13151) [S]). sign/ZO optimizer는 더 견고하다([2510.00761](https://arxiv.org/abs/2510.00761) [S]).
- **공격.**
  - Egashira et al.은 블록 스케일을 고정한 구간 제약과 PGD로 NF4/LLM.int8()/FP4를 공격한다([2405.18137, NeurIPS 2024](https://arxiv.org/abs/2405.18137) [V]).
  - GGUF k-quant에서는 정확한 preimage 구간을 계산할 수 없어 오차 기반 근사 구간을 쓴다([2505.23786, ICML 2025](https://arxiv.org/abs/2505.23786) [V]).
  - Zhan et al.은 outlier 주입으로 공격을 AWQ/GPTQ/AutoRound까지 확장한다([2605.15152, ICML 2026 workshop](https://arxiv.org/abs/2605.15152) [V]).
- **방어.**
  - Gaussian 노이즈 방어([2108.09187](https://arxiv.org/abs/2108.09187) [S]): Egashira Table 6에서 σ=10⁻³은 효과가 있었지만 σ=10⁻²에서는 HumanEval이 붕괴했다. outlier 공격 앞에서는 σ=10⁻³부터 효용이 떨어지는데도 공격이 유지되었다(2605.15152 Fig. 2). QVec([2606.20254](https://arxiv.org/abs/2606.20254) [V])도 이 방어가 불안정하다고 보고한다.
  - 결정론적 라운딩 재설계: EFRAP([2405.12725](https://arxiv.org/abs/2405.12725) [S]), FlipGuard([2606.28962](https://arxiv.org/abs/2606.28962) [S]), QuantGuard([2606.29239](https://arxiv.org/abs/2606.29239) [V], 적응형 공격자 포함).
  - seeded subtractive dither 자체는 고전적 기법이다([1802.02271](https://arxiv.org/abs/1802.02271), [2607.18745](https://arxiv.org/abs/2607.18745) [S]).
  - 이를 QCB 방어로 쓴 논문이나 구현은 쿼리 80여 개로 검색했지만 찾지 못했다.

### 수학
**정리 7.1 (영역 i, SR 항등식의 기저 위상 적용).** u_ij ~ U[−½,½)가 독립이고 |d_ij| ≤ Δ이면 다음이 성립한다.
- P(셀 변경) = |d_ij|/Δ
- E[D̃_ij] = d_ij
- Var(D̃_ij) = Δ|d_ij| − d_ij²

층 출력은 D̃X = DX + NX이고, N은 평균 0이며 원소가 서로 독립이다. 등방 입력에서 SNR ≈ ‖D‖_F²/(Δ‖D‖_1 − ‖D‖_F²) ≈ mean|d|/Δ다.

몬테카를로(Δ=1, d ~ N(0, 0.05²), 2×10⁶): 회귀 기울기 1.006(비편향), 분산 0.0402 vs 예측 0.0374, SNR 0.0664 vs 예측 0.0669.

**보조정리 7.1′ (영역 ii).** u≡0이면 |d|<Δ/2에서 D̃=0이다(편향 −d). 2-bit처럼 위상이 비균일하면 (i)과 (ii) 사이를 보간한다.

**matched-input SNR.** SNR_x = (Σ_i d_i x_i)² / Σ_i(Δ|d_i| − d_i²)x_i². 델타를 학습시킨 입력에서는 신호가 코히어런트하게 더해지므로 SNR_x는 등방 SNR의 최대 hidden dim배가 된다. 따라서 durability 따름정리는 추측으로 격하한다. **추측 7.2:** 4-bit 복귀율은 SNR_x와 곡률비로 예측된다.

**따름정리 (공격).** |d_ij| < Δ(½ − |u_ij|)이면 Q(W+D)=Q(W)다. 이것이 Egashira의 구간 제약이다. outlier 주입은 Δ를 키워 이 구간을 넓힌다.

**설계 1 (cell-aligned sparse deltas).** D ∈ {−Δ,0,+Δ}^{sparse}는 정수 코드에 무손실로 머지된다. 다만 이것은 LoTA-QAF ternary 머지와 MANSU magnitude floor의 변형이다. 남는 것은 unlearning용 top-s ±Δ proximal step과 1.58-bit 델타 압축 겸용 정도다. 그룹 스케일이 바뀌지 않아야 한다는 조건이 붙는다.

**설계 2 (subtractive dither).** 정의는 Q_u(w) = Δ·round(w/Δ + u) − Δu이고, u ~ U[−½,½)는 비밀 시드에서 그룹별 PRNG로 생성한다. Schuchman의 정리([Schuchman 1964; Gray & Stockham 1993] [U: 고전])에 따라 오차는 입력과 독립인 균일분포를 따르고 MSE는 Δ²/12다. 따라서 배포 모델은 W_fp + E이고 E ⊥ W_fp다.

이 설계는 원리적 무력화가 아니다. E의 분포는 공개되어 있으므로 공격자는 EOT로 노이즈 조건부 유해 모델을 겨냥할 수 있다. 비밀 시드가 막는 것은 특정 실현값을 겨냥하는 공격뿐이다. 예측: 구간형 공격은 무력화된다. outlier형 공격에 대한 효과는 아직 판단할 수 없다.

원안의 "방어자가 샘플링으로 검사 가능"은 결정론적 양자화에서도 똑같이 할 수 있으므로 장점에서 뺐다. 비용도 있다. dequant 단계에서 PRNG를 다시 생성해야 하므로 계산 비용이 0이 아니다. 비감산 SR은 오차가 입력에 의존한다.

증명 가능한 것: 정리 7.1, 보조정리 7.1′, dither의 입력 독립성.

추측:
- 추측 7.2
- merge retention이 RTN ≪ SR/dither < GPTQ 순서인지
- 2-bit에서 dither의 손실
- GPTQ와 dither의 결합

### 판정 실험(1–2 GPU·일)
1. **머지.** Llama-3-8B에 LoRA 어댑터 여러 개를 붙인다. (a) BF16 base에 머지한 뒤 직접 양자화하는 경우와 (b) int4 base에 머지하는 경우를 각각 RTN/SR/dither/GPTQ로 대조한다. 예측: (a)의 기울기는 ≈1이고 분산은 정리 7.1을 따른다. (b) RTN의 retention은 낮다.
2. **unlearning.** FailureLLMUnlearning(MUSE, Llama-2-7B)에서 GA/NPO/SURE의 D를 뽑는다. 먼저 |Δw|/Δ와 코드 변경률을 측정한다. 그다음 4-bit 복귀율을 등방 SNR, SNR_x, 곡률비, optimizer에 회귀한다.
3. **방어.** eth-sri 코드로 Phi-2/Gemma-2b NF4/INT4 공격 모델을 만들고 비밀 dither를 적용한다. 비교 대상은 Gaussian σ=10⁻³, QVec, FlipGuard, QuantGuard다. 공격 측은 EOT 적응형 공격자와 outlier 공격을 쓴다. 탐지 지표로 층별 Δ²/12 대 ‖W_group‖²를 본다. 예측: 구간형 공격의 ASR은 노이즈 기준선까지 떨어지고, 효용 손실은 RTN 대비 1점 이내다.
4. **cell-aligned delta unlearning.** 4-bit 복귀가 0인지 확인하고 MANSU, LoTA-QAF와 비교한다.

### 평가
novelty는 낮다(원안 기준). 정리 7.1은 SR 항등식이고 2609.21834가 선점했다. 설계 1은 LoTA-QAF와 MANSU의 변형이다. "통합"은 머지 영역에서 틀렸다. 실패 수치가 on-grid base의 결정론적 편향에서 나왔기 때문이다.

남는 것은 세 가지다.
- (a) 두 영역 정리와 SNR_x를 durability 예측량으로 검정하는 일
- (b) Zhang이 측정하지 않은 |Δw|/Δ와 코드 변경률을 방법별로 측정하는 일. QUAIL의 버킷 overlap은 측정이 한 번뿐이다.
- (c) 비밀 시드 subtractive dither 방어와 그 적응형 공격 평가. 선행이 없는 것은 이것뿐이며 novelty는 중간이다.

원안 그대로는 메인 트랙에 낼 수 없으므로 부록 A로 내린다. 본문에 남기려면 (c)를 중심으로 하고 (a)를 메커니즘 분석으로 붙인 보안 논문으로 바꿔야 한다. 그 경우 메인 트랙 여부는 경계선이다.

리스크:
- EOT 공격자가 노이즈 조건부 행동을 학습하면 방어의 범위가 구간형 공격 차단으로 줄어든다.
- 2-bit에서의 정확도 손실과 PRNG 커널 비용이 크면 적용 범위를 4-bit 이상으로 한정해 적는다.

---
