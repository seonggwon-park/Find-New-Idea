# Idea 7 판정: 양자화 셀 정리 (requantized delta, SNR, LoRA merge / unlearning 역전 / QCB 공격 통합, subtractive dither 방어)

판정일: 2026-10-11. 다운로드한 논문 본문은 신뢰하지 않는 데이터로 다뤘고, 인용은 원문 그대로 옮겼다. 위치는 scratchpad/papers/<id>/<id>.txt의 줄 번호다.

## 판정 요약

- **novelty: low** (아이디어 원문 그대로 기준). 표에 쓸 말: "낮음(정리·설계1 선점: 2609.21834·LoTA-QAF·MANSU; dither 방어만 중간)".
- **정리 7.1은 새로운 수학이 아니다.** Bernoulli 교차 확률 p=|d|/Δ, E[D̃]=d, Var=Δ²p(1−p)=Δ|d|−d²은 고전적인 stochastic rounding 항등식이다(Gupta et al. 1502.02551, El Arar et al. 2207.10321). 아이디어는 RNG 대신 기저 가중치의 셀 위상을 무작위성의 원천으로 쓴 것뿐이다. 같은 구조(균일 잔차 위상 → 교차 확률 η|u_i|/s_i → 기대 endpoint 이동 = shadow update)는 이미 **2609.21834** Model 2와 Corollary 6(App. D.3, 줄 490–520, 2645–2690)에 증명과 함께 있다. 2609.21834에 없는 것은 분산식, 스칼라 SNR, 그리고 merge·unlearning·attack 연결이다. 분산식은 Bernoulli에서 한 줄이면 나온다.
- **통합 주장은 merge 영역에서 틀렸다.** 아이디어가 근거로 인용한 수치("2–34%만 남는다", 2609.18063 §3.3)는 int4 base, 즉 이미 격자 위에 있는 base(u≡0)에서 측정되었다. 이 영역에서 sub-step 델타는 결정론적으로 지워진다(편향이며, 비편향 노이즈가 아니다). Scale-QLoRA 2609.04526 §5.3은 "reads like quantization noise swamping a small update. It is not."라고 명시하고 cos(Q(W+∆)−W, ∆)=0.017을 보고한다. 정리 7.1이 예측하는 retention 기울기는 ≈1인데 측정값은 0.34/0.02다.
- **설계 1(cell-aligned ±Δ 델타)은 거의 선점되었다.** LoTA-QAF 2505.18724가 정수 코드 위의 ternary Ŵ∈{−1,0,1}을 lossless merge하고 t-SignSGD로 학습한다. MANSU 2605.15138은 unlearning 업데이트를 NF4 bin 폭 이상으로 rescale하는 magnitude floor를 쓰고 Lemma 1(quantization survival)을 제시한다.
- **설계 2(비밀 시드 subtractive dither 방어)는 선행을 찾지 못했다.** 검색 범위는 arXiv 3개 노트의 80여 쿼리, 추가 쿼리 3개, WebSearch이며 시스템 구현(llama.cpp, bitsandbytes 등)도 찾지 못했다. 다만 메커니즘 자체는 고전이다(Choi et al. 1802.02271의 seeded dither + 디코더 측 상쇄). 비결정 섭동 방어도 이미 있다. Gaussian noise 방어는 Ma 2108.09187, Egashira 2405.18137 Table 6에 있고, 결정론적 rounding 재설계는 EFRAP, FlipGuard, QuantGuard가 한다. 따라서 남는 기여는 calibrated·subtractive·입력 독립 노이즈를 보안 목적으로 쓰는 것과 그 적응형 공격 평가다. "원리적 무력화"는 과장이다(아래 전제 오류 5).

## 읽은 논문과 겹침

| 논문 | 무엇이 겹치나 (위치·수치) | 무엇이 남나 | 분류 |
|---|---|---|---|
| 2609.21834 Beyond Shadow Weights (math.OC, 2026-09-18) | Model 2 "conditional phase θ_i\|q ≈ Unif[−1/2,1/2]", "crossing probability p_i = η\|u_i\|/s_i"; Corollary 6 "E[q_i⁺ − q_i \| q] = −s_i sign(u_i) p_i" (줄 490–520, 2647–2690). 위상 균일성은 3/4/8-bit에서 경험적으로 확인되고 2-bit에서는 중심 집중 (Fig. 2) | Var=Δ\|d\|−d², SNR 스칼라, merge/unlearning/attack 해석, dither | preempts_part (정리 7.1의 핵심 확률 부분) |
| 2609.04526 Scale-QLoRA (2026-09-03) | Claim "(two cases, one reason)": on-grid base + sub-step 델타면 "re-quantizing W+∆ returns W" (줄 1035–1045). 140k 블록의 99.4%에서 base가 재구성 최적. cos=0.017 (줄 1118). Llama Banking77에서 RTN은 이득의 15%만 유지하고 GPTQ는 94% 유지 (줄 736–738). stochastic rounding은 QAT-LoRA를 3개 중 2개 데이터셋에서 붕괴시킴 (AGNews 57%, 줄 1804) | FP-base(u 균일) 영역의 비편향/분산 진술 | contradicts_premise (merge 통합) |
| 2609.18063 The Other Half of the Memory Wall (2026-09-16) | §3.3: "LoRA deltas (RMS 10⁻³) sit below the 4-bit group step ... 34% ... attention projection, 2% on a dense projection ... logits level 18%" (줄 326–330). retention=Σ(actual·δ)/Σδ² (줄 352). base는 int4 on-grid | 없음 (현상 보고만 있음) | contradicts_premise (기울기 ≈1 예측과 충돌). 인용 [U]는 [V]로 고쳐야 함 |
| 2605.15138 MANSU (2026-05-14) | Lemma 1 "Any update \|∆θ_i\| ≥ w_i changes the quantized value" (줄 486). magnitude floor rescale 식 (4) (줄 355–370). Prop. 2 "silently erased by NF4" (줄 489–494). "47–828× below the NF4 bin width" | 비편향성. MANSU의 "silently erased"는 FP base에서 틀림: 원소별로 확률 \|d\|/w로 살아남음. 정확한 ±Δ 정수 코드 델타. SNR | preempts_part (durability 따름정리, 설계 1의 unlearning 판) |
| 2505.18724 LoTA-QAF (2025-05) | Ŵ = sign(∆W)·I{\|∆W\|>ω} ∈ {−1,0,1}, "losslessly merged into W_int", overflow prevention, t-SignSGD (줄 317–331, 370–381) | unlearning·durability·델타 압축으로의 해석. "SNR=∞" 진술 | preempts_part (설계 1의 merge 메커니즘) |
| 2410.16454 Catastrophic Failure of LLM Unlearning via Quantization (ICLR 2025) | §5 "the weight changes in f_unlearn must exceed the quantization step size ∆" (줄 806–822, int4 12.5 vs int8 0.78). dither·노이즈 방어에 대한 논의 없음. 완화책은 SURE | 확률적 교차(\|d\|/Δ)와 비편향성. Zhang의 "반드시 Δ를 넘어야 한다"는 FP base에서 엄밀하지 않음 | preempts_part (구간 이론은 있음. 원문의 "이론은 없다"는 틀림) |
| 2405.18137 Exploiting LLM Quantization (NeurIPS 2024) | Gaussian noise 방어: "At σ = 10⁻³ adding noise proves to be an effective defense ... removing the security contrast while maintaining utility" (Phi-2, LLM.int8(), Table 6, 줄 1006–1020). 근거로 Ma et al. [14]를 듦. dither·subtractive는 논의 안 함. "rigorous investigations"가 필요하다고 함 (줄 1111) | calibrated subtractive dither, 입력 독립성 보장, 비밀 시드 | supports_premise (설계 2의 동기) |
| 2605.15152 Widening the Gap: Outlier Injection (ICML 2026) | Gaussian 방어 실패: "utility begins to drop clearly from σ = 10⁻³, while the attack remains effective ... complete utility collapse (σ = 5·10⁻³)"; "driven by large inserted outliers that are unaffected by small noise" (줄 741–770). 통계 검정으로 탐지 (§C.2). 재스케일 방어는 c=4096에서 실패 (Table 3) | dither 아래에서 outlier 공격이 어떻게 되는지는 미검증 (영 붕괴가 "w + 분산 Δ²/12 노이즈"로 바뀜) | contradicts_premise (작은 노이즈 방어 일반). dither에 대해서는 미정 |
| 2606.29239 QuantGuard (2026-06, v3 2026-08) | learnable(결정론적) rounding. 방어 인지 적응형 공격자가 feasible interval을 좁힘 (App. E, 줄 2444–2461). 비편향·SNR·dither 없음 | 비밀 무작위화 방어 | adjacent_only |
| 2606.20254 QVec (2026-06) | δ=Q(W)−W가 "structured behavioral transformation rather than random rounding noise" (줄 1338). Gaussian noise baseline이 "noticeable instability", 4-bit에서 CA가 크게 하락 (줄 832–834). 비편향·SNR 없음 | dither (δ ⊥ W) | adjacent_only (설계 2 framing 지지) |
| 2605.02196 DurableUn (2026-05) | "INT4 induces recovery of up to 22×"; "merged-model NF4 deployment does not exhibit the INT4 recovery effect" (줄 19, 63) | SNR 예측이 이 차이를 설명해야 함 | contradicts_premise (일반성) |
| 2609.27355 Retain-Forget Landscapes (COLM 2026) | "quantization injects a deterministic parameter perturbation" (줄 195). MUSE에서 13% → 54% (줄 203). Theorem 1(곡률비) | 경쟁 설명 (sharpness) | adjacent_only / 경쟁 가설 |
| 2601.15538 QUAIL | "weight changes must exceed ∆/2"; 평균 변화 2.97e-5, ∆≈0.268 → 버킷 overlap >99.9% (줄 105, 484–488, 573) | p=\|d\|/Δ≈1e-4와 정량적으로 일치. 정리의 예측과 맞으므로 실험 2의 근거로 쓸 수 있음 | supports_premise |

추가 검색 3건 (2026-09-15 이후 date-sorted, "moving target / keyed quantization", "stochastic rounding + delta/LoRA/merging"): 새 선행은 없었다. 2609.37887(Behavioral Capacity Certificates)은 PAC-Bayes 논문이라 무관하다. WebSearch 1건에서도 dither 방어 논문이나 구현은 나오지 않았다. 2602.13151, 2510.00761, 1802.02271, 2411.19530의 본문은 다시 읽지 않았고 discover 노트의 인용에 의존했다.

## 전제 오류

1. **"정리 7.1이 LoRA 머지에서 효과가 2–34%만 남는 이유를 설명한다"는 틀렸다.** 근거 수치(2609.18063 §3.3)와 Scale-QLoRA는 on-grid base(u≡0)에서 나왔다. 이 영역에서는 sub-step 델타가 결정론적으로 0이 되고, 정리의 균일 위상 가정은 성립하지 않는다. 정리대로라면 retention 기울기가 ≈1이어야 하지만 측정값은 0.34/0.02/0.18이다. 정리 7.1은 FP(BF16) base를 직접 양자화하는 경우에만 적용된다. 실험 1(QLoRA 어댑터 → 머지 → 재양자화)은 정리가 적용되지 않는 영역에 놓여 있다.
2. **"[2609.18063] ID 미확인 [U]"는 틀렸다.** 2609.18063(Lin et al., "The Other Half of the Memory Wall", AutoArk, 2026-09-16)의 §3.3이 34%/2%/18% 수치의 출처다. 이 논문은 MoE SSD serving 논문이고 어댑터는 unmerged로 배포한다. [V]로 바꾸고 "66–98% 지워짐"은 "가중치 수준 retention 34%(attn)·2%(dense), logits 18%"로 정확히 옮겨야 한다.
3. **"unlearning 역전에 이론은 없다"는 틀렸다.** Zhang 2410.16454 §5에 구간 논증이 있다. QUAIL 2601.15538은 ∆/2 논증을, MANSU 2605.15138은 Lemma 1과 Prop. 1–2를, DurableUn 2605.02196은 Prop. 1을, 2609.27355는 Theorem 1을 제시한다. 정확히 쓰면 "비편향·분산·SNR 형태의 진술은 없다"이다.
4. **"merge의 유일한 닫힌 해법은 QA-LoRA"는 낡은 주장이다.** LoTA-QAF 2505.18724(ternary 정수 코드 lossless merge), IntLoRA 2410.21759, Scale-QLoRA 2609.04526(code-invariant scale-only 어댑터로 bit-exact merge)가 있다.
5. **"subtractive dither가 양자화 조건부 공격을 원리적으로 무력화한다"는 과장이다.**
   - (a) 배포 모델이 W_fp+E(E ⊥ W_fp)라는 것은 사실이다. 하지만 공격자는 seed 없이도 E의 분포를 안다. 따라서 FP에서는 무해하고 노이즈 E 아래에서는 유해한 모델을 겨냥할 수 있다. outlier 주입으로 Δ를 키워 해당 층을 노이즈가 지배하게 만들면 "노이즈 조건부" 행동을 학습시킬 수 있다. 비밀 시드가 막는 것은 특정 실현값을 겨냥하는 것뿐이다.
   - (b) "방어자가 노이즈 샘플링으로 직접 검사할 수 있다"는 이점은 결정론적 경우에도 똑같이 있다. 방어자는 Q(W)를 직접 평가할 수 있다. 어려운 점은 유해 행동과 트리거를 모른다는 것이고, 이 점은 dither로 바뀌지 않는다.
   - (c) "outlier 주입으로 Δ가 커지면 정확도 하락으로 탐지된다"는 미검증이다. 2605.15152의 Gaussian 실험은 σ≥10⁻³에서 효용이 붕괴하는 것을 보였을 뿐이다.
   - (d) Δ²/12 입력 독립성은 dequant 시점에 seed로 u를 다시 빼야 성립한다. 커널이 그룹별 PRNG를 재생성해야 하므로 "저장 비용 0"이어도 계산 비용은 0이 아니다. 비감산 SR은 오차가 입력에 의존하고 MSE도 더 크다.
6. **"safety SFT 델타는 mean|d|≪Δ라 4-bit에서 지워진다"는 일반화할 수 없다.** 2606.29581(discover 노트 인용): AWQ INT4에서 8개 모델 중 7개가 FP16 대비 ASR 1.6pp 이내다. DurableUn: merged NF4에서는 복귀가 없다. 2602.13151: LoRA는 업데이트를 집중시켜 오히려 살아남는다.
7. **(유도상의 약점) 등방 입력 SNR ‖D‖_F²/(Δ‖D‖_1)은 델타가 바꾸려는 입력에서의 SNR이 아니다.** 델타가 학습된 입력 x(예: forget set)에서는 신호 Σ_i d_i x_i가 코히어런트하게 더해진다. 노이즈는 Σ_i(Δ|d_i|−d_i²)x_i²로 더해지므로 SNR이 최대 hidden dim 배까지 커진다. 비편향성만 보면 선형 1차 효과는 기대값으로 보존된다. 그래서 역전을 설명하려면 비선형(곡률) 항이나 matched-input SNR이 필요하다. 이는 2609.27355의 sharpness 설명과 직접 경쟁한다.

## 수정 요구

1. 제목과 가설에서 "통합"을 두 영역 정리로 바꾼다.
   - (i) FP base, 균일 위상: 비편향 + 분산 Δ|d|−d².
   - (ii) on-grid base(QLoRA, NVFP4/MXFP4 체크포인트): |d|<Δ/2이면 결정론적으로 삭제(편향 −d).
   - Scale-QLoRA 2609.04526과 2609.18063을 (ii)의 증거로 인용한다. 실험 1은 BF16 base LoRA 머지 후 직접 양자화하는 경우와 QLoRA 머지 경우를 대조하도록 바꾼다.
2. 정리 7.1을 "고전 SR 항등식(Gupta 2015, El Arar 2207.10321)을 기저 위상에 적용한 것"으로 낮춰 쓴다. 2609.21834 Model 2와 Corollary 6을 교차확률·비편향성의 선행으로 명시한다. 남는 기여는 분산과 SNR의 집계, 그리고 그 예측 검정이다.
3. "이론은 없다"를 지운다. Zhang §5, QUAIL, MANSU Lemma 1, DurableUn Prop. 1, 2609.27355 Thm 1을 인용한다. 정리 7.1이 MANSU의 "silently erased"와 Zhang의 "must exceed ∆"를 확률적으로 바로잡는다는 점(원소별로 확률 |d|/Δ로 생존하며 기대값 보존)을 기여로 적는다.
4. 설계 1은 LoTA-QAF(ternary 정수 코드 merge)와 MANSU(magnitude floor)의 변형이라고 밝힌다. 새로운 점은 "unlearning용 top-s ±Δ proximal step + 델타 압축 겸용"으로 좁힌다.
5. 설계 2는 "원리적 무력화"를 지우고 다음 내용으로 바꾼다. "공격을 결정론적 실현값 겨냥에서 분포 겨냥(노이즈 조건부 행동)으로 바꿔 강제한다. box형 공격(Egashira, GGUF)은 폭 Δ의 균일 노이즈에 무력화될 것으로 예측되고, outlier형(2605.15152)은 미정이다."
   - 평가에 추가할 것: (a) QuantGuard 식의 방어 인지 적응형 공격자, (b) dither 분포를 알고 E[유해 | W+E]를 최적화하는 EOT 공격자, (c) Gaussian noise baseline(σ=10⁻³)과 QVec, FlipGuard 비교.
   - Choi 1802.02271을 seeded subtractive dither 메커니즘의 선행으로 인용한다.
   - dequant 커널에서 PRNG 재생성 비용을 명시한다.
6. 2609.18063의 [U]를 [V]로 바꾸고 수치를 정정한다(전제 오류 2).
7. "safety SFT 델타가 4-bit에서 취약하다"를 "모델과 방법에 따라 다르다"로 고친다(2606.29581, DurableUn App. D, 2602.13151).
8. SNR 예측량을 matched-input SNR로 보강한다. 2609.27355의 sharpness와 2510.00761의 optimizer 축을 공변량으로 넣은 회귀를 실험 2에 추가한다.

## 새 가설

1. **matched-input SNR이 역전을 예측한다** (단계: PTQ 평가 / unlearning durability).
   - 정의: SNR_x = (Σ_i d_i x_i)² / Σ_i(Δ|d_i| − d_i²) x_i². x는 forget set 활성이다.
   - 등방 SNR보다 4-bit 지식 복귀율을 더 잘 설명할 것이다. full-FT unlearning(Zhang)과 LoRA unlearning(2602.13151)의 차이도 설명할 것이다.
   - 근거: 정리 7.1 분산식, 비편향성에서 오는 선형 효과 보존, 2602.13151의 LoRA 집중 효과.
2. **merge에서 rounding 규칙에 따른 retention 순서는 RTN ≪ SR/dither < GPTQ다** (단계: adapter merge·재양자화).
   - on-grid base에서 RTN은 결정론적으로 삭제한다(편향). SR이나 subtractive dither는 편향을 없애지만 분산 Δ|d|가 남는다. GPTQ는 오차 피드백(ΣΔ 유사)으로 노이즈를 낮은 Hessian 방향으로 성형해 높은 retention을 낸다.
   - 근거: Scale-QLoRA의 RTN 15% vs GPTQ 94%, SR이 QAT-LoRA를 2/3 데이터셋에서 붕괴시킨 결과, 2609.21834의 위상 모델.
3. **outlier 주입 공격은 dither 아래에서 "층 노이즈화"로 바뀌므로 그룹 스케일 팽창 통계로 탐지할 수 있다** (단계: 배포 양자화기 / 보안 검사).
   - 지표: 층별 Δ²/12 대 ‖W_group‖² 비율.
   - 반대로 적응형 공격자는 노이즈 조건부 유해 행동을 학습할 수 있다. dither 방어의 실효성은 이 EOT 공격에 대한 ASR로 판정한다.
   - 근거: 2605.15152의 Gaussian 방어 실패와 §C.2 통계 탐지, Egashira Table 6에서 box형 공격이 σ=10⁻³에 무력화된 결과.

생략한 것: 2602.13151, 2510.00761, 2606.29581, 1802.02271 본문 재확인(discover 노트 인용에 의존). Gupta 1502.02551과 El Arar 2207.10321 원문 확인. vLLM과 TensorRT-LLM에 대한 별도 웹 검색(WebSearch는 1회만 사용).
