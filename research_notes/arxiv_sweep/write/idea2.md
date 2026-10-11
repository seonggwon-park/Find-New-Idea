## 아이디어 2. SSM/하이브리드 recurrent state의 error-feedback 양자화: 동일 비트 회계, deadband 하한, erasure 분기

### 가설
재귀 상태 h_t = a_t h_{t−1} + u_t 재양자화의 (i) RTN deadband, (ii) SR 오차 누적, (iii) 정확 잔차 1차 EF의 |d_t| ≤ Δ 는 출판되었거나 출판된 증명의 따름정리다. 남는 가설: (H1) 동일 총 비트에서 "INT(b) + b_r-bit RTN 잔차"는 INT(b+b_r)과 같거나 나쁘다(사전 등록된 부정 예측). Weyl-dithered 잔차만 이길 여지가 있다. (H2) 감쇠율 비트 할당은 deadband 영역에서 깨지며 채널별 최소 비트폭 하한으로 고쳐진다. (H3) EF/dither 이득은 Mamba-2/Nemotron-H(erasure 없음)에서 크고 GDN/KDA(erasure)에서 0에 가깝다.

### 문헌상 공백
- **상태 양자화는 이미 다수다("0건"은 틀렸다).** Quamba는 h를 양자화하지 않지만([2410.13229](https://arxiv.org/abs/2410.13229) [V], ICLR 2025) Quamba2는 캐시된 상태를 정적 per-group 8-bit로 양자화한다(App. E; open-loop) ([Quamba2 2503.22879](https://arxiv.org/abs/2503.22879) [V], ICML 2025). eMamba는 INT24 계산/INT17 저장 상태를 FPGA/ASIC에 구현했다(소형 모델) ([eMamba 2508.10370](https://arxiv.org/abs/2508.10370) [V], ESWEEK 2025). Q-Mamba(ACL 2025 Findings, arXiv 미등재 [V])는 W8A8H4, 이후 [DAMP 2608.27513](https://arxiv.org/abs/2608.27513), [STEPQuant 2609.38169](https://arxiv.org/abs/2609.38169), [LeapQuant 2609.38166](https://arxiv.org/abs/2609.38166) [S]가 있다.
- **(i)–(iii)의 선점.** [Low-Discrepancy Dither 2609.39185](https://arxiv.org/abs/2609.39185) [V]: Lemma 1(Write erasure) = deadband, Corollary 1(RTN 선형, Weyl ≤ C s(1+log T)), Lemma 2(per-write MSE RTN s²/12, SR s²/6), App. A 부분합 |e_T| ≤ max_m |R_m|. ΣΔ는 "상태 캐시 크기가 두 배"라며 기각한다. [When Can Depth Replace Precision? 2607.23390](https://arxiv.org/abs/2607.23390) [V]: Prop. 7(Fixed-grid freeze), Thm 8 증명의 s_k = e_k + r_k 가 아래 g_t 기법이다. [When Quantization Breaks Memory 2609.04490](https://arxiv.org/abs/2609.04490) [V]: GRU/LSTM에서 write deadband(8→4-bit에서 99.40%), EF, 2/3/4-bit residual memory를 구현했다(정리 없음). a = 1의 RTN/SR/Kahan 삼분법: [2010.06192](https://arxiv.org/abs/2010.06192) [V] Thm 1.
- **산업 데이터.** [Nemotron 3 Super 2604.12374](https://arxiv.org/abs/2604.12374) [V] Sec. 4.3, Table 9(early checkpoint): FP32 기준선 SSM cache를 FP16 RTNE로 바꾸면 livecodebench completion token +36.95%(W16A16)/+40.27%(W8A8), pass@1 손실 없음. FP16+SR(Philox<5>) 또는 INT16 + block-128 스케일로 해소, 출시 checkpoint는 FP16+SR(Tables 6–7). [Nemotron-Labs-3-Puzzle 2607.04371](https://arxiv.org/abs/2607.04371) [V]는 NVFP4만 FP16+SR, FP8은 FP32(Table 2). [Nemotron 3 Ultra 2606.15007](https://arxiv.org/abs/2606.15007) [S]: INT8 SR이 RTN보다 낫고 checkpointing C = 8 제안(Table 16). vLLM RFC #55196: Falcon-H1-1.5B FP8 KV 용량 이득 512/1024 토큰에서 1.00×, 코멘트에서 Qwen3.5-4B GDN INT8 상태 GSM8K 81→41%, EF로 81% 복구(자기보고) ([vllm issue 55196](https://github.com/vllm-project/vllm/issues/55196) [V]).
- **반대 증거.** [Ternary Mamba 2606.18114](https://arxiv.org/abs/2606.18114) [V]는 W1.58A16 weight-only QAT(상태 FP16)이고 "이력 의존 오차"는 가중치 오차, 결론은 QAT다. §6.3/Fig. 5의 "ruling out sigma-delta"는 feature 축 ΣΔ라 시간축 EF를 배제하지 않는다. [Why GDN Survives 4-Bit 2609.04098](https://arxiv.org/abs/2609.04098) [S]: GDN 상태 오차는 32K까지 plateau(relS ≈ 12–13%). [Topollai–Choromanska 2603.16731](https://arxiv.org/abs/2603.16731) [V](preprint)는 양자화 EMA의 유사 병리(stalling)이며 SSM 언급은 없다(IIR 유비는 우리 것).
- **남은 공백:** 동일 비트 회계, deadband-보정 할당, erasure 분기, KV + 상태 공동 예산.

### 수학
대각 채널: h_t = a_t h_{t−1} + u_t, a_t = exp(Δ_t A) ∈ (0,1), 스텝 Δ.

(1) RTN: d_t = a_t d_{t−1} + e_t, |e_t| ≤ Δ/2, 최악 Δ/(2(1−a))(느슨함: 토이 3.4 vs 500). |(1−a_t)ĥ_{t−1} − u_t| < Δ/2 지속 시 동결(= 2609.39185 Lemma 1, 2603.16731 Eq. 5). 2603.16731 Prop. 3.3의 정량형은 부동소수 ULP·χ²₁ 가정이라 INT 스텝에 그대로 옮겨지지 않는다.

(2) SR: per-write MSE Δ²/6(f 균일). Var d_t = (Δ²/6)(1−a^{2t})/(1−a²): t ≪ 1/(1−a)에서 ≈ Δ²t/6, 이후 표준편차 Δ/√(6(1−a²)) 에서 포화(a = 0.999에서 ≈ 9.1Δ; 토이 최대 12.9와 부합).

(3) ΣΔ: v_t = a_t ĥ_{t−1} + u_t + r_{t−1}, ĥ_t = Q(v_t), r_t = v_t − ĥ_t. g_t := d_t + r_t = a_t g_{t−1} + (1−a_t) r_{t−1} 이므로 |g_t| ≤ (Δ/2)(1 − ∏a_s).

정리 2. 정확 잔차 하에서 모든 t, a_t ∈ [0,1)에 대해 |d_t| ≤ Δ. 이는 2607.23390 Thm 8의 축약 특수형이자 2609.39185 App. A의 한 줄 따름정리(잔차 r_{t−1} − r_t 가 telescoping), 고전적으로는 1차 ΣΔ 안정성(Daubechies–DeVore 2003)과 ESS의 ‖(1−z⁻¹)/(1−az⁻¹)‖₁ = 2([2506.01404](https://arxiv.org/abs/2506.01404) [S])다.

(4) 저장 회계: a-가중 변형 v_t = a_t ĥ_{t−1} + u_t + a_t r_{t−1} 이면 ĥ_t + r_t = h_t 가 정확히 성립한다. 정확 잔차 ΣΔ는 두 limb 무손실 저장이지 압축이 아니다. b_r-bit 잔차의 최악 Δ·2^{−(b_r+1)}/(1−a), 표준편차 Δ·2^{−b_r}/√(24(1−a)) 는 INT(b+b_r) RTN 상한과 같고, 잔차 양자화기가 미세 스케일 deadband를 재도입한다. 토이(a = 0.999, max|d_t|/Δ) RTN 3.4, SR 12.9, ΣΔ(정확) 0.58, ΣΔ(4-bit 잔차) 2.0(최종 0.05). 판정 노트 재시뮬레이션(Δ = 0.05, max/rms): a = 0.999에서 ΣΔ+4-bit 2.00/0.49 vs RTN(+4 bit) 1.56/0.39, a = 0.99에서 1.06/0.32 vs 0.61/0.13. (d)가 (e)에 진다.

(5) 할당: b_i = ½log₂(σ²_e/((1−a_i²)θ)), 느린 채널에 +½log₂(1/(1−a_i²)) 비트(1−a = 10⁻³에서 약 +4.5). 이는 [2609.30950](https://arxiv.org/abs/2609.30950) [V] Prop. 1(Γ_i = ρ_i/(C·expm1(2s_i)))과 eq. (4) b_i = b̄ + ½log₂(c_i/GM(c))이며, DAMP의 1/(1−a_eff²), STEPQuant L_u 도 같다. 2609.30950은 deadband 영역에서 lag-one 상관 최대 0.23, 이득 1.7–2.6× 과대평가를 인정한다. (H2): write margin M = 2|δ|/Δ(2609.04490)에 대해 P(M < 1) ≤ τ 인 하한 b_i ≥ log₂(range_i/|δ_i|_q) 위에서 eq. (4)를 적용한다. KV + 상태 공동 예산은 선행 미발견 [S].

증명 가능한 것: (1)–(4)(기존 결과의 특수형). 추측: (H1) dithered 잔차 limb의 우위, (H2) 하한의 격차 해소, (H3).

### 판정 실험(2–3 GPU·일)
Mamba-2 2.7B, Falcon-H1-1.5B, Nemotron-H-8B, GDN 하이브리드(Qwen3.5-4B). 상태 INT8/INT6/INT4 fake-quant: (a) RTN, (b) SR, (b′) Weyl, (c) ΣΔ 정확 잔차(무손실 상한), (d) ΣΔ + 4-bit RTN 잔차, (d′) ΣΔ + 4-bit Weyl 잔차, (e) 동일 총 비트 INT(b+4) RTN/SR/Weyl, (f) checkpointing C = 8, LeapQuant window 16, (g) FP16+SR. 측정: GSM8K, NIAH 4K–32K, 4K 스텝 KL, completion length, 오차 t-성장, deadband fraction. 성공 기준: (d)가 (e)를 못 이김(사전 등록), (d′)가 (e)-Weyl을 유의하게 이김, (H2) 하한 + eq. (4) > eq. (4), EF/dither 이득 Mamba-2 ≫ GDN.

### 평가
novelty 낮음. deadband·RTN/SR 성장·부분합은 2609.39185, g = d + r 기법은 2607.23390, EF + k-bit 잔차는 2609.04490와 vLLM RFC, 감쇠율 할당은 2609.30950/DAMP/STEPQuant가 선점했다. 시변 a_t 균일형만 미명시이나 한 줄 따름정리다. 원안의 "ΣΔ + 4-bit 잔차의 메모리 이득"은 (4)로 무너지므로 원안은 부록 A로 내린다. 남는 (H1)–(H3)와 KV + 상태 공동 예산은 workshop 또는 하이브리드 서빙 논문의 한 절 규모다.

---
