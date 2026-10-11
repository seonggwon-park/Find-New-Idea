## 아이디어 5. PTQ 캘리브레이션 격차의 3항 분해: 의존 토큰의 n_eff, 이상치가 지배하는 댐핑 척도, 학습 회전·VQ "과적합"의 이동 해석

### 가설
캘리브레이션-보류 격차 = 목적 불일치(n→∞에서도 남음) + 분포 이동(H_T ≠ H_C) + 표본 분산(Ĥ − H). 경험→모집단 상한은 이미 증명되어 있고, 남는 가설은 셋이다. (i) 시퀀스 내 토큰 상관으로 n_eff ≪ n이고, 기존 i.i.d. 상한은 n_eff로 써야 맞는다. (ii) massive-activation 채널이 mean(diag H)를 부풀려 mean diag 비례 댐핑은 bulk 방향을 과수축한다; 최적 λ는 bulk 척도 × (d/n_eff)^p, p ∈ {1/3, 1/2, 1}. (iii) 학습 회전·코드북의 "과적합"은 P에 비례하는 표본 분산이 아니라 분포 이동이고, 백색화된 Hessian 차이가 이를 예측한다.

### 문헌상 공백
- 경험 연구. 1,800개 모델 연구의 실제 결론은 양자화가 pruning보다 캘리브레이션에 덜 민감하다는 것(GPTQ zero-shot 범위 0.9–1.6%, SparseGPT 2.4–4.8%) ([On the Impact of Calibration Data in PTQ and Pruning 2311.09755, ACL 2024](https://arxiv.org/abs/2311.09755) [V]). "GPTQ는 과적합"은 이 논문이 관련연구에서 인용한 [ZeroQuant(4+2) 2312.08583](https://arxiv.org/abs/2312.08583) [S]의 주장이다. "새 모델은 덜 민감"의 원출처는 [Outliers and Calibration Sets have Diminishing Effect 2405.20835](https://arxiv.org/abs/2405.20835) [V], "도메인은 GPTQ에만 영향"은 [Quantization Hurts Reasoning? 2504.04823, COLM 2025](https://arxiv.org/abs/2504.04823) Sec. 5 [V]다.
- 이론 선행 연구가 있다(이전 판의 "전부 경험적 / RMT 논문 없음"은 철회).
  - 분해식 E|zᵀ(w−q)|² = (1/m)‖X(w−q)‖² + (w−q)ᵀ(E zzᵀ − XᵀX/m)(w−q): [Provable PTQ 2508.04853](https://arxiv.org/abs/2508.04853) Remark 3.8 [V].
  - 주 선점 [Generalization Behavior of OPTQ and the Role of Regularization 2609.31560](https://arxiv.org/abs/2609.31560) [V]: Thm 3.1, λ > 2R² log(N/ε)이면 확률 1−ε로 모든 데이터 의존 r에 대해 E|xᵀr|² ≤ (1 − √(2R² log(N/ε)/λ))⁻¹((1/m)‖Xr‖² + λ‖r‖²). Remark 4.1 eq.(35) λ = β‖X‖_F²/(m^{1/3} min{m,N}^{2/3})가 적은 표본에서 0.01 규칙을 이김. 한계: 합성 데이터, i.i.d., R²이 outlier에 지배됨.
  - 균일수렴 [SARQC 2605.05693](https://arxiv.org/abs/2605.05693) Thm 3.1, R²M_X²√(log(2|Q_R|/δ)/2n) [V]. 이동 [DRQ 2610.11226](https://arxiv.org/abs/2610.11226) Thm 3.1/3.2(tr(CΔ)≤α 모호집합, 최악 손실 L + αS_C)와 Thm B.2 "more calibration data need not improve the selected weights" [V]; [TASA 2607.00908](https://arxiv.org/abs/2607.00908) Thm 1, tr(H_test H_α⁻¹)가 α에 엄격 볼록, 단일 도메인에서 유효랭크 5.8→3.3(Table 19) [V].
  - 측정·수축 [DASH-Q 2604.13806, EuroMLSys'26](https://arxiv.org/abs/2604.13806) [V]: Ĥ_n 대 Ĥ_4096(n=8…2048), LW 인용 D+ρO 수축에서 ρ≈0, GPTQ PPL은 n>2⁸에서 흔들리거나 악화(Sec. 7.2). 유한 캘리브레이션 추정량의 최적 수축 α* = μ²/(μ²+σ²)은 [CoreQ 2602.05902](https://arxiv.org/abs/2602.05902) [V](대상이 Ĥ+λI 아님).
  - 댐핑의 다른 해석: ridge = 상류 노이즈 강건성([Activation Denoising 2610.07522](https://arxiv.org/abs/2610.07522) Thm 2 [V]); QEP는 평균 대각(100×) ([2504.09629](https://arxiv.org/abs/2504.09629) [S]).
- 회전·VQ. SpinQuant은 800→128 표본에서 PPL 불변(A.3, Table 11) ([SpinQuant 2405.16406](https://arxiv.org/abs/2405.16406) [V]); QuIP#/QTIP Hessian은 6144 시퀀스([2402.04396](https://arxiv.org/abs/2402.04396), [2406.11235](https://arxiv.org/abs/2406.11235) [V]). 관찰된 격차는 분포 간이다: [DartQuant 2511.04063](https://arxiv.org/abs/2511.04063) Table 1, LLaMA-2-7B PTB PPL 38.24(PTB 캘리브레이션) 대 45.13(Wiki) [V].
- 도메인 이동. 다국어 캘리브레이션이 Llama-3.1-8B GPTQ PPL을 최대 3.52 낮춤(AWQ ≤0.35) ([Calibrating Beyond English 2601.18306, EACL 2026](https://arxiv.org/abs/2601.18306) [V]). Qwen3-4B-Thinking INT3 GPTQ에서 자기생성 Math-CoT 캘리브레이션이 28%→65%(Fig. 5) ([Quantization Inflates Reasoning 2606.25519](https://arxiv.org/abs/2606.25519) [V]); 자기생성 추론 트레이스가 1.58-bit 붕괴를 막음 ([Attend to Your Own Thoughts 2608.01078](https://arxiv.org/abs/2608.01078) [V]). n 증가에 코드 정확도 57.67%(16)→34.03%(128) ([2510.10618, NeurIPS 2025](https://arxiv.org/abs/2510.10618) [V]).
- 남은 공백: MP/LW의 GPTQ Ĥ 적용, 상관 토큰의 n_eff, 이동 예측량 비교, P 대 격차. 시퀀스 단위 휴리스틱은 [MaCa 2602.07465](https://arxiv.org/abs/2602.07465) [V]와 [RSQ 2503.01820](https://arxiv.org/abs/2503.01820) [S]에 있으나 n_eff 추정은 없다.

### 수학
(1) 표본복잡도. 가산형 proxŷ + ‖H−Ĥ‖_op‖ΔW‖_F²는 ‖Σ‖_op ≫ 평균 고유값인 LLM에서 거의 공허하므로 George–Saab 곱셈형에서 출발한다. 핵심 양은 ‖(Σ+λI)^{−1/2}(Ĥ−Σ)(Σ+λI)^{−1/2}‖이다. 새 기여 후보: (a) β-mixing/블록 독립 아래 m을 n_eff로 바꾼 버전(블록 matrix Bernstein), (b) R² 대신 유효랭크 r(Σ)로 쓴 조건 [S].

(2) 댐핑. 경쟁 예측: George–Saab eq.(35)를 H = XᵀX/m으로 정규화하면 m≥N에서 λ ≈ 평균 고유값 × (d/n)^{1/3}; 고전 LW 수축 강도는 대략 d/n; MP bulk 폭은 √(d/n). (DASH-Q는 대각 방향 수축.) 새 가설은 척도: λ > 2R² log(N/ε)도 mean diag도 이상치에 지배되므로 bulk 척도(median/trimmed diag)를 써야 한다 [S].

(3) "과적합". O(√(P/n_eff))는 SARQC의 |Q|를 P로 바꾼 것뿐이다. "P≈8×10⁶ 대 1.6×10⁶ 토큰 → O(1)" 산술은 삭제(스칼라 제약 n·d ≈ 6.6×10⁹, Stiefel 100 iteration). 예측: 분포 내 격차는 P에 무관, 분포 간 격차는 (4)에 비례.

(4) 공변량 이동. 초과 = tr(ΔW(H_T − H_C)ΔWᵀ). 후보는 ‖H_C^{−1/2}(H_T−H_C)H_C^{−1/2}‖_op, TASA의 tr(H_T H_C⁻¹), DRQ의 S_C; 비백색화 ‖H_T − H_C‖_op는 기준선. 기여는 증명이 아니라 어느 것이 열화를 정렬하는지의 측정이다.

증명 가능한 것: (1)(a), (4)의 상한. 추측: n_eff/n(0.01–0.1 예상이나 GPTQ의 128 시퀀스 정체·DASH-Q 대각 안정으로 보아 측정 대상), p, bulk 척도의 이득.

### 판정 실험(2–4 GPU·일)
Llama-3-8B, W3/W2. (a) n ∈ {8, 32, 128, 512, 2048} × λ 격자에서 캘리브레이션·보류 proxy와 end KL, 최적 λ(n)의 지수 p 적합. (b) 토큰 위치 블록 부트스트랩으로 n_eff 추정; 같은 토큰 예산에서 짧은 시퀀스 다수 대 긴 시퀀스 소수 비교. (c) 댐핑: 고정 1%, George–Saab eq.(35), LW, DASH-Q, QEP 평균 대각, bulk 척도, split-half CV. (d) GPTQ, AWQ, SpinQuant, AQLM/QTIP의 분포 내·간 격차 대 P(각 방법의 캘리브레이션 예산 명시). (e) 영어/다국어/추론 캘리브레이션에서 세 이동 예측량과 열화의 순위 상관. 성공 기준: 3항 분해에서 분산 항이 n_eff로, 분포 간 항이 백색화 예측량으로 정렬된다. 단조성 기준(반례: DASH-Q Sec. 7.2, 2510.10618, DRQ Thm B.2)과 출처 없는 "2-bit 증분 ≈0.3 PPL"은 삭제.

### 평가
novelty 낮음. 증명 가능한 부분은 2609.31560(주 선점: 곱셈형 상한 + 표본수 의존 λ), 2508.04853, 2605.05693, 2610.11226, 2607.00908에 같거나 더 강한 형태로 있고, 측정·수축은 2604.13806이 먼저 했다. 남는 것(n_eff, 이상치 지배 댐핑 척도, 지수 판별, 이동 예측량 비교)은 경험적 검증이다. main-track 가능성 낮음: n_eff/n이 매우 작거나 bulk 척도 λ가 W2/W3에서 George–Saab·DASH-Q를 분명히 이길 때만 workshop 이상이고, 아니면 부록 A로 옮긴다. 리스크: "128 시퀀스로 충분, 격차는 이동과 목적 불일치"(DRQ 재확인)에 그칠 가능성이 높다.

---
