## 아이디어 9. RoPE 쌍 혼합의 범위 모델: pre/post-RoPE 이득을 (양자화기, 범위 공유 구간 G, RoPE base)의 함수로, 그리고 회전된 평균 차감

### 가설
RoPE는 쌍 (2f, 2f+1)을 각도 mθ_f로 회전한다. "pre/post-RoPE 이득의 부호는 양자화기에 따라 바뀐다"는 이미 출간되었고, 주장은 크기의 정량 모델이다: 이득은 범위 통계를 공유하는 토큰 구간 G의 위상 sweep G·θ_f, 쌍 비대칭 max₁/max₂, 채널 평균 대 산포 a/δ, 양자화기 클래스의 함수다. 비대칭(zero-point) per-channel에서는 outlier 채널 평균(DC)의 회전이 손실을 지배하므로, post-RoPE에 k_m − R_m μ 를 저장하고 q·R_m μ 를 로짓에 더하면 과거 key 재회전 커널 없이 pre-RoPE 이득을 회수할 수 있다.

### 문헌상 공백
- 정성적 화해는 선점됨. KVQuant([2401.18079, NeurIPS 2024](https://arxiv.org/abs/2401.18079) [V]): "0.82"는 App. H Table 10 한 점(LLaMA-7B, int3, 그룹 없이 오프라인 보정한 per-channel K: 7.05 → 6.23)이고, App. P Table 20(per-token K)에선 부호가 반전되며(int3 pre 14.68 / post 10.87) 기제는 말로만 설명된다. AQUA-KV([2501.19392](https://arxiv.org/abs/2501.19392) [V]): Quanto는 post가 약간 낫고 HIGGS는 Hadamard 때문에 무차이(App. D 7.03/7.05; predictor 포함 수치). YAKV(KV offloading, [2604.08426](https://arxiv.org/abs/2604.08426) [V]): 4-bit HIGGS에서 App. L Table 10 정확도 93.36/93.73, Hadamard 귀속은 ablation 없음. 공백은 정량·대역 분해 설명뿐이다.
- "Hadamard/VQ면 이득 소멸"의 반례: RotateKV([2501.16383](https://arxiv.org/abs/2501.16383) [V]) Table 2 2-bit key MSE pre/post 회전+재배열 0.247/0.605(회전만 0.632/0.670); TaSQ([2610.03027](https://arxiv.org/abs/2610.03027) [V]) Fig. 1d pre-RoPE k-means VQ 재구성 MSE 35% 낮음; AnTKV([2506.19505](https://arxiv.org/abs/2506.19505) [V])는 시각화(Fig. 8)뿐. Interface-Aware KV([2609.05764, ICCAD 2026](https://arxiv.org/abs/2609.05764) [V])의 post-RoPE Hadamard는 하드웨어 선택이다.
- 등변 클래스: [2608.13365](https://arxiv.org/abs/2608.13365) [V] Lemma 2(centraliser = 쌍별 SO(2) 회전, 역방향 증명)와 §6.2(2차 모멘트 최적이 max−min 범위를 통제 못함); FPTQuant([2506.04985, ICML 2026](https://arxiv.org/abs/2506.04985) [S])·WUSH-KV([2609.38121](https://arxiv.org/abs/2609.38121) [S]).
- 할당: Block-GTQ([2606.24033](https://arxiv.org/abs/2606.24033) [V]): Lemma 2(E ≤ Σ_i‖q^(i)‖‖e^(i)‖), query 포함 점수 s_i = ½E[‖q^(i)‖²+‖k^(i)‖²](Eq. 1), Theorem 1(Σ s_i 4^{−b_i} greedy 최적). 대역 주장은 없고 TQ-MSE는 회전 불변. AATC([2608.14191](https://arxiv.org/abs/2608.14191) [V]) Thm 1은 q_c² 가중 reverse water-filling. SpectrumQuant(ACL 2026, arXiv 없음 [S]) Prop. 1은 피크 k* = Gθ_c/π 로 G·θ 의존성을 이미 쓴다.
- 주파수 전제: KV-Latent([2507.11273, ACL 2025](https://arxiv.org/abs/2507.11273) [V])의 "noise"는 양자화 잡음이 아니다(근거 철회). 최대 outlier는 저주파: Massive Spikes([2606.02288](https://arxiv.org/abs/2606.02288) [V]) Fig. 3c sink 채널 i ≈ 55/64. PM-KVQ([2505.18610, ICLR 2026](https://arxiv.org/abs/2505.18610) [S]): 최저주파 주기 54,410 토큰.
- YaRN: Q-ROAR([2509.14391, AAAI 2026 Student Abstract](https://arxiv.org/abs/2509.14391) [V])와 풀 버전([2510.00028](https://arxiv.org/abs/2510.00028) [V])은 같은 연구다. W4 악화는 YaRN s=8에서 완만(16K FP16 2.438 / RTN 2.517)하고 s=32·131K에서 5.386 / 9.958 / Q-ROAR 7.869(완화, 복구 아님); PPL>100은 NTK/linear 본문 서술뿐. Eq. (13)–(14)의 Δ_i²/6·η_i(m)은 "largest for high-frequency i" 휴리스틱이고 η_i(m)은 유도되지 않았다.

### 수학
복소 표현: 쌍 f의 key k_f ∈ ℂ, query q_f ∈ ℂ, logit(m, n) = Re Σ_f q_f k̄_f e^{iθ_f(m−n)}.

따름정리 9.1 (2608.13365 Lemma 2의 귀결). 스케일·코드북 결정 통계가 쌍별 회전에 불변이면 pre/post 오차 분포가 같다. 포함: 랜덤 Hadamard + 고정 격자, TQ-MSE, 전원 균일 각 격자, 가환 코드북(CommVQ [2506.18879](https://arxiv.org/abs/2506.18879) [S]). 제외: PolarQuant-Wu([2502.00527](https://arxiv.org/abs/2502.00527) [S])의 그룹 min–max 각 격자, 재배열, 학습 VQ.

보조정리 9.2 (범위 모델, 정정). G = 범위 공유 구간(KIVI: 그룹 크기; KVQuant: 보정·배포 길이), outlier 채널 평균 a, 산포 δ.
(a) 대칭 max-abs, G·θ_f ≥ π: 두 채널 범위 ≈ max₁ → 쌍 MSE 비 ≤ 2max₁²/(max₁²+max₂²) ≤ 2.
(b) 비대칭 min–max: 범위 폭 2δ → 2(a+δ), 비 ≈ ((a+δ)/δ)², 상한 없음.
(c) per-token: 혼합이 토큰별 skew를 줄여 post 우세(KVQuant T20).
G·θ_f ≪ 1 이면 비 ≈ 1. 시뮬레이션(a=5, δ=1, G=128, INT3; 판정 노트 수치 재현): G·θ ≥ π 대칭 1.5, 비대칭 6.6–6.7; G·θ ≈ 0 1.0–1.1. 총 이득 ≈ Σ_f 1[G·θ_f ≳ π]·[ρ_f − 1]·E_f. 혼합 쌍 수(d=128, 재계산): base 1e4, G = 32/128/512/2048/8192 → 17/26/36/46/55 (of 64); base 5e5 → 12/19/25/32/39. KVQuant 체제에선 대부분이 혼합되며 "고주파 집중"은 작은 G에서만 성립한다.

따름정리 9.3 (회전된 평균 차감). μ = pre-RoPE 채널 평균이면 저장 k_m − R_m μ 는 (b)의 DC 항을 없애고 q·R_m μ 는 위치만으로 계산되며, 잔차엔 (a)형 혼합만 남는다. 단순 post-RoPE 평균 차감([2608.02691](https://arxiv.org/abs/2608.02691) [S])은 G·θ_f ≥ π 쌍에서 평균이 ≈ 0이라 효과가 없어야 한다.

원안의 할당 b_f = ½log₂(|q_f|²σ_f²/θ)는 Block-GTQ 목적의 연속 완화(곱 가중은 그 Table 15에서 95.12 < 97.36 NIAH)이자 AATC Thm 1의 재표기라 삭제하고, rate에 ρ_f를 곱하는 수정만 남긴다 [S]. YaRN은 고주파 쌍을 보존하고 저주파만 보간하므로 고정 G의 혼합 쌍 집합은 저주파 쪽에서만 바뀌며, 이것이 η_i(m)의 폐형식 후보다.

증명 가능: 9.2(a)–(c), 9.3 항등식. 추측: 0.82의 대역 귀속, YaRN 정량 예측.

### 판정 실험(1–2 GPU·일)
Llama-2-7B(base 1e4), Llama-3.1-8B(5e5), Qwen3-8B. (a) per-channel INT2/3 key, 대칭·비대칭 × pre/post, G ∈ {32, 128, 2048, 보정 길이}: 대역별 MSE, PPL, NIAH/RULER. 예측: 이득 ∝ 혼합 쌍 수 × ρ_f, 비대칭 ≫ 대칭, base 5e5에서 작음. (b) 랜덤 Hadamard → 소멸, 보정 회전+재배열 → 잔존. (c) 전원 각 격자 → 불변, min–max 각 격자 → 비불변. (d) per-token K → post 우세. (e) 회전된 평균 차감 vs pre-RoPE vs 단순 post 평균 차감. (f) Block-GTQ qk_avg vs ρ_f 보정 할당. (g) YaRN s=8, 32에서 (a).

### 평가
novelty 낮음. 정성적 화해는 KVQuant App. H/P·AQUA-KV, 9.1은 2608.13365·FPTQuant·WUSH-KV, 할당은 Block-GTQ·AATC, G·θ 의존성은 SpectrumQuant, YaRN 기제는 2510.00028이 이미 쥐고 있다. 원안의 "선행 설명 없음", "Hadamard/VQ면 소멸", "MSE 약 2배", "저번호 채널이 노이즈 대부분"은 철회한다. 남는 것은 대칭/비대칭 범위 모델, 양자화기를 변수로 둔 통제 실험, 0.82의 대역 귀속(이상 분석 노트 수준)과 회전된 평균 차감이다. (e)가 post-RoPE 경로에서 pre-RoPE 이득 대부분을 회수할 때만 KV 양자화 논문의 한 절이 되고, 아니면 부록 A로 내린다.

---
