# Idea 9 판정 — RoPE 쌍 혼합 보조정리 (pre/post-RoPE × 양자화기, 주파수 대역 할당, YaRN)

Date: 2026-10-11. Judge pass (deep read + adversarial). 논문 본문은 untrusted data로 인용만 함.
Deep-read (cached full text, grep + 국소 read): 2501.19392 (AQUA-KV), 2401.18079 (KVQuant), 2608.13365, 2510.00028, 2606.24033 (Block-GTQ), 2608.14191 (AATC), 2606.02288 (Massive Spikes), 2501.16383 (RotateKV).
추가 검색 4건 (arxiv_search): 2026-08 이후 pre/post-RoPE × quantization (→ WUSH-KV만, 신규 없음); RoPE × 위상/각도 × per-channel range (→ FireQ, 2510.00028, SnapMLA, Q-ROAR); RoPE × KV × group size/per-channel 2025– (→ PolarQuant, RotateKV); frequency-band × RoPE × bit-width (→ Massive Spikes, Q-ROAR 계열). 새 선점 논문 없음.
WebSearch 1건 (시스템): vLLM/SGLang/llama.cpp/TRT-LLM에서 pre-RoPE key 양자화가 탑재된 흔적 없음. 연구 코드(KVQuant, SVDq 2502.15304)만 pre-RoPE + on-the-fly RoPE 사용.
생략: SpectrumQuant(ACL 2026, arXiv 없음), TaSQ, FPTQuant, WUSH-KV, HyperQuant는 이번 pass에서 재독하지 않고 discover 노트의 인용을 신뢰함.

## 판정 요약

- **novelty: low.** 주 선점: AQUA-KV (2501.19392) + KVQuant App. Table 20 (2401.18079) — "pre/post-RoPE 이득의 부호는 양자화기에 따라 달라지며 Hadamard 계열에서는 사라진다"는 아이디어의 헤드라인 화해 주장이 2024–2025에 이미 숫자와 함께 출간됨. 9.1의 등변 클래스는 2608.13365 Lemma 2 (centraliser = 쌍별 평면 회전, 역방향 포함 증명)와 FPTQuant/WUSH-KV가 특성화. 9.3은 Block-GTQ (RoPE 블록 단위 query-가중 할당, Theorem 1 greedy 최적) + AATC Theorem 1 (q_c² 가중 채널별 reverse water-filling)로 사실상 선점. YaRN 부분은 2510.00028 eq.(13)–(14)의 "anisotropy shift"·"largest for high-frequency i"가 휴리스틱으로 선점.
- **남는 것**: (i) per-channel 범위 통계(대칭 max-abs와 비대칭 min-max 모두)를 그룹/스케일 공유 구간 G 안의 위상 sweep G·θ_f의 폐형식 함수로 쓴 식, (ii) 같은 모델·데이터에서 양자화기 종류(대칭 per-channel, 비대칭 per-channel, per-token, 랜덤 Hadamard, 보정된 회전+재배열, 극좌표, 학습 VQ)를 변수로 둔 통제 실험, (iii) "어느 쌍이 KVQuant의 0.82를 만드는가"에 대한 대역별 귀속. 셋 다 짧은 분석 노트 수준이며, 방법 novelty는 없음.
- **보조정리 9.2 자체에 정량 오류**: "쌍 MSE 약 2배" 상한은 대칭 max-abs에서만 성립. KIVI/KVQuant 계열의 비대칭(zero-point) per-channel 양자화기에서는 부호가 일정한 outlier 채널의 평균(DC)이 post-RoPE에서 회전 진동으로 바뀌어 범위가 [a−δ,a+δ]에서 [−a−δ, a+δ]로 넓어지므로 비율이 (a/δ)² 차수로 무한히 커질 수 있음. 자체 시뮬레이션(평균 5, 표준편차 1인 outlier 채널, G=128, INT3): Gθ ≥ π에서 post/pre MSE = 대칭 1.51, 비대칭 6.7; Gθ = 0.013에서는 0.99–1.11.

## 읽은 논문과 겹침

| 논문 | 겹치는 부분 | 남는 novelty |
|---|---|---|
| AQUA-KV 2501.19392 (Jan 2025) — preempts_part (헤드라인) | §"Positional embeddings": "the uniform quantizer (Quanto) works slightly better post-RoPE, while HIGGS works equally well in both cases due to its use of the Hadamard transform"; "the optimal predictor for post-RoPE compression needs to be rotation-equivariant". App. D "Predictor Order": HIGGS pre 7.03 / post 7.05 PPL (LongBench 44.26/44.13); Quanto-gs64 (per-token) pre 10.90 / post 10.33. | 유도 없음, 주파수/G 의존성 없음, per-channel 경우 없음. 주의: 이 수치는 AQUA-KV predictor가 붙은 residual 양자화 결과라 순수 양자화기 비교가 아님. |
| KVQuant 2401.18079 (NeurIPS 2024) — preempts_part | Table 10 (LLaMA-7B, Wikitext-2, per-channel K / per-token V): int3 post 7.05, pre 6.23 (fp16 5.68). App. Table 20 (per-token K): int4 6.02/5.98, int4-gs128 5.76/5.77, int3 14.68/10.87, int3-gs128 6.28/6.17 (pre/post) — per-token에선 post가 우세. 기제를 말로 설명: "part of the magnitude from the outlier channel will be shifted to the smaller channel. This partially mitigates the impact of the outlier channel on skewing the quantization range". §3.2: RoPE가 "mixing pairs of channels by different amounts for different positions". 또한 per-channel 스케일은 "offline calibration ... per-channel quantization without grouping" (l.336). | 식, G·θ_f, 대역 분해 없음. 중요: KVQuant per-channel은 그룹이 아니라 오프라인 보정 스케일이므로 실효 G는 보정/배포 길이 전체 — 9.2의 G 해석을 바꿔야 함. |
| 2608.13365 (Aug 2026) — preempts_part (9.1, range-vs-moment 논점) | Lemma 2: 서로 다른 RoPE 각에서 O(d_h) centraliser = 쌍별 SO(2) 회전 곱(역포함 증명, App. C). §6.2 "Objective mismatch": 동적 INT4는 "tokenwise max − min range over groups that span multiple frequency pairs"를 쓰므로 위상 평균 2차 모멘트 최적이 범위를 통제하지 못함. 고주파 쌍은 C_k,S_k≈0 (L=2048에서 3B 모델 64쌍 중 17쌍 |(C_k,S_k)|<0.01). 쌍별 회전만 쓰면 full-head Hadamard보다 PPL이 나쁨. | pre/post 저장 비교 없음, per-channel(토큰 축) 범위 미분석, G·θ_f 식 없음. |
| Block-GTQ 2606.24033 (Jun 2026) — preempts_part (9.3) | §2.1 "norm-based block statistics are RoPE-invariant—we can therefore compute the energy score in pre-RoPE coordinates"; 이상적 가중 s*_i = E[‖q^(i)‖‖k^(i)‖] (query 가중, l.192), 대리 s_i = ½E[‖q‖²+‖k‖²] (Eq.1); 목적 Σ s_i 4^{-b_i}, Theorem 1 greedy 정수 할당 최적. | 연속 rate reverse water-filling 형태 b_f = ½log₂(|q_f|²σ_f²/θ)와 "위상은 분산에 무관" 언급만 추가. 로컬 양자화기 TQ-MSE가 회전 불변이라 pre/post 문제 자체가 없음. |
| AATC 2608.14191 (Aug 2026) — preempts_part (9.3) | Theorem 1: D_K = Σ_{i,c} a_i²‖v_iW_O−o‖² q_c² E[(δk_ic)²]; 채널별 bit을 reverse water-filling으로 할당. | RoPE 없음. RoPE 쌍 = 두 채널이라 9.3은 이 정리의 쌍 단위 재표기에 가까움. |
| Rethinking RoPE Scaling 2510.00028 (+Q-ROAR 2509.14391) — preempts_part (YaRN, 축정렬 격자 기제) | Eq.(13)–(14): 회전된 2D 쌍에 축정렬 per-channel mid-rise 양자화기, E‖ẑ−z‖² ≈ Δ_i²/6·η_i(m), "Anisotropy shift: Per-channel quantization grids are axis-aligned; RoPE rotates the signal relative to these grids"; "position-dependent logit error that is typically largest for high-frequency i and large \|m−n\|". l.293: PPL>100은 NTK/linear 경우. | η_i(m)은 유도되지 않음(정의만). G·θ_f, 양자화기 클래스 구분, KV-cache pre/post 비교 없음. 실험은 weight W4 중심. |
| Massive Spikes 2606.02288 (Jun 2026) — contradicts_premise | Fig.3c: sink를 만드는 top-2 채널이 "strictly localize in low-frequency bands"이며 "high-index subspaces (i ≈ 55 out of 64)"; "~80% of cases, the top-2 channels form an exact RoPE 2D pair (i, i + d/2)". | 가장 큰 key outlier는 저주파(Gθ_f ≪ 1)·대칭 쌍(max₁≈max₂)에 있어 9.2가 차이 0을 예측하는 영역. "outlier 쌍이 고주파에 몰림" 추측과 충돌. KVQuant 이득이 중·고주파 비대칭 쌍 또는 저주파 DC 성분의 장거리 회전에서 온다는 것은 측정이 필요한 별도 가설. |
| RotateKV 2501.16383 (2025) — contradicts_premise ("Hadamard → 이득 소멸" 전면 주장) | Table 2 (LLaMA2-7B, 2-bit key MSE, pre/post): Rotation 0.632/0.670; Rotation+Reordering 0.247/0.605; Reordering+Rotation 0.477/0.568. "the quantization error increases by 145% after applying RoPE". | 순수 회전에서는 거의 불변(+6%)이라 9.1의 좁은 형태는 지지됨. 보정된 재배열·그룹 구조를 얹으면 불변성이 깨짐. |

discover 노트에서만 확인(이번에 재독 안 함): YAKV = 2604.08426 "KV Cache Offloading for Context-Intensive Tasks", App. L Table 10 (93.36/93.73, 90.83/91.05, 과제 정확도); TaSQ 2610.03027 Fig.1d pre-RoPE VQ 35% 낮은 오차; NSNQuant 2505.18231 Tab.11 Hadamard+VQ에서도 5.29/5.33; FireQ 2505.20839 Thm 3.1 쌍 공유 스케일; HyperQuant 2606.23406 §6 "pre- and post-RoPE quantization are statistically equivalent"(RHT+lattice); PM-KVQ 2505.18610 저주파 주기 54K 토큰 → 짧은 보정이 실패; SpectrumQuant (ACL 2026 long 1732) Prop.1 RoPE 스펙트럼 피크 k* = Gθ_c/π, DCT 4/2-bit 대역 분할.

## 전제 오류

1. "양자화기 의존성을 설명한 논문이 없다" — **틀림.** AQUA-KV (2501.19392 §Positional embeddings + App. D)가 Quanto(post 우세) vs HIGGS(무차이, Hadamard 때문)를 명시. KVQuant App. Table 20이 per-token에서 부호 반전과 기제를 기술. HyperQuant §6도 RHT 하 등가성을 명시.
2. "Hadamard/VQ 양자화기는 어차피 채널을 섞으므로 pre-RoPE 이득이 사라진다" — **과잉 일반화.** RotateKV Table 2 (회전+재배열 0.247 vs 0.605), TaSQ Fig.1d (pre-RoPE VQ 35% 낮음), NSNQuant Tab.11 (작지만 0이 아님)이 반례. 성립하는 것은 분포 무관(랜덤 Hadamard + 고정 격자, 회전 불변 코드북) 양자화기뿐.
3. "쌍 MSE ≈ 2배" — **비대칭 per-channel 양자화기에서 틀림.** KIVI/KVQuant식 zero-point 양자화에서는 부호가 일정한 outlier 채널의 평균이 회전으로 진동하여 비율이 (평균/표준편차)² 차수로 커짐. 시뮬레이션 기준 대칭 1.5, 비대칭 6.7 (평균/표준편차=5).
4. "G = 양자화 그룹" — **KVQuant에는 해당 없음.** KVQuant의 per-channel 스케일은 그룹 없이 오프라인 보정되므로, 실효 G는 시퀀스(배포) 길이 전체. Llama-2(base 1e4, d=128)에서 Gθ_f ≥ π인 쌍 수: G=32 → 17/64, 128 → 26, 512 → 36, 2048 → 46, 8192 → 55. Llama-3(base 5e5): 12, 19, 25, 32, 39. 즉 KVQuant 설정에선 고주파 집중이 아니라 대부분의 쌍이 혼합됨.
5. "저번호(고주파) RoPE 채널이 노이즈의 대부분" (KV-Latent 2507.11273) — **오인용.** KV-Latent §3.3.2의 "noise"는 head dim 축소 시 RoPE 점수-거리 함수의 진동이지 양자화 잡음이 아님.
6. outlier 쌍이 고주파에 몰려 있다는 암묵 전제 — **최대 outlier(sink bias)에 대해서는 반대.** Massive Spikes 2606.02288 Fig.3c (i≈55/64, 80% 대칭 쌍); Massive Values 2502.01563 본문도 저주파(단, 부록 라벨이 모델별로 엇갈려 신뢰도 낮음).
7. "Block-GTQ = key-energy 할당, 9.3이 query 가중을 결합" — **틀림.** Block-GTQ의 s*_i = E[‖q^(i)‖‖k^(i)‖]와 대리 s_i는 이미 query를 포함(§2.3, Eq.1).
8. "Q-ROAR/Rethinking: YaRN × W4 파괴적 상호작용 (NTK/linear PPL>100)" — **과장.** PPL>100은 NTK/linear 경우(2510.00028 l.293). YaRN s=8에서는 W4 기준선이 FP16 근처(Tab.2: 16K FP16 2.438 / RTN 2.517 / AWQ 2.476), 크게 무너지는 것은 s=32 @131K (FP16 5.386 / RTN 9.958 / AWQ 8.700 / Q-ROAR 7.869).
9. YAKV [2604.08426, U] — 실재 확인. 논문 제목은 "KV Cache Offloading for Context-Intensive Tasks"이고 수치는 PPL이 아니라 과제 정확도 2점(App. L Table 10).
10. "극좌표 양자화(PolarQuant-Wu)는 불변" — **조건부.** PolarQuant-Wu의 θ 양자화기는 그룹 min–max 비대칭이라 회전 불변이 아님(post-RoPE 각은 (0,2π)에 퍼짐). 균일한 전원 각 격자일 때만 불변.

## 수정 요구

- 헤드라인 "2년간 상반된 보고를 처음 화해"를 철회하고, AQUA-KV(2501.19392)와 KVQuant App. Table 20을 이 정성적 화해의 최초 출처로 인용. 기여를 "그 관찰에 대한 폐형식 범위 모델 + 통제 실험"으로 재정의.
- 9.1: 새 정리로 내세우지 말고, 2608.13365 Lemma 2 / FPTQuant / WUSH-KV App. A의 centraliser 특성화를 인용한 한 줄 따름정리로 축소. "Hadamard/VQ" 대신 "스케일·코드북 결정 통계가 쌍별 회전에 불변인 양자화기"로 조건을 명시하고 RotateKV(재배열), TaSQ/AnTKV(학습 VQ), PolarQuant-Wu(min–max 각)를 반례로 명기.
- 9.2: (a) 대칭 max-abs와 비대칭 min-max를 분리. 후자에서는 채널 평균(DC)의 회전이 지배항이며 비율이 2로 상한되지 않음. (b) G를 "범위 통계를 공유하는 토큰 구간"으로 정의(KIVI는 그룹 크기, KVQuant는 보정·배포 길이). (c) per-token 따름정리(혼합이 토큰별 범위를 줄여 post가 유리)를 추가해 KVQuant T20 / AQUA-KV Quanto를 설명. (d) "고주파 집중" 예측은 쌍 비대칭(max₁/max₂)과 대역 에너지를 곱한 형태로 쓰고, sink 쌍(저주파·대칭)은 0 기여로 명시.
- 9.3: 삭제하거나, Block-GTQ의 AM-GM 대리와 AATC Theorem 1을 RoPE 쌍에 적용한 비교 항목으로 강등.
- YaRN: 2510.00028 eq.(14) η_i(m)의 폐형식화로 포지셔닝. s=8은 경미, s=32에서만 심각, PPL>100은 NTK/linear로 정정. YaRN은 고주파 쌍을 보존하므로 저주파 쌍의 혼합만 바뀐다는 점을 정확히 서술.
- KV-Latent 인용 삭제. Massive Spikes(2606.02288), Round-and-Round(2410.06205), PM-KVQ(2505.18610), SpectrumQuant(ACL 2026), 2608.13365, AATC(2608.14191)를 추가.
- 실험 (b)를 "랜덤 Hadamard vs 보정 회전+재배열"로 분할하고, (c)는 균일 전원 각 격자와 min–max 각 격자를 분리. 비대칭 per-channel INT2/3 조건을 추가.
- novelty 표기를 "중"에서 "낮음(부분 선점 다수, 폐형식만 잔존)"으로 하향.

## 새 가설

1. **회전된 평균 차감으로 post-RoPE 저장에서도 pre-RoPE 이득 회수** (KV cache 양자화: key 저장 형식). 근거: 비대칭 per-channel에서 손실의 대부분이 채널 평균의 회전에서 옴(시뮬레이션 6.7× vs 대칭 1.5×). sink bias가 사실상 상수 key 벡터(2606.02288). R_m μ는 위치만으로 계산 가능하고, q·R_m μ는 logit에 닫힌 형태로 더해 줄 수 있음. 예측: post-RoPE에 k_m − R_m μ_pre를 저장하면 KVQuant Table 10의 0.82 PPL 차이 대부분이 사라지고, pre-RoPE 전용 커널(on-the-fly RoPE)이 필요 없음. 단순 post-RoPE 평균 차감(2608.02691)은 고주파 쌍에서 평균이 ≈0으로 상쇄되므로 효과가 없어야 함.
2. **스케일 공유 구간과 RoPE base가 pre-RoPE 이득의 크기를 결정** (KV cache 양자화: 그룹 크기와 스케일 보정 설계). 근거: Gθ_f ≥ π인 쌍 수가 Llama-2(base 1e4)에서 G=2048일 때 46/64, Llama-3(base 5e5)에서는 32/64. PM-KVQ의 54K 토큰 주기 관찰. 예측: KIVI식 G=32 그룹에서는 pre/post 차이가 작고, 오프라인 per-channel(KVQuant)이나 긴 스트리밍 창에서는 크며, base가 큰 모델일수록 줄어듦. Llama-2-7B와 Llama-3.1-8B에서 G ∈ {32, 128, 2048, 전체}로 검증 가능.
3. **분포 적응형 변환의 pre-RoPE 이득 = 보정 통계의 위치 정상성** (회전/변환 설계 단계). 근거: RotateKV 재배열(0.247 vs 0.605), TaSQ 학습 코드북(35%), NovaKV가 보정 길이 밖에서 급격히 나빠짐(TaSQ Fig.7). 대조적으로 HIGGS/RHT는 무차이. 예측: 재배열·코드북을 post-RoPE에서 위치 구간별로 따로 보정하면 차이가 Σ_f 1[Gθ_f ≥ π]·E_f에 비례해 줄어듦. 즉 9.2의 혼합 인자가 적응형 변환에도 같은 대역 구조로 나타남.

Sources (WebSearch): https://arxiv.org/pdf/2502.15304 (SVDq, pre-RoPE 설정), https://arxiv.org/pdf/2401.18079 (KVQuant), https://rocm.blogs.amd.com/artificial-intelligence/turboquant-vllm-agentic/README.html (vLLM TurboQuant, post-RoPE 계열 배포 사례)
