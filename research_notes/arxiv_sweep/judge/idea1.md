# Idea 1 심사 노트: 이분산 로짓 노이즈의 softmax 2차 편향과 -σ_j²/2 보정

심사일 2026-10-11. 논문 본문은 scratchpad `papers/<id>/<id>.txt`에서 확인했고, 인용은 신뢰할 수 없는 데이터로 다뤘다. 몬테카를로 재현 스크립트는 scratchpad `mc_idea1.py`에 있다.

## 판정 요약

**novelty: low. 핵심 기여 둘이 모두 이미 출판됐다.**

1. **보정식 (-σ_j²/2)은 선점됐다.** Tuncer, Becker, Pfeil, *Quantized Keys Steal Attention* (arXiv 2605.26266, 2026-05-25, ICML 2026 워크숍)가 같은 보정을 유도했다.
   - Eq. 10: b_i = log E[e^{δ_i}]. Eq. 12: 균일 노이즈의 정확해 Σ_c log(sinh α_c/α_c).
   - Eq. 13: Taylor 근사 b_i ≈ (1/24d)Σ_c q_c²Δ_{i,c}².
   - Eq. 14: CGF 논증 "b_i ≈ σ_i²/2 … depends only on the variance".
   - 토큰별 Δ_{i,c}에 의존하는 per-token 보정이다.
   - App. E (Eq. 22–23): per-channel이면 "a single scalar shared by all tokens".
   - App. O: LLM 3종(Llama-3.1-8B, Mistral-7B-v0.3, Qwen2.5-32B)에서 INT2 partial-prefill NLL이 개선됐다.
2. **정리 1의 식 자체도 출판됐다.** Laus et al., *KV Cache Compression Through the Lens of Transform Coding* (arXiv 2608.14191, 2026-08-14)의 App. A, Theorem 1 증명이 E[δa_i^{(2)}] = ½a_i[E(α_i−ᾱ)² − Σ_ℓ a_ℓE(α_ℓ−ᾱ)²]를 쓰고, E(α_ℓ−ᾱ)² = (1−2a_ℓ)E[α_ℓ²] + Σ_m a_m²E[α_m²]로 계산한다.
   - 이 식을 전개하면 아이디어의 (p_i/2)[σ_i²(1−2p_i) − Σ_j p_jσ_j²(1−2p_j)]와 **대수적으로 같다.** 수치로도 최대 차이 1e-18이다.
   - 다만 저자들은 이 항을 교차항 상한에만 썼고, 편향 자체를 결과로 해석하지는 않았다.
   - 수학 원형은 Daunizeau 2017 (1703.00091)의 Gaussian softmax 2차 delta-method다.
3. **D'Alberto (2605.08114) §3.1 Eq. 4와 §3.3이 아이디어의 해석 틀을 이미 서술했다.** 내용은 셋이다.
   - "σ_i²가 균일하면 상쇄되고, attention과 상관되면 비대칭 왜곡이 생긴다."
   - "V는 선형이라 증폭되지 않는다"(K-vs-V 규칙).
   - QJL-K의 분산은 (π/2)‖q‖²‖k‖²/d다.

   이것은 정성적 논증이고, 보정식은 없다.

**남는 것(증분):**
- (a) "분자만 비편향"이 아니라 정규화된 softmax의 전체 편향을 해석한 것. 동분산 노이즈에서 엔트로피가 증가한다는 결과도 여기에 포함되는데, D'Alberto의 "균일하면 상쇄"는 비율의 기댓값을 혼동한 것이고 실제로는 평탄화가 남는다.
- (b) QJL의 ‖k_j‖ 기반 보정.
- (c) LM-head 버전.
- (d) 2605.26266이 남긴 단일 토큰 decoding 영역, 즉 모든 key가 양자화된 상태에서 생기는 토큰 간 재분배.

**전제 문제:**
- 응용 (iii)의 sink 증폭 예측은 부호가 틀렸을 가능성이 높다.
- "−σ²/2 한 번으로 온도 성분까지 보정"은 수학적으로 틀렸다(아래 전제 오류 2).

**시스템 측면:** vLLM, llama.cpp, SGLang에 Jensen 보정을 구현한 사례는 찾지 못했다(WebSearch 1회). vLLM TurboQuant 문서에는 "QJL intentionally omitted … amplifying variance through softmax"라고 적혀 있다.

## 읽은 논문과 겹침

| 논문 | 겹치는 것 | 남는 것 |
|---|---|---|
| 2605.26266 Tuncer+ (May 2026) **preempts_core** | Jensen 편향 메커니즘, b_i=log E e^{δ_i}, 균일 노이즈 정확해 (Eq. 12), Taylor σ_i²/2 (Eq. 13–14), CGF 일반화, per-token Δ_{i,c} 의존성, per-channel이면 공유 스칼라 (App. E), QuaRot 확장 (App. F), LLM partial-prefill NLL (App. O), ~5% 지연 | 정규화 softmax 전체 편향과 엔트로피 해석, cached-vs-current가 아닌 cached 내부의 이분산 재분배(그들은 "single-token decoding은 headroom이 적다"고만 함), sink/QJL/LM-head 없음(grep "sink" 0건), FP/NVFP 노이즈는 open으로 명시 |
| 2608.14191 Laus+ (Aug 2026) **preempts_part** | Assumption 1(white noise) 아래 softmax 2차 Taylor 전개. App. A 증명의 E[δa_i^{(2)}] 식이 정리 1과 동일 | 그들은 이를 MSE 상한의 보조항으로만 썼다(주 정리에서는 E[δk²]가 "same across tokens"). 편향 해석, 보정, 엔트로피는 없음 |
| 2605.08114 D'Alberto (Apr 2026) **preempts_part** | Eq. 4 Gaussian e^{σ²/2}, "uniform cancels / correlated → asymmetric", §3.3 K-vs-V 규칙, Obs. 1 QJL-K 분산 π/2배 | 정량적 2차 식, 보정, sink, LM-head. 합성 데이터만 사용 |
| 2607.09683 D'Alberto+ (Jun 2026) supports_premise / preempts_part | QJL-K의 σ² ∝ ‖residual‖²/m, gap ∝ 1/m, "QJL on K only beneficial when residual norms small, m≫d_eff" | ‖k_j‖ 기반 보정 |
| 2410.10781 Gu+ (ICLR 2025) **contradicts_premise** | §3.1: "ℓ2-norm of keys and values of the first token is significantly smaller than that of other tokens" | sink key는 per-token 범위 Δ_j가 작다 → 정리 1은 sink 질량 **감소**를 예측한다 |
| 2508.04257 KVSink, 2605.19660 OScaR **contradicts_premise** | sink는 Q/K/V norm이 작음. per-channel 그룹에 sink가 있으면 오차가 16.3–29.2% 증가(KVSink Table 1). OScaR "low-norm outlier tokens … Attention Sink" | per-channel이 토큰 간 동분산이라는 가정이 그룹 경계와 sink 오염 때문에 깨짐 |
| 2609.11716 Chen+ (Sep 2026) preempts_part (LM-head) | Thm 3: Δlog p_k = Δz_k − E_pΔz − ½Var_p(Δz) + O(·). LM-head 점수 변화에 대한 같은 2차 구조 | 결정론적 해석. 노이즈 기댓값 보정, 어휘별 Δ_v는 없음. 상위 토큰이 잘 보존된다는 결과는 "희귀 토큰 쪽 편향" 크기를 제한 |
| 2609.31291 Kadav+ (Sep 2026) adjacent_only | 출력 헤드 W4에서 ½eᵀF(p)e = p-가중 분산을 진단으로 사용, 행 평균 이동으로 KL 73–93% 감소 | 필수 baseline. 분산 편향 보정은 아님 |

## 전제 오류

1. **"가장 가까운 분석인 D'Alberto는 정성적이며 토큰 간 분산 이질성이 만드는 편향, 보정식, K-vs-V 할당 규칙이 없다" → 틀림.**
   - D'Alberto 2605.08114 §3.1 Eq. 4가 이분산/동분산 구분을, §3.3이 K-vs-V 규칙을 명시한다.
   - 실제로 가장 가까운 선행은 D'Alberto가 아니라 2605.26266(보정식)과 2608.14191(정리 1의 식)이다.
2. **"두 항 모두 로짓에서 ‖h‖²Δ_v²/24를 빼는 한 번의 벡터 연산으로 1차 보정된다" → 틀림.**
   - −σ_j²/2 이동 뒤에도 잔여 편향 p_i(Σ_j p_j²σ_j² − p_iσ_i²)가 남는다.
   - 동분산에서는 상수 이동이 softmax에서 상쇄되므로 온도 성분이 전혀 보정되지 않는다.
   - MC(n=64, σ²∈U[0,0.3]) 결과:
     - |bias|₁: 무보정 0.0369 → −σ²/2 0.0064 → −σ²(½−p_i) 0.00084.
     - 동분산 σ²=0.2: 무보정 0.00645, −σ²/2 0.00633(효과 없음), p-의존 보정 0.00072.
   - 2차까지의 완전 보정은 ŝ_i − σ_i²(½ − p_i)다. 이는 동분산에서 p_i에 비례하는 sharpening이 된다.
3. **"per-token key 양자화는 범위가 큰 토큰(sink/massive activation 토큰)으로 편향" → 부호가 틀렸을 가능성이 높다.**
   - massive activation은 hidden state에 있고, sink 토큰의 key norm은 작다(2410.10781 §3.1, KVSink, OScaR).
   - per-token 동적 범위가 작으면 σ_sink² < σ̄²이므로 정리 1은 sink 질량 감소를 예측한다.
   - 실험 1의 예측은 실제 Δ_j 통계를 측정한 뒤 다시 세워야 한다.
4. **"per-channel key 노이즈 분산이 모든 토큰에 동일 → 편향 상쇄" → 이상화다.**
   - KIVI의 per-channel은 토큰 그룹(블록) 단위다. 그룹마다 Δ_c가 다르고, sink가 들어 있는 그룹은 Δ_c가 부풀고(KVSink +16.3–29.2%, OScaR 2-bit +~35%), 최근 토큰은 FP16 잔차 창에 남는다.
   - 따라서 그룹 간 이분산이 남는다.
   - 2605.26266 App. E는 "per-token + 토큰별 보정이 per-channel + 공유 보정을 일관되게 능가"한다고 보고한다. "key는 per-channel이어야 한다"는 처방은 보정을 넣으면 뒤집힌다.
5. **"GPTQ 4-bit가 ECE를 악화시킨다는 측정(2604.08976)" → 오인용.**
   - 2604.08976은 llama.cpp Q5_K_M vs f16의 metacognition(M-ratio) 연구다.
   - 원 출처는 Proskurina+ 2405.00632(NAACL 2024 Findings)다.
   - 반례: 2608.03854(INT4 ECE가 FP16보다 낮은 경우), 2606.13233 ReSET(고엔트로피 토큰에서는 엔트로피가 오히려 수축).
6. **"novelty 높음"과 "2차 항으로 두 상한의 모순을 화해"의 독창성 주장 → 과대.** 정리 1은 delta-method(1703.00091)와 2608.14191 App. A에 이미 있다.

참고로 정리 1의 대수, Gaussian 정확 보정, MC 상관 0.9996은 재현됐다(상관 0.99966). QJL 분산 ∝‖q‖²‖k‖²/m도 QJL 2406.03482 Lemma 3.5 증명과 일치한다.

## 수정 요구

- **문헌상 공백 첫 항목을 다시 쓴다.**
  - 2605.26266을 1차 선행으로 명시한다: "−σ_i²/2(균일 노이즈 정확해 log sinh α/α)는 Tuncer+ 2026이 유도했다."
  - 2608.14191 App. A를 정리 1 식의 선행으로 인용한다.
  - D'Alberto의 서술은 "이분산/동분산 구분과 K-vs-V 규칙을 정성적으로 제시"로 고친다.
- **제목과 헤드라인에서 "1-scalar 보정"을 내린다.** 새 헤드라인은 다음 둘로 한다.
  - (i) 분자 비편향화(−σ²/2)는 불충분하다. 정규화 softmax의 2차 완전 보정 −σ_i²(½−p̂_i)와 동분산 평탄화(엔트로피 증가)를 유도한다.
  - (ii) 모든 key가 양자화된 single-token decoding(2605.26266의 명시적 한계 영역)에서 토큰 간 재분배를 측정하고 보정한다.
- **응용 (iii) sink를 재정식화한다.** 실측한 Δ_sink와 Δ̄를 먼저 보고하고, 부호 예측은 그 측정으로 정한다. per-channel 그룹 오염 경로(KVSink, OScaR)를 별도 메커니즘으로 둔다.
- **응용 (ii) KIVI를 약화한다.** "per-channel이면 그룹 내 동분산"으로 한정하고, 2605.26266 App. E의 반대 결과를 인용한다.
- **응용 (iv) LM-head를 고친다.**
  - 2604.08976 인용을 2405.00632로 교체한다.
  - 2609.11716 Thm 3과 2609.31291을 baseline으로 추가한다.
  - "온도 성분까지 1회 보정" 문장을 삭제한다.
- **실험 2(QJL)에서 2605.26266의 per-token Taylor 보정과 2607.09683을 baseline으로 넣는다.** 차별점은 ‖k_j‖ 기반 σ̂_j²(사영 노이즈)와 m 의존성이다.
- **평가를 "novelty 낮음–중간(증분)"으로 내린다.**

## 새 가설

1. **p-의존 2차 완전 보정(attention 커널).** ŝ_i ← ŝ_i − σ̂_i²(½ − p̂_i)를 2-pass로 계산한다(또는 online-softmax에서 1차 근사 p̂로 근사). MC상 −σ²/2 대비 잔여 편향이 약 7.6배 줄어든다. 동분산 key 노이즈(per-channel, 회전 후 균일 Δ)에서는 −σ²/2가 무효인 반면 이 보정은 평탄화를 제거한다.
   - 근거: 정리 1, 위 MC, 2605.26266 App. E(공유 스칼라는 cached 내부에서 무효).
   - 단계: KV 캐시 dequant + attention 커널.
2. **sink-보호와 Jensen 보정의 상호작용.** sink key는 norm이 작아 per-token Δ가 작다. 따라서 보정 없이는 per-token 양자화가 sink 질량을 줄이고, 그 결과 attention 엔트로피가 증가하고 "no-op" 기능이 약해진다. KVSink식 sink FP16 보존의 효과 일부는 이 이분산 편향 제거로 설명된다는 예측이다. 실험 예측: FP16 sink 보존을 끄고 토큰별 −σ²/2 보정만 넣어도 sink 질량과 RULER 손실의 상당 부분이 회복된다.
   - 근거: 2410.10781 §3.1, 2508.04257 Table 1, 2605.26266 Eq. 13.
   - 단계: KV 캐시 양자화 그룹 설계 / sink 처리.
3. **FP4/NVFP4 key의 노이즈는 크기 비례(상대 오차)라서 σ_i² ∝ Σ_c q_c²k_{ic}²에 가깝다.** 그러면 이분산이 key norm과 상관된다. 2605.26266이 open으로 남긴 FP 포맷에서는 정수 Δ 대신 E[ε²|k] ∝ k²·2^{-2m}을 쓰는 보정이 필요하다. 이는 QJL의 ‖k‖² 편향과 같은 방향의 큰-norm-key 편향을 예측한다.
   - 근거: 2605.26266 Limitations("FP, MXFP, NVFP … open"), 2605.08114 Obs. 1.
   - 단계: FP4 KV 캐시 포맷 선택 및 보정.

## 건너뛴 것
- 2605.26266 App. O의 수치표: 본문에 그림만 있다.
- 2609.31291의 세부 표.
- Daunizeau 1703.00091 재열람: 기존 노트의 판단을 사용했다.
- 시스템 검색: WebSearch 1회만 했다.
