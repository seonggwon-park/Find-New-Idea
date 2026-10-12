## 아이디어 1. 이분산 로짓 노이즈의 softmax 2차 편향: 분자 보정(−σ²/2)을 넘는 p-의존 완전 보정과 single-token decoding의 토큰 간 재분배

### 가설
양자화된 key(또는 LM-head 가중치)가 로짓에 넣는 평균 0 노이즈의 분산 σ_j²가 토큰마다 다르면, E[softmax]는 분산이 큰 토큰 쪽으로 편향된다. 분산이 같으면 평탄화만 남는다. 분자를 비편향으로 만드는 −σ_j²/2 보정은 이미 출판됐다. 그러나 이 보정은 불충분하다. 정규화된 softmax에는 2차 잔여 편향이 남고, 동분산 성분은 전혀 보정되지 않는다. 2차 완전 보정 −σ_j²(½ − p_j)는 두 문제를 모두 제거하며, 그 이득은 모든 key가 양자화되는 single-token decoding에서 가장 크다.

### 문헌상 공백
- **보정식은 선점됐다.** [Quantized Keys Steal Attention 2605.26266](https://arxiv.org/abs/2605.26266) [V](ICML 2026 워크숍, 비디오 diffusion)이 다음을 유도했다.
  - 균일 노이즈의 정확해 b_i = log E[e^{δ_i}] = Σ_c log(sinh α_c/α_c) (Eq. 10–12)
  - Taylor 근사 b_i ≈ σ_i²/2 (Eq. 13–14)
  - per-channel일 때의 공유 스칼라 (App. E)
  - LLM INT2 partial-prefill NLL 개선 (App. O)

  대상은 cached→current 질량 이동이다. single-token decoding, sink, QJL, LM head, 엔트로피, FP 노이즈는 다루지 않는다.
- **정리 1의 식도 출판됐다.** [Transform Coding KV 2608.14191](https://arxiv.org/abs/2608.14191) [V] App. A의 E[δa_i^{(2)}]가 정리 1과 대수적으로 같다. 다만 MSE 상한의 보조항으로만 쓰였다. 원형은 Gaussian softmax delta-method([1703.00091](https://arxiv.org/abs/1703.00091) [S])다.
- **D'Alberto는 같은 틀을 정성적으로 제시했다.** [D'Alberto 2605.08114](https://arxiv.org/abs/2605.08114) [V]는 §3.1 Eq.(4) e^{σ_i²/2}와 "균일하면 상쇄, 점수와 상관되면 비대칭", 그리고 §4 K–V 규칙을 제시했다. QJL-K 분산비는 (π/2)/ε_B ≥ π/2, 동일 예산에서는 2π다. 닫힌 편향식과 보정은 없다. 또 "균일하면 상쇄"는 틀렸다: 평탄화가 남는다. [2607.09683](https://arxiv.org/abs/2607.09683) [V]은 σ² ∝ ‖잔차‖²/m을 보였다.
- **QJL**([2406.03482, AAAI 2025](https://arxiv.org/abs/2406.03482) [V])은 softmax 이전의 비편향(Lemma 3.2)과 점수별 (1±3ε)(Theorem 3.6)만 증명한다. vLLM은 softmax 분산 증폭을 이유로 QJL을 제외했다 [V].
- **KIVI**([2402.02750, ICML 2024](https://arxiv.org/abs/2402.02750) [V])의 축 선택은 모두 경험적 근거다. 정리와 편향 분해가 없고, 스트리밍은 구현상 이유다.
  - K per-channel: outlier 채널(점수 오차 47.00→9.60, Table 2)
  - V per-token: 희소성 84.3%와 출력 오차 49.89→3.55
- **KIVI 계열 후속.**
  - [KVLinC 2510.05373](https://arxiv.org/abs/2510.05373) [V]는 KIVI 축을 유지하고 V에만 Hadamard를 쓴다. 근거는 경험적 설명이다.
  - [OSCAR 2605.17757](https://arxiv.org/abs/2605.17757) [V]는 logit 오차의 2차 모멘트만 다뤄 편향과는 인접하지 않는다. sink·recent 보호를 끄면 점수가 0.00이다(Table 5).
- **어텐션 민감도 이론.** [2608.20988](https://arxiv.org/abs/2608.20988) [V](1차 Jacobian; peaked 맹점을 스스로 인정)과 [SpectrumKV 2606.08635](https://arxiv.org/abs/2606.08635) [V](e^{±2η}; Lemma 2는 스케치, 엔트로피 임계는 Conjecture 1)의 "반대 직관"은 절대오차 대 상대오차의 차이다. 둘 다 편향을 다루지 않는다.
- **sink.** sink의 key norm은 작다([2410.10781, ICLR 2025](https://arxiv.org/abs/2410.10781) §3.1 [V], [KVSink 2508.04257, COLM 2025](https://arxiv.org/abs/2508.04257) [V], [OScaR 2605.19660](https://arxiv.org/abs/2605.19660) [V]). sink가 섞인 per-channel 그룹은 오차가 16.3–29.2% 크다(KVSink Table 1).
- **출력 분포.**
  - GPTQ 4-bit 보정오차 증가의 출처는 [2405.00632, Findings of NAACL 2024](https://arxiv.org/abs/2405.00632) [V]다. 2604.08976은 오인용이다.
  - 반례: NF4에서 ECE 방향이 혼재하고 적합 T = 1.25–1.67이다([2608.03854](https://arxiv.org/abs/2608.03854) [S]).
  - [ReSET 2606.13233](https://arxiv.org/abs/2606.13233) [V]: 전체 모델 W4A4에서 엔트로피가 양방향으로 이동한다. LM head를 분리하지 않았고 보정도 없다.
  - [2609.11716](https://arxiv.org/abs/2609.11716) [V] Thm 3: Δlog p_k = Δz_k − E_pΔz − ½Var_p(Δz). 결정론적이다.
  - [2609.31291](https://arxiv.org/abs/2609.31291) [V]: 행 평균 이동으로 KL을 73–93% 줄인다.
  - 노이즈 기댓값 보정은 없다.

### 수학
설정은 다음과 같다. ŝ_j = s_j + ε_j, E ε_j = 0, Var ε_j = σ_j², 토큰 간 독립, p = softmax(s).

정리 1 (2차 편향; delta-method, 2608.14191 App. A와 동치). ∂²p_i/∂s_j² = p_i[(δ_ij − p_j)² − p_j(1 − p_j)]이므로

E[p̂_i] − p_i ≈ (p_i/2)·[σ_i²(1 − 2p_i) − Σ_j p_jσ_j²(1 − 2p_j)] + O(σ⁴).

- 동분산: σ²p_i(‖p‖₂² − p_i). 평균 맵의 엔트로피는 2차에서 증가한다(Chebyshev 합 부등식).
- p_i ≪ 1: E[p̂_i]/p_i ≈ 1 + ½(σ_i² − σ̄_p²), σ̄_p² = Σ_j p_jσ_j².

보조정리 2 (완전 보정).
- ŝ_j − σ_j²/2는 분자를 비편향으로 만든다(CGF). 그래도 잔여 편향 p_i(Σ_j p_j²σ_j² − p_iσ_i²)가 남고, 동분산에서는 효과가 0이다.
- ŝ_i − σ_i²(½ − p_i)는 정리 1의 편향을 2차까지 정확히 상쇄한다. 동분산에서는 p_i에 비례하는 sharpening이 된다. p̂를 대입해 생기는 오차는 3차다.

수치(부록 C)는 n=64, σ_j² ∈ U[0, 0.3]에서 측정했다.
- 예측과 측정의 상관은 0.9996, 최대 상대 오차는 3.9%다.
- 동분산일 때 엔트로피는 3.0635 → 3.0681로 증가한다.
- 심사 MC의 |bias|₁(무보정 / −σ²/2 / 완전 보정): 이분산 0.0369 / 0.0064 / 0.00084, 동분산(σ²=0.2) 0.00645 / 0.00633 / 0.00072.

적용:
- **QJL.** QJL Eq.(5)에서 Var = (‖q‖²‖k_j‖²/m)(π/2 − cos²θ_j)가 나오지만, 논문은 이를 명시하지 않는다. 이 분산은 key의 노름과 q와의 정렬에 따라 달라지므로 이분산이다. ν_j = ‖k_j‖는 이미 저장되어 있어, 보정에는 key당 곱셈 몇 번이면 된다.
- **per-token key.** σ_j² = Σ_c q_c²Δ_j²/(12d)(2605.26266 Eq. 7)이다. sink는 norm이 작아 Δ_sink < Δ̄일 가능성이 높고, 그러면 sink 질량 **감소**가 예측된다. 초안의 "증폭"은 부호가 틀렸을 가능성이 높다.
- **per-channel key.** 동분산은 그룹 안에서만 성립한다. 그룹 간 Δ_c 차이, sink 오염, 잔차 창 때문에 그룹 사이의 이분산은 남는다. 2605.26266 App. E에서는 per-token에 보정을 더한 쪽이 per-channel보다 낫다. 따라서 "key는 per-channel"은 보정이 없을 때만 맞는 처방이다.
- **FP4 key.** 상대오차이므로 σ_i² ∝ Σ_c q_c²k_ic²·2^{−2m}이고, 큰 norm 쪽 편향이 예측된다 [S].
- **LM head.** 이분산 성분 σ_v² = ‖h‖²Δ_v²/12는 −σ_v²(½ − p_v)로 보정된다. 온도 성분은 −σ²/2로 보정되지 않으므로, 초안의 "1회 보정"은 철회한다.

증명 가능한 것은 정리 1, 보조정리 2, 엔트로피 증가다. 추측으로 남는 것은 두 가지다.
- σ²가 2차 전개가 성립할 만큼 작은가. QJL m=256에서는 전개가 깨지고 winner flip이 우세해질 수 있다.
- sink 질량 변화의 부호.

### 판정 실험(1–2 GPU·일)
Llama-3.1-8B와 Qwen3-8B, fake-quant, single-token decoding으로 한다.
0. Δ_j, Δ_c, Δ_sink/Δ̄를 측정해 sink의 부호 예측을 고정한다.
1. 설정: key per-token vs per-channel(4/3/2비트) × {무보정, −σ²/2(2605.26266), −σ²(½−p̂)} × sink FP16 보존 on/off.
   - 측정: sink 질량, 어텐션 엔트로피, LongBench/RULER.
   - 예측: sink 질량 변화가 ½(σ_sink² − σ̄_p²)를 따르고, 완전 보정이 sink 보존 효과의 일부를 대체한다.
2. QJL/TurboQuant(m ∈ {128, 256, 512})에서 비교한다.
   - baseline: 2605.26266 per-token Taylor, 2607.09683.
   - 제안: ‖k_j‖ 기반 −σ̂²/2, −σ̂²(½−p̂).
   - 측정: NIAH·RULER. 예측: PolarQuant 단독 수준에 접근한다.
3. LM head INT4(g128)에서 ECE·엔트로피·MMLU를 측정한다. baseline은 2609.31291의 행 평균 이동이다.

실패 기준: decoding에서 완전 보정의 이득이 −σ²/2 대비 노이즈 수준이거나, sink 질량 변화의 부호가 실험 0의 예측과 반대면 기각한다.

### 평가
novelty는 낮음–중간(증분)이다.

선점된 것:
- −σ²/2 보정과 공유 스칼라는 2605.26266에 있다.
- 정리 1의 식은 2608.14191 App. A에 있다(원형 1703.00091).
- 해석 틀은 D'Alberto가 정성적으로 제시했다.

초안의 "단일 설명" 중 무너진 것:
- sink 증폭은 부호가 반대일 가능성이 높다.
- KIVI 처방은 보정을 넣으면 뒤집힌다.

남는 것:
- (a) 정규화 softmax의 잔여 편향과 p-의존 완전 보정(동분산 평탄화 포함).
- (b) 2605.26266이 남긴 single-token decoding의 cached 내부 재분배.
- (c) QJL 사영 노이즈와 FP4 상대오차용 σ̂_j².
- (d) LM-head 판(2609.11716과 2609.31291이 인접).

main-track 단독은 낮다. (a)+(b)가 decoding RULER/LongBench에서 2605.26266보다 뚜렷이 나으면 중간이고, 아니면 부록 A로 내린다.

---
