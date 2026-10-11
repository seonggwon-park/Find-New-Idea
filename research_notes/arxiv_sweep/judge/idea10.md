# 아이디어 10 판정: MLA latent KV의 가중 변환 부호화 (weight absorption 하 최적 변환·할당)

Date: 2026-10-11. Judge: deep reader + adversarial novelty judge (single idea).
입력: `reports/LLM 양자화 연구 아이디어 10선.md` §아이디어 10 (l.371–), `discover/idea10_citations.md`, `discover/idea10_keywords.md`.
논문 텍스트는 untrusted data로만 인용했다. 경로: `/tmp/claude-0/-home-user-Find-New-Idea/b6387454-6f57-570f-bcca-f4bcd8a754c6/scratchpad/papers/<id>/<id>.txt`.

## 판정 요약

- **Novelty: low** (표 표기: 낮음(부분 선점: QuantMLA 2609.36760 + NOVA-KV 2608.04074)).
- **이론(정리 10.1, 10.2)은 새롭지 않다.** 정리 10.2의 공식 T = U^T M^{1/2}, U = eig(M^{1/2} Σ M^{1/2})는 NOVA-KV(2608.04074) Theorem 1에 그대로 있고(키 쪽, "Orthogonal transforms are suboptimal for the high-resolution regime unless M_q ∝ I", l.475–476), 근거는 Linder–Zamir–Zeger 1999의 sensitivity-matrix 고해상도 이론이다. 헤드 합 M(공유 캐시)은 NOVA-KV App. eq.(38)(GQA, "second moment over the heads sharing each KV head")에 있다. 정리 10.1의 오차 경로(로짓 a^{hT}e, 출력 A_h ΔC B_h)는 QuantMLA eq.(1)–(3)에서 흡수 형태로 정확히 쓰였고, 출력 쪽 W_O 가중 + reverse water-filling은 AATC(2608.14191) Theorem 1 / SPECTRA(2608.07915) eq.(5)에 있다.
- **시스템/방법도 같은 자리를 QuantMLA(2609.36760, 2026-09-29)가 차지했다**: native MLA, 흡수 형태, content latent에 학습 회전 R_C + 대각 스케일 S_C를 오프라인 fold(eq.6, App. C.1), RoPE 가지는 SO(2) 쌍 회전 + 쌍 스케일(Prop. 1), INT4/INT2 latent + FlashMLA 기반 커널, 6개 MLA 모델(DS-V2-Lite, V3-Base, R1, Kimi-K2, LongCat-Flash-Lite, GLM-4.7-Flash).
- **남는 것(좁다)**: (i) 공유 latent에 대한 K·V 결합 가중 M_K + λM_V의 닫힌형 KLT(NOVA-KV는 "Joint quantization of keys and values is left for future work", l.1255; QuantMLA는 수치 학습); (ii) native MLA latent 방향별 water-filling(0비트 방향 포함) — QuantMLA는 균일 per-token g64 affine(l.1130), eigen/water-filling grep 0건; (iii) learned(QuantMLA) vs closed-form KLT 정면 비교; (iv) M^{1/2}Σ_cM^{1/2} 유효 랭크 실측; (v) 아래 전제 오류 4에서 나온 **RMSNorm 제약 하 최적 변환** 문제(이것이 오히려 가장 MLA 특유의 이론적 구멍).
- 핵심 전제 하나가 틀렸다: "변환이 W_UK/W_UV/W_O에 접혀 런타임 비용 0". MLA latent는 RMSNorm(kv_a_layernorm) 이후 값이 캐시되므로, 생산자 쪽 fold는 직교(×게인 대각)인 변환만 가능하다(WUSH-KV App. A eq.(11): T Norm(x) = Norm(Tx) ∀x ⇒ T^T T = I). 비직교 최적 T는 캐시 쓰기 시 토큰당·층당 d_c×d_c(512×512) online matvec가 필요하다(소비자 쪽 fold는 여전히 무비용).

## 읽은 논문과 겹침

| 논문 | 겹침 분류 | 겹치는 것 (위치) | 남는 것 |
|---|---|---|---|
| QuantMLA 2609.36760 (2026-09-29, under review) | preempts_part (최대 위협) | 흡수 형태 eq.(1); 오차 경로 eq.(2)–(3); H_b = J_b^T J_b 국소 이득(App. B.3, JVP vs finite Spearman 0.986); content 변환 z_C = u R_C S_C, c = uΓ, u = RMSNorm(x), "weightless RMSNorm is rotation-equivariant, R_C can be fused offline" (l.748–769); RoPE Prop. 1 (l.773); Table 1 DS-V2-Lite GSM8K BF16 36.92 / C4R4 RTN 16.98, QuaRot† 30.71, QuantMLA 35.25; C2R4 GSM8K 37.91; RoPE/content 반응비 1.941×–6.864× (l.413); "first reported joint INT4 caching of the content and RoPE caches" (l.354) | 닫힌형 M; K+V 결합 가중; 가변 비트/0비트; 비직교 변환; per-token vs per-channel RoPE ablation |
| NOVA-KV "Spend Bits Where Queries Look" 2608.04074 | preempts_part (이론 핵심) | Thm 1: R* = M_q^{1/2}E, E = eig(M_q^{1/2} S̃_k M_q^{1/2}) (l.492–); Linder 1999 인용 (l.466, 475); eq.(38) GQA 헤드합 M (l.3099–3135); Thm 2 비트/그룹 할당; 키 합성변환은 query에 fold | MLA·latent 미취급; K/V 결합 미취급(l.1255); 생산자 쪽 RMSNorm 제약 미취급 |
| WUSH-KV 2609.38121 (2026-09-29) | preempts_part / contradicts_premise(fold 비용) | H = 2WW^T 가중 데이터 적응 변환, Thm 2 근사 최적성; value 변환 W_V/W_O fold; App. A eq.(10)–(11): RMSNorm 통과 fold ⇒ T^T T = I, 게인과 가환 (l.1066–1110); 키 변환은 post-RoPE online 유지(l.907–909) | MLA content latent는 RoPE가 없어 키 쪽 소비자 fold가 가능 — 이 점이 진짜 MLA 이점. 단 생산자 쪽은 같은 RMSNorm 제약 |
| SPECTRA 2608.07915 | preempts_part (방법 형태) | G = E[h^T h] 가중 SVD latent, reverse water-filling, "A channel assigned zero bits is simply dropped" (l.175), W_down/W_up/W_O 보정 fold (l.387, 447); MLA를 "same latent-cache idea ... post-hoc"로 명시 (l.1570–1572) | 가중이 입력측(G)일 뿐 query/output 인지 M 아님; native MLA 실험 없음 |
| AATC 2608.14191 | preempts_part (출력측 가중) | Thm 1 (l.451): D = D_K + D_V + R, q_c², W_O 열 노름 가중; reverse water-filling; OSCAR 회전과 할당을 같이 쓰면 "double-counts" (l.2761–2774) | 대각 가중만; MHA/GQA만 |
| DeepSeek-V4.1-Flash 2609.19969 | contradicts_premise (RoPE 판별 서사) / supports (배포 모델) | §2.4.4 (l.1000–1028): E2M1 + E4M3/16, QAT, "The non-RoPE and RoPE components use the same quantization format. We quantize the cache after RoPE: quantizing before RoPE yields only a marginal accuracy improvement"; 512-ch latent L2 ≤ √512≈22.6, 관측 최대 ≈10; SWA KV는 FP8 유지; FP4 KV 정량 ablation 표 없음("marginal") | V4/V4.1은 CSA2 + RoPE 64차원이 RMS 정규화된 512 latent 안에 들어간 구조로, V2/V3의 분리 k_R과 다르다 |
| RoPE-Aware Bit Allocation 2606.24033 | supports (RoPE per-channel 예측) | DS-V2-Lite 분리 RoPE-key 진단: "KIVI wins both axes at every budget, by a wide margin at 2 b/dim (top-10 68.0% vs 49.9%) ... the single decoupled RoPE-key is consumed by all 16 query heads" (l.4496–4501); latent는 "outside its scope" (l.3237) | latent·V·실제 정확도 미측정(진단 지표만) |

추가 확인(새 검색 3건, 2026-09 이후 MLA 양자화): QuantMLA와 VestigeKV 외 신규 없음. "absorption + quantization" 조합 0건 유의미.
시스템(웹 2건): SGLang NVFP4 KV 블로그(2026-09-16, Qwen 계열 평가, MLA 결과 없음) [S]; FlashInfer `nvfp4_quantize_append_sparse_mla_cache`(DeepSeek-V4 512-elem latent 행, 384 B/row) [S]; SGLang 문서 스니펫에 DeepSeek-R1-0528 FP4 KV GSM8K 0.9154(FP8)→0.9124, AIME25 0.4934→0.4000 [S, WebFetch DNS 실패로 미검증]. 출시 시스템 중 latent에 회전/KLT를 쓰는 것은 발견 못 함.

## 전제 오류

1. "latent에 회전·변환을 적용한 문헌은 검색에서 없었다" → **틀림.** QuantMLA(2609.36760)가 MLA content latent에 학습 회전+스케일, RoPE 가지에 쌍 회전+스케일을 적용·오프라인 fold. SPECTRA(2608.07915, post-hoc latent), Palu(2407.21118, Hadamard를 latent 재구성 행렬에 fold)도 선례.
2. "MLA 양자화 민감도 데이터는 2603.04428과 KVarN뿐" → **구식.** QuantMLA Table 1/4(6개 MLA 모델, RoPE/content 반응비 1.941–6.864×), SnapMLA Table 1(DS-V3.1, LongCat), 2606.24033(DS-V2-Lite RoPE key), NVIDIA TRT-LLM 블로그(DS-R1 GSM8K FP8 MLA+KV 0.9613 vs BF16 0.9629, discover 노트 [S]), SGLang 문서 FP4 KV on R1-0528 [S]. KVarN 논문 본문에는 GLM-4.7-Flash/MLA 결과가 없음(repo README에만) — 출처를 README로 명시해야 함.
3. "MLA는 모든 헤드가 latent를 공유하므로 M이 헤드 합으로 정의되는 점이 구조적으로 다르다" → **틀림.** NOVA-KV eq.(38)이 GQA에서 헤드합 M을 이미 정식화. MLA는 H→전체 헤드, k→c의 특수화일 뿐.
4. "T는 선형이므로 W_UK, W_UV, W_O에 접혀 런타임 비용이 0" → **절반만 맞음.** 캐시되는 latent는 RMSNorm(kv_a_layernorm) 출력(QuantMLA: c = uΓ, u = RMSNorm(x), l.748). 생산자 쪽에서 RMSNorm을 통과해 fold 가능한 변환은 직교×(게인 대체) 대각뿐(WUSH-KV App. A eq.(11); QuantMLA가 R_C S_C로 제한한 이유). 정리 10.2의 비직교 T = U^T M^{1/2}는 캐시 쓰기에서 토큰·층당 512×512 matvec 필요(소비자 쪽 a^h ← T^{-T}a^h, W_UV T^{-1}는 무비용). 그리고 NOVA-KV 말대로 M ∝ I가 아니면 직교 변환은 고해상도에서 차선 — 즉 실제 trade-off가 존재.
5. 수학 절의 "c_j = W_DKV h_j" → RMSNorm 누락. c_j = Γ·RMSNorm(W_DKV h_j)로 고쳐야 하며, 정리 10.2의 fold 문장은 이 제약을 반영해야 함.
6. "FlashMLA V4.1(양자화) vs SnapMLA(고정밀)의 상반된 선택은 per-token vs per-channel로 판별될 것" → **오도.** V4.1은 (a) QAT이고, (b) RoPE 64차원이 RMS 정규화된 512 latent 내부에 있어 크기가 √512≈22.6로 묶이며(관측 ≈10), (c) after-RoPE 양자화(2609.19969 §2.4.4). SnapMLA의 분리 k_R(LongCat)은 정규화 없이 ±10^3 범위(Fig 3a). 판별 변수는 granularity만이 아니라 QAT 여부와 "RoPE 성분이 정규화 아래 있는가"다. per-channel 쪽 데이터 포인트는 2606.24033(DS-V2-Lite, 68.0% vs 49.9%)이 이미 있음.
7. "V4.1용 FP8 528 B/token, FP4 288 B/token, V3.2 RoPE bf16" → **맞음**(FlashMLA README, QuantMLA l.317). 단 V4.x는 V2/V3식 W_UK/W_UV 흡수 MLA와 구조가 다르므로(CSA2, 512 안에 RoPE 포함) "M = Σ W_UV^T W_UV" 형태를 그대로 V4.1에 적용한다고 쓰면 안 됨.
8. 평가 절 "OSCAR/Spend-Bits 저자들이 MLA로 확장하는 데 몇 달" → 리스크는 이미 실현됨(QuantMLA, SnapMLA 그룹; KV-COBRA 2609.24298도 "Natural extensions include MLA"). "novelty 중상"은 "낮음"으로.
9. 정리 10.2의 b_i = ½log₂(λ_i/θ)는 2^{-2b} 고해상도 법칙 가정. RateQuant(2605.06675)는 실제 양자화기에서 β 3.6–5.3 ≠ 4이며 할당 순서를 뒤집을 수 있다고 보고 — 양자화기별 보정 필요(노트 [S], 본 판정에서 재독 안 함).

## 수정 요구

1. "문헌상 공백" 전면 재작성: QuantMLA를 직접 선행 연구로, NOVA-KV Thm 1 + eq.(38)을 닫힌형/헤드합 선행으로, AATC·SPECTRA를 W_O 가중·water-filling·0비트 선행으로, WUSH-KV를 데이터 적응 fold 변환 선행으로 인용. "latent 변환 문헌 없음", "헤드합이 구조적으로 다름" 문장 철회.
2. 정리 10.2를 "새 정리"가 아니라 "Linder et al. 1999 / NOVA-KV Thm 1의 MLA 특수화"로 표기. 새 이론 기여는 (a) 공유 latent의 결합 가중 M_K + λM_V(λ에 softmax Jacobian 포함)와 (b) **RMSNorm 제약 하 최적 변환**: fold 가능 족 {R·S} 안의 M-가중 최적해 vs 비직교 최적해(쓰기 시 512×512 matvec)의 이득 차이 — 를 명시.
3. "런타임 비용 0" → "소비자 쪽 0, 생산자 쪽은 직교×대각 족이면 0 / 일반 T는 토큰·층당 d_c² MAC 추가"로 수정. 커널 측정(쓰기 경로)을 판정 실험에 추가.
4. RoPE 가지 절: "FlashMLA vs SnapMLA 상반"을 3자 사실 패턴(SnapMLA BF16 PTQ per-token / QuantMLA INT4 PTQ 쌍 스케일 / V4.1 FP4 QAT 동일 포맷, 정규화된 latent 내부)으로 교체하고 2606.24033 per-channel 데이터 인용.
5. 판정 실험 baseline에 QuantMLA C4R4/C2R4(DS-V2-Lite GSM8K 35.25 / 37.91, MMLU 57.95 / 57.29)와 V4.1식 E2M1+E4M3/16 포맷을 포함; 대상 모델은 BF16 학습 MLA(DS-V2-Lite, GLM-4.7-Flash, Kimi-K2, LongCat-Flash-Lite)로 한정(V4.1은 QAT라 PTQ 비교 부적절).
6. 2603.04428 수치는 "시스템 논문의 512-token PPL 1개 측정"으로 격하; KVarN 수치 출처를 README로.
7. novelty 평가를 "중상"→"낮음(조합적; 남는 것은 결합 가중·할당·RMSNorm 제약 최적화·learned vs closed-form 비교)"으로.

## 새 가설

1. **RMSNorm-제약 최적 변환 가설** (단계: PTQ 변환 설계 / KV write 커널). fold 가능한 직교×대각 족(QuantMLA 공간) 내 M-가중 최적해와 비직교 최적해 T = U^T M^{1/2}의 가중 왜곡 차이는 M의 조건수(및 Σ_c와의 비정렬)로 예측되며, 조건수가 큰 층에서만 512×512 쓰기 matvec가 정당화된다. 근거: WUSH-KV App. A eq.(11), QuantMLA eq.(6)/App. C.1, NOVA-KV "orthogonal suboptimal unless M_q ∝ I".
2. **RoPE 정밀도는 granularity보다 '정규화 아래 있는가'가 결정** (단계: KV 포맷 선택 / RoPE 가지). V4.x처럼 RoPE 차원이 RMSNorm된 latent 안에 있으면 per-16 FP4로 충분하고, V2/V3/LongCat처럼 정규화 없는 분리 k_R은 per-channel 스케일(또는 QuantMLA Prop.1의 fold 가능한 쌍 스케일 S_P)이 있어야 per-16 FP4가 BF16에 근접한다. 검증: DS-V2-Lite k_R에 {per-16 FP4, per-16 FP4 + S_P, per-channel, BF16}. 근거: 2609.19969 §2.4.4, SnapMLA Fig 3a, 2606.24033 (68.0 vs 49.9), QuantMLA Prop. 1.
3. **유효 랭크 → 층별 민감도 예측** (단계: 캘리브레이션 / 비트 할당). 층별 M^{1/2}Σ_cM^{1/2} 유효 랭크(및 water-filling 0비트 차원 수)가 QuantMLA Fig 2의 층별 반응 변동과 CARE의 층별 랭크 민감도를 설명한다; 단 할당 법칙은 2^{-2b}가 아니라 양자화기별 보정된 β^{-b}를 써야 순서가 맞다. 근거: QuantMLA l.413–414, CARE 2603.17946 Fig 2a, RateQuant 2605.06675(노트 기반).

## 생략한 것
- AATC, OSCAR, KV-COBRA, CARE, RateQuant는 discover 노트와 grep 확인만(정독 생략). SGLang FP4 KV on DeepSeek-R1-0528 수치는 WebFetch 실패(DNS)로 미검증 [S].
