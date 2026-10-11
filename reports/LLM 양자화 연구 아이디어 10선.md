# LLM 양자화 연구 아이디어 10선

작성일: 2026-10-10. 대상: LLM 파이프라인 전체(사전학습, PTQ, 서빙, 후학습/RL, 보안)에서 main-track 수준의 novelty와 contribution을 노릴 수 있는 양자화 연구 가설 10개. 각 항목은 (1) 문헌상 공백과 근거, (2) 가설, (3) 정리 수준으로 증명 가능한 부분과 추측에 머무는 부분, (4) 며칠 안에 판정 가능한 최소 실험, (5) 냉정한 novelty·리스크 평가로 구성한다.

---

## 0. 요약과 판정표

| # | 아이디어 | 파이프라인 단계 | 핵심 수학 도구 | 판정 실험 비용 | novelty | main-track 가능성 |
|---|---|---|---|---|---|---|
| 1 | 이분산(heteroscedastic) 로짓 노이즈가 softmax에 만드는 2차 편향: QJL 실패·KIVI 비대칭·sink 증폭·LM-head 온도 이동의 단일 설명 + 1-scalar 보정 | 서빙(KV, attention, LM head) | softmax Hessian, CGF, Jensen | 1–2 GPU·일 | 높음 | 높음 |
| 2 | SSM/하이브리드 recurrent state의 ΣΔ(error-feedback) 양자화와 T-균일 오차 상한 | 서빙(Mamba/hybrid) | IIR 필터·noise shaping, deadband 해석 | 2–3 GPU·일 | 높음(논문 0건) | 높음(MLSys/ICLR) |
| 3 | 블록 스케일 마이크로포맷(NVFP4/MXFP4/UE5M3)의 극값통계 왜곡 이론 | 포맷·사전학습·PTQ 공통 | 극값통계(Gumbel), Lloyd–Max, 비균일 그리드 편향 | CPU 수치 + 실가중치 수일 | 중상 | 중상 |
| 4 | FP4 사전학습의 LR-decay 말기 격차: 분산(η에 비례) vs 편향 바닥(Δ²tr H/24)의 서명 분리 | 사전학습 | SGD 정지분포, 2차 섭동, edge-of-stability | 수십 GPU·일(≤1B) | 중상 | 중상 |
| 5 | 캘리브레이션 표본복잡도(Marchenko–Pastur)와 공변량 이동: VQ/학습 회전의 과적합 가설 | PTQ | 공분산 추정 집중부등식, Ledoit–Wolf, Weyl | 2–4 GPU·일 | 중 | 중 |
| 6 | 양자화 롤아웃 RL의 엔트로피 인지 IS 보정: QeRL(노이즈=탐색) vs FlashRL(편향) 통합 | 후학습(RL) | 토큰 KL ≈ ½δᵀJδ, PG 편향 상한, 80/20 fork | 1–2 노드·수일 | 중상 | 중상(빠르게 움직이는 분야) |
| 7 | 양자화 셀(preimage box) 정리: LoRA 머지 SNR, unlearning 역전, 양자화 조건부 공격을 하나로; 비밀 subtractive dither 방어 | 후학습(PEFT)·보안 | 고해상도 양자화 가정, Schuchman dither 정리 | 1–2 GPU·일 | 높음 | 높음 |
| 8 | 추론 모델 PTQ의 엔트로피 가중 캘리브레이션 | PTQ(reasoning) | Fisher 가중 = collision entropy 보조정리 | 1–2 GPU·일 | 중 | 중(실증 결과 크기에 좌우) |
| 9 | RoPE 쌍 혼합 보조정리: pre/post-RoPE 논쟁 해소 + 주파수 대역 비트 할당 + YaRN 상호작용 | 서빙(KV)·어텐션 | 2×2 회전 등변성, 그룹 내 위상 혼합 | 1–2 GPU·일 | 중 | 중 |
| 10 | MLA latent KV의 가중 변환 부호화(weight-absorption 하의 최적 KLT) | 서빙(MLA 계열) | 가중 MSE 변환 부호화, reverse water-filling | 2–3 GPU·일 | 중상 | 중상 |

판정의 전제: 1, 2, 7은 "이론이 명확하고 선행 논문이 사실상 없음"이라 가장 안전하다. 3, 4, 10은 이론은 명확하지만 경쟁 그룹이 몇 달 안에 도달할 수 있는 영역이다. 5, 6, 8, 9는 실험 결과의 크기가 논문의 운명을 결정한다.

### 0.1 문헌 스윕 방법과 한계

두 단계로 진행했다.

1. **1차(스니펫 스윕, 2026-10-10 오전)**: 6개 영역(사전학습, 가중치 PTQ, 활성화·KV·어텐션 서빙, 후학습·RL·추론·안전, 이론, 인접 분야)을 병렬로 스윕했다. 노트 원문은 `research_notes/LLM 양자화 연구 아이디어 10선/01~06_*.md`(약 450KB, 인용 약 400건). 이 단계에서는 네트워크 정책이 arXiv 등 전문 호스트를 차단해 검색엔진 요약·공식 README·교과서 기억으로 재구성했다.
2. **2차(arXiv 전문 대조 스윕, 2026-10-10 오후~10-11)**: 호스트 차단을 해제한 뒤 `scripts/arxiv_fetch.py`/`arxiv_search.py`로 전문(PDF→텍스트)을 받아, 보고서가 인용한 arXiv ID 84건 전부를 인용 문장 단위로 대조했다(`research_notes/arxiv_sweep/verify/`). 결과: 확인 42건, 부분 확인(수치·귀속·표현 수정 필요) 40건, 오귀속 2건, 존재하지 않는 ID 0건. 이어서 정리·수치를 원문에서 추출(`theorems/`, 40편), 아이디어별 선행 연구 탐색(`discover/`, 키워드·인용 이웃·반대 증거 3각도)과 적대적 novelty 판정(`judge/`), 2026년 6–10월 신규 이론 스캔(`recent/`)을 수행했다. 수정 내역은 0.3절에 정리했다.

인용 뒤의 표기는 다음을 뜻한다.

- [V] arXiv 전문(또는 공식 문서)에서 해당 주장과 수치를 직접 확인함
- [S] 초록/스니펫 수준에서만 확인됨(수치는 저자 자기보고)
- [U] 이 세션에서 ID 또는 존재가 확인되지 않음

2차 스윕 뒤에도 [S]로 남은 항목은 전문 대조 대상(84건)에 들지 않은 보조 인용이다. 전문 대조에서 "부분 확인"으로 판정된 항목은 본문을 수정한 뒤 [V]로 올렸다.

### 0.2 스윕에서 확인된, 아이디어의 공통 토대가 되는 사실

1. 엄밀한 이론은 거의 전부 "층별 MSE proxy" 위에 있다(QuIP 비간섭성 정리, GPTQ=Babai nearest-plane, 격자/TCQ 율-왜곡). proxy에서 end-to-end 손실로 가는 다리는 (a) HIGGS linearity theorem — 국소 최소점 주변 2차 Taylor 전개에서 나온 E[PPL(Ŵ)] ≈ PPL(W*) + Σ_l α_l t_l² (t_l² = 층별 상대 Frobenius 오차, 3–8bit에서만 검증, OPT-125M에서만 가정 검증) — 와 (b) Lipschitz 활성화 가정 하의 다층 확장뿐이며, Provable PTQ는 층별 ℓ₂/ℓ∞ 상한만 주고 PPL 보장은 "열린 문제"라고 명시한다 ([QuIP 2307.13304, NeurIPS 2023](https://arxiv.org/abs/2307.13304) [V], [GPTQ=Babai 2507.18553, ICLR 2026](https://arxiv.org/abs/2507.18553) [V], [Provable PTQ 2508.04853](https://arxiv.org/abs/2508.04853) [V], [HIGGS 2411.17525, NAACL 2025](https://arxiv.org/abs/2411.17525) [V]). 초안의 "KL≈¼‖δ‖²"는 HIGGS에 없는 식이라 삭제했다.
2. 회전은 ℓ₂ 노름, Hessian 스펙트럼, 토큰 간 동적 범위를 바꾸지 못한다(직교성의 결과). SpinQuant는 무작위 회전 100회 시행 간 zero-shot 평균 최대 13점(W4A4 LLaMA-2 7B), 무작위 Hadamard 간에도 최대 6점 편차를 보고한다(Sec 2.2, Fig. 4; 논문은 "Haar"라는 표현을 쓰지 않는다). 이를 "비간섭성 보장이 손실을 통제하지 못한다"고 읽는 것은 이 보고서의 해석이다 ([SpinQuant 2405.16406, ICLR 2025](https://arxiv.org/abs/2405.16406) [V]).
3. 정밀도 스케일링 법칙(Kumar et al. [2411.04330, ICLR 2025](https://arxiv.org/abs/2411.04330) [V], Ouyang et al. [2411.17691, ACL 2025](https://arxiv.org/abs/2411.17691) [V])은 적합식이다. 기작은 Kumar 부록 H의 sharpness/edge-of-stability 추측(저자 스스로 "speculative")과 Ouyang Sec 4의 "말기 가중치 변동폭이 작아 양자화가 그 범위를 벗어난다"는 직관뿐이며 검증은 없다. Kumar 부록 G는 cosine vs warmup+linear decay를 30M 모델에서 비교해 토큰/파라미터 비에 따른 열화가 스케줄과 무관하게 지속된다고 보고한다. 2025–26에 "LR이 감쇠하면 손실과 양자화 오차가 분리된다"는 반론이 나왔고, 이 논문은 peak-LR·WSD vs cosine·weight decay·LAWA 개입 실험과 Hessian trace·sharpness 측정까지 이미 수행했다 ([Catalan-Tatjer et al. 2510.06213, ICLR 2026](https://arxiv.org/abs/2510.06213) [V]).
4. 서빙 측 KV 양자화 이론은 추정량 수준(QJL 비편향 내적, TurboQuant Beta 주변분포)에 머물고, 어텐션 출력이나 토큰 수준 오차를 다루는 정리가 없다. QJL 잔차는 커뮤니티에서 "softmax가 분산을 증폭해 품질을 해친다"고 보고되어 vLLM 구현에서 제외됐다 ([vLLM TurboQuant docs](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/turboquant/) [V]; 문서 주석이지 벤치마크는 아니다).
5. 후학습 측에서는 "preimage box"(한 양자화 모델의 FP 원상)가 공격(Egashira et al. [2405.18137](https://arxiv.org/abs/2405.18137) [V]), unlearning 역전([2410.16454, ICLR 2025](https://arxiv.org/abs/2410.16454) [V]; 형식적 정리는 없고 Sec 5의 Δ_int4 ≫ Δ_int8 간격 비교가 전부), LoRA 머지 실패([QA-LoRA 2309.14717](https://arxiv.org/abs/2309.14717) [V]; QA-LoRA 자체의 머지는 그룹 상수를 zero-point에 흡수하므로 구성상 정확하고, 상한이 없는 것은 비제약 LoRA+PTQ 머지)를 관통하지만 누구도 통합 정리로 쓰지 않았다.

### 0.3 arXiv 전문 대조로 수정한 항목

전체 표는 `research_notes/arxiv_sweep/verify/SUMMARY.md`(84건), 섹션별 수정 사항은 `verify/by_section.md`에 있다. 본문에 반영한 주요 수정:

| arXiv id | 초안의 서술 | 전문 대조 결과 | 반영 |
|---|---|---|---|
| 2604.08976 | GPTQ 4-bit가 ECE를 악화시킨다는 측정 | 그 문장은 이 논문의 related work이며 원출처는 Proskurina et al. 2405.00632 (Findings of NAACL 2024); 2604.08976 자체는 GGUF Q5_K_M vs f16 비교 | 2405.00632로 재귀속 |
| 2608.01078 | 최신 모델은 캘리브레이션 데이터에 덜 민감 | 원출처는 Paglieri et al. 2405.20835 (ICML 2024 WS); 2608.01078의 자체 결과는 반대(통상 캘리브레이션으로 ternary PTQ가 수학·코딩에서 붕괴, 자가 생성 추론 흔적으로 복구) | 2405.20835로 재귀속, 2608.01078은 아이디어 8의 근거로 이동 |
| 2411.17525 | "KL≈¼‖δ‖²", 1차 근사 | 식이 논문에 없음; 2차 Taylor에서 나온 층별 상대 MSE 선형식 | 0.2절 수정 |
| 2405.16406 | "Haar 회전 간 13점" | "Haar" 없음; 무작위 회전 100회 최대 13점, Hadamard 간 6점 | 0.2절 수정 |
| 2503.22879, 2508.10370 | "SSM 상태 양자화 학술 논문 0건" | Quamba2는 캐시된 SSM 상태를 정적 per-group 8-bit로 양자화(App. E); eMamba는 INT24 계산/INT17 저장 고정소수점 상태를 FPGA/ASIC에 구현 | 아이디어 2의 공백을 "피드백 없는 open-loop 정적 양자화만 존재"로 재정의 |
| 2510.06213 | "남는 기여는 Δ² vs tr(H) 분해와 개입 실험" | 개입 실험(peak LR, WSD vs cosine, weight decay, LAWA)과 Hessian trace·sharpness 측정을 이미 수행 | 아이디어 4의 기여를 서명 분리 예측과 전환 시점 예측으로 축소 |
| 2509.25149 | MXFP4 2.5% vs NVFP4 1.5% | 8B/1T, BF16 기준 상대 학습손실 오차(마지막 8블록 BF16 유지); 12B/10T 격차는 FP8 기준; 전환 실험은 BF16만, forward 전환이 1.5%→0.5% | 아이디어 3·4 수치 정정 |
| 2606.20381 | "정량 이론이 없다" | bin 수준 닫힌 형태 Eq.(2) (2q_i−q_{i−1}−q_{i+1})/4와 곱셈 누적 항등식이 있음; 분포 수준·블록 크기 의존 닫힌 형태는 없음 | 아이디어 3의 공백을 분포·블록 수준으로 좁힘 |
| 2605.08114 | "이질 분산이 만드는 체계적 편향 논의 없음" | Sec 3.1 Eq.(4) e^{σ²/2} 인자와 "σ²가 점수와 상관될 때 위험" 서술이 있음; fair-budget 인자는 2π | 아이디어 1의 선행 연구로 명시 |
| 2510.05373 | KVLinC가 KIVI 축을 "반대로" | KIVI 축을 유지하고 V에만 Hadamard; K Hadamard가 per-channel에서 해롭다는 관찰 | 아이디어 1·9 수정 |
| 2505.23786 | preimage box 계산 | k-quant는 정확한 preimage 구간 계산이 불가능해 오차 기반 근사 구간 사용 | 아이디어 7 수정 |
| 2605.15152 | Egashira et al., ICML 2026 | 1저자 Zhan; ICML 2026 workshop | 아이디어 7 수정 |
| 2604.06798 | MoBiE [S] | 저자 철회(2026-04-20) | 부록 A에서 [U]로 강등 |
| 2604.08826 | HiFloat4 포맷 제안 | 포맷은 2602.11287; 2604.08826은 에뮬레이션 사전학습 연구 | 아이디어 3 수정 |
| 2601.22813 | Quartet II 병목 = micro-scale SR 분산 | 병목은 원소별 FP4 SR 분산(RTN 대비 MSE 2.5배); 해법이 확률성을 FP8 micro-scale로 옮기는 것 | 아이디어 3·4 수정 |
| 2604.08426, 2609.18063 | [U] | 존재·주장 확인 | [V]로 상향 |

---

## 아이디어 1. 이분산 로짓 노이즈의 softmax 2차 편향 정리: 비편향 KV 추정량이 왜 해를 끼치고, 왜 key는 per-channel이어야 하는가

### 가설
양자화된 key(또는 LM-head 가중치)가 로짓에 주입하는 노이즈의 분산이 토큰(또는 어휘)마다 다르면, 노이즈가 평균 0이어도 softmax의 기대값은 "분산이 큰 토큰 쪽으로" 편향된다. 그 편향은 2차에서 닫힌 형태로 계산되며, 분산이 토큰마다 같으면 정확히 상쇄되어 온도 상승(평탄화)만 남는다. 이 하나의 정리가 (i) QJL/TurboQuant 잔차가 실제 서빙에서 해를 끼치는 이유, (ii) KIVI의 "key는 per-channel, value는 per-token" 비대칭이 경험적으로 옳았던 이유, (iii) per-token key 양자화가 attention sink 질량을 증폭하는 이유, (iv) 4-bit LM head가 보정오차(ECE)와 엔트로피를 바꾸는 이유를 동시에 설명한다.

### 문헌상 공백
- QJL은 "비편향 내적 추정량, 최소 왜곡"을 증명했지만 ([QJL 2406.03482, AAAI 2025](https://arxiv.org/abs/2406.03482) [V]) vLLM 구현은 QJL을 제외했고, 그 사유를 "5개 이상 독립 그룹의 합의: softmax가 분산을 증폭"으로 적었다 [V]. 가장 가까운 분석인 D'Alberto의 노트([2605.08114](https://arxiv.org/abs/2605.08114) [S])는 "QJL이 K의 내적 분산을 π/2배 키우고 softmax가 Jensen으로 증폭한다"는 정성적 주장이며, 토큰 간 분산 이질성이 만드는 체계적 편향, 보정식, K-vs-V 할당 규칙은 없다.
- KIVI의 per-channel K / per-token V는 "outlier 채널"과 "스트리밍 편의"로 정당화됐고 최적성 증명이 없다 ([KIVI 2402.02750](https://arxiv.org/abs/2402.02750) [V]). KVLinC는 반대로 "raw key per-channel + Hadamard value per-token"이 최선이라 보고했는데 설명이 없다 ([2510.05373](https://arxiv.org/abs/2510.05373) [S]).
- 어텐션 민감도 이론은 1차 Jacobian 상한([2608.20988](https://arxiv.org/abs/2608.20988) [S])과 e^{±2η} 승법 상한([SpectrumKV 2606.08635](https://arxiv.org/abs/2606.08635) [S])뿐이고, 둘은 "날카로운 어텐션이 더 민감하다/덜 민감하다"로 서로 반대 직관을 준다. 2차 항을 쓰면 이 둘이 화해된다.
- LM head 쪽은 ReSET([2606.13233](https://arxiv.org/abs/2606.13233) [S])이 "NVFP4가 토큰 엔트로피를 이동시킨다"고 관찰했고, GPTQ 4-bit가 ECE를 악화시킨다는 측정([2604.08976](https://arxiv.org/abs/2604.08976) [S])이 있으나 유도된 보정은 없다.

### 수학
설정: 참 로짓 s ∈ ℝⁿ, 추정 로짓 ŝ_j = s_j + ε_j, E ε_j = 0, Var ε_j = σ_j², 토큰 간 독립. p = softmax(s), p̂ = softmax(ŝ).

정리 1 (2차 편향). ∂²p_i/∂s_j² = p_i[(δ_ij − p_j)² − p_j(1 − p_j)] 이므로

E[p̂_i] − p_i ≈ (p_i/2) · [ σ_i²(1 − 2p_i) − Σ_j p_j σ_j² (1 − 2p_j) ] + O(σ⁴).

- 동분산(σ_j² ≡ σ²): 괄호가 2σ²(‖p‖₂² − p_i)로 줄어 E[p̂_i] − p_i ≈ σ² p_i(‖p‖₂² − p_i). 질량이 p_i > ‖p‖₂²인 토큰에서 작은 토큰으로 이동하며, Chebyshev 합 부등식(p_i와 log(1/p_i)가 역순)으로 평균 어텐션 맵의 Shannon 엔트로피가 2차에서 증가함을 보일 수 있다. 즉 동분산 노이즈는 온도 상승과 같은 방향이다.
- 이분산, p_i ≪ 1 영역: E[p̂_i]/p_i ≈ 1 + ½(σ_i² − σ̄_p²), σ̄_p² = Σ_j p_j σ_j². 각 토큰의 상대 어텐션 이득은 "어텐션 가중 평균 분산 대비 초과 분산의 절반"이다.

Gaussian 노이즈에서는 E[exp(ŝ_j)] = exp(s_j + σ_j²/2)가 정확히 성립하므로(일반 노이즈는 누적생성함수 K_j(1)), 보정 ŝ_j − σ̂_j²/2 는 분자를 비편향으로 만든다. 이것이 "1-scalar 보정"이다.

이 식을 몬테카를로(n=64, σ_j² ∈ U[0, 0.3])로 검증했다. 예측과 측정의 상관 0.9996, 최대 상대 오차 3.9%였다. 동분산 사례에서 평균 맵의 엔트로피는 3.0635 → 3.0681로 증가했다.

적용:
- QJL: 추정량 분산은 Var ≈ (π/2)‖q‖²‖k_j‖²/m (m = 사영 차원) 이므로 σ_j² ∝ ‖k_j‖². 즉 QJL은 "큰 노름 key"에 어텐션을 체계적으로 더 준다. ‖k_j‖는 QJL이 이미 저장하므로 보정 비용은 key당 곱셈 한 번이다.
- per-token key 양자화: σ_j² = ‖q‖²Δ_j²/(12 d) 가 토큰의 범위 Δ_j에 따라 달라져 이분산 → 범위가 큰 토큰(sink/massive activation 토큰)으로 편향. per-channel key 양자화: 노이즈 분산 Σ_c q_c²Δ_c²/(12d)가 모든 토큰에 동일 → 편향 상쇄, 온도 상승만 남음. 이것이 KIVI 비대칭의 이론적 이유다. value는 softmax를 거치지 않고 선형으로 들어가므로 per-token이 무해하다.
- LM head: per-row 스케일의 INT4 가중치는 어휘별 분산 ‖h‖²Δ_v²/12 를 만들어 범위가 큰 임베딩 행(희귀 토큰) 쪽으로 샘플링을 편향시키고, 동분산 성분은 온도를 올린다. 두 항 모두 로짓에서 ‖h‖²Δ_v²/24 를 빼는 한 번의 벡터 연산으로 1차 보정된다.

증명 가능한 것: 정리 1(테일러 2차, 독립성 가정), Gaussian 정확 보정, 동분산 평탄화의 엔트로피 증가. 추측에 머무는 것: 실제 모델에서 2차 전개가 유효할 만큼 σ²가 작은가(QJL m=256에서는 σ²가 수 단위일 수 있어 전개가 깨질 수 있고, 그 경우 "winner flip" 설명이 우세해진다. 둘 다 같은 실험으로 판별된다).

### 판정 실험(1–2 GPU·일)
Llama-3.1-8B, Qwen3-8B, fake-quant.
1. key를 per-token vs per-channel로 4/3/2비트 양자화하고 (a) sink 토큰(BOS·구분자)의 어텐션 질량 변화, (b) 평균 어텐션 엔트로피 변화, (c) LongBench/RULER를 측정. 예측: per-token에서만 sink 질량이 증가하고 그 크기가 ½(σ_sink² − σ̄²)에 비례.
2. TurboQuant/QJL(m ∈ {128, 256, 512})에 −σ̂_j²/2 보정을 넣고 뺀 채 NIAH·RULER·multi-hop을 비교. 예측: 보정이 QJL 포함 변형을 PolarQuant 단독 수준으로 회복.
3. LM head INT4(g128)에서 ECE·엔트로피·MMLU를 보정 전후로 비교. 예측: 보정이 ECE 악화의 대부분을 제거.
실패 기준: per-token/per-channel 간 sink 질량 차이가 노이즈 수준이거나, 보정이 RULER를 개선하지 못하면 가설 기각.

### 평가
novelty 높음. 유도는 학부 수준이지만 서빙 현장의 실제 실패(QJL 제외), 널리 쓰이는 경험 규칙(KIVI), 그리고 두 상한의 모순을 하나의 식으로 설명하고 비용 0의 보정을 준다. 리스크: D'Alberto([2605.08114](https://arxiv.org/abs/2605.08114))와 OSCAR([2605.17757](https://arxiv.org/abs/2605.17757))가 인접하므로 "이분산 편향과 보정"을 전면에 세워 차별화해야 한다. 2차 전개가 깨질 정도로 σ²가 크면 정리는 정성적 설명으로 격하된다.

---

## 아이디어 2. SSM/하이브리드 recurrent state의 ΣΔ(error-feedback) 양자화: T와 감쇠율에 균일한 오차 상한

### 가설
Mamba/SSD/GDN 류의 재귀 상태 h_t = a_t h_{t−1} + u_t 를 매 스텝 재양자화할 때 (i) RTN은 느린 채널(a_t ≈ 1)에서 "deadband"로 상태를 동결시키고, (ii) stochastic rounding은 분산이 Δ²t/12 로 발산하며, (iii) 1차 error feedback(ΣΔ 변조)은 상태 오차를 모든 t와 모든 a ∈ [0,1)에 대해 Δ 이하로 결정론적으로 묶는다. 하이브리드 모델에서는 Mamba 상태 페이지가 KV 페이지 크기를 고정하므로 이 문제가 짧은 문맥에서 FP8 KV의 이득을 0으로 만든다.

### 문헌상 공백
- 학술 논문으로 SSM "상태" 양자화를 다룬 것은 0건이다. Quamba/Quamba2는 가중치·입력·B/C를 양자화하고 ([2410.13229](https://arxiv.org/abs/2410.13229), [2503.22879](https://arxiv.org/abs/2503.22879) [S]), eMamba는 "상태를 양자화하면 비트폭이 폭발하고 재양자화하면 정보가 사라진다"는 딜레마만 적었다 ([2508.10370](https://arxiv.org/abs/2508.10370) [S]). Ternary Mamba는 사후 점별 보정(Kalman gain, James–Stein)이 perplexity를 악화시킨다고 보고하며 "오차가 이력 의존적이어서 점별 출력 보정으로 되돌릴 수 없다"고 썼다 ([2606.18114](https://arxiv.org/abs/2606.18114) [S]). 이는 루프 안의 피드백이 필요하다는 뜻이다.
- Nemotron 3 Super는 Mamba 상태를 기본 FP32로 두고, FP16으로 캐스팅하면 W8A8에서 verbosity가 최대 40% 증가했다고 보고했다 ([2604.12374](https://arxiv.org/abs/2604.12374) [S]). Nemotron-Labs-3-Puzzle도 상태는 FP16 유지 ([2607.04371](https://arxiv.org/abs/2607.04371) [S]).
- vLLM RFC #55196는 Falcon-H1-1.5B에서 FP8 KV가 512/1024 토큰에서 1.00×(이득 없음)임을 보이고, 한 기여자가 INT8 상태로 Qwen3.5-4B GSM8K 81→41%, error feedback으로 81% 복구를 자기보고했다 ([vllm issue 55196](https://github.com/vllm-project/vllm/issues/55196) [V]). 논문도, 이론도 없다.
- 최적화기 상태 양자화에서 동일 현상("EMA 상태 동결, 리셋으로 복구")을 모델링한 Topollai–Choromanska([2603.16731](https://arxiv.org/abs/2603.16731) [S])가 있다. SSM 상태와 EMA는 수학적으로 같은 1차 IIR 재귀다.

### 수학
대각 SSM 채널 하나: h_t = a_t h_{t−1} + u_t, a_t = exp(Δ_t A) ∈ (0,1). 양자화 스텝 Δ.

(1) RTN 재양자화 ĥ_t = Q(a_t ĥ_{t−1} + u_t). 오차 d_t = a_t d_{t−1} + e_t, |e_t| ≤ Δ/2. 최악의 경우 |d_t| ≤ (Δ/2)·Σ_s ∏_{r>s} a_r → Δ/(2(1−a)). 또한 |(1−a_t)ĥ_{t−1} − u_t| < Δ/2 가 지속되면 Q(a_tĥ_{t−1}+u_t) = ĥ_{t−1}로 상태가 동결된다(deadband). 느린 채널은 정의상 스텝당 변화가 작으므로 바로 이 영역에 있다.

(2) SR: E e_t = 0, Var e_t ≤ Δ²/4(균일 가정 시 Δ²/12). Var d_t = Σ_s ∏_{r>s} a_r² Var e_s → a≈1에서 ≈ Δ²t/12. 랜덤워크 발산.

(3) ΣΔ error feedback: v_t = a_t ĥ_{t−1} + u_t + r_{t−1}, ĥ_t = Q(v_t), r_t = v_t − ĥ_t, |r_t| ≤ Δ/2. 그러면 ĥ_t = a_tĥ_{t−1} + u_t + r_{t−1} − r_t 이고 g_t := d_t + r_t 는 g_t = a_t g_{t−1} + (1 − a_t) r_{t−1} 을 만족한다. 따라서 |g_t| ≤ (Δ/2)·Σ_s (1−a_s)∏_{r>s} a_r = (Δ/2)(1 − ∏_s a_s) ≤ Δ/2 (telescoping). 결론:

정리 2. 1차 error feedback 하에서 |d_t| ≤ |g_t| + |r_t| ≤ Δ 가 모든 t, 모든 a_t ∈ [0,1)에 대해 성립한다. 반면 naive RTN은 Δ/(2(1−a)), SR은 표준편차 Δ√(t/12) 이다.

잔차 r_t를 b_r 비트로 저장하면(범위 [−Δ/2, Δ/2]) 추가 오차 Δ·2^{−(b_r+1)}/(1−a) 가 최악의 경우 생기지만 확률적 표준편차는 Δ·2^{−b_r}/√(24(1−a)) 로, b_r = 4, 1−a = 10⁻³에서 약 0.4Δ다. 토이 시뮬레이션(a = 0.999, T = 2×10⁴)에서 최대 |d_t|/Δ는 RTN 3.4, SR 12.9, ΣΔ(정확 잔차) 0.58, ΣΔ(4비트 잔차) 2.0(최종 0.05)이었다.

채널별 비트 할당: SR 또는 잔차 노이즈 하에서 정상상태 분산이 ∝ Δ_i²/(1−a_i²) 이므로 reverse water-filling은 b_i = ½log₂(σ²_e/((1−a_i²)θ))를 준다. 즉 감쇠가 느린 채널에 ½log₂(1/(1−a_i²)) 비트를 더 준다(1−a = 10⁻³면 약 +5비트). A 행렬은 고정이고 Δ_t 분포는 캘리브레이션 가능하므로 오프라인 계산된다.

하이브리드 확장: Mamba 상태 페이지와 어텐션 KV 페이지를 하나의 메모리 예산으로 보고 두 종류의 "분산"에 같은 water-filling을 적용한다.

증명 가능한 것: 정리 2와 (1),(2)의 상한, water-filling 규칙(Gaussian 가정). 추측: 실제 모델에서 deadband가 지배적 실패 모드인지, 선택적 재귀(입력 의존 a_t)에서 평균 감쇠로 충분한지, 4비트 잔차가 실용적 메모리 이득(예: INT8+INT4 = 12비트 vs FP16)에 머무는지.

### 판정 실험(2–3 GPU·일)
Mamba-2 2.7B, Falcon-H1-1.5B, Nemotron-H-8B. 상태를 INT8/INT6/INT4로 fake-quant하여 (a) RTN, (b) SR, (c) ΣΔ 정확 잔차(상한), (d) ΣΔ + 4비트 잔차, (e) 동일 총 비트의 naive. 측정: GSM8K, NIAH 4K–32K, 채널별 상태 오차의 t-성장, 느린 채널(작은 |A|Δ_t) 대 빠른 채널의 열화 기여. 성공 기준: (d)가 (e)를 유의하게 이기고 오차 성장이 포화하며, 열화가 느린 채널에 집중됨을 확인.

### 평가
novelty 높음(논문 0건, 산업 RFC가 문제의 실재를 보여줌). 이론은 ΣΔ 변조와 IIR 필터 이론의 재적용이지만 SSM 맥락에서는 새롭고, 최적화기 상태 staleness와의 통합도 가능하다. 리스크: Quamba2/SGLang이 이미 INT8 상태를 어떤 트릭으로 처리하고 있다면 방법의 참신성이 줄어든다(이론과 할당 규칙은 남는다). 커널 구현 없이 fake-quant만으로는 MLSys에서 약하다.

---

## 아이디어 3. 블록 스케일 마이크로포맷의 극값통계 이론: (블록 크기, 원소 포맷, 스케일 포맷, 라운딩 방향)의 닫힌 형태 왜곡

### 가설
RHT 후 블록 원소가 근사 Gaussian이라는 표준 가정 아래, 블록 최대값 M_B의 극값 분포(Gumbel)와 원소 그리드의 Lloyd–Max 해석을 결합하면 NVFP4(16, E2M1, E4M3 스케일), MXFP4(32, E2M1, UE8M0), UE5M3, INT4+스케일 등 모든 마이크로포맷의 기대 왜곡을 닫힌 형태로 비교할 수 있다. 이 이론은 "MXFP8 스케일 지수는 올림해야 수렴한다"(Mishra et al.)는 레시피 수준 사실, E2M1의 shrinkage bias(UFP4), Four-over-Six 휴리스틱을 하나의 틀에서 설명하고 최적 포맷을 도출한다.

### 문헌상 공백
- 포맷 비교는 전부 경험적이다: Egiazarian et al.은 "NVFP4·MXFP4 모두 손실이 있고 MXFP4는 약 10% 상대 하락"([2509.23202, ICLR 2026](https://arxiv.org/abs/2509.23202) [S]); Meta는 소프트웨어 스케일 조정으로 MXFP4 격차를 1% 미만으로 ([2603.08713](https://arxiv.org/abs/2603.08713) [S]); NVIDIA는 "MXFP4 상대오차 ≈2.5% vs NVFP4 1.5%"([2509.25149](https://arxiv.org/abs/2509.25149) [S]); Mishra et al.은 UE8M0 지수를 내림하면 발산하고 올림하면 8B/15T에서 BF16과 일치한다고 보고 ([2506.08027](https://arxiv.org/abs/2506.08027) [S]).
- 스케일 포맷 제안들(UE5M3 [2609.02846](https://arxiv.org/abs/2609.02846) [S], Four-over-Six [2512.02010](https://arxiv.org/abs/2512.02010) [S], HiFloat4 [2604.08826](https://arxiv.org/abs/2604.08826) [S])은 모두 실험 기반이다.
- UFP4는 "E2M1의 RTNE가 음의 기대 라운딩 오차(shrinkage bias)를 가지며 RHT가 이를 증폭한다"고 주장 ([2606.20381](https://arxiv.org/abs/2606.20381) [S]); 자동 리뷰는 효과 분리가 안 됐다고 지적했다. 정량 이론이 없다.
- Sun et al.의 FP 스케일링 법칙은 (E, M, B)에 대한 적합식이다 ([2501.02423](https://arxiv.org/abs/2501.02423) [U: 내용 미확인]).

### 수학
블록 x_1..x_B iid N(0, σ²). M_B = max|x_i|. E[M_B] ≈ σ√(2 ln(2B)), Gumbel 척도 σ/√(2 ln 2B).

스케일 s = M_B/g_max 를 스케일 포맷으로 양자화 → s_q. 원소는 x_i/s_q 를 그리드 G에서 RTN, g_max에서 클리핑.

- E4M3 스케일: s_q/s ∈ 1 ± 2^{−4} 근사 균일 → 왜곡 계수 ≈ 1 + O(2^{−8}).
- UE8M0 올림: s_q/s = 2^U, U ~ U[0,1) (로그 균일) → 균일 그리드 영역에서 원소 MSE ∝ s_q² 이므로 E[4^U] = 3/(2 ln 2) ≈ 2.16배. 블록 크기 32 vs 16은 E[M_32]/E[M_16] ≈ √(ln 64/ln 32) ≈ 1.10을 더한다.
- UE8M0 내림: s_q ∈ (s/2, s] → 최대 원소가 클리핑되어 (M − s_q g_max)² 의 편향된 오차가 "가장 큰 원소"에 집중된다. E[(1 − 2^{−U})²] = 1 − 2(0.7213) + 0.541 ≈ 0.098 이고 M² ≈ 2 ln(2B)σ² 이므로 클리핑 MSE 기여 ≈ 0.098·2 ln(2B)σ²/B.

E2M1 그리드 {0, .5, 1, 1.5, 2, 3, 4, 6}에서의 수치 계산(2×10⁵ 블록): MSE/σ² = NVFP4형 0.0092, MXFP4 올림 0.0133, MXFP4 내림 0.0414, 정확 스케일 B=16 0.0089. RMS 비 MXFP4(올림)/NVFP4 = 1.20, MXFP4(내림)/NVFP4 = 2.12. 즉 (i) 라운딩 방향 하나가 MSE를 3.1배 바꾸고 그 오차가 최대 원소(기울기 신호가 큰 원소)에 편향되어 쌓인다는 점이 Mishra et al.의 발산 관찰과 정합하며, (ii) 단순 Gaussian 모델이 주는 올림 MXFP4/NVFP4 비 1.20은 NVIDIA 보고치 1.67에 못 미친다. 이 격차 자체가 연구 질문이다(실제 블록의 heavy tail, E2M1 비균일 그리드와 max-abs 스케일의 상호작용, 측정 지표 차이 중 무엇이 남은 0.4를 설명하는가).

E2M1 shrinkage bias: 비균일 그리드에서 RTN의 결정경계는 중점이고, 감소하는 밀도에서는 셀 내 조건부 평균이 중점 아래에 있으므로 큰 셀([4,6])에서 수축 편향이 생긴다. 수치: E[|Q(x)|−|x|]/E|x| = −0.66%, E[Q(x)x]/E[x²] = 0.9968. 작지만 체계적이고 층을 거치며 승법적으로 쌓인다(UFP4 주장의 정량화). 결정경계를 중점에 두는 것이 MSE 최적이므로 편향은 MSE 최적성의 대가이며, 비편향을 원하면 SR 또는 경계 이동으로 분산을 사야 한다. 깊이 L에서 (L·bias)² > 추가 분산이면 비편향 라운딩이 유리하다는 간단한 판정식이 나온다.

증명 가능한 것: Gaussian 블록 가정 하의 모든 기대 왜곡은 1차원 적분으로 정확히 계산되고, UE8M0 올림의 2.16배, 내림의 클리핑 항, shrinkage 편향의 부호는 해석적으로 나온다. 추측: 실제 RHT 후 블록이 Gaussian에 얼마나 가까운지(토큰 수준 스파이크가 남는다는 보고가 있다: [OffQ 2606.07116](https://arxiv.org/abs/2606.07116) [S]), 학습 안정성과 왜곡의 연결(클리핑 편향 → 발산)은 가설이다.

### 판정 실험(CPU 수치 + 수일)
1. 합성 Gaussian·Laplace·Student-t 블록에서 모든 (B, G, 스케일 포맷, 라운딩) 조합의 MSE와 편향을 계산하고 해석식과 대조.
2. Llama-3.1-8B·Qwen3-8B의 실제 가중치·활성화·기울기(짧은 학습 런) 블록에 RHT를 적용한 뒤 같은 표를 재계산. 예측: 실 데이터 비율이 1.2를 넘어 NVIDIA의 1.67에 접근하면 잔여 격차는 heavy tail로 설명됨.
3. 300M 모델 FP4 사전학습에서 스케일 라운딩 올림/내림만 바꾼 두 런의 손실 격차와 발산 여부를 비교(Mishra et al. 재현의 축소판).
실패 기준: 실 데이터에서 포맷 순위가 해석식 예측과 다르면 Gaussian 가정이 쓸모없다는 결론.

### 평가
novelty 중상. 공학적으로 결정된 포맷 규칙에 처음으로 정량 이론을 주는 "이해 논문"이며 하드웨어 포맷 논쟁(UE5M3, HiFloat4)에 직접 개입한다. 리스크: 이론이 쉬워 경쟁자가 빨리 따라올 수 있고, 핵심 가치는 실 데이터와의 정합 여부에 달려 있다. 실제 사전학습 안정성과의 연결은 실험이 비싸다.

---

## 아이디어 4. FP4 사전학습의 LR-decay 말기 격차: 분산 바닥과 편향 바닥의 서명 분리

### 가설
저정밀 학습의 손실 격차는 두 바닥의 합이다. (A) 비편향 SR 기울기 노이즈가 만드는 정지분포 초과손실 ≈ (η/4)·tr(C_q)는 학습률 η에 비례해 decay와 함께 사라진다. (B) forward RTN 가중치 양자화 오차 b(w), |b| ≤ Δ/2 가 만드는 ½E[bᵀHb] ≈ (Δ²/24)·tr(H)는 η와 무관하다. 따라서 LR decay 동안 BF16 런의 초과손실은 떨어지는데 FP4 런은 (B)에 막혀 상대 격차가 벌어진다. 또 edge-of-stability로 η가 작아지면 곡률 tr(H)가 커져 (B) 자체가 decay 중 커진다. 이 모델은 "forward 텐서를 BF16으로 바꾸면 이득의 대부분이 온다"(NVIDIA)와 "decay 중 PTQ 오차가 손실에서 분리된다"(Catalan-Tatjer et al.)를 동시에 예측하고, "BF16 전환이 발산을 못 고쳤다"(Nemotron 3 Ultra)는 바닥이 아닌 불안정성 문제라고 분류한다.

### 문헌상 공백
- NVIDIA NVFP4 12B/10T: 안정 구간 <1% 격차가 decay 구간에서 >1.5%로 벌어지고, decay 중 BF16/MXFP8 전환이 완화하며 "forward 전환이 대부분의 이득"이라 보고 ([2509.25149](https://arxiv.org/abs/2509.25149) [S]). 원인 ablation은 없다.
- 2026년 그룹들이 각자 다른 병목을 주장한다: Quartet II는 micro-scale SR 분산([2601.22813](https://arxiv.org/abs/2601.22813) [S]), Dong et al.은 hot channel([2602.02047](https://arxiv.org/abs/2602.02047) [S]), TetraJet-v2는 bin 경계 가중치 진동([2510.27527](https://arxiv.org/abs/2510.27527) [S]), UFP4는 shrinkage bias([2606.20381](https://arxiv.org/abs/2606.20381) [S]), Full-Stack FP4는 미양자화 구성요소([2607.04422](https://arxiv.org/abs/2607.04422) [S]). 같은 LR 지점에서 이들을 분리한 실험이 없다.
- Nemotron 3 Ultra는 전 텐서 BF16 전환이 발산을 해결하지 못했다고 보고([2606.15007](https://arxiv.org/abs/2606.15007) [S]) — NVFP4 논문과 모순.
- Catalan-Tatjer, Ajroldi, Geiping은 32B/15T 궤적까지 분석해 "LR이 감쇠하면 검증 손실과 양자화 오차가 분리된다"고 보고하고 데이터량이 본질적으로 해롭다는 통설에 반박했다 ([2510.06213, ICLR 2026](https://arxiv.org/abs/2510.06213) [S]). 이는 PTQ 측 관찰이며, 학습 중 저정밀 격차의 분해로 확장되지 않았다.
- Edge of stability: λ_max(H) ≈ 2/η (GD, [Cohen et al. 2103.00065](https://arxiv.org/abs/2103.00065) [U: 고전, 미재확인]); 적응형 최적화기는 전처리된 sharpness 기준([2207.14484](https://arxiv.org/abs/2207.14484) [U]).

### 수학
2차 근사 L(w) ≈ L* + ½(w−w*)ᵀH(w−w*). SGD(학습률 η, 기울기 노이즈 공분산 C + C_q)의 정지분포에서 E[L] − L* ≈ (η/4)tr(C + C_q) (작은 η, 노이즈가 곡률과 독립이라는 표준 근사). forward 양자화 w_q = w + b(w): 오프셋이 셀 내 균일하면 E[b] = 0, E[b²] = Δ²/12 이고 추가 초과손실 ≈ (Δ²/24)tr(H). 비균일 그리드(E2M1)에서는 E[b] = b̄ ≠ 0 (아이디어 3) 이어서 1차 항 ∇Lᵀb̄ 가 추가된다.

예측 1: 격차(η) = A·η + B(η), B = (Δ²/24)tr(H(η)) + 1차 편향항. decay에서 A·η → 0, B는 유지되거나 tr(H) 증가로 커진다.
예측 2: forward만 BF16으로 바꾸면 B가 사라지고 A·η만 남아 격차가 η에 비례해 소멸. backward만 바꾸면 B는 그대로.
예측 3: 그룹 크기를 16→8로 줄여 Δ²를 4분의 1로 하면 B가 약 4분의 1.
예측 4: 최종 η가 다른 두 런(WSD 스케줄)에서 격차의 η-절편이 B.
예측 5: 전환 기준 — 전환 포맷의 (Δ'²/24)tr(H_final) 이 BF16 런의 최종 SGD 바닥 (η_final/4)tr(C) 보다 충분히 작으면 MXFP8 전환으로 충분하고, 아니면 BF16이 필요.

가중치 진동(TetraJet)은 b(w)의 동적 표현이다: master weight가 bin 경계 근처에 있으면 forward 오차가 매 스텝 부호를 바꿔 η로 줄지 않는 Δ/2 크기의 노이즈를 forward에 주입한다. 즉 진동은 (B)의 분산 성분이다.

증명 가능한 것: 2차 모델에서의 두 바닥의 분리와 η-의존성(표준 SGD 정지분포 해석). 추측: 실제 Adam·transformer에서 근사의 정확도, tr(H)의 decay 중 증가, 1차 편향항의 크기.

### 판정 실험(수십 GPU·일)
125M–1B Llama형 모델, WSD 스케줄, NVFP4(또는 시뮬레이션). decay 구간 여러 지점에서 (i) BF16 대비 격차, (ii) Hutchinson tr(H)와 power-iteration λ_max, (iii) ablation: forward만 BF16 / backward만 BF16 / 그룹 16→8 / SR forward, (iv) 최종 η 두 값. 회귀 격차 ≈ aη + cΔ²tr(H_t) 의 적합도와 c의 이론값 1/24 근접성. 실패 기준: forward 전환이 격차의 절반 미만을 제거하거나 격차가 Δ²에 비례하지 않으면 모델 기각.

### 평가
novelty 중상. 2026년 다섯 그룹이 각각 주장하는 병목을 하나의 2-바닥 모델로 분류하는 "진단 논문"이며, 전환 시점·포맷 선택 기준을 준다. Catalan-Tatjer et al.이 PTQ 측에서 같은 분리를 관찰했으므로 "학습 중 저정밀 격차의 분해"로 범위를 명확히 해야 한다. 리스크: ≤1B 실험은 10T 토큰 현상의 대리로 약하다는 비판을 받는다. 상수 c가 1/24에서 크게 벗어나면 정량 이론으로서의 가치가 줄고 정성 진단만 남는다.

---

## 아이디어 5. PTQ 캘리브레이션 이론: 표본복잡도(Marchenko–Pastur)와 공변량 이동, 그리고 VQ·학습 회전의 과적합

### 가설
GPTQ류의 Hessian Ĥ = (1/n)Σx_ixᵢᵀ 는 토큰이 시퀀스 내에서 강하게 상관되어 유효 표본수 n_eff ≪ n 이고, 추정 오차 ‖Ĥ − H‖는 √(d/n_eff) 로 스케일한다. 이 오차는 H^{−1}의 작은 고유값 방향에서 증폭되며, GPTQ의 1% 댐핑은 이를 가리는 ad hoc 수축이다. 학습 회전(SpinQuant, d²/2 자유도)과 코드북(AQLM/VPTQ/QTIP, 수백만 파라미터)은 GPTQ보다 훨씬 많은 파라미터를 같은 캘리브레이션에 적합시키므로 캘리브레이션 과적합이 더 크고, 2-bit SOTA 증분(≈0.3 PPL)의 일부는 과적합·표본 분산이다.

### 문헌상 공백
- 캘리브레이션 데이터만 바꿔 1,800개 모델을 만든 연구는 "성능이 캘리브레이션 세트에 따라 변하고 GPTQ는 과적합한다"고 보고 ([Williams & Aletras 2311.09755, ACL 2024](https://arxiv.org/abs/2311.09755) [S]); 후속은 "새 모델은 덜 민감"([2608.01078](https://arxiv.org/abs/2608.01078) [S]); 도메인 일치는 GPTQ엔 돕고 AWQ엔 안 돕는다 ([OpenReview pfw3saHzGU](https://openreview.net/forum?id=pfw3saHzGU) [S]). 전부 경험적이다.
- 스윕 노트가 명시적으로 적은 공백: "H를 n 표본으로 추정할 때의 MP형 오차와 GPTQ 업데이트의 관계를 다룬 RMT 논문 없음", "VQ/TCQ/학습 회전의 캘리브레이션 표본복잡도 없음", "2-bit 증분이 캘리브레이션 표본 노이즈 안에 있음".
- 다국어·추론 도메인 이동: 영어 캘리브레이션이 저자원 언어를 더 해치고 다국어 캘리브레이션이 PPL을 최대 3.52 낮춤 ([2601.18306](https://arxiv.org/abs/2601.18306) [S]); Qwen3-4B INT3에서 WikiText→Math-CoT 캘리브레이션이 정확도 28%→65% ([2606.25519](https://arxiv.org/abs/2606.25519) [S]). 이동량과 열화를 연결하는 식이 없다.

### 수학
(1) 표본복잡도. x_i가 공분산 Σ의 sub-Gaussian이면 ‖Ĥ − Σ‖_op ≲ ‖Σ‖(√(d/n_eff) + d/n_eff) w.h.p. (Vershynin 공분산 추정 정리; heavy-tail 토큰에서는 로그 인자 추가). LDLQ/GPTQ가 Ĥ로 선택한 라운딩 ΔW의 참 proxy 손실은 tr(ΔW Ĥ ΔWᵀ) + tr(ΔW(H − Ĥ)ΔWᵀ) ≤ proxŷ + ‖H − Ĥ‖_op‖ΔW‖_F² 이므로 상대 초과는 O(√(d/n_eff))·(‖ΔW‖_F²/proxŷ). 두 번째 인자는 H의 조건수가 클 때(저랭크 활성화 Hessian — QuIP가 보고한 "대다수 H의 고유값 1/4 미만이 최대값의 1% 초과") 커진다. 따라서 ε-상대 초과를 위해 n_eff ≳ d/ε² · κ_eff.

(2) 댐핑. Ĥ_λ = Ĥ + λI 는 Ledoit–Wolf 수축의 특수형이다. 최적 λ는 Σ의 스펙트럼과 d/n_eff 의 함수로 닫힌 형태(LW 공식)로 추정되며, 고정 1%가 아니라 √(d/n_eff)·평균 고유값 규모로 스케일한다는 예측을 준다.

(3) 과적합. 파라미터 수 P인 양자화 절차(회전 d(d−1)/2, 코드북 K·2^b·d_g 등)의 캘리브레이션-보류 간 proxy 손실 격차는 균일수렴 논증으로 O(√(P/n_eff))이다. SpinQuant R의 P ≈ 8×10⁶(d=4096) 대 캘리브레이션 ≈ 1.6×10⁶ 토큰(800×2048)은 격차가 O(1)일 수 있음을 뜻한다.

(4) 공변량 이동. 타깃 분포 Hessian H_T ≠ 캘리브레이션 H_C 이면 초과 = tr(ΔW(H_T − H_C)ΔWᵀ) ≤ ‖H_T − H_C‖_op‖ΔW‖_F². 즉 "활성화 2차 모멘트의 연산자 노름 거리"가 도메인 이동 열화의 자연스러운 예측량이며, 언어·추론 도메인 결과를 하나의 축으로 정렬할 수 있어야 한다.

증명 가능한 것: (1)–(4)는 표준 도구(Vershynin, Weyl, LW, 균일수렴)의 조합으로 엄밀하다. 추측: 실제 n_eff(토큰 상관 때문에 시퀀스 수에 가까울 가능성), 과적합 격차가 보고된 SOTA 증분과 같은 크기인지.

### 판정 실험(2–4 GPU·일)
Llama-3-8B, W3/W2. (a) 캘리브레이션 시퀀스 수 n ∈ {8, 32, 128, 512, 2048}과 λ를 격자로 바꾸며 캘리브레이션 proxy, 보류 토큰 proxy, end KL을 측정하고 격차를 (d/n_eff)^p로 적합. (b) 시퀀스 재표집으로 n_eff 추정. (c) 고정 1% 댐핑 vs LW-최적 λ. (d) GPTQ, AWQ, SpinQuant, AQLM/QTIP의 캘리브레이션-보류 격차를 파라미터 수 P에 대해 플롯. (e) 영어/다국어/추론 캘리브레이션에서 ‖H_T − H_C‖_op 와 타깃 열화의 상관. 성공 기준: 격차가 P와 1/n_eff에 단조 증가하고 VQ·회전 방법의 격차가 보고된 2-bit 증분(0.3 PPL)과 같은 자릿수.

### 평가
novelty 중. 도구는 교과서지만 PTQ 문헌의 명시적 공백을 채우고, "2-bit SOTA 경쟁의 일부는 과적합"이라는 결론이 나오면 분야에 교정 효과가 크다. 리스크: 결론이 "GPTQ는 128 샘플로 충분"에 그치면 기여가 작다. 과적합 가설이 기각될 수도 있다(그 자체로도 정보지만 논문은 약해진다).

---

## 아이디어 6. 양자화 롤아웃 RL의 엔트로피 인지 중요도 보정: QeRL(노이즈=탐색)과 FlashRL(편향)의 통합

### 가설
양자화 롤아웃 정책 μ와 BF16 학습 정책 π의 토큰 수준 KL은 ≈ ½δᵀJδ, J = diag(p) − ppᵀ 로, 같은 로짓 섭동 δ에 대해 p가 뾰족하면 작고 평평하면 크다. 따라서 불일치는 소수의 고엔트로피 "fork" 토큰에 집중되고, 시퀀스 길이 T가 아니라 fork 수 k ≪ T가 편향·분산을 지배한다. 이로부터 (i) IS 보정을 고엔트로피 토큰에만 적용하면 시퀀스 곱의 분산 폭발("length trap")을 피하면서 편향 상한이 O(T²Δ_max)에서 O(k·Δ_max)로 줄고, (ii) QeRL이 관찰한 "양자화 노이즈가 엔트로피를 올려 탐색을 돕는다"는 같은 토큰에서 일어나는 같은 현상의 다른 면이며, (iii) 정책이 날카로워지며 로짓 규모가 커질 때 δ가 커져 KL이 다시 증가한다는 AIS의 관찰도 같은 식에서 나온다.

### 문헌상 공백
- FlashRL/"Your Efficient RL Framework Secretly Brings You Off-Policy RL"은 INT8/FP8 롤아웃이 off-policy를 만들고 TIS(C≈2)로 BF16 수준을 회복한다고 보고 ([Flash-RL repo](https://github.com/yaof20/Flash-RL) [V], NeurIPS 2025 [S]); verl 문서는 토큰 TIS 편향 O(T²Δ_max), 시퀀스 IS의 length trap(1.1^100 ≈ 13,780)을 적었다 ([verl rollout_corr_math](https://github.com/verl-project/verl/blob/main/docs/algo/rollout_corr_math.md) [V]).
- QeRL(NVIDIA, ICLR 2026)은 NVFP4+LoRA 롤아웃에서 "양자화 노이즈가 정책 엔트로피를 올려 탐색을 향상"한다고 주장하고 IS 보정은 쓰지 않는다 ([2510.11696](https://arxiv.org/abs/2510.11696) [V: README]). FP8-RL([2601.18150](https://arxiv.org/abs/2601.18150) [S]), Jet-RL([2601.14243](https://arxiv.org/abs/2601.14243) [S]), QuRL([2602.13953](https://arxiv.org/abs/2602.13953) [S]), HiFloat4/Rollout-ResQ("학습 정밀도를 올리면 FP4 롤아웃이 더 나빠진다", [2607.26515](https://arxiv.org/abs/2607.26515) [S]), AIS("KL이 정책 날카로워짐과 함께 단조 증가, 초기엔 탐색 도움 후기엔 편향", [2605.13907](https://arxiv.org/abs/2605.13907) [S])가 서로 다른 처방을 낸다. 엔트로피, KL(μ‖π), 보상을 한 런에서 함께 보고한 논문이 없다.
- 양자화 추론 모델에서 토큰 KL이 고엔트로피 위치에 집중된다는 관찰은 Meta FAIR가 보고 ([Lotfi, Kirichenko, Li, Liu 2606.00206](https://arxiv.org/abs/2606.00206) [V]). RLVR의 이득이 20% 고엔트로피 fork 토큰에서 온다는 결과도 있다 ([Beyond the 80/20 Rule 2506.01939, NeurIPS 2025](https://arxiv.org/abs/2506.01939) [V]). 둘을 롤아웃 불일치 보정에 연결한 연구는 없다.

### 수학
(1) 토큰 KL. KL(π_t‖μ_t) ≈ ½δ_tᵀJ_tδ_t. 등방 δ(분산 σ²)에서 E[KL_t] ≈ (σ²/2)(1 − ‖p_t‖₂²). 1 − ‖p‖₂² = 1 − e^{−H₂(p)} 는 collision entropy의 단조함수다. 뾰족한 p에서 KL_t → 0.

(2) PG 편향. 목표 g = E_π[∇log π·A], 추정 ĝ = E_μ[∇log π·A]. |g − ĝ| ≤ Σ_t 2·E[TV(π_t, μ_t)]·sup|∇log π·A| 이고 Pinsker로 TV_t ≤ √(KL_t/2) ∝ σ√(1 − ‖p_t‖²)/2. 고엔트로피 토큰만 기여한다. 임계 τ 이상 엔트로피 토큰에만 IS를 적용하는 추정량의 편향은 저엔트로피 토큰의 잔여 TV 합(작음)과 절단 편향으로 분해되며, 시퀀스 곱은 k개 인자만 포함해 분산이 C^k 로 묶인다.

(3) 날카로워짐. 상대 양자화 오차에서 δ ∝ ‖z‖(로짓 규모) 이므로 RL이 로짓을 키우면 δ²가 커지고, 동시에 (1 − ‖p‖²)는 줄어든다. 두 효과의 곱이 KL의 비단조 궤적(AIS)을 만든다. 이는 측정 가능한 예측이다.

증명 가능한 것: (1)의 2차 식, (2)의 편향 상한과 k-의존성(표준 off-policy 논증). 추측: 실제 δ가 등방·토큰 독립에 가까운지, k/T가 충분히 작은지(80/20 논문은 ~20%), 엔트로피 게이팅이 상수 TIS보다 최종 정확도를 높이는지.

### 판정 실험(1–2 노드·수일)
Qwen2.5-Math-1.5B/7B 또는 Qwen3-1.7B/4B, GRPO/DAPO, vLLM FP8/INT8/NVFP4 롤아웃. 암: (a) 무보정(QeRL식), (b) 토큰 TIS C=2(verl 기본), (c) 엔트로피 게이트 TIS(H_t > τ에서만), (d) BF16 롤아웃. 모든 런에서 토큰 KL(μ‖π), 토큰 엔트로피, 보상, AIME24/MATH-500을 함께 기록. 예측: KL 질량의 80% 이상이 상위 20% 엔트로피 토큰에 있고, (c)가 (b)와 같은 정확도에 더 낮은 분산·더 긴 안정 학습을 보이며, (a)의 초기 보상 가속이 후기 정확도 손실로 바뀐다.

### 평가
novelty 중상. 분야가 매우 빠르다(2026년에만 5편). 기여의 핵심은 "두 상반된 서술이 같은 토큰 수준 양에서 나온다"는 통합과 fork-게이팅 보정이다. 리스크: IcePop, Trust Region Masking([2512.23075](https://arxiv.org/abs/2512.23075) [S]) 등 토큰 마스킹 변형이 이미 유사한 효과를 낼 수 있어, 엔트로피 기준의 차별성을 실험으로 보여야 한다. RL 실험의 분산이 커 3 seed 이상이 필요하다.

---

## 아이디어 7. 양자화 셀 정리: 후학습 델타 대 셀 폭 — LoRA 머지, unlearning 역전, 양자화 조건부 공격의 통합과 subtractive dither 방어

### 가설
그룹별 균일 양자화기(스텝 Δ)에서 기저 가중치 W의 셀 내 오프셋 u가 [−½, ½)에 균일하다고 보면(고해상도 가정), 후학습 델타 D(|d| ≪ Δ)를 더해 재양자화한 결과 D̃ = Q(W+D) − Q(W)는 비편향(E D̃ = D)이지만 원소 분산이 Δ|d| − d² 인 노이즈를 가진다. 그 SNR은 ‖D‖_F²/(Δ‖D‖_1) ≈ mean|d|/Δ ≪ 1. 이 하나의 양이 (a) LoRA 머지가 "효과는 2–34%만 남는" 이유, (b) 효용 제약 unlearning이 4-bit에서 역전되는 이유(델타가 셀 내부에 머묾), (c) 공격자가 FP에선 무해하고 양자화 후 유해한 모델을 만들 수 있는 이유(델타를 셀 내부에 가둠)를 설명한다. 반대로 공격자가 결정론적 라운딩 맵을 이용한다는 점에서, 비밀 시드의 subtractive dither 양자화는 양자화 모델을 "FP 모델 + 입력 독립 균일 노이즈"로 만들어 양자화 조건부 공격을 원리적으로 무력화한다.

### 문헌상 공백
- 머지 문제의 유일한 닫힌 해법은 QA-LoRA의 "zero-point에 접기"이며 어댑터를 그룹 상수로 제약한다 ([2309.14717, ICLR 2024](https://arxiv.org/abs/2309.14717) [V: repo]). 머지 오차 상한은 없다. "델타가 4-bit 그룹 스텝 아래라 66–98%가 지워진다"는 2026 시스템 논문 보고는 이 세션에서 ID를 확인하지 못했다 ([2609.18063] [U]).
- Egashira et al.은 preimage box를 계산해 PGD로 FP 가중치를 수리하는 공격을 NF4/INT8/FP4 ([2405.18137, NeurIPS 2024](https://arxiv.org/abs/2405.18137) [V]), GGUF 9종 ([2505.23786, ICML 2025](https://arxiv.org/abs/2505.23786) [S]), outlier 주입으로 AWQ/GPTQ까지 ([2605.15152, ICML 2026](https://arxiv.org/abs/2605.15152) [S]) 확장했다. Gaussian 노이즈 방어는 outlier 주입(Δ 확대)에 깨졌다. dither를 방어로 검토한 문헌은 검색에서 나오지 않았다.
- unlearning 역전: 효용 제약 unlearning 모델은 FP에서 잊은 지식 21%를 유지하는데 4-bit 후 83%로 복귀 ([Zhang et al. 2410.16454, ICLR 2025](https://arxiv.org/abs/2410.16454) [V: repo]). SURE(salient 가중치에 큰 LR)가 완화한다. 이론은 없다.
- 스윕 노트가 명시적으로 적은 공백: "preimage box 폭 vs 후학습 델타 노름"을 통합 정리로 쓴 논문 없음.

### 수학
정리 7.1 (재양자화 델타). u_ij ~ U[−½,½) 독립, |d_ij| ≤ Δ 이면 P(셀 변경) = |d_ij|/Δ, E[D̃_ij] = d_ij, Var(D̃_ij) = Δ|d_ij| − d_ij². 층 출력에서 D̃X = DX + NX, N은 평균 0·독립 원소. 등방 입력에서 SNR ≈ ‖D‖_F²/(Δ‖D‖_1 − ‖D‖_F²) ≈ mean|d|/Δ.
몬테카를로(Δ=1, d ~ N(0, 0.05²), 2×10⁶): 회귀 기울기 1.006(비편향), 분산 0.0402 vs 예측 0.0374, SNR 0.0664 vs 예측 0.0669.

따름정리 (durability). 개입 D가 재양자화에서 "살아남는다"는 것은 SNR ≫ 1, 즉 mean|d| ≫ Δ 와 동치다. 효용 제약 unlearning·safety SFT·LoRA는 모두 mean|d| ≪ Δ 이므로 취약하고, 사전학습된 능력은 전체 가중치 규모가 Δ ≫ 이어서 견고하다(Decoding Compressed Trust의 "4-bit는 신뢰성을 보존"과 unlearning 역전이 모순이 아님을 설명).

따름정리 (공격). 공격자는 반대로 D의 모든 원소를 셀 내부(|d_ij| < Δ(½ − |u_ij|))에 두어 Q(W+D) = Q(W)를 보장한다. 이것이 Egashira의 box 제약이며, outlier 주입은 Δ를 키워 box를 넓힌다.

설계 1 (cell-aligned sparse deltas). D ∈ {−Δ, 0, +Δ}^{sparse} 로 제약한 델타는 정수 코드에 정확히 머지되고(손실 0), 동시에 1.58-bit 델타 압축(BitDelta/Delta-CoMe 계열)이며, SNR = ∞인 durable unlearning이다. 학습은 top-s 좌표의 ±Δ 이동을 선택하는 projected/proximal step으로 한다.

설계 2 (subtractive dither 방어). Q_u(w) = Δ·round(w/Δ + u) − Δu, u ~ U[−½,½) 비밀 시드(그룹당 PRNG, 저장 비용 0). Schuchman의 정리([Schuchman 1964; Gray & Stockham 1993, IEEE T-IT "Dithered quantizers"] [U: 고전])에 의해 오차 e = Q_u(w) − w 는 입력과 독립인 균일 분포, MSE Δ²/12 (RTN의 고해상도 MSE와 동일). 따라서 배포 모델은 W_fp + E, E ⊥ W_fp. 공격자는 결정론적 라운딩 맵이 아닌 독립 균일 노이즈를 통과해야 유해 행동을 발현시켜야 하며, 이는 "FP 모델이 Δ²/12 노이즈에 취약한가"로 환원되어 방어자가 노이즈 샘플링으로 직접 검사할 수 있다. outlier 주입으로 Δ를 키우면 dither 오차도 커져 정확도 하락으로 탐지된다.

증명 가능한 것: 정리 7.1(고해상도 가정), durability 동치, dither의 입력 독립성(고전). 추측: SNR 하나가 서로 다른 unlearning 방법·LR·그룹 크기에 걸쳐 역전 비율을 단조 예측하는가; dither가 2-bit처럼 비고해상도 영역에서 RTN보다 얼마나 나쁜가(RTN은 셀 내 밀도를 이용해 Δ²/12보다 작은 MSE를 얻는다); GPTQ식 오차 보상과 dither의 결합.

### 판정 실험(1–2 GPU·일)
1. 머지: Llama-3-8B, QLoRA 어댑터 여러 개(랭크·LR 다양) → 머지·재양자화 → 태스크 효과 유지율을 SNR = ‖D‖_F²/(Δ‖D‖_1)에 대해 플롯. 예측: 단조 S-곡선.
2. unlearning: FailureLLMUnlearning 코드(MUSE, Llama-2-7B)로 GA/NPO/RMU/SURE 변형의 D를 추출, SNR과 4-bit 후 지식 복귀율의 상관.
3. 방어: eth-sri 공격 코드로 Phi-2/Gemma-2b에 NF4/INT4 공격 모델 생성 → 비밀 dither 양자화 → 공격 성공률과 효용(MMLU) 비교. 예측: 성공률이 무작위 노이즈 모델의 기준선으로 떨어지고 효용은 RTN 대비 1점 이내.
4. cell-aligned sparse delta로 unlearning 후 4-bit에서 복귀율 0 확인.

### 평가
novelty 높음. 세 문헌(PEFT, unlearning, 보안)이 각자 경험적으로 알던 것을 하나의 스칼라 예측량과 두 설계로 묶고, dither 방어는 증명 가능한 보안 보장을 준다. 리스크: QuantGuard([2606.29239](https://arxiv.org/abs/2606.29239) [S])·QVec([2606.20254](https://arxiv.org/abs/2606.20254) [S]) 등 2026 방어와 비교해야 한다. dither가 저비트에서 정확도를 해치면 방어의 실용성이 제한된다(그 경우 4-bit 이상에서만 유효하다고 범위를 정직하게 적는다).

---

## 아이디어 8. 추론 모델 PTQ의 엔트로피 가중 캘리브레이션

### 가설
토큰 t에서 로짓 섭동 δ가 만드는 end KL은 ≈ ½δᵀJ_tδ 로, 등방 δ에서 (σ²/2)(1 − ‖p_t‖²)에 비례한다. 즉 같은 층별 오차라도 고엔트로피 토큰에서 손실 기여가 크다. GPTQ의 XᵀX는 모든 토큰을 동등 가중하고 WikiText/C4는 저엔트로피 연속 토큰이 대부분이라, 추론 트레이스의 고엔트로피 fork에서 생기는 손실을 과소 적합한다. 토큰 가중 w_t = 1 − ‖p_t‖₂² (한 번의 forward로 계산)을 Hessian에 넣는 것만으로 추론 벤치마크 열화와 CoT 길이 증가(hesitation 토큰 과표집)가 줄어든다.

### 문헌상 공백
- Liu et al.은 추론 모델 PTQ를 체계 평가하고 자가생성 추론 트레이스를 캘리브레이션에 썼다 ([2504.04823, COLM 2025](https://arxiv.org/abs/2504.04823) [V: repo]). Qwen3-4B INT3에서 WikiText→Math-CoT 캘리브레이션이 28%→65%, CoT 30k→12k 토큰 ([2606.25519](https://arxiv.org/abs/2606.25519) [S]). 도메인 효과는 크지만 "무엇이" 효과를 내는지(엔트로피 프로필 vs 어휘·도메인)는 분리되지 않았다.
- Meta FAIR는 양자화-FP 간 토큰 KL이 고엔트로피 위치에서 최대이고 그곳에서 "wait/but/alternatively"가 과표집되어 CoT가 길어지며 실패의 최대 52%가 정답을 중간에 얻고도 확정하지 않는 경우라고 보고 ([2606.00206](https://arxiv.org/abs/2606.00206) [V]). 디코딩 측 로짓 페널티만 제안했다.
- GuidedQuant는 end-loss 기울기로 출력 채널을 가중하지만(비용: 1×A100 0.3h@7B, 6–7×A100 2.7h@70B) 토큰 믹스는 바꾸지 않는다 ([2505.07004, ICML 2025](https://arxiv.org/abs/2505.07004) [S]). 엔트로피 가중 캘리브레이션 토큰을 다룬 논문은 검색에서 없었다.

### 수학
보조정리 8.1. 로짓 섭동 δ_t ~ (0, σ²I)에 대해 E[KL(p_t‖softmax(z_t+δ_t))] ≈ (σ²/2)·tr(J_t) = (σ²/2)(1 − ‖p_t‖₂²). 따라서 토큰의 Fisher 가중은 collision entropy의 단조함수다. 층 출력 섭동 ε_t가 로짓에 Jacobian G_t로 전달되면 KL_t ≈ ½ε_tᵀG_tᵀJ_tG_tε_t 이고, G_t가 토큰 간 대략 등방·동질이라는 가정 아래 토큰 가중 Hessian H_w = Σ_t w_t x_tx_tᵀ, w_t = 1 − ‖p_t‖² 가 end KL의 1차 대리가 된다(GuidedQuant의 대각·토큰 수준 근사).

따름정리 (importance sampling 해석). 배포 분포에서의 w_t 분포와 캘리브레이션 분포에서의 w_t 분포가 다르면 H_w는 토큰 수준 공변량 이동(아이디어 5의 (4))의 특수형이다. 추론 트레이스는 평균 w_t가 높고 꼬리가 두껍다(fork).

증명 가능한 것: 보조정리 8.1(2차). 추측: G_t 동질성 가정, 가중만으로 도메인 교체 효과의 상당 부분을 재현하는지.

### 판정 실험(1–2 GPU·일)
R1-Distill-Qwen-7B, Qwen3-8B(thinking). GPTQ/GPTAQ W4/W3 g128에서 캘리브레이션 (a) C4, (b) 자가생성 CoT, (c) CoT + w_t 가중, (d) C4 + w_t 가중(가중 효과를 도메인에서 분리), (e) C4에서 w_t 상위 분위만 추출(엔트로피 층화). AIME24/25, MATH-500, GPQA, LiveCodeBench, CoT 길이, "정답 미확정" 비율(2606.00206 프로토콜), 3 seed. 성공 기준: (d)가 (a)와 (b) 사이의 격차 중 절반 이상을 회복하고 CoT 길이 증가를 줄임. (d)≈(a)면 효과는 도메인·어휘이고 가설 기각.

### 평가
novelty 중. 방법은 사실상 한 줄이지만 "Math-CoT 캘리브레이션 28→65%"라는 큰 효과의 기작을 분리하고 비용 0의 처방을 준다. 리스크: 효과가 작거나 도메인 효과에 흡수되면 workshop 수준. 평가 벤치마크(AIME ≤120문항)의 분산이 커 seed가 중요하다.

---

## 아이디어 9. RoPE 쌍 혼합 보조정리: pre/post-RoPE 논쟁의 해소, 주파수 대역 비트 할당, YaRN 상호작용

### 가설
RoPE는 각 (2f, 2f+1) 채널 쌍에 각도 mθ_f의 2×2 회전을 가한다. per-channel key 양자화는 "토큰 그룹에 걸쳐 채널별 범위"를 잡는데, post-RoPE에서는 그룹 내 토큰들이 서로 다른 각도로 회전되어 쌍의 outlier 채널 에너지가 두 채널에 섞인다. 섞임의 정도는 그룹 내 위상 범위 G·θ_f 에 의해 결정된다: 고주파 쌍(G·θ_f ≫ 1)은 완전히 섞여 범위가 두 채널 모두 max₁로 커지고(MSE 약 2배), 저주파 쌍(G·θ_f ≪ 1)은 거의 섞이지 않아 pre/post 차이가 없다. Hadamard/VQ 양자화기는 어차피 채널을 섞으므로 pre-RoPE의 이득이 사라진다. 이 보조정리는 KVQuant("pre-RoPE +0.82 PPL")와 YAKV("Hadamard에선 이득 없음")의 상반된 보고를 양자화기 종류의 함수로 화해시키고, "저번호(고주파) RoPE 채널이 노이즈의 대부분"이라는 관찰과 Block-GTQ의 주파수 블록 할당을 예측한다.

### 문헌상 공백
- pre-RoPE 지지: KVQuant([2401.18079, NeurIPS 2024](https://arxiv.org/abs/2401.18079) [V]), AnTKV(pre-RoPE 클러스터가 더 조밀, [2506.19505](https://arxiv.org/abs/2506.19505) [S]), TaSQ([2610.03027](https://arxiv.org/abs/2610.03027) [S]). 반대: YAKV는 Hadamard 기반 HIGGS 양자화기에서 pre-RoPE 이득이 없다고 보고 ([2604.08426] [U: 이 세션에서 미확인]); Interface-Aware KV는 Hadamard를 post-RoPE에 적용 ([2609.05764](https://arxiv.org/abs/2609.05764) [S]). 양자화기 의존성을 설명한 논문이 없다.
- 주파수 구조: KV-Latent는 저번호 RoPE 채널이 노이즈 대부분 ([2507.11273](https://arxiv.org/abs/2507.11273) [S]); Block-GTQ는 2-D 주파수 블록별 에너지 비트 할당 ([2606.24033](https://arxiv.org/abs/2606.24033) [S]); Q-ROAR와 "Rethinking RoPE scaling"은 YaRN × W4가 파괴적으로 상호작용(NTK/linear는 PPL>100)하고 대역별 스케일로 복구한다고 보고 ([2509.14391](https://arxiv.org/abs/2509.14391), [2510.00028](https://arxiv.org/abs/2510.00028) [S]). 세 현상을 하나의 복소계수 표현에서 분석한 이론은 없다.

### 수학
복소 표현: 쌍 f의 key를 k_f ∈ ℂ, query를 q_f ∈ ℂ 로 두면 logit(m, n) = Re Σ_f q_f k̄_f e^{iθ_f(m−n)}.

보조정리 9.1 (등변 양자화기의 불변성). 쌍 단위로 회전 등변(Q(Rx) = RQ(x), 또는 MSE가 회전 불변)인 양자화기에서는 key MSE와 로짓 오차 분포가 pre/post-RoPE에서 동일하다. 극좌표(크기·각) 양자화와 쌍 보존 VQ가 이에 해당한다(PolarQuant-Wu, CommVQ의 RoPE 가환 코드북).

보조정리 9.2 (per-channel max-abs의 혼합 손실). 그룹 G 토큰에 걸친 채널별 범위를 쓰는 양자화기에서, outlier 쌍(max₁ ≫ max₂)의 post-RoPE 범위는 두 채널 모두 ≈ max₁·max_{m∈G}|cos/sin(mθ_f + φ)| 이다. G·θ_f ≥ π 이면 두 채널 범위가 모두 max₁ → 쌍 MSE ≈ 2·max₁²/(12·4^b) 로 pre-RoPE(max₁² + max₂²)의 약 2배. G·θ_f ≪ 1 이면 범위가 거의 변하지 않아 이득 없음. 따라서 pre-RoPE의 총 이득 = Σ_f [혼합 인자(Gθ_f) − 1]·E_f 이며 고주파 대역에 집중된다. Llama RoPE에서 θ_i = base^{−2i/d} 이므로 저번호 채널이 고주파다(KV-Latent 관찰과 일치).

따름정리 9.3 (대역별 할당). 로짓 오차 분산 = ½Σ_f |q_f|²E|ε_f|² (위상은 크기에 영향 없음). reverse water-filling은 b_f = ½log₂(|q_f|²σ_f²/θ) 를 준다 — Block-GTQ의 key 에너지 할당에 query 에너지 가중(OSCAR/Spend-Bits의 논점)을 결합한 형태다.

YaRN 상호작용: 위치 보간은 θ_f를 대역별로 축소해 (i) 어떤 쌍이 그룹 내에서 섞이는지를 바꾸고(저주파 대역의 Gθ_f가 더 작아져 pre/post 차이가 줄고, 고주파는 유지), (ii) W_Q/W_K 양자화 오차가 만드는 q_f, k_f 섭동의 상대 위치 의존성을 확장된 범위에서 누적시킨다. 두 효과를 같은 식에서 추적할 수 있다는 것이 통합의 요지이며, Q-ROAR의 대역별 스케일은 (ii)의 1차 보정으로 해석된다.

증명 가능한 것: 9.1, 9.2(max-abs 범위의 기하), 9.3(Gaussian 가정). 추측: 실제 모델에서 outlier 쌍이 고주파에 몰려 있는지, YaRN 상호작용의 정량 예측.

### 판정 실험(1–2 GPU·일)
Llama-3.1-8B, Qwen3-8B. (a) per-channel INT2/INT3 key를 pre vs post RoPE로, 그룹 G ∈ {32, 128, 512}에서 대역별 MSE와 NIAH/RULER 측정. 예측: 이득이 Gθ_f ≥ π 대역에 집중하고 G가 커질수록 저주파로 확장. (b) Hadamard 양자화기에서 반복 → 예측: 이득 소멸. (c) 극좌표 쌍 양자화 → 예측: 불변. (d) 9.3 할당 vs 균일 할당. (e) YaRN 4×에서 (a) 반복.

### 평가
novelty 중. 보조정리는 간단하지만 2년간 상반되게 보고된 질문을 양자화기 종류의 함수로 닫고, 측정 가능한 대역 예측을 준다. Block-GTQ·Q-ROAR·OSCAR가 각 조각을 이미 쥐고 있어 "통합 해석 + 검증" 논문으로 자리매김해야 하며, 방법 novelty는 낮다.

---

## 아이디어 10. MLA latent KV의 가중 변환 부호화: weight absorption 하의 최적 변환과 할당

### 가설
MLA 디코드(weight absorption)에서 latent c_j ∈ ℝ^{d_c}의 양자화 오차 e_j는 (K 역할) 모든 헤드의 로짓에 a^hᵀe_j (a^h = W_UK^{hᵀ}q^h, 흡수된 query)로, (V 역할) 출력에 W_UV^hΣ_jp_je_j 로 들어간다. 따라서 올바른 왜곡은 등방 MSE가 아니라 M = Σ_h E[a^ha^{hᵀ}] + λΣ_hW_UV^{hᵀ}W_UV^h 로 가중된 MSE이고, 고전 결과에 의해 최적 변환 부호화는 y = M^{1/2}c 의 KLT 기저에서 M^{1/2}Σ_cM^{1/2} 의 고유값에 reverse water-filling 하는 것이다. 이 변환은 선형이므로 W_UK, W_UV, W_O에 접혀 런타임 비용이 0이며, 스펙트럼이 급격히 감쇠하면 다수 방향에 0비트가 할당되어 sub-4-bit latent가 FP8 품질을 낸다.

### 문헌상 공백
- FlashMLA는 V4.1용 FP8 latent(528 B/토큰, e4m3 + 32당 e8m0 스케일)와 FP4(288 B/토큰, e2m1 + 16당 e4m3)를 출시하고 RoPE 64차원도 양자화하지만 정확도 표가 없다 ([FlashMLA repo](https://github.com/deepseek-ai/FlashMLA) [V]). V3.2 포맷은 RoPE 부분을 bf16로 유지했었다 [S]. SnapMLA는 위치 부분을 고정밀로 두라고 주장 ([2602.10718](https://arxiv.org/abs/2602.10718) [S]) — 상반된다.
- MLA 양자화 민감도 데이터는 단일 프리프린트(4-bit KV PPL +2.8% GQA vs +3.0% MLA, [2603.04428](https://arxiv.org/abs/2603.04428) [S])와 KVarN의 GLM-4.7-Flash k4v2 AIME25 동률(0.94× throughput) ([KVarN repo](https://github.com/huawei-csl/KVarN) [V])뿐이다. 스윕 노트: "weight absorption 하 latent 양자화 이론 없음(W_UK, W_UV, RoPE 분리 가지로의 오차 전파)". latent에 회전·변환을 적용한 문헌은 검색에서 없었다.
- MHA에서는 query 공분산 가중 변환이 Hadamard를 이긴다는 주장이 있다 (OSCAR [2605.17757](https://arxiv.org/abs/2605.17757) [S], Spend Bits Where Queries Look [2608.04074](https://arxiv.org/abs/2608.04074) [S]). MLA의 latent는 모든 헤드가 공유하므로 가중 행렬 M이 헤드 합으로 정의되는 점이 구조적으로 다르고, 변환을 가중치에 접을 수 있다는 점은 MHA의 online Hadamard(QuaRot)와 달리 디코드 비용이 0이다.

### 수학
DeepSeek-V2/V3 MLA: c_j = W_DKV h_j (d_c = 512), k_j^C = W_UK c_j, v_j^C = W_UV c_j, 분리 k_j^R = RoPE(W_KR h_j) (64차원, 헤드 공유). score^h_{tj} = a^{hᵀ}c_j + q_t^{Rᵀ}k_j^R, a^h = W_UK^{hᵀ}q_t^h. 출력 o^h = W_UV^hΣ_jp_jc_j → W_O에 흡수.

정리 10.1 (가중 왜곡). 양자화 오차 e_j, 공분산 Σ_e 에 대해 로짓 오차 에너지 E Σ_h(a^{hᵀ}e)² = tr(Σ_eM_K), M_K = Σ_hE[a^ha^{hᵀ}]; 출력 오차 에너지 ≤ tr(Σ_eM_V), M_V = Σ_hW_UV^{hᵀ}W_UV^h (어텐션 가중 평균의 볼록성으로 토큰 길이에 무관). 합성 가중 M = M_K + λM_V (λ는 로짓 오차의 softmax Jacobian 전달을 포함한 교환비).

정리 10.2 (최적 변환). Gaussian latent, 가중 MSE tr(Σ_eM), 고해상도 가정에서 최적 변환 부호화는 T = U^ᵀM^{1/2} (U: M^{1/2}Σ_cM^{1/2}의 고유벡터)로 변환한 뒤 고유값 λ_i에 b_i = max(0, ½log₂(λ_i/θ)) 를 할당한다(가중 MSE ⇔ 백색화 후 표준 MSE; 표준 변환 부호화 정리). T는 c에 선형이므로 캐시는 Tc를 저장하고 a^h ← T^{−ᵀ}a^h, W_UV ← W_UVT^{−1} 로 접힌다.

RoPE 가지: k^R은 헤드 공유·64차원·‖q^R‖가 크므로 per-token 스케일이면 아이디어 1의 이분산 편향이 가장 크게 나타나는 자리다. 정리 1에 의해 per-channel 스케일 또는 고정밀 유지가 예측되며, FlashMLA V4.1(양자화)과 SnapMLA(고정밀)의 상반된 선택은 "per-token인가 per-channel인가"로 판별될 것이다.

증명 가능한 것: 10.1(선형대수), 10.2(고전 가중 변환 부호화). 추측: M^{1/2}Σ_cM^{1/2}의 유효 랭크가 실제로 낮은가(그래야 이득이 크다), M_K의 query 통계가 도메인에 걸쳐 안정적인가(아이디어 5의 공변량 이동이 여기에도 적용된다).

### 판정 실험(2–3 GPU·일)
DeepSeek-V2-Lite(16B, d_c=512), GLM-4.7-Flash 또는 Kimi 계열. (a) 캘리브레이션에서 M_K, M_V, Σ_c를 추정하고 M^{1/2}Σ_cM^{1/2} 스펙트럼의 유효 랭크 측정. (b) latent를 4/3/2 비트로 raw per-token(FlashMLA 유사) vs T-변환 + water-filling 으로 fake-quant; 로짓 오차, NIAH/RULER, AIME 비교. (c) k^R을 per-token/per-channel/BF16으로 바꿔 sink 질량과 정확도 측정. 성공 기준: 같은 비트에서 T-변환이 로짓 오차를 유의하게 줄이고 3-bit 변환이 4-bit raw와 동등.

### 평가
novelty 중상. DeepSeek·Kimi·GLM이 모두 MLA로 이동한 지금 산업 영향이 크고, 변환이 가중치에 접힌다는 점이 MHA 대비 구조적 이점이다. 리스크: OSCAR/Spend-Bits 저자들이 MLA로 확장하는 데 몇 달이면 충분하다. λ 선택과 softmax 비선형 전달이 이론의 깔끔함을 해친다(아이디어 1과 결합하면 보완된다).

---

## 부록 A. 검토 후 순위에서 내린 후보와 이유

A1. 과학습 취약성의 기작(토큰 수 vs LR 스케줄·sharpness). 가설은 "Kumar/Ouyang의 D^γ 효과는 상당 부분 cosine decay의 말기 곡률 상승"이었고 QiD ≈ ΣΔ_g²tr(H_g)/24 분해와 재평탄화 개입을 제안할 계획이었다. 그러나 Catalan-Tatjer, Ajroldi, Geiping([2510.06213, ICLR 2026](https://arxiv.org/abs/2510.06213) [S])이 32B/15T 궤적에서 "LR decay 후 손실과 양자화 오차가 분리된다"고 이미 보고했다. 남는 기여는 Δ² vs tr(H) 분해와 개입 실험이며, 아이디어 4에 흡수했다.

A2. MoE 라우팅 플립 분해. "양자화 활성화가 top-k 라우팅을 뒤집어 Lipschitz로 묶이지 않는 불연속 오차를 만든다; 마진 분포로 플립 확률 2Φ(−m/σ)를 예측한다"는 가설은 여전히 유효하지만, 2026년에 VSRAQ([2606.05688](https://arxiv.org/abs/2606.05688) [S]), MoBiE([2604.06798](https://arxiv.org/abs/2604.06798) [S]), EAQuant(라우팅 KL 최소화, [2506.13329](https://arxiv.org/abs/2506.13329) [S]), GEMQ(1.5-bit에서 40% 토큰 재라우팅, [2605.23078](https://arxiv.org/abs/2605.23078) [S])가 측정과 방법을 이미 냈다. 남는 틈은 "손실의 플립 성분 vs 평활 성분 분해"와 RL 롤아웃 불일치에서의 MoE 특이성(아이디어 6의 따름정리)이다.

A3. 추측 디코딩 수락률 법칙. 1 − α = E[TV(p,q)] ≤ √(KL/2), KL ≈ ½δᵀJδ, δ ∝ 2^{−b} 이므로 log₂(1 − α) ≈ −b + const(드래프트 비트당 거부율 절반)라는 예측과 최적 self-draft 정밀도의 닫힌 해. 검색에서 비트폭-수락률 모델은 없었다(QSpec, QuantSpec, ML-SpecQD는 수치만 보고 [S]). 유도가 전문가에게 자명해 novelty가 낮고 상수가 모델별로 달라 법칙이 느슨할 위험이 있어 내렸다. 저비용 workshop 논문으로는 적합하다.

A4. 회전된 KV의 격자/TCQ 부호화로 TurboQuant 상수 2.7 → 1에 접근. 이론 노트가 지적한 대로 2.7 ≈ √3π/2(Panter–Dite 스칼라 패널티)이므로 E8(0.65 dB)/TCQ(~1 dB) 부호화가 상수를 줄일 것이라는 예측은 정확히 계산 가능하다. 그러나 NestQuant([2502.09720](https://arxiv.org/abs/2502.09720) [S])가 E8 격자를 W4A4KV4에 이미 썼고 CQ/CommVQ가 VQ를 썼으므로 방법 novelty가 낮다. 2-bit value가 병목(cos-sim 0.94)이라는 커뮤니티 측정과 결합하면 실용적이긴 하다.

A5. 변환–포맷 공동 설계(Great Inversion의 4개 공개문제, [2608.25188](https://arxiv.org/abs/2608.25188) [S]). 단일 저자 프리프린트가 문제를 이미 정식화했고 저자가 후속을 낼 가능성이 높아 직접 경쟁이 된다.

A6. prefill/decode 비대칭 이론. "prefill 오차는 문맥 표현의 평균화된 노이즈로, decode 오차는 샘플링 분포에 직접 들어가 분기 과정으로 복합된다"는 모델. Mix-Quant·Disaggregated Quantization([2605.20315](https://arxiv.org/abs/2605.20315), [2609.26333](https://arxiv.org/abs/2609.26333) [S])의 경험적 결론에 이론을 붙이는 것이지만 실험 설계가 모호해 내렸다.

## 부록 B. 공통 리스크와 평가 위생

- 평가 지표: 4-bit 이상에서 perplexity는 방법 간 차이를 보이지 못하고(OmniQuant 표에서 Llama-2-7B W4 g128 방법 간 0.03–0.05 PPL), 추론은 perplexity와 분리된다(GPTQ v1/v2: ARC-c 동일, GSM8K 0.40 vs 0.76, [GPTQv2 README](https://raw.githubusercontent.com/Intelligent-Computing-Lab-Yale/GPTQv2/main/README.md) [V]). 모든 아이디어의 판정 실험은 KL·토큰 수준 지표·추론 벤치마크를 함께 보고해야 한다.
- 캘리브레이션 표본 분산(아이디어 5)이 0.3 PPL 수준의 "SOTA 증분"을 만들 수 있으므로 seed와 CI 없는 비교는 무의미하다.
- fake-quant로 끝나는 서빙 논문은 시스템 학회에서 약하다. 아이디어 1, 2, 10은 최소한 하나의 실제 커널 측정을 붙여야 한다.
- 2026년 ID 다수가 [S]다. 특히 2606.15007(Nemotron 3 Ultra), 2609.18063(LoRA 머지 소거), 2604.08426(YAKV)은 존재·내용을 PDF로 확인하기 전까지 인용하지 말 것.

## 부록 C. 이 보고서에서 수치로 검증한 유도

스크립트는 세션 스크래치패드에서 실행했다(numpy 몬테카를로).
- 아이디어 1: 이분산 2차 편향식, n=64, σ_j² ∈ U[0, 0.3], 4×10⁵ 표본: 예측-측정 상관 0.9996, 최대 상대 오차 3.9%. 동분산: 상관 0.9995, 평균 맵 엔트로피 3.0635 → 3.0681.
- 아이디어 3: E2M1 블록 MSE/σ²: NVFP4형(16, E4M3) 0.0092, MXFP4 올림(32, UE8M0) 0.0133, MXFP4 내림 0.0414, 정확 스케일 B=16 0.0089. E2M1 shrinkage: E[|Q(x)|−|x|]/E|x| = −0.66%.
- 아이디어 2: a=0.999, T=2×10⁴, Δ=0.02, 입력 노이즈 10⁻³: max|d_t|/Δ = RTN 3.36, SR 12.92, ΣΔ(정확) 0.58, ΣΔ(4-bit 잔차) 2.03(최종 0.05).
- 아이디어 7: Δ=1, d ~ N(0, 0.05²), 2×10⁶: E[D̃]/d 기울기 1.006, Var 0.0402(예측 0.0374), SNR 0.0664(예측 0.0669).
