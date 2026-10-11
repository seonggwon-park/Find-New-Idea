# LLM 양자화 연구 아이디어 10선

작성일: 2026-10-10. 대상: LLM 파이프라인 전체(사전학습, PTQ, 서빙, 후학습/RL, 보안)에서 main-track 수준의 novelty와 contribution을 노릴 수 있는 양자화 연구 가설 10개. 각 항목은 (1) 문헌상 공백과 근거, (2) 가설, (3) 정리 수준으로 증명 가능한 부분과 추측에 머무는 부분, (4) 며칠 안에 판정 가능한 최소 실험, (5) 냉정한 novelty·리스크 평가로 구성한다.

---

## 0. 요약과 판정표

2026-10-11 개정: arXiv 전문 대조와 적대적 novelty 판정(0.4절) 결과, 초안의 10개 아이디어 중 9개가 2026년 논문에 핵심을 선점당했다. 아래 표는 개정 판정이며, 각 행의 "남는 기여"만이 연구 대상으로 의미가 있다.

| # | 아이디어(개정 제목 요약) | 파이프라인 단계 | 판정 실험 비용 | novelty(개정) | main-track(개정) |
|---|---|---|---|---|---|
| 1 | softmax 2차 편향: −σ²/2를 넘는 p-의존 완전 보정, single-token decoding 재분배 | 서빙(KV, LM head) | 1–2 GPU·일 | 낮음–중간(선점: 2605.26266 보정식, 2608.14191 정리식) | 낮음(2605.26266 대비 decoding 이득이 뚜렷할 때만 중간) |
| 2 | 재귀 상태 error-feedback: 동일 비트 회계, deadband 하한, erasure 분기 | 서빙(Mamba/hybrid) | 2–3 GPU·일 | 낮음(선점: 2609.39185, 2607.23390, 2609.04490, 2609.30950) | 낮음(원안은 부록 A급, 잔여 가설은 workshop급) |
| 3 | 블록 스케일 포맷 왜곡: PoT 스케일 라운딩과 gain 편향 | 포맷·사전학습·PTQ | CPU 수치 + 수일 | 낮음–중하(선점: 2601.19026, 2210.05470, 2605.20402, 2510.25602) | 낮음(편향 가설 확인 시 workshop/단신) |
| 4 | FP4 LR-decay 말기 격차: 분산 바닥 대 편향 바닥, 전환 조건 | 사전학습 | 수십 GPU·일(≤1B) | 중간(이론 선점: 2601.22101, 2510.10693, 2505.19115; 진단·전환 기준만 신규) | 중(진단 실험 논문) |
| 5 | 캘리브레이션 격차 3항 분해: n_eff, 댐핑 척도, 회전·VQ 과적합 | PTQ | 2–4 GPU·일 | 낮음(선점: 2609.31560 외) | 낮음(n_eff·댐핑 실증이 강할 때만 workshop) |
| 6 | 양자화 롤아웃 RL 토큰 단위 불일치 지도와 보정 벤치마크 | 후학습(RL) | 1–2 노드·수일 | 낮음(선점: 2609.32444, 2607.22186, 2605.13907, 2512.23075) | 낮음(ε_t의 ‖z‖ 의존 확인 시 중간) |
| 7 | 양자화 셀 정리의 두 영역과 비밀 시드 subtractive dither 방어 | 후학습(PEFT)·보안 | 1–2 GPU·일 | 낮음(선점: 2609.21834, 2505.18724, 2605.15138; dither 방어만 중간) | 원안 불가, dither 방어로 축소 시 보안 트랙 경계 |
| 8 | 추론 모델 캘리브레이션 도메인 효과 분해: 엔트로피 대 부분공간 정렬 | PTQ(reasoning) | 3–4 GPU·일 | 낮음–중(선점: 2505.07004, 2503.01820, 2605.08755) | 낮음–중(2×2 분리 실험에서 독립 효과가 나올 때만) |
| 9 | RoPE 쌍 혼합의 범위 모델과 회전된 평균 차감 | 서빙(KV) | 1–2 GPU·일 | 낮음(선점: 2501.19392, 2401.18079) | 낮음(분석 노트 수준) |
| 10 | MLA latent 가중 변환 부호화: RMSNorm 제약 하 닫힌형, K·V 결합 가중 | 서빙(MLA 계열) | 2–3 GPU·일 | 낮음(선점: 2609.36760, 2608.04074) | 중하(같은 비트에서 QuantMLA 대비 우위 시) |

판정의 전제(개정): 초안은 1, 2, 7을 "선행 논문이 사실상 없어 가장 안전"하다고 했으나, 이 판단은 전문 호스트가 차단된 상태의 스니펫 검색에 근거했고 틀렸다. 세 아이디어 모두 2026년 5–9월 논문이 핵심을 선점했다. 현재 독립된 main-track 기여로 성립할 가능성이 남은 것은 아이디어 4(진단 실험 논문)뿐이며, 나머지는 각 행의 좁은 증분이 실험으로 강하게 확인될 때만 workshop 이상이 된다. 스윕에서 새로 얻은 가설은 0.5절과 아이디어 11 이후에 평가했다.

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
4. 서빙 측 KV 양자화 이론은 초안 작성 시 추정량 수준(QJL 비편향 내적, TurboQuant Beta 주변분포)에 머문다고 보았으나 이는 틀렸다. 2026년에 어텐션 출력 오차를 직접 다루는 정리가 여럿 나왔다: WitCert([2607.28699](https://arxiv.org/abs/2607.28699) [V], Theorem 1: |ε_t| ≤ c_t이면 TV(p, p̃) ≤ ½(A² − 1), A = E_p̃[e^c]; 부호 없는 최악 경우 상한), 2차 softmax 응답을 포함한 변환 부호화 분석([2608.14191](https://arxiv.org/abs/2608.14191) [V] App. A), 1차 Jacobian 민감도([2608.20988](https://arxiv.org/abs/2608.20988) [V]). 부호를 가진 편향과 분산 항을 함께 다룬 것은 2605.26266의 보정식뿐이다(아이디어 1). QJL 잔차는 커뮤니티에서 "softmax가 분산을 증폭해 품질을 해친다"고 보고되어 vLLM 구현에서 제외됐다 ([vLLM TurboQuant docs](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/turboquant/) [V]; 문서 주석이지 벤치마크는 아니다).
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


### 0.4 적대적 novelty 재판정 (2026-10-11)

아이디어마다 판정 에이전트 1개가 선행 연구 탐색 노트(`research_notes/arxiv_sweep/discover/`, 키워드·인용 이웃·반대 증거 3각도)에서 가장 위협적인 논문을 골라 전문을 읽고, 이론 선점·시스템 선점·사실 전제의 세 관점에서 판정했다(`judge/idea*.md`). 판정의 핵심 근거 중 아이디어 1(2605.26266 Eq. 13–14 "the Taylor correction is exactly half the score-space noise variance")과 아이디어 2(2609.39185 §관련연구 "sigma–delta … need the previous rounding error to be stored, which doubles the size of a state cache")는 원문을 직접 다시 대조했다.

| # | 개정 전 novelty | 개정 후 | 결정적 선점 근거 |
|---|---|---|---|
| 1 | 높음 | 낮음–중간 | 2605.26266 Eq. 10–14(−σ²/2 보정, CGF 논증, per-channel 공유 스칼라 App. E); 2608.14191 App. A(정리 1과 대수적으로 동일) |
| 2 | 높음 | 낮음 | 2609.39185 Lemma 1·Corollary 1(deadband, RTN/SR 증가 법칙)과 ΣΔ 저장 2배 반론; 2607.23390 Thm 8(EF carry 증명) |
| 3 | 중상 | 낮음–중하 | 2601.19026 eq.(1)–(5)(Gaussian 블록 왜곡 정확 계산, UE5M3 제안, ICLR 2026); 2210.05470(Gumbel 블록 최대 + PoT 스케일) |
| 4 | 중상 | 중간 | 2601.22101 eq.(18), 2510.10693 Thm VI.3(η 무관 양자화 바닥), 2505.19115 §4.1 |
| 5 | 중 | 낮음 | 2609.31560 Thm 3.1·eq.(35)(OPTQ 일반화 상한, 표본수 의존 λ) |
| 6 | 중상 | 낮음 | 2609.32444 Lemma 3.1·cap 1+λ(1−p_t); 2607.22186 Eq. 19·Prop. 1–2 |
| 7 | 높음 | 낮음(dither 방어만 중간) | 2609.21834 Model 2·Corollary 6(교차확률 η|u|/s, 비편향 이동) |
| 8 | 중 | 낮음–중 | 2505.07004 Sec. 3.2(토큰 가중 Hessian); 2605.08755(softmax-Fisher, 부분공간 정렬 설명) |
| 9 | 중 | 낮음 | 2501.19392 App. D, 2401.18079 App. P Table 20(양자화기별 pre/post-RoPE 부호 반전) |
| 10 | 중상 | 낮음 | 2609.36760(native MLA latent 변환·kernel); 2608.04074 Thm 1·eq.(38)(M^{1/2} 가중 KLT) |

해석: 초안의 오류는 수학이 아니라 문헌 공백 판단에 있었다. 각 아이디어의 유도(부록 C의 수치 검증 포함)는 대부분 맞았고, 판정 에이전트가 정리 1의 식과 2608.14191 식의 동치를 수치로 재확인했다(최대 차 1e-18). 그러나 "아무도 하지 않았다"는 주장은 9건에서 틀렸다. 선점 논문의 상당수가 2026년 5–9월에 나왔다는 점은 이 분야의 속도가 몇 달 단위라는 뜻이며, 개정된 증분도 같은 위험을 진다.

---

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

## 아이디어 3. 블록 스케일 마이크로포맷 왜곡 이론의 확장: PoT 스케일 라운딩 계열과 편향(gain)

### 가설
Gaussian 블록의 최대값 밀도 × E2M1 구간 적분으로 기대 MSE를 계산하는 틀은 이미 있다(Fasoli). 남은 가설은 OCP v1 스케일 규칙의 학습 문제가 MSE에서 오지 않는다는 것이다. OCP와 올림의 Gaussian MSE는 사실상 같다. 원인은 최대 원소의 체계적 축소(편향)의 승법 누적이다. 이 틀을 PoT 라운딩 계열과 gain으로 확장하면 Mishra 레시피, UFP4 shrinkage, Li의 미유도 교차항을 한 틀에서 설명한다.

### 문헌상 공백
- **해석 이론은 이미 있다**("전부 경험적"은 틀렸다).
  - Fasoli et al.: 정확한 블록 최대값 밀도 eq.(5)에 E2M1 Voronoi 구간 MSE eq.(2)–(4)를 결합한다. χ²≈2·10⁻⁹이고 UE4M3 스케일에서 4·10⁻⁸(App. F)이다. UE4M3 subnormal에 의한 블록 크기 역전을 보이고 UE5M3를 제안한다 ([Is Finer Better? 2601.19026, ICLR 2026](https://arxiv.org/abs/2601.19026) [V]).
  - Soloveychik et al.: Gumbel(Lemma 2), PoT 올림 BFP 상한(Prop. 2), 4비트 최적 n=64를 준다. 정수 가수만 다룬다 ([2210.05470](https://arxiv.org/abs/2210.05470) [V]).
  - Li/Wu/Shen: UE8M0 올림 γ=2^U 모델(eq.9), E[γ]=1/ln2, RMSE 0.57. cos(e_scale,e_grid)≈−0.66은 "we do not derive in closed form"이라고 명시한다(Remark 3.3) ([2605.20402](https://arxiv.org/abs/2605.20402) [V]).
  - Chen et al.: Theorem 1–2가 UE8M0(ρ=1.5 고정)과 E4M3 스케일의 QSNR 닫힌 형태다 ([INT v.s. FP 2510.25602](https://arxiv.org/abs/2510.25602) [V]).
  - MXAttention Theorem 1: E8M0 최적 경계 7.25는 log2 M의 분포와 무관하다. Table 1은 OCP=e_floor(8)로 공식화한다 ([2607.24377](https://arxiv.org/abs/2607.24377) [V]).
  - Egiazarian et al.도 Sec. 3.1에 해석적 MSE 경계(Lemma 1–2)를 둔다. 회전 후 kurtosis는 0.02–0.05다 ([2509.23202, ICLR 2026](https://arxiv.org/abs/2509.23202) [V]).
- **경험적 보고(정정).**
  - Meta(ICML 2026, PTQ): OAS+MBS로 MXFP4를 61.25에서 66.50으로 올린다. NVFP4는 67.02다 ([2603.08713](https://arxiv.org/abs/2603.08713) [V]).
  - NVIDIA의 "MXFP4 2.5% vs NVFP4 1.5%"는 8B/1T에서 BF16 대비 **학습 손실** 상대오차다(Fig. 6a). 텐서 오차가 아니다 ([2509.25149](https://arxiv.org/abs/2509.25149) [V]).
  - Mishra et al.: OCP v1(⌊log2 amax⌋−emax, 사실상 내림)은 843M(Fig. 5)에서 "training issues"로 조기 중단됐다(v1 초록은 "divergence"). ceil+E4M3는 8B/15T에서 val-ppl 차이 <0.50%다(fake-quant). MXFP8 결과이며, 원인은 저자가 "We hypothesize"라고 쓴다 ([2506.08027](https://arxiv.org/abs/2506.08027) [V]).
- **포맷 제안(정정).**
  - UE5M3는 Hu et al. [2509.17791](https://arxiv.org/abs/2509.17791) [V]가 도입했고, Fasoli가 MSE 분석으로 제안했다. [2609.02846](https://arxiv.org/abs/2609.02846) [V]는 단일 seed 에뮬레이션 후속 실증이다.
  - Four-over-Six는 E4M3 스케일을 유지하는 블록 스케일 **선택** 알고리즘이다. 사전학습에서는 MAE로 선택한다 ([2512.02010](https://arxiv.org/abs/2512.02010) [V]).
  - HiFloat4 포맷은 [2602.11287](https://arxiv.org/abs/2602.11287) [S]가 제안했고, [2604.08826](https://arxiv.org/abs/2604.08826) [V]는 사전학습 연구다.
- **UFP4.** 국소 균일 밀도의 bin 수준 닫힌 형태 Eq.(2) (2q_i−q_{i−1}−q_{i+1})/4(q=2에서 −0.125)와 승법 누적 Eq.(3)–(5)가 있다. 분포 수준 식, 블록·스케일 의존, α 측정은 없다. RHT 증폭의 근거는 ΔSQNR(MSE형)뿐이다 ([2606.20381](https://arxiv.org/abs/2606.20381) [V]).
- Sun et al.(ICML 2025)은 (N, D, E, M, B)에 대한 경험적 적합식이다 ([2501.02423](https://arxiv.org/abs/2501.02423) [V]).
- **남는 공백(좁다).**
  - (i) OCP, ceil, UOS, E4M3, UE5M3를 **편향(gain)·최대 원소 편향**까지 비교한 결과가 없다.
  - (ii) cos≈−0.66 유도가 없다.
  - (iii) 밀도 기울기 shrinkage의 분포 수준 크기와 깊이 판정식이 없다.
  - (iv) Gumbel 점근식은 편의 이상이 아니다.

### 수학
블록 iid N(0,σ²), M=max|x_i|. 정확 밀도는 Fasoli eq.(5)를 쓰고, 점근은 E[M]≈σ√(2 ln 2B)다. 원소는 E2M1 {0,.5,1,1.5,2,3,4,6}에서 RTN하고 6에서 포화한다.

- **E4M3 스케일**: s_q/s∈1±2⁻⁴다. UE4M3 subnormal에서는 블록 크기 역전이 생긴다 ([2605.08565](https://arxiv.org/abs/2605.08565) [V]). "32 vs 16 → E[M] 비 1.10" 단조성은 범위 항을 넣어야 성립한다.
- **UE8M0 올림(ceil(log2(M/6)))**: γ=2^U에서 균일 근사 E[γ²]=E[4^U]=3/(2 ln 2)≈2.16이다. Li 모델의 2차 모멘트이며, E2M1 비균일성 때문에 실제 비는 이보다 작다.
- **OCP v1(⌊log2 M⌋−2, 정규화 최대 [4,8))** ([2310.10537](https://arxiv.org/abs/2310.10537) [V]): 올림과 다른 곳은 클리핑 대역 M/X∈(6,8)뿐이다. 해당 블록 비율은 Gaussian B=32에서 29.6%, MXFP8에서 16.9%다.
- **MSE/σ²**(2×10⁵ 블록): NVFP4형 0.0092, 올림 0.0133, OCP 0.0132, UOS 0.0125, 정확 B=16 0.0089. 이전 판의 "내림 0.0414"는 항상 클리핑하는 순수 floor(log2(M/6)) 규칙의 값이며 OCP가 아니다. "MSE 3.1배"와 "RMS 비 2.12"는 철회한다.
- **차이는 편향에 있다**: MXFP8에서 OCP/올림의 상대 MSE 비는 1.22배이지만, 최대 원소 편향은 −1.28% vs −0.08%이고 gain은 0.9972 vs 0.9993이다. MXFP4 B=32의 gain은 올림 0.990, OCP 0.980, UOS 0.988, NVFP4 0.996이다.
- **Shrinkage 분해**: UFP4 구간 편향은 전역 균일 밀도에서 합이 0이다(0.25·(−0.125)+0.75·(−0.125)+1.5·(−0.25)+1·(+0.5)=0). 순 편향은 밀도 기울기, deadzone, 클리핑에서 나온다. Gaussian B=16의 구간별 오차는 q=2 −0.112, q=4 −0.164, q=6 +0.545다. 순 1−α≈0.0033은 거의 상쇄된 잔차이고 부록 C의 −0.66%와 같은 계산이다(판정 재계산은 −0.70%, α 0.9959 [S]).
- **판정식**: 중점 경계는 MSE 최적이므로 편향은 그 대가다. (L·bias)²가 추가 분산보다 크면 비편향 라운딩이 유리하다. RMSNorm이 균일 감쇠를 상쇄하는지는 미확인이다 [S].

증명 가능한 것: e_floor/e_ceil 계열의 MSE·gain·최대 원소 편향을 1차원 적분으로 정확히 계산하는 것. H2: cos≈−0.66은 올림 조건 γ≥1에서 오는 B 무관 상수로 유도된다. 추측: H1(편향 누적이 불안정의 원인), RHT 후 Gaussian성. OffQ는 Hadamard가 고노름 토큰을 토큰별 상수 오프셋으로 바꿀 뿐임을 보인다(PTQ) ([OffQ 2606.07116](https://arxiv.org/abs/2606.07116) [V]).

### 판정 실험(CPU 수치 + 수일)
1. Fasoli 공개 코드로 즉시 재현한다. 그 위에 OCP/ceil/UOS/E4M3/UE5M3 × B × Gaussian·Laplace·t3의 MSE, gain, 최대 원소 편향, 클리핑 비율, cos(e_scale,e_grid)를 계산한다.
2. Llama-3.1-8B·Qwen3-8B의 RHT 후 블록에서 같은 표를 만든다. 기준점은 Egiazarian Fig. 3과 Fasoli Fig. 2이며, NVIDIA 손실 비율과는 비교하지 않는다.
3. 300M MXFP8/MXFP4 사전학습을 4개 arm으로 돌린다: (a) 올림, (b) 실제 OCP, (c) OCP + 블록별 gain 보정(MSE 유지, 편향 제거), (d) UOS. 클리핑 비율을 기록한다.
실패 기준: (c)도 불안정하면 H1을 기각한다. 실 데이터의 gain 순위가 해석식과 다르면 Gaussian 틀은 무용하다.

### 평가
novelty 낮음~중하. 핵심인 닫힌 형태 기대 왜곡은 Fasoli(2601.19026)가 출판했다. Soloveychik, Chen, Li, MXAttention이 나머지를 나눠 선점했다. 남는 기여는 편향 축의 확장이다. 메인트랙 가치는 실험 3이 H1을 확인하는지에 달렸다. 확인되지 않으면 부록 A로 내리는 것이 맞다.

---

## 아이디어 4. FP4 사전학습의 LR-decay 말기 격차: 분산 바닥과 편향 바닥의 서명 분리, 그리고 말기 정밀도 전환의 조건

### 가설
저정밀 학습의 손실 격차는 두 바닥의 합이다. (A) 비편향 SR 기울기 노이즈가 만드는 초과손실 ≈ (η/4)tr(C_q)는 η에 비례하며, decay와 함께 사라진다. (B) forward RTN 가중치 오차가 만드는 ½E[bᵀHb] ≈ (Δ²/24)tr(H)는 고정된 H에서 η와 무관하다. 이 구조는 장난감 모델에서 이미 증명되었다(아래). 주장은 이것이 FP4 LLM의 decay 말기 격차를 정량 설명한다는 것이다. decay 중 tr(H)가 오르므로(측정된 사실) (B)도 커진다. (B)는 다시 둘로 나뉜다. (B0)는 고정 w에서 forward만 BF16으로 바꾸면 즉시 사라지는 평가 편향, (B1)은 η>0 학습으로만 회복되는 재최적화 성분이다. NVIDIA의 10T 늦은 전환이 덜 회복된 것은 B1이 있다는 신호이다. Nemotron 3 Ultra의 발산은 바닥 문제가 아니라 불안정성 문제이다.

### 문헌상 공백
- **두 바닥 구조는 알려져 있다.**
  - ECO §3.4 식 (18): master weight가 있는 SGDM에서 lim_{η→0} E[(∇f(x̂))²] = L²σ²이다(1D 2차). master weight가 없으면 오차는 ∝1/η이다 ([2601.22101](https://arxiv.org/abs/2601.22101) [V]). 이를 손실로 환산하고 σ²=Δ²/12를 넣으면 (Δ²/24)tr(H)이다(이 보고서의 환산).
  - Ichikawa et al.: Prop. VI.1의 SGD 바닥은 η에 비례하고, Thm VI.3은 η와 무관한 바닥 σ_ψ²Δ²p(1−p)을 준다 ([2510.10693](https://arxiv.org/abs/2510.10693) [V]).
  - Chmiel et al.: §4.1은 ½η²σ_q²tr(H) 항과 √3 SNR 전환 기준을 준다. App. B.2는 편향 rounding의 η-무관 바닥 μ_ε²/(2λ)을 준다. §5 QAF(backward만 BF16, LR rewarm)는 7B/1T 격차를 완전히 닫았다 ([FP4 All the Way 2505.19115, NeurIPS 2025](https://arxiv.org/abs/2505.19115) [V]).
  - 인접 결과: SWALP noise ball ([1904.11943, ICML 2019](https://arxiv.org/abs/1904.11943) [V]), Quartet의 forward=eff_N / backward=eff_D ([2505.14669](https://arxiv.org/abs/2505.14669) [S]).
- **NVIDIA NVFP4 12B/10T** ([2509.25149](https://arxiv.org/abs/2509.25149) [V]).
  - FP8 대비 상대 검증 손실 오차가 안정 구간 1% 미만에서 decay 구간 1.5% 남짓으로 벌어진다(Fig. 2).
  - 정밀도 전환은 BF16만 실험했고, MXFP8은 언급만 했다.
  - App. D: 8.2T에서 forward만 전환하면 1.5%→0.5%로 줄고, backward만 전환하면 개선이 없다(Chmiel과 충돌). 10T 전환은 "presumably" LR이 낮아서 덜 회복된다.
  - decay 기작(편향 대 분산, tr(H))을 분리한 ablation은 없다.
- **같은 LR 지점에서 분리된 적 없는 병목 주장들.**
  - Quartet II: backward 원소별 SR 분산(SR 23.5 vs RTN 9.0 ×1e-3)을 FP8 micro-scale SR(MS-EDEN 9.4)로 줄인다 ([2601.22813, ICML 2026](https://arxiv.org/abs/2601.22813) [V]).
  - Dong et al.: hot channel. GLA-1.3B, cosine, 상관 증거이다 ([2602.02047](https://arxiv.org/abs/2602.02047) [V]).
  - TetraJet-v2: bin 경계 진동 ([2510.27527, ICML 2026](https://arxiv.org/abs/2510.27527) [V]).
  - UFP4: shrinkage bias. bin 조건부 Eq. 2, FP32 스케일 ([2606.20381](https://arxiv.org/abs/2606.20381) [V]).
  - Full-Stack FP4: optimizer와 attention까지 범위를 넓혔고, 격차 감소는 projection LoRA-SVD에서 나온다(1.40%→0.61%) ([2607.04422](https://arxiv.org/abs/2607.04422) [V]).
- **Nemotron 3 Ultra** ([2606.15007](https://arxiv.org/abs/2606.15007) [V]).
  - 전 텐서 BF16 분기가 손실을 "substantially recovers"했다.
  - 격차는 첫 5B 토큰 0.27/0.28/0.25%에서 74B 후 0.33/0.34/0.03%이다.
  - 발산은 BF16 분기에서도 재현되었다. NVFP4 논문과 모순이 아니며, 즉시 회복분과 재최적화분을 가를 데이터이다.
- **Catalan-Tatjer, Ajroldi, Geiping** ([2510.06213, ICLR 2026](https://arxiv.org/abs/2510.06213) [V]).
  - 최대 32B(OLMo2) / 15T(Apertus-8B) 궤적에서 "once learning rates decay, validation loss and quantization error diverge"(WSD).
  - LR·스케줄·weight decay·LAWA 개입을 수행했다.
  - 160M에서 decay 중 모든 고유값과 trace가 상승한다(Fig. 9). 큰 peak LR은 tr(H)와 PTQ 오차를 함께 낮춘다(Fig. 26).
  - PTQ만 다루고 정량 적합은 없다.
- **반론과 교란:** shadow weight 대 endpoint ([2609.21834](https://arxiv.org/abs/2609.21834) [S]), FP32 수렴 시점에 시작되는 INT4 PTQ 붕괴 ([2604.15167](https://arxiv.org/abs/2604.15167) [S]), BF16 master weight의 update swamping ([2010.06192](https://arxiv.org/abs/2010.06192) [S]).
- **EoS.** GD에서 λ_max ≈ 2/η([Cohen et al. 2103.00065, ICLR 2021](https://arxiv.org/abs/2103.00065) [V]), Adam에서 λ₁(P⁻¹H) ≈ 38/η([2207.14484](https://arxiv.org/abs/2207.14484) [V]). 둘 다 top 고유값만 제약하므로 tr(H) 상승을 설명하지 못한다. 동기로만 쓴다.

### 수학
2차 근사 L(w) ≈ L* + ½(w−w*)ᵀH(w−w*)를 쓴다. SGD(학습률 η, 노이즈 공분산 C + C_q)의 정지분포에서 E[L] − L* ≈ (η/4)tr(C + C_q)이다. forward 양자화 w_q = w + b(w)에서 셀 내 오프셋이 균일하면 E[b²] = Δ²/12이고, 초과손실은 ≈ (Δ²/24)tr(H)이다. (Ichikawa p(1−p)의 셀 내 평균; 위치와 그리드가 균일할 때만). E2M1에서는 b̄ ≠ 0이어서(UFP4 Eq. 2, 아이디어 3) 1차 항 ∇Lᵀb̄가 더해진다. absmax 스케일의 Δ_g는 η와 λ에 따라 변하므로(Catalan-Tatjer Sec. 6.1) 측정된 E[Δ_g²]를 쓴다.

예측 1: 격차(η) = A·η + B0 + B1, B0 = (1/24)Σ_g E[Δ_g²]tr(H_g(η)) + 1차 편향항.
예측 2(조건부): forward 전환은 B0를 즉시 없앤다. B1은 남은 ∫η dt에 비례해서만 회복된다. backward 전환이 무효인 것은 backward 양자화기가 비편향일 때뿐이고, 편향이 있으면 μ_ε²/(2λ) 바닥이 생긴다. 이것이 NVIDIA와 Chmiel의 충돌에 대한 설명 후보이다.
예측 3: B는 측정된 E[Δ_block²]에 비례한다. 블록 16→8이 Δ²를 1/4로 만들지는 않는다(극값 통계). 대조군은 E1M2(b̄≈0)이다.
예측 4: 최종 η가 다른 두 WSD 런의 η-절편이 B이다. 수렴 상태를 통제해야 한다.
예측 5: (Δ'²/24)tr(H_final) ≪ (η_final/4)tr(C)이면 MXFP8로 충분하다. 전환 시점은 남은 ∫η dt로 정한다. 이 기준을 √3 SNR 기준과 비교한다.

가중치 진동(TetraJet)은 b(w)의 동적 표현이다. bin 경계 근처 master weight가 매 스텝 오차 부호를 바꿔 η로 줄지 않는 Δ/2 노이즈를 낸다. 즉 (B)의 분산 성분이다.

알려진 것: 2차·선형 모델에서 두 바닥의 분리(ECO 식 18, Ichikawa Thm VI.3, Chmiel App. B.2). 따라서 이론 몫은 새 정리가 아니라 블록 스케일 FP4, tr(H(η_t)), B0/B1로 확장한 진단 모델이다. 추측: Adam·transformer에서 근사의 정확도, FP4 런의 tr(H), 1차 항 크기, B1의 ∫η 비례.

### 판정 실험(수십 GPU·일)
설정: 125M–1B Llama형, WSD, NVFP4(또는 시뮬레이션), FP32 master weight.

decay 구간 여러 지점에서 다음을 측정한다.
- (i) BF16 대비 격차.
- (ii) FP4 런의 Hutchinson tr(H_g), λ_max, E[Δ_g²].
- (iii) 영-스텝 forward 전환으로 B0를, k 스텝 BF16 학습으로 B1을 잰다.
- (iv) 같은 LR 지점 ablation: forward만 BF16, backward만 BF16(SR 대 RTN), 블록 8, E1M2, SR forward.
- (v) 최종 η 두 값.
- (vi) decay 대신 LAWA.

분석: 격차 ≈ aη + c·ΣE[Δ_g²]tr(H_g,t)로 회귀해 적합도를 보고, c가 1/24에 가까운지 본다.

실패 기준: B0 예측이 영-스텝 측정과 2배 넘게 어긋나거나, backward 편향 유무가 backward 전환 효과를 예측하지 못하면 모델을 기각한다.

### 평가
novelty는 중이다. 2-바닥 이론은 ECO·Ichikawa·Chmiel이, decay 중 tr(H) 상승은 Catalan-Tatjer가 이미 보였다. 신규 정리는 없다.

남는 신규성은 다음과 같다.
- FP4 사전학습 *중* 격차의 aη + cΔ²tr(H) 적합.
- matched ablation과 FP4 런 위의 tr(H) 측정.
- B0/B1 분해.
- NVIDIA–Chmiel 충돌을 backward 편향 조건으로 해소하는 전환 기준. 기존 기준은 NVFP4 App. D의 경험칙뿐이다 [S].

메인 트랙 가능성은 중이다(진단 실험 논문). 리스크: ≤1B 실험은 10T 현상의 대리로 약하다. c가 1/24에서 크게 벗어나거나 B1이 지배적이면 정성 진단만 남는다.

---

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

## 아이디어 6. 양자화 롤아웃 RL의 토큰 단위 불일치 지도: 엔트로피·KL·advantage 결합 측정과 엔트로피 인지 보정의 양자화 벤치마크

### 가설
롤아웃 정책 μ(FP8/INT8/NVFP4)와 학습 정책 π의 토큰 KL은 로짓 섭동 δ에 대해 ≈ ½δᵀJδ(J = diag(p) − ppᵀ)이므로 같은 δ라도 평평한 위치에서 크다. (i) 양자화 롤아웃 RL 학습 중 KL(μ‖π) 질량은 엔트로피 상위 위치에 몰리고, 엔트로피는 위치별 불일치 D̄_t의 사전 예측자다. (ii) KL 질량 상위 토큰은 고엔트로피 fork, 깨진 응답의 토큰, 저엔트로피 희귀 표본으로 나뉘며, 그 비율이 포맷과 학습 단계에 따라 바뀌어 엔트로피 게이트의 성패를 정한다. (iii) 양자화에서는 δ ∝ ‖z‖(로짓 규모)이므로 bf16 엔진 불일치에서 본 log-odds 변위 ε_t의 확신도 불변성이 깨지고, 최적 cap은 1+λ(1−p_t)·g(‖z_t‖) 꼴이 된다. QeRL의 "노이즈=탐색"과 FlashRL/AIS의 "불일치=편향"은 같은 양이 아니다. 전자는 롤아웃과 학습기가 같은 NVFP4 가중치를 쓰는 on-policy 섭동(δ의 1차 효과, 엔트로피)이고 후자는 off-policy 불일치(2차 효과, KL(μ‖π))다. 두 효과의 토큰 위치 겹침은 측정 대상이다.

### 문헌상 공백
- 처방: FlashRL은 INT8/FP8 롤아웃을 토큰 TIS(C≈2)로 보정한다 ([Flash-RL repo](https://github.com/yaof20/Flash-RL) [V]; NeurIPS 2025 virtual 항목만 있고 arXiv 논문 없음, 본 트랙/워크숍 미확인 [S]). verl 문서는 토큰 TIS 편향 O(T²Δ_max)와 length trap(1.1^100 ≈ 13,780)을 적었다 ([verl rollout_corr_math](https://github.com/verl-project/verl/blob/main/docs/algo/rollout_corr_math.md) [V]). QeRL은 롤아웃과 학습기가 같은 NVFP4 가중치를 공유해 IS 없이 clipped ratio만 쓴다(§1, App. F Alg. 1). 엔트로피 상승은 LoRA RL에서의 관찰이며(Fig. 5, 14), rebuttal에서는 BF16 LoRA+AQN이 89.7로 NVFP4 LoRA+AQN 90.8의 이득 대부분을 재현한다(GSM8K 7B) ([QeRL 2510.11696, ICLR 2026](https://arxiv.org/abs/2510.11696) [V]). FP8-RL은 토큰 TIS/MIS C=2 ([2601.18150](https://arxiv.org/abs/2601.18150) [V], 기술 보고서), Jet-RL은 통일 FP8 흐름(on-policy, IS 없음; [2601.14243](https://arxiv.org/abs/2601.14243) [V]), QuRL은 양자화 old actor를 behaviour로 둔 decoupled PPO+TIS+적응 clip ([2602.13953, ICLR 2026](https://arxiv.org/abs/2602.13953) [V]). HiFloat4: 롤아웃만 FP4로 두고 학습을 BF16으로 올리면 최종 GSM8K 정확도가 전부 FP4보다 낮다(Qwen2.5-3B Table 1, MXFP4 50.57 대 73.31; 단일 런, TIS 5.0 적용; [2607.26515](https://arxiv.org/abs/2607.26515) [V]). AIS(FP8 롤아웃+BF16 학습기): KL(π_roll‖π_train)이 "grows monotonically during training as the policy sharpens"(Sec. 3, 정리 없는 관찰; 곡선은 LLaDA-8B GSM8K Fig. 4뿐, ESS 0.95 초과→0.70 미만). α는 배치 스칼라이고 엔트로피는 쓰지도 측정하지도 않는다 ([AIS 2605.13907](https://arxiv.org/abs/2605.13907) [V]).
- 선점: ENTR은 App. B Eq.(19) E_µ[δ²] = Var_µ[∆] + KL², Prop. 2 KL ≈ ½E[δ²], Prop. 1 엔트로피 envelope E_µ[δ²] ≤ R(e^R−1)(1−µ_max) ≤ R(e^R−1)H_t, 마스크 δ²/(H+ε) ≤ τ(Eq. 23–24)를 냈다. 양자화는 다루지 않는다 ([ENTR 2607.22186](https://arxiv.org/abs/2607.22186) [V]). DVP Prop. 4.2 |Δ_a| ≤ 2ε_max(1−p_a) ([2512.23087](https://arxiv.org/abs/2512.23087) [V])와 CIS Lemma 3.1 k_t − 1 = (1−p_t)(e^{ε_t} − 1)도 같은 (1−p) 구조다. CIS의 cap min(k_t, 1+λ(1−p_t))(Eq. 4)은 확신 토큰에서 보정을 끄는 연속 게이트로 계단형 엔트로피 게이트를 선점한다. bf16 엔진 불일치만 실험했고 ε_t는 확신도와 거의 무관했다(Spearman −0.028) ([CIS 2609.32444](https://arxiv.org/abs/2609.32444) [V]). MXFP4 분해는 App. D.4 Eq.(27)에서 η ~ N(0, σ_η²I)로부터 T_eff ≈ √(1+2σ_η²/Var(Δℓ)) > 1, H[π_Q] > H[π]를 온도 근사로 유도했다 ([2605.20402](https://arxiv.org/abs/2605.20402) [V]). TRM은 Mixed 상한 O(T)(Thm 3.4, "tighter when divergence is sparse")와 위치별 D̄_t를 유지하는 Adaptive 상한(Thm 3.6)을 증명했다. 마스킹은 max_t KL 위반 시퀀스 전체를 버리는 시퀀스 단위이고(§4.2), Remark 4.1은 토큰 단위 마스킹이 단조 개선 보장을 회복하지 못한다고 논증한다 ([TRM 2512.23075, ICML 2026](https://arxiv.org/abs/2512.23075) [V]). 엔트로피로 묶은 IS 비율은 ESPO가 제안했다 ([2512.00499](https://arxiv.org/abs/2512.00499) [S]). 통합 서사는 AIS §1의 "double-edged sword"(초기 탐색, 후기 불안정)가 선점했고, AIS는 이를 가설로 남겼다(§4.2, App. F).
- 반대 증거: full-pipeline FP8에서 엔트로피 급등을 이끄는 것은 엔트로피 > 1.0인 깨진 응답이다(양의 advantage 6.64%, lower-bound over-clip 토큰의 약 90%). FP8 롤아웃+BF16 학습은 TIS로 안정적이었다 ([2609.22870](https://arxiv.org/abs/2609.22870) [V]). TRIAGE(양쪽 NVFP4 W4A4)에서 해로운 불일치는 방향성 영역(A<0, δ<0)에 있고, 상위 5% 세그먼트에 amplifying-tail 토큰의 약 1/4이 몰린다 ([TRIAGE 2610.07043](https://arxiv.org/abs/2610.07043) [V]). Score Centering은 원인을 분산이 아닌 drift로 보고, 양자화 sampler에서 SC 단독이 IS 계열과 같거나 낫다고 한다 ([2609.20807](https://arxiv.org/abs/2609.20807) [V]). 저엔트로피 희귀 표본은 E[δ²|q] ∝ (1−q)/q로 비율 분산을 지배한다(ENTR Eq. 7).
- 남은 공백: 엔트로피·KL·보상을 한 런에서 함께 그린 연구는 있다(FP8-RL Sec. 2.4.3 Fig. 11, TRIAGE). 없는 것은 (a) 양자화 롤아웃 RL 학습 중 토큰 단위 KL–엔트로피–advantage 부호의 결합 분포, (b) CIS·ENTR·TRIAGE·Score Centering을 FP8/INT8/NVFP4 롤아웃에서 비교한 벤치마크다. 추론에서는 위치별 KL(BF16‖AWQ-3bit)과 BF16 엔트로피의 Spearman ρ = 0.92가 보고됐다(Fig. 4c, 한 모델·한 설정의 순위 상관; [Lotfi 외 2606.00206](https://arxiv.org/abs/2606.00206) [V]). 상위 20% 엔트로피 토큰만 학습한 DAPO는 Qwen3-32B AIME'24 55.83→63.54, 8B는 동률이었고 KL은 측정하지 않았다 ([Beyond the 80/20 Rule 2506.01939, NeurIPS 2025](https://arxiv.org/abs/2506.01939) [V]).

### 수학
(1) 토큰 KL(따름정리). KL(π_t‖μ_t) ≈ ½δ_tᵀJ_tδ_t. 등방·i.i.d. δ(분산 σ²)에서 E[KL_t] ≈ (σ²/2)tr(J_t) = (σ²/2)(1 − ‖p_t‖₂²), 1 − ‖p‖₂² = 1 − e^{−H₂(p)}. Fisher 2차 근사의 한 줄 계산이고, ENTR Eq.(19)·Prop. 2에 E[Var_µ∆] = σ²(1−‖µ‖²)를 넣은 특수 경우다. 새로운 것은 collision entropy 형태뿐이다. 등방 가정이 양자화에서 깨지는지(δ ∝ ‖z‖, outlier 채널)는 검증 대상이다.

(2) PG 편향. g = E_π[∇log π·A], ĝ = E_μ[∇log π·A]. |g − ĝ| ≤ Σ_t 2·E[TV(π_t, μ_t)]·sup|∇log π·A|, Pinsker로 TV_t ≤ √(KL_t/2) ∝ σ√(1 − ‖p_t‖²)/2. 이는 TRM Thm 3.4/3.6의 특수 경우다. τ 이상 엔트로피 토큰에만 IS를 쓰면 편향은 저엔트로피 토큰의 잔여 TV 합과 절단 편향으로 분해되고, 시퀀스 곱은 C^k로 묶인다. 저엔트로피 토큰의 보정을 끄면 편향이 주는 것이 아니라, 그 토큰의 TV가 작다는 가정 아래서만 잔여가 작다. 이 아이디어의 몫은 "엔트로피가 D̄_t의 사전 예측자"라는 경험적 연결이다. 기대 KL 논증은 저엔트로피 희귀 표본의 비율 꼬리를 다루지 못한다.

(3) 날카로워짐(이 아이디어의 예측). δ ∝ ‖z‖이면 RL이 로짓을 키울 때 δ²는 커지고 (1 − ‖p‖²)는 줄어 KL 궤적이 비단조일 수 있다. AIS는 단조 증가만 관찰했으므로 반대 증거가 있다. 꺾이는 시점을 ‖z‖·‖p‖² 궤적으로 정량 예측해야 반증 가능하다.

증명 가능한 것: (1)의 2차 식, (2)의 편향 분해. 추측: δ의 등방·토큰 독립성, k/T(80/20 논문의 20%는 정의상 80분위), 엔트로피 게이트가 CIS 연속 cap보다 나은지.

### 판정 실험(1–2 노드·수일)
Qwen2.5-Math-1.5B/7B 또는 Qwen3-1.7B/4B, GRPO/DAPO, vLLM FP8/INT8/NVFP4 롤아웃. 암: (a1) QeRL식 정렬(양쪽 같은 저비트 가중치), (a2) 저비트 롤아웃+BF16 학습기 무보정, (b) 토큰 TIS C=2, (c) 엔트로피 게이트 TIS(H_t > τ), (c′) CIS cap, (c″) ENTR 마스크, (e) TRIAGE 방향 게이트, (f) Score Centering, (d) BF16 롤아웃. 모든 런에서 토큰 KL(μ‖π), 엔트로피(Shannon과 1−‖p‖²), advantage 부호, 보상, AIME24/MATH-500을 기록한다. 진단: KL 질량 상위 토큰을 fork/깨진 응답/저엔트로피 희귀 표본으로 3분류하고, ε_t의 확신도 불변성(ρ ≈ −0.03)이 양자화에서 유지되는지(가설 iii) 본다. 예측: KL 질량의 80% 이상이 상위 20% 엔트로피 토큰에 있고, (c)는 (b)와 같은 정확도에 더 낮은 분산을 보이되 (c′)와의 차이는 작으며, (a2)의 초기 보상 가속은 후기 손실로 바뀐다. 3 seed 이상.

### 평가
novelty 낮음. 이론 (1)은 따름정리이고 ENTR·DVP·CIS가 같은 (1−p) 메커니즘을 증명했다. 엔트로피 게이트는 CIS cap의 계단 근사, 편향 상한은 TRM의 특수 경우다. 통합 서사는 AIS가 선점했고, QeRL은 on-policy라 FlashRL과 같은 양이 아니다. 남는 기여는 측정과 벤치마크다. 토큰 KL–엔트로피–advantage 결합 분포, KL 질량 3분류, 기존 보정법(CIS, ENTR, TRIAGE, Score Centering, IcePop, AIS)의 FP8/INT8/NVFP4 비교가 그것이다. 가설 (iii)이 확인되면 양자화 특이적 cap이라는 방법 기여가 생긴다. main-track: 측정만으로는 낮음(분석·워크숍 수준), (iii) 확인 시 중간. 부록 A 경계에 있다. 2026년 양자화 롤아웃 RL 논문만 10편이 넘는다(위 인용 외 [QaRL 2604.07853](https://arxiv.org/abs/2604.07853), [QUADS 2607.15810](https://arxiv.org/abs/2607.15810), [ACRL 2607.24062](https://arxiv.org/abs/2607.24062), [TRACE 2610.07767](https://arxiv.org/abs/2610.07767) [S]). 리스크: 고엔트로피가 붕괴 증상일 수 있어(2609.22870) 엔트로피 게이트가 깨진 응답을 더 크게 보정할 수 있고, RL 분산이 크다.

---

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

## 아이디어 8. 추론 모델 PTQ 캘리브레이션의 도메인 효과 분해: 엔트로피 프로필 대 Hessian 부분공간 정렬

### 가설
generic에서 자가생성 CoT로 캘리브레이션을 바꿀 때 GPTQ가 얻는 큰 이득에는 두 후보 원인이 있다. (i) 활성 Hessian 부분공간 정렬과 (ii) 토큰 엔트로피 프로필이다. 등방 로짓 섭동에서 토큰 end KL은 (σ²/2)(1 − ‖p_t‖²)이다. (ii)가 맞다면 forward 한 번으로 얻는 w_t = 1 − ‖p_t‖₂²를 Hessian에 넣은 H_w = Σ_t w_t x_tx_tᵀ만으로 generic 데이터에서도 이득의 상당 부분을 재현하고 CoT 길이 증가를 줄일 수 있어야 한다. 이 가중은 GuidedQuant 토큰 가중을 backward 없이, 채널 등방으로 근사한 것이다. 초안의 전제 "C4는 저엔트로피, 추론 트레이스는 고엔트로피"는 실측과 반대일 수 있어 0단계에서 먼저 잰다.

### 문헌상 공백
- **도메인 효과**
  - GPTQ에서만 나타난다. R1-Distill-Qwen-1.5B를 WikiText2 대신 NuminaMath로 캘리브레이션하면 평균 +9.8%다 ([2504.04823, COLM 2025](https://arxiv.org/abs/2504.04823) [V]).
  - Qwen3-4B-Thinking INT3 GPTQ g128을 자가생성 OpenR1-Math로 캘리브레이션하면 정확도 28%→65%, CoT 30k→12k다(Sec. 7.2, Fig. 5). 다만 저자들은 "캘리브레이션 최적화는 두 축 모두 QAT보다 엄격히 아래"라고 결론짓는다 ([2606.25519](https://arxiv.org/abs/2606.25519) [V]).
- **경쟁 가설과 반대 증거**
  - LAQuant는 도메인 효과를 부분공간 정렬로 설명한다(Sec. 4.1 Eq. (2), principal-angle cosine c̄_k).
  - 같은 논문의 측정이 이 아이디어에 불리하다. E2E 메트릭은 30배 이상 비등방이고 "no scalar reweighting can repair it"이다(App. C Table 6). softmax-Fisher 감독은 top-1 일치율을 오히려 떨어뜨린다(88.44% vs 93.95%). AIME25 트레이스의 약 56%는 H < 0.1이다 ([LAQuant 2605.08755](https://arxiv.org/abs/2605.08755) [V]).
  - 다른 측정에서도 Qwen3 CoT의 50.64%가 H < 10⁻²였다 ([2506.01939, NeurIPS 2025](https://arxiv.org/abs/2506.01939) [V]).
- **토큰 가중 Hessian은 선점됐다**
  - 초안은 GuidedQuant가 "토큰 믹스를 바꾸지 않는다"고 했으나 틀렸다. GuidedQuant의 H_k = Xᵀ Diag(mean_{j∈J_k}(∂ℓ/∂z_j)²) X(Eq. (4), Sec. 3.2)는 토큰 × 출력 채널 그룹을 가중하는 block-diagonal 근사다. 비용은 gradient caching 7B 0.3h(1×A100), 70B 2.7h(6×A100, Table 9)이고, 여기에 Hessian caching 7B 0.3h가 더해진다 ([GuidedQuant 2505.07004, ICML 2025](https://arxiv.org/abs/2505.07004) [V]).
  - RSQ는 H = 2XR²Xᵀ를 휴리스틱 가중으로 쓰고 r_min = 0.01 floor를 둔다 ([RSQ 2503.01820, TMLR 2026](https://arxiv.org/abs/2503.01820) [V]).
  - OAC Eq. (11)의 x π(1−π) xᵀ는 1 − ‖p‖²의 이진형이다 ([2405.15025](https://arxiv.org/abs/2405.15025) [V]). BRECQ Sec. 3.3은 대각 FIM 가중을 쓴다 ([2102.05426, ICLR 2021](https://arxiv.org/abs/2102.05426) [V]).
  - DPQ의 entropy-only 표본 선택은 arm (e)의 표본 수준판이다 ([DPQ 2608.21019, EMNLP Findings 2026](https://arxiv.org/abs/2608.21019) [V]).
  - LRM 프루닝에도 토큰 가중 공분산이 있다 ([OBC-Prune 2609.17890](https://arxiv.org/abs/2609.17890) [S]).
- **2606.00206의 범위**
  - KL(FP‖Q)과 BF16 Shannon 엔트로피의 Spearman ρ = 0.92(Fig. 4c), 그리고 "실패의 최대 52%"(139/265, GPT-5 judge)는 모두 한 설정의 결과다. 그 설정은 R1-Distill-Qwen-1.5B, AWQ 3-bit, Pile 캘리브레이션, MATH-500이다.
  - 같은 논문의 GPTQ-3bit는 이미 Numina 자가생성 트레이스로 캘리브레이션돼 있다(71.6%/6.8K 토큰 vs AWQ 47.0%/23.4K). 이 비교는 알고리즘 차이와 교란된다.
  - ρ는 보조정리 자체가 예측하는 값이라 캘리브레이션 결함의 증거가 아니다. 이 논문이 낸 처방은 디코딩 측 페널티뿐이다 ([Lotfi et al. 2606.00206](https://arxiv.org/abs/2606.00206) [V]).
- **저엔트로피 쪽 실패**: NVFP4는 저엔트로피 기호 토큰에서 오표집을 늘린다 ([ReSET 2606.13233](https://arxiv.org/abs/2606.13233) [V]). w_t → 0인 곳이 보호되지 않는다는 뜻이다.
- **남는 공백**: 정확히 1 − ‖p‖²를 GPTQ 토큰 가중으로 쓴 논문은 없다. 도메인, 엔트로피, 부분공간 정렬을 인과적으로 분리한 실험도 없다.

### 수학
**보조정리 8.1 (표준 결과의 trace 형태).** J_t = diag(p_t) − p_tp_tᵀ이고, δ_t ~ (0, σ²I)이면 E[KL] ≈ (σ²/2)tr(J_t) = (σ²/2)(1 − ‖p_t‖₂²)이다. 1 − ‖p‖²는 Gini/Tsallis-2이며 1 − e^{−H₂}로 collision entropy의 단조함수다. 2차 전개 자체는 이미 알려져 있다(BRECQ, OAC Eq. (11), LAQuant Prop. 1 M_E2E = JᵀH_KL J, Lemma 1 σ_max(H_KL) ≤ 2 max p_i(1−p_i) ≤ 1/2). 새로운 것은 이 trace를 토큰 가중으로 쓰는 점뿐이다.

**층 전달.** 층 출력 섭동이 Jacobian G_t로 로짓에 전달되면 KL_t ≈ ½ε_tᵀG_tᵀJ_tG_tε_t이다. on-policy 트레이스에서는 GuidedQuant의 토큰별 Σ_j g_tj² 기댓값이 tr(G_tᵀJ_tG_t)와 같다. 따라서 GuidedQuant-on-traces가 이상형이고, w_t는 G_t를 버린 근사다. 이 근사가 기대는 G_t 동질성 가정은 LAQuant Table 6과 충돌한다.

**따름정리 8.2.** 최종 은닉에서 tr(G_tᵀJ_tG_t) = Σ_i p_i‖u_i‖² − ‖Σ_i p_i u_i‖²이다(u_i는 언임베딩 행, 최종 norm 제외). 이 값도 forward만으로 정확히 계산된다.

**IS 해석.** 캘리브레이션과 배포에서 w_t 분포가 다르면 H_w는 토큰 수준 공변량 이동(아이디어 5의 (4))이 된다. 이동의 방향은 아직 측정되지 않았다.

증명 가능한 것은 8.1과 8.2다. G_t 동질성과 가중만으로 도메인 효과를 재현하는지는 추측이다.

### 판정 실험(약 3–4 GPU·일 [S])
- **0단계**: C4, WikiText, 트레이스에서 w_t 분포와 c̄_k를 재고, KL을 1−‖p‖²와 H에 대해 함수형으로 비교한다.
- **모델**: R1-Distill-Qwen-7B, Qwen3-8B(thinking). GPTQ/GPTAQ W4/W3 g128로 알고리즘을 고정한다.
- **기본 arm**: (a) C4, (b) 자가생성 CoT, (c) CoT + w_t, (d) C4 + w_t, (e) C4 w_t 상위 분위.
- **추가 arm**: 같은 트레이스 위 (f) GuidedQuant g=1·4, (g) RSQ-AttnCon, (h) DPQ entropy-only, (i) CoT+generic 혼합, (j) 8.2 가중.
- **2×2 분리**: c̄_k를 맞춰 재표집한 C4 × 엔트로피를 평탄화한 트레이스.
- **가중 floor**: w_t ← max(ε, w_t^γ).
- **지표**: AIME24/25, MATH-500, GPQA, LiveCodeBench, CoT 길이, overthinking 비율(2606.00206의 GPT-5 judge 프로토콜, 정의 명시), 저엔트로피 기호 토큰의 꼬리 질량, K/V RMSE. 3 seed.
- **성공 기준**: (d)가 (a)–(b) 격차의 절반 이상을 회복하고 c̄_k를 통제한 뒤에도 그 회복이 남는다. (c) 또는 (j)는 (f)와 동등해야 한다.
- **기각 조건**: (d)≈(a)이거나, c̄_k를 맞춘 C4가 (b)를 재현한다.

### 평가
**novelty: 낮음–중.** 방법은 GuidedQuant, RSQ, OAC/BRECQ 계열의 forward-only, 채널 등방 특수형이다. 보조정리 8.1은 교과서 수준이다. 남는 기여는 둘이다. (i) backward 없는 closed-form 가중(8.1, 8.2), (ii) 도메인 효과의 인과 분해.

**리스크.**
- 트레이스가 뾰족하다는 실측이 이미 있다.
- LAQuant의 Fisher 희석 결과와 ReSET의 저엔트로피 실패가 가설에 불리하다.
- 2606.25519는 QAT가 캘리브레이션 최적화보다 우월하다고 결론지었다.
- GuidedQuant가 7B에서 약 0.6h라 비용 이점이 작다.
- AIME는 분산이 커서 seed가 중요하다.

**main track: 낮음–중.** 2×2 실험이 엔트로피 프로필의 독립 효과를 보이면 기작 논문이 된다. 그렇지 않으면 workshop 수준이다. 0단계에서 트레이스 w_t가 C4보다 낮고 c̄_k가 효과를 설명하면 부록 A로 내린다.

---

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

## 아이디어 10. MLA latent KV의 가중 변환 부호화: RMSNorm 제약 하 최적 변환과 K·V 결합 가중 (QuantMLA의 닫힌형 대안)

### 가설
MLA 디코드(weight absorption)에서 캐시되는 latent c_j = Γ·RMSNorm(W_DKV h_j)의 오차 e_j는 로짓에 a^{hᵀ}e_j (a^h = W_UK^{hᵀ}q^h)로, 출력에 W_UV^hΣ_jp_je_j로 들어가므로 왜곡은 M = Σ_hE[a^ha^{hᵀ}] + λΣ_hW_UV^{hᵀ}W_UV^h 가중 MSE다. 이 가중 KLT는 NOVA-KV Thm 1의 특수화이고, native MLA latent의 학습 회전·스케일은 QuantMLA가 이미 했다. 남는 가설: (i) 공유 latent의 K·V 결합 가중 닫힌형 변환 + 방향별 비트 할당(0비트 포함)이 같은 비트에서 QuantMLA(학습 변환, 균일 비트)와 동등 이상이다. (ii) RMSNorm 때문에 생산자 쪽에 무비용으로 접히는 변환은 직교×대각 족뿐이고, 이 족과 비직교 최적해의 격차는 Σ_c와 M의 비가환성이 정한다 — 격차가 큰 층에서만 쓰기 경로 d_c×d_c matvec가 정당화된다. (iii) RoPE 가지의 정밀도 요구는 granularity만이 아니라 "RoPE가 정규화 아래 있는가"와 QAT 여부가 정한다.

### 문헌상 공백
- **직접 선행**: [QuantMLA 2609.36760](https://arxiv.org/abs/2609.36760) [V] (2026-09-29, under review) — 흡수 형태 오차 경로(eq.1–3), content latent에 학습 회전 R_C + 대각 S_C 오프라인 fold(eq.6, App. C.1, "weightless RMSNorm is rotation-equivariant"), RoPE 가지 쌍 회전 + 쌍 스케일(Prop. 1), INT4/INT2 FlashMLA 기반 커널, 6개 MLA 모델. DS-V2-Lite GSM8K BF16 36.92 / RTN C4R4 16.98 / QuantMLA C4R4 35.25, C2R4 37.91. 없는 것: 닫힌형 M, K+V 결합 가중, 가변 비트(균일 per-token g64 affine), 비직교 변환.
- **닫힌형 이론**: [Spend Bits Where Queries Look (NOVA-KV) 2608.04074](https://arxiv.org/abs/2608.04074) [V] — GQA 모델만(Llama-3.1-8B, Qwen3-8B/4B, GPT-OSS-20B). Thm 1: R⋆_{K,p} = M_q^{1/2}E_{1:p}, E = eig(M_q^{1/2}S̃_kM_q^{1/2}), M_q = QᵀQ, RRᵀ = M_q (Linder–Zamir–Zeger 1999); App. F.5 eq.(38)에서 GQA M_q는 이미 헤드 합. 2-bit RULER에서 QuaRot/OSCAR를 이긴다(Table 2, Qwen3-8B 128K: 75.4 vs 25.3 vs 0.0, BF16 83.4) — VQ일 때만이고 같은 변환 + 스칼라 양자화는 0.0으로 붕괴(Table 4). Thm 2(등부피 그룹화 + 고정율 VQ = 가변율 최적)가 할당의 선행. "Joint quantization of keys and values is left for future work"(Sec. 6).
- **회전·가중·fold**: [OSCAR 2605.17757](https://arxiv.org/abs/2605.17757) [V] (GQA, preprint) — query 2차 모멘트 고유기저를 Hadamard와 합성(R_K = U_Q H_Had P_br)하면 Hadamard 단독을 크게 이긴다(Table 2, Qwen3-8B: BF16 70.84 / QuaRot-INT2 10.14 / OSCAR 69.416); C_Q는 GQA 그룹 헤드 평균(App. A.4), R_V는 가중치에 흡수·R_K만 online, 비트 할당 없음(App. D). [AATC 2608.14191](https://arxiv.org/abs/2608.14191) [S]: Thm 1 D = D_K + D_V + R(q_c²·W_O 열 노름 대각 가중) + reverse water-filling. [SPECTRA 2608.07915](https://arxiv.org/abs/2608.07915) [V]: 입력측 G = E[hᵀh] 가중 latent, water-filling·0비트 채널 drop, W_O fold; MLA는 언급만. [WUSH-KV 2609.38121](https://arxiv.org/abs/2609.38121) [V] App. A eq.(10)–(11): RMSNorm을 통과하는 fold ⇒ TᵀT = I.
- **RoPE 가지(3자 사실 패턴)**: [SnapMLA 2602.10718](https://arxiv.org/abs/2602.10718) [V] (NeurIPS 2026 포스터 목록 [S]) — latent FP8 E4M3 per-token, RoPE BF16. 근거는 LongCat-Flash-Thinking의 분포(RoPE ±10³ vs content ±10¹, Fig. 3a)와 32K 층별 오차 "error explosion"(App. G Config A)뿐, 정리·RoPE 양자화 벤치마크 ablation·채널 그룹 스케일 시험 없음(per-token absmax와 교락). FP8 latent만으로도 DS-V3.1 AIME-25 87.92→85.42(Table 1, CI 없음). [DeepSeek-V4.1-Flash 2609.19969](https://arxiv.org/abs/2609.19969) [V] §2.4.4: E2M1 + E4M3/16 QAT, "The non-RoPE and RoPE components use the same quantization format", RoPE 64차원이 RMS 정규화된 512 latent 안(L2 ≤ √512 ≈ 22.6). 이 포맷은 [FlashMLA repo](https://github.com/deepseek-ai/FlashMLA) [V]의 FP8 528 B/FP4 288 B per token, 정확도 표 없음; V3.2는 RoPE bf16 [S]. per-channel 쪽: [RoPE-Aware Bit Allocation 2606.24033](https://arxiv.org/abs/2606.24033) [V] DS-V2-Lite 분리 RoPE key에서 KIVI가 모든 예산에서 우세(2 b/dim top-10 68.0% vs 49.9%, 진단 지표만).
- **민감도 데이터**: [Agent Memory Below the Prompt 2603.04428](https://arxiv.org/abs/2603.04428) [V] Table 9 — group-64 균일 4-bit(흡수 전 per-head K/V, latent 아님) PPL Gemma 3 12B −0.7%(노이즈), Llama 3.1 8B +2.8%, DeepSeek-Coder-V2-Lite +3.0%; CI 없음·WikiText-2 7,935 토큰이라 GQA↔MLA 차이를 뒷받침하지 못한다. KVarN GLM-4.7-Flash k4v2 AIME25 동률(0.94× throughput)은 논문 본문이 아닌 [KVarN repo](https://github.com/huawei-csl/KVarN) README [V].
- 철회: "latent 변환 문헌 없음", "헤드 합 M이 MLA에서 구조적으로 다름"(차이는 전 헤드 공유 + W_UK 흡수 경로라는 정도 차), "MHA에서는"(선행 평가는 GQA).

### 수학
DeepSeek-V2/V3 MLA: c_j = Γ·RMSNorm(W_DKV h_j) (d_c = 512), k_j^C = W_UK c_j, v_j^C = W_UV c_j, 분리 k_j^R = RoPE(W_KR h_j) (64차원, 헤드 공유, 정규화 없음). score^h_{tj} = a^{hᵀ}c_j + q_t^{Rᵀ}k_j^R. 출력 o^h = W_UV^hΣ_jp_jc_j → W_O에 흡수. V4.x는 RoPE가 정규화된 latent 안에 있어 이 형태를 그대로 적용하지 않는다.

정리 10.1 (가중 왜곡; NOVA-KV eq.(5)/(38), QuantMLA eq.(2)–(3)의 흡수 형태). 로짓 오차 에너지 E Σ_h(a^{hᵀ}e)² = tr(Σ_eM_K), M_K = Σ_hE[a^ha^{hᵀ}]; 출력 오차 에너지 ≤ tr(Σ_eM_V), M_V = Σ_hW_UV^{hᵀ}W_UV^h (볼록성으로 길이 무관). M = M_K + λM_V (λ는 softmax Jacobian 전달을 포함한 교환비). 새것은 두 항이 같은 캐시 벡터에 걸린다는 점뿐이다.

정리 10.2 (최적 변환; Linder et al. 1999 / NOVA-KV Thm 1의 MLA 특수화). Gaussian·고해상도에서 T = UᵀM^{1/2} (U: M^{1/2}Σ_cM^{1/2}의 고유벡터), b_i = max(0, ½log₂(λ_i/θ)). 소비자 쪽 a^h ← T^{−ᵀ}a^h, W_UV ← W_UVT^{−1}는 무비용. 생산자 쪽은 RMSNorm과 가환해야 W_DKV에 접히므로 직교(×Γ 대체 대각)만 무비용이고(WUSH-KV eq.(11)), 일반 T는 토큰·층당 d_c² = 512² MAC가 쓰기 경로에 추가된다. 2^{−2b} 법칙은 근사다: NOVA-KV App. H.3 적합 α ≈ 1.82–1.88, [RateQuant 2605.06675](https://arxiv.org/abs/2605.06675) [S] β 3.6–5.3(순서 역전 가능). 고정폭 서빙에는 NOVA-KV Thm 2의 등부피 그룹화를 쓴다.

명제 10.3 (RMSNorm 제약 격차; 이 보고서의 유도 [S]). 좌표별 독립 스칼라 양자화·Gaussian·고해상도·최적 할당에서 T = DR (D 대각, R 직교)의 가중 왜곡 ∝ [Π_i(RΣ_cRᵀ)_ii(RMRᵀ)_ii]^{1/d_c}·2^{−2b̄}로 D와 무관하다(d_i²와 d_i^{−2}가 상쇄). 모든 가역 T의 하한은 Hadamard 부등식으로 (detΣ_c·detM)^{1/d_c}이고 정리 10.2의 T가 달성한다. 따라서 격차 G ≥ 1, 등호 ⇔ Σ_c와 M이 가환, G ≤ [Π_i(U_ΣᵀMU_Σ)_ii/detM]^{1/d_c}. 대각 D는 per-token 공유 스케일·균일 비트에서만 범위 균등화로 의미가 있다(QuantMLA S_C의 역할). NOVA-KV의 "orthogonal ... suboptimal unless the queries are isotropic"은 VQ 셀 형태를 포함한 Linder 조건이며, 스칼라 + 할당에서는 가환 조건으로 완화된다는 것이 이 명제의 주장이다.

RoPE 가지: 정규화 없는 분리 k^R(V2/V3/LongCat)에 per-token 스케일을 쓰면 RoPE 이상치가 토큰 스케일을 정하고 아이디어 1의 이분산 편향이 가장 크다. 정리 1은 per-channel(또는 QuantMLA Prop. 1의 fold 가능한 쌍 스케일 S_P)을 예측하고, V4.1처럼 정규화 아래 + QAT이면 동일 포맷으로 충분할 수 있다. 단 NOVA-KV Table 22(GPT-OSS-20B 2-bit 스칼라: 스케일 그룹 4개 4.0 BPE NIAH-8K 27.0 vs INT3 3.5 BPE 82.8)가 보이듯 이 예측은 비트폭에 조건부다.

증명 가능한 것: 10.1(선형대수), 10.2(고전 결과), 10.3(Hadamard 부등식). 추측: M^{1/2}Σ_cM^{1/2}의 유효 랭크가 낮은가, 층별 G가 쓰기 비용을 정당화할 만큼 큰가, M_K가 도메인에 걸쳐 안정적인가(아이디어 5).

### 판정 실험(3–4 GPU·일)
BF16 학습 MLA만: DeepSeek-V2-Lite, GLM-4.7-Flash, Kimi-K2 또는 LongCat-Flash-Lite (V4.1은 QAT라 제외). (a) M_K, M_V, Σ_c 추정, 유효 랭크와 층별 G 측정. (b) latent 4/3/2 비트: raw per-token, E2M1 + E4M3/16(V4.1식), QuantMLA C4R4/C2R4(DS-V2-Lite GSM8K 35.25/37.91, MMLU 57.95/57.29), {직교×대각 최적, 비직교 T} × {균일, water-filling/등부피 그룹}; 로짓 오차, GSM8K/MMLU, NIAH/RULER, AIME. (c) k^R ∈ {per-16 FP4, per-16 FP4 + S_P, per-channel, BF16}, sink 질량과 정확도. (d) 쓰기 경로 512×512 matvec 커널 지연. 성공 기준: 같은 비트에서 닫힌형 결합 가중이 QuantMLA와 동등 이상, G가 큰 층에서만 비직교 T의 유의한 추가 이득, 3-bit 할당이 4-bit 균일과 동등.

### 평가
novelty 낮음(부분 선점: QuantMLA 2609.36760 + NOVA-KV 2608.04074). 가중 KLT(NOVA-KV Thm 1), 헤드 합 M(NOVA-KV eq.(38), OSCAR), W_O 가중·water-filling·0비트(AATC, SPECTRA), RMSNorm fold 제약(WUSH-KV), native MLA 학습 변환·RoPE 쌍 스케일·INT4/INT2 커널(QuantMLA)이 모두 2026년에 나왔다. "novelty 중상", "런타임 비용 0", "확장에 몇 달"은 철회한다 — 리스크는 이미 실현됐고 [KV-COBRA 2609.24298](https://arxiv.org/abs/2609.24298) [S]도 "Natural extensions include MLA"라고 쓴다. 남는 것: K·V 결합 가중 닫힌형, 방향별 가변 비트, 명제 10.3의 격차-쓰기 비용 trade-off, learned vs closed-form 정면 비교, 유효 랭크 실측. 메인트랙은 "QuantMLA의 닫힌형·할당 대안"으로서 같은 비트에서 C2R4보다 낫거나 캘리브레이션 비용이 크게 낮을 때만 가능하고, 아니면 부록 A로 내린다. NOVA-KV Table 4(비직교 변환 + 스칼라 붕괴)가 FP4 스칼라 서빙에서의 핵심 리스크이고, λ 선택과 softmax 비선형 전달은 여전히 이론을 해친다.

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
