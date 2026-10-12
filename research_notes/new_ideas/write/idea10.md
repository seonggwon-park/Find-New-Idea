## 새 아이디어 10. 검색 헤드별 극값(EVT) 꼬리 보정으로 KV 양자화의 사용 가능 문맥 길이를 예측하고, 목표 길이에 맞춰 비트를 배분하기

### 가설

수학은 새 결과가 아니다. 아래 "수학" 절에서 보듯 2609.22101 Thm 1의 분산 치환 따름정리다. 남은 기여는 **경험 법칙 하나와 할당기 하나**다. 합성 수치 확인(아래 "수치 확인" 절)의 결과를 반영해 원래 가설을 다음과 같이 좁힌다.

- **H1 (예측).** 각 retrieval head h에 대해 4K 토큰 이하의 문맥 여러 개에서 세 가지를 측정한다.
  - needle 마진 m_h
  - 양자화된 방해 로짓의 상위 꼬리 Q̂_{h,b}(x). 프롬프트 수백 개에서 모은 경험 꼬리를 쓴다.
  - needle 자체의 양자화 잡음
  
  그다음 E[(1 − Q̂_{h,b}(m_h + σ_q z))^N] = 1/2을 풀어 실패 시작 길이 N50_h(b)를 구한다. 이 값은 well-scaled 2–4bit KV에서 needle과 RULER-NIAH의 50% 실패 길이를 8K–128K 구간에서 **2배 이내**로 예측해야 한다.
  - 원래 법칙 ln N* = m²/(2(σ_d²+σ_q²))은 꼬리가 Gaussian일 때만 성립하는 특수형이다. 그 선두 차수(leading-order) 근사는 예측식으로 쓰지 않는다.
  - Gaussian 모멘트로 꼬리를 외삽하는 방식은 기준선으로 함께 보고한다.
- **H2 (할당).** 평균 비트 예산을 고정하고, 목표 길이 N에서 Σ_h acc_h(N, b_h)를 최대화하도록 헤드별 비트를 배분한다(DP). 이 배분은 길이와 무관한 logit-MSE 할당보다 목표 길이에서의 정확도가 높아야 하고, 그 차이는 목표 길이가 길수록 커져야 한다. logit-MSE 할당은 Block-GTQ와 KVTuner 계열이다.
- **반증 기준.** 다음 중 하나라도 해당하면 법칙을 기각한다.
  - 3개 모델 계열 중 2개 이상에서 예측 N50이 2배 넘게 빗나감
  - well-scaled 2–3bit에서 정확도 하락이 8K–128K 동안 평평함. 2605.20868이 naive INT4V에서 보고한 패턴이다.
  - H2의 이득이 1pp 미만이거나 길이에 따라 커지지 않음

### 문헌상 공백과 선행 연구 위치

- [Context Poisoning as Extreme-Value Attention Interference in Long-Context Language Models 2609.22101](https://arxiv.org/abs/2609.22101)
  - Thm 1(txt L251–311): needle 1개(S⋆ ≤ γ)와 iid N(0,σ²) 방해 로짓 N개를 놓고 A_N ≤ a0+(1−a0)[Φ((γ+τlog((1−ρ)/ρ))/σ)]^N을 증명한다. 정확도를 유지하려면 γ ≥ σ√(2 log N)+O(1)이어야 한다(L306–308). [V]
  - L1296–1324: Gemma-3-4B에서 Gaussian 가정이 모든 N에서 기각되고, N=100에서는 독립성도 기각된다(Ljung–Box p<10⁻⁹⁹). 저자들은 Thm 1을 "a diagnostic baseline rather than a calibrated quantitative bound"로 규정한다. [V]
  - 이 논문에는 양자화가 없다. 하지만 σ가 모든 로짓 잡음을 흡수하므로, 본 아이디어의 수학 핵심을 그대로 포함한다.
- [KV Cache Compression Through the Lens of Transform Coding 2608.14191](https://arxiv.org/abs/2608.14191)
  - white-noise 가정 아래 attention 왜곡이 key 기여와 value 기여로 가법 분해된다는 것을 보인다.
  - Fig. 1(L1179–1215): 2bit에서 "AATC var-only degrades sharply as the context grows". [V] 관측은 있고 법칙은 없다.
- [Why Gated DeltaNet Survives 4-Bit Quantization 2609.04098](https://arxiv.org/abs/2609.04098)
  - L495, L552: weight cost는 위치에 따라 줄고, FP8-KV cost는 "small, rises with position"이다. 저자들은 이 KV cost가 scale 효과이며 §7의 calibrated scales로 제거된다고 설명한다. [V]
  - S05의 결과 (2)는 이미 관측으로 보고되어 있다.
- [WitCert 2607.28699](https://arxiv.org/abs/2607.28699)
  - raw fp8 22.8은 "six hard RULER-4096 tasks"(L1869–1873)에서 나온 수치로, 4K 문맥이다. [V]
  - 저자들은 이 캐시를 "broken fp8 KV cache"라 부르고 scaling, outlier bypass, K=int8로 복구한다.
  - 길이 법칙으로 설명할 수 있는 사례가 아니다.
- [RoPE-Aware Bit Allocation for KV-Cache Quantization (Block-GTQ) 2606.24033](https://arxiv.org/abs/2606.24033)
  - Lemma 2(L3076, L3128)는 query-key 쌍마다의 상한이다. 길이 항이 없다. [V]
- [Runtime-Certified Bounded-Error Quantized Attention 2605.20868](https://arxiv.org/abs/2605.20868)
  - TV 상한이 "independent of sequence length N"이다(L900). [V]
  - naive INT8K/INT4V에서 RULER가 8K–128K "across all context lengths"에 걸쳐 −44~−56pp 떨어진다(L1844–1850). [V]
  - 길이 구동 가설에 대한 반례 후보다.
- [Quantized Keys Steal Attention 2605.26266](https://arxiv.org/abs/2605.26266): 합(질량) 판독에서 생기는 −σ²/2 편향 보정은 이미 나와 있다. 본 아이디어는 argmax 판독만 다룬다. [S]
- [SemKV 2608.28911](https://arxiv.org/abs/2608.28911): (2.0, 2.322]bit 구간의 quality cliff를 "model-quantizer pair" 의존으로 보고한다. 길이 함수로는 보지 않는다. [V(노트)]
- [Compression for Quadratic Similarity Queries 1307.6609](https://arxiv.org/abs/1307.6609), [The Minimal Compression Rate for Similarity Identification 1312.2063](https://arxiv.org/abs/1312.2063)
  - "압축률 대 식별 가능한 DB 크기" 교환관계를 다루는 고전 정보이론 연구다. [S]
  - 스칼라 EVT 임계 ln N < SNR/2는 M-ary orthogonal signalling의 교과서 결과다.
- 비-arXiv: vLLM 블로그(2026-04-22)의 128k NIAH FP8 91→13 붕괴는 Tensor Core FP32 누산 버그였다. 저장값 잡음과 무관하다. [S]

**남은 공백.** KV 양자화에 대한 ln N*(b) 또는 b(N) 법칙을 실제 retrieval head에서 진술하거나 적합한 논문은 이번 검색(쿼리 26회, 생성 단계 N1–N5 포함 30회 이상)에서 발견되지 않았다. 따라서 열려 있는 것은 두 가지다.

- (a) 짧은 문맥에서 헤드별로 보정해 장문맥의 실패 시작 길이를 비트폭별로 예측하는 프로토콜
- (b) 목표 문맥 길이를 입력으로 받는 헤드별 비트 할당

현재 할당기들은 모두 길이 무관이다. Block-GTQ는 RoPE 대역, KVTuner, RateQuant, RDKV, AATC는 rate-distortion, SemKV는 cliff 보간에 기반한다. DuoAttention과 RazorAttention은 retrieval head를 full cache로 둘 뿐 비트 법칙이 없다. [S]

### 수학 (증명 가능한 것과 추측)

**모형.** 방해 로짓을 D_j = σ_d X_j + σ_q e_j, needle 로짓을 m + σ_q z₀로 둔다. X_j는 iid 표준화 변수, e_j와 z₀는 iid N(0,1)이다. 판독은 argmax다.

- **명제 1 (정확식, 교과서).** P_N = E_{z₀}[(1 − Q_D(m+σ_q z₀))^N]이다. N50은 P_N = 1/2의 해이고, 계산은 1차원 구적법 한 번이면 끝난다. 독립성만 가정하면 되는 항등식이다.
- **따름정리 (Gaussian, 2609.22101 Thm 1의 σ² → σ_d²+σ_q² 치환).** X가 Gaussian이면 ln N* ≈ m²/(2(σ_d²+σ_q²))이고, 따라서 ln N*_q/ln N*_FP ≈ 1/(1+r)이다(r = σ_q²/σ_d²). **새 내용은 없다.**
  - 수치 확인 결과, 선두 차수 근사는 N50을 4.9–10.7배 과소 예측한다. 원인은 log log N 크기의 유한-N 보정이다.
  - 반면 비율 법칙은 r ≤ 0.33에서 3.5% 이내로 맞는다.
- **관찰 (새로운 점, 정성적).** 꼬리가 무거우면 비율 법칙 1/(1+r)이 무너진다. 예를 들어 Student-t(df=4)에서는 Gaussian 양자화 잡음이 방해 꼬리를 거의 바꾸지 못하고, N50도 거의 움직이지 않는다.
  - 따라서 "KV 양자화가 사용 가능 문맥을 다항적으로 줄인다"는 S05 주장은 **Gaussian 꼬리에서만** 성립한다.
  - 실제 헤드는 2609.22101 L1296–1324가 보였듯 Gaussian이 아니다. 따라서 이 주장은 실제 헤드에서는 검증되지 않은 **추측**이다.
- **추측 (핵심, 미증명).**
  - 실제 헤드에서 방해 로짓의 꼬리가 위치(RoPE 감쇠)나 길이에 따라 비정상적으로 변하지 않는다면, 짧은 문맥에서 모은 경험 꼬리로 장문맥 N50을 예측할 수 있다.
  - 근접 중복 hard negative가 k개씩 묶여 있으면 extremal index가 1/k로 줄어 N50이 k배 늘어난다. 이 효과는 N_eff로 보정해야 한다.

### 판정 실험 (≈1–1.5 GPU·주)

- **모델(3계열).** Llama-3.1-8B-Instruct(128K), Qwen3-8B, Mistral 또는 Gemma 계열 장문맥 모델 1개.
- **과제.**
  - 합성 needle: 2609.22101식 same-format hard negative를 쓰고, 문맥은 4K–128K로 늘린다.
  - RULER NIAH 하위과제.
- **KV 팔(arm).**
  - K/V를 INT2/3/4로 양자화한 설정. per-channel K, per-token V, outlier bypass를 넣어 well-scaled로 만든다.
  - FP8 e4m3. 2-level 누산 커널로 vLLM 버그를 통제한다.
  - BF16.
  - scale, outlier, 누산 실패는 별도 진단 팔로 분리한다.
- **보정(calibration).**
  - 4K 이하 프롬프트 500개 이상(헤드당 방해 로짓 200만 개 이상)으로 retrieval head를 찾는다.
  - 각 헤드에서 m_h, 경험 꼬리 Q̂_{h,b}, σ_q(b)를 측정한다.
- **예측기 비교.**
  - 경험 꼬리 EVT(본안)
  - Gaussian 모멘트 EVT
  - Block-GTQ식 logit-MAE 회귀
  - SemKV식 측정 cliff 보간
- **지표.**
  - 예측 N50 대 실측 N50의 배수 오차
  - 비트폭별 ln N50의 기울기
  - 할당 팔(평균 2.5bit, 3bit): uniform, logit-MSE(DP), KVTuner/RateQuant 재현, 본안. 목표 길이 32K, 64K, 128K에서 NIAH와 RULER 점수를 잰다.
- **성공 기준.**
  - 본안 예측의 배수 오차 중앙값 ≤ 2이고, Gaussian 모멘트와 logit-MAE보다 유의하게 작다.
  - 할당 이득이 128K에서 3pp 이상이고 32K 이득보다 크다.
- **실패 기준.** "가설" 절의 반증 기준과 같다.

### 수치 확인

스크립트는 `scripts/new_ideas/idea10_check.py`(numpy만 사용, seed 고정, 약 14초)다. 각 블록의 출력은 다음과 같다.

**A. 선두 차수 법칙과 정확식 비교 (σ_d=1)**
- m=4, r=0일 때 정확한 N50은 2.189e4이고 법칙은 2981이다(7.34배 차이). 같은 비율이 m=5에서 9.01, m=6에서 10.70이다. 즉 선두 차수 법칙은 2배 기준을 **통과하지 못한다**.
- 비율 ln N50_q/ln N50_FP 대 1/(1+r):

| r | 실측 비율 | 1/(1+r) |
|---|---|---|
| 0.06 | 0.943–0.945 | 0.943 |
| 0.33 | 0.767–0.778 | 0.752 |
| 1.0 | 0.535–0.559 | 0.500 |

  비율 법칙은 30% 기준 안에 든다.

**B. t(df=4) 꼬리, 4K×1000 프롬프트로 보정**

| 조건 | 실제 N50 | Gaussian 예측 | 경험 꼬리 예측 | 선두 차수 법칙 |
|---|---|---|---|---|
| m=5, r=0 | 656 | 2.41e6 (3668배 과대) | 657.3 (1.002배) | 407배 과대 |
| m=6, r=0 | 1310 | 7.16e8 (5.5e5배) | 1318 (1.007배) | 5.11e4배 과대 |

- r=0.33에서도 경험 꼬리 예측은 0.977–0.993배로 맞는다.
- 양자화가 N50에 주는 효과는 656→552, 1310→1157에 그친다. ln 비율은 0.973–0.983으로, 비율 법칙이 예측하는 0.752와 크게 다르다.

**C. Gaussian 키(d=128)에서 측정한 r과, FP N50=1M인 헤드에 대한 예측 N50**

| 형식 | r (per-token) | 예측 N50 | outlier 4채널 + per-token | outlier 4채널 + per-channel |
|---|---|---|---|---|
| INT2 | 0.254 | 7.46e4 | 83 | 7922 |
| INT3 | 0.046 | 5.43e5 | 1.58e4 | 2.73e5 |
| INT4 | 0.010 | 8.62e5 | — | — |
| FP8 | 7.0e-4 | 9.89e5 | — | — |

**D. 16헤드, 평균 3bit 할당 (목표 길이 기준 DP 대 logit-MSE DP)**

| 목표 길이 | 8K | 32K | 128K | 512K | 2M |
|---|---|---|---|---|---|
| 이득 | +1.35pp | +1.78pp | +3.52pp | +5.60pp | +6.39pp |

128K에서 FP16 대비 남은 격차 14.2pp 가운데 약 25%를 회수한다.

**판정.**
- 지지되는 것
  - 경험 꼬리 예측기
  - 길이에 따라 커지는 할당 이득. 단 합성 오라클 기준이다.
  - "fp8는 well-scaled라면 길이 붕괴를 만들 수 없다"는 premise 반박
- 약화된 것
  - 닫힌형 ln N* = m²/(2s²)를 예측식으로 쓰는 것
  - 1/(1+r) 법칙의 일반성. heavy tail에서는 깨진다.
- 주의: 할당 이득은 같은 모형에서 최적화·평가한 in-sample 상한이다.

### 평가

- **신규성: 2/5 (낮음).**
  - 이론은 2609.22101 Thm 1과 교과서 EVT의 분산 치환에 불과하다.
  - 실제로 새로운 것은 짧은 문맥 경험 꼬리 보정 프로토콜과, 길이를 목표로 하는 헤드별 할당기다.
- **메인 트랙 가능성: 낮음(약 15–20%).** 3계열 모델에서 2배 이내 예측이 나오고, 128K 할당 이득이 강한 2026 할당기들(Block-GTQ, AATC, RateQuant, SemKV) 대비 3pp 이상이면 borderline이 된다. 그렇지 않으면 워크숍 분석 논문이다.
- **최대 위험.**
  1. 실제 KV 붕괴는 scale, outlier, 커널 문제(WitCert 4K, vLLM 누산 버그, 2605.20868의 평평한 하락)이고, 잘 스케일하면 INT3 이상에서는 효과가 너무 작다. 수치 C에서 INT4 well-scaled의 N50 감소는 14%다.
  2. 방해 꼬리가 무거우면 양자화가 길이 한계를 거의 바꾸지 않아 "법칙"이 공허해진다(수치 B).
  3. 위치에 따라 꼬리가 변하는 RoPE 비정상성 때문에 짧은 문맥 보정이 깨진다.
- **선점 위험.**
  - 2609.22101 저자(양자화 축 추가), AATC 2608.14191 저자(길이 의존 할당), SemKV 저자(cliff의 길이 함수화). 모두 수개월 안에 가능해 보인다.
