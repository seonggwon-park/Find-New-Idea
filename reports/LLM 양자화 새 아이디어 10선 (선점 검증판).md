# LLM 양자화 새 아이디어 10선 (선점 검증 통과판)

작성일: 2026-10-11. 이 문서는 「LLM 양자화 연구 아이디어 10선」의 1차 아이디어 10개 중 9개가 2026년 논문에 선점된 뒤, 처음부터 다시 찾은 아이디어 10개다. 이번에는 아이디어를 쓰기 전에 선점 여부를 먼저 검증했다.

---

## 0. 요약

### 0.1 결론부터

- 후보 40개를 만들고 23개를 적대적 선점 검증에 넣었다. 선점 1개, 전제 오류 3개, "좁혀서 생존" 19개였다. 핵심이 통째로 살아남은 후보는 하나도 없었다.
- 아래 10개는 "좁혀서 생존"한 것 중 상위 10개다. 여기서 "선점되지 않음"은 "남는 기여를 정확히 좁혔을 때 그 부분을 낸 논문이 이번 검색(후보당 arXiv 쿼리 12–40회, 웹 검색 2회 이상, 위협 논문 3–5편 전문 확인)에서 발견되지 않았다"는 뜻이다. "비슷한 연구가 없다"는 뜻이 아니다.
- novelty는 최고 3/5이다. 단독 NeurIPS/ICLR main-track 가능성이 20%를 넘는다고 볼 근거가 있는 것은 아이디어 1뿐이다. 나머지는 workshop, 단신, 혹은 더 큰 논문의 한 절 규모로 보는 것이 정직하다.
- 수치 확인에서 10개 중 8개가 원래 주장의 일부를 반증당해 약화됐다. 각 절의 "수치 확인"에 무엇이 지지되고 무엇이 틀렸는지 적었다.

### 0.2 판정표

| # | 아이디어 | 단계 | novelty | 단독 main-track 가능성 | 판정 실험 비용 | 수치 확인 스크립트 |
|---|---|---|---|---|---|---|
| 1 | V 캐시 토큰 축 noise shaping (0바이트 오차 피드백) | 서빙(KV) | 3/5 | 낮음–중간(약 20–30%) | 약 3 GPU·일 | `scripts/new_ideas/idea01_check.py` |
| 2 | DP replica 간 공유 시드 SR의 평균화 붕괴 진단 | 학습(FP4 backward, DP) | 2/5 | 낮음(<10%) | Step 0 1시간, 이후 1–2 GPU·주(조건부) | `scripts/new_ideas/idea02_check.py` |
| 3 | Haar 회전 GPTQ와 WaterSIC 최악 간극의 정확한 점근식 | PTQ 이론 | 3/5 | 낮음(ISIT·TMLR 또는 한 절) | CPU + 약 2 GPU·일 | `scripts/new_ideas/idea03_check.py` |
| 4 | FP4 RL Wgrad의 토큰 축 공유 순열 | 학습(FP4 RL backward) | 2.5/5 | 낮음(10–15%) | 약 1–1.5 GPU·주 | `scripts/new_ideas/idea04_check.py` |
| 5 | 양자화 KL 앵커·teacher의 보상 shaping 분해 | 후학습(RL·OPD) | 2/5 | 낮음(workshop) | 약 3–4 GPU·주 | `scripts/new_ideas/idea05_check.py` |
| 6 | 오차 피드백 반영 WUSH(Schur pivot 블록 변환) | PTQ(weight-only) | 3/5 | 낮음 | 약 4 GPU·일 | `scripts/new_ideas/idea06_check.py` |
| 7 | 저비트 키 프록시 top-k의 경험적 베이즈 사후 질량 랭킹 | 서빙(희소 어텐션·eviction) | 2/5 | 낮음(10–15%) | 3–5 GPU·일 | `scripts/new_ideas/idea07_check.py` |
| 8 | QeRL형 양자화 RL의 그룹 공유 섭동과 롤아웃별 재추출 | 후학습(RL) | 2/5 | 낮음(10–15%) | 약 250 GPU·시간 | `scripts/new_ideas/idea08_check.py` |
| 9 | PTQ 층간 오차 상쇄의 부분공간 분해와 오차 셰이핑 | PTQ·분석 | 2/5 | 낮음(10–15%) | 약 1.5 GPU·주 | `scripts/new_ideas/idea09_check.py` |
| 10 | 검색 헤드별 EVT 꼬리로 사용 가능 문맥 길이 예측과 길이 목표 비트 배분 | 서빙(KV·장문맥) | 2/5 | 낮음(15–20%) | 약 1–1.5 GPU·주 | `scripts/new_ideas/idea10_check.py` |

권장 순서: 비용 대비 판정력이 가장 큰 것은 아이디어 1(실제 어텐션 행의 R·β·h 측정만으로 4 GPU·시간 안에 kill gate 통과 여부가 갈린다), 아이디어 3(CPU와 2 GPU·일, 이론 노트로 완결 가능), 아이디어 6(4 GPU·일)이다. 아이디어 2는 Step 0(2-GPU, 1시간)으로 Megatron/TE 기본 설정의 SR 난수 공유 여부만 먼저 확인할 가치가 있다.

### 0.3 방법

1. **후보 생성(4관점 × 10개).** 서빙, 학습, PTQ 이론·포맷, 단계 간·인접 분야 관점의 에이전트가 각자 10개를 냈다. 근거는 이전 스윕의 arXiv 전문 노트(`research_notes/arxiv_sweep/`)에서 찾은 구체적 모순·한계·미설명 수치만 허용했다. 각 후보는 제안 전에 자체 arXiv 검색 4회 이상을 거쳤다. 기록: `research_notes/new_ideas/generate/`.
2. **선별(40 → 23).** 중복(S05·X03, S09·X02, P01·P02·P09)을 합치고, 기존 보고서에서 이미 채점된 것(T07)과 근거가 약한 것을 뺐다. 기록: `research_notes/new_ideas/triage/shortlist.json`.
3. **적대적 선점 검증(23개).** 후보마다 "선점됨"을 기본 판정으로 두는 반박 에이전트 1개가 arXiv 쿼리 12회 이상(2025–2026 날짜 제한 포함), 웹·GitHub 검색 2회, 위협 논문 3–5편 전문 grep을 수행했다. 기록: `research_notes/new_ideas/priorart/*.md`, 통합 판정 `priorart/ALL_VERDICTS.json`.
4. **작성과 수치 확인(상위 10개).** 섹션마다 좁혀진 가설만 쓰고, 핵심 정량 주장을 numpy 합성 실험으로 확인했다. 결과가 주장과 어긋나면 주장을 약화했다. 10개 스크립트는 모두 다시 실행해 섹션에 인용된 수치와 대조했다(반올림 또는 출력값에서 계산한 값 외 불일치 없음, 실행 19–65초).

인용 표기: [V]는 노트 또는 이번 작성 단계에서 논문 원문과 대조한 것, [S]는 초록·스니펫 또는 저자 자기보고 수준이다. 모든 arXiv id는 `research_notes/new_ideas/` 또는 `research_notes/arxiv_sweep/` 노트에 근거가 있다. 예외 1건(아이디어 9의 PulseQuant 2609.33384)은 작성 단계에서 새로 찾은 것이며 arXiv 메타데이터로 존재를 확인했다.

### 0.4 한계

- 수치 확인은 전부 합성 데이터다. 실제 LLM에서의 효과 크기는 각 절의 판정 실험으로만 결정된다.
- 선점 검증은 후보당 에이전트 1개가 수행했다. 1차 보고서의 실패 원인(스니펫 의존)은 제거했지만, 다른 이름으로 같은 일을 한 논문을 놓쳤을 가능성은 남는다. 특히 2026년 9–10월 논문은 몇 주 단위로 쌓이고 있다.
- 이 분야는 선점 속도가 매우 빠르다. 23개 후보의 선점 근거 중 상당수가 2026년 8–10월 논문이었다. 아래 아이디어도 같은 위험을 진다.

---

## 새 아이디어 1. V 캐시의 토큰 축 오차 피드백(noise shaping): 어텐션 확률을 재구성 저역통과 필터로 쓰는 0바이트 기법

### 가설

write-once KV 캐시에서 V를 토큰 순서대로 1차 오차 피드백으로 기록한다.

- 기록 규칙: q_t = Q(v_t − u_{t−1}), u_t = q_t − (v_t − u_{t−1}).
- carry u는 write kernel 안에만 있다. 헤드당 O(d)이고 영구 저장은 0바이트다.
- 효과: 어텐션 출력 오차가 σ²Σ_t p_t²에서 σ'²Σ_t (p_t − p_{t+1})²로 바뀐다.
- σ'²/σ² = h는 범위 headroom에 따른 원소 오차 분산 증가분이다.

예측은 세 가지다.

1. **헤드별 이득 법칙**: 헤드별 출력 MSE 비(shaped/RTN)는 h·R을 따른다. 여기서 R = Σ(Δp)²/Σp²이다. h·R < 1인 헤드에서만 이득이 있다. 이 헤드들은 diffuse, local, recency 헤드이며, sink 토큰과 최근 FP residual 창은 고정밀로 둔다.
2. **비트 등가**: shaped 2-bit V가 unshaped 3-bit V와 같아지려면 h·R ≲ σ₃²/σ₂² ≈ (3/7)² ≈ 0.18이어야 한다. 원 가설은 "h·R < 1이면 3-bit와 동등"이라고 했는데, 수치 확인 결과 틀렸다. 이 문턱으로 약화한다.
3. **K에는 적용 금지**: 같은 shaping을 K에 적용하면 오히려 나빠진다.

반증 기준은 둘이다.

- Llama-3.1-8B, Qwen3-8B의 4–32K 실제 어텐션 행에서 attention-weighted mean h·R이 0.7을 넘으면 기각한다.
- 선택적 shaping의 2-bit 다운스트림 이득이 RTN-2→RTN-3 격차의 25% 미만이면 기각한다.

### 문헌상 공백과 선행 연구 위치

선행 연구는 다음과 같다.

- [WitCert 2607.28699](https://arxiv.org/abs/2607.28699)
  - zero-storage subtractive dither를 KV 캐시에 이미 구현했다.
    - 양자화식은 x̂ = s(round(x/s+ξ)−ξ)이다(l.264-266).
    - ξ는 Philox로 (request, layer, KV head, token, channel)에서 생성한다(l.866).
    - "stored at zero cost"라고 명시한다(l.631). [V]
  - Table 9(l.1240-1268; Qwen2.5-7B, 8k, INT8 K+V, 실제 V와 실제 출력)의 상대 출력 오차는 RTN 2.751e-2, dither 2.717e-2이다.
  - 저자들은 "1.01×, quality neutral"이라고 쓰고, 이전의 1.7–1.8× 주장을 철회했다. [V]
  - 즉 "디더가 토큰 평균으로 오차를 줄인다"는 부분은 구현도 되었고, 8비트에서는 경험적 null까지 출판되었다.
- [Low-Discrepancy Dither for Quantized Recurrent State Caches 2609.39185](https://arxiv.org/abs/2609.39185)
  - SSM 상태용 디더를 다룬다.
  - sigma-delta는 이전 오차를 저장해야 해서 "doubles the size of a state cache"라고 기각한다(l.594-597).
  - attention 모델의 KV 양자화에서는 "errors are not fed back"이라고 쓴다(l.603-604). [V]
- [KV Cache Compression Through the Lens of Transform Coding (AATC) 2608.14191](https://arxiv.org/abs/2608.14191)
  - Assumption 1에서 오차가 토큰·채널에 걸쳐 독립이라고 가정한다(l.409-412).
  - 토큰 상관을 활용하려면 "computationally intractable token varying bit allocation"이 필요하다고 쓴다(l.437-441). [V]
- [When Can Depth Replace Precision? 2607.23390](https://arxiv.org/abs/2607.23390)
  - depth/microstep 축의 telescoping carry를 다룬다.
  - Daubechies–DeVore, Güntürk의 고전 sigma-delta와 SDQ-LLM을 인용한다(l.639-644). [V]
- 가중치·상태 쪽 유사물
  - [SDQ-LLM 2510.03275](https://arxiv.org/abs/2510.03275)는 가중치 sigma-delta다.
  - GPFQ [2010.15979](https://arxiv.org/abs/2010.15979)/[2201.11113](https://arxiv.org/abs/2201.11113)는 greedy 오차 피드백 가중치 양자화다.
  - [2609.04490](https://arxiv.org/abs/2609.04490)은 recurrent write-back 오차 피드백이다. [S: abstract 수준]
- 주변 연구
  - [2610.01889](https://arxiv.org/abs/2610.01889)(SR 대 RN 리덕션 √n 대 n), [2609.05641](https://arxiv.org/abs/2609.05641)(독립 잡음 디더 행렬곱 하한), [2605.13810](https://arxiv.org/abs/2605.13810)(디더 RHT 비편향·MSE), [KVarN 2606.03458](https://arxiv.org/abs/2606.03458)(decode 누적 오차), [VC-Attention 2609.15810](https://arxiv.org/abs/2609.15810)(V 블록 평균 smoothing).
  - 이들 중 토큰 축 noise shaping을 다루는 것은 없다. [V/S]

남는 것은 다음과 같다. KV/V 캐시의 **토큰 축** 오차 피드백, noise shaping, sigma-delta는 이번 검색(arXiv API 쿼리 약 25회, 웹/GitHub 3회)에서 발견되지 않았다. 2026-10-11 재검색에서도 KV×{noise shaping, error feedback, error diffusion, sigma-delta} 쿼리는 0건이었다. 미출판으로 보이는 기여는 다음과 같다.

- (1) 영구 저장 없는 write-time 1차/L차 피드백. 이는 2609.39185의 저장 반론에 대한 답이다.
- (2) 출력 오차가 σ'²Σ(Δp)²라는 명제.
- (3) h를 포함한 헤드 선택 규칙 h·R < 1.
- (4) V 전용 예측.
- (5) AATC가 "intractable"이라 한 토큰 상관을 신호가 아닌 *오차*의 구조로 싸게 활용하는 점.

디더와 deadzone 부분은 기여가 아니고 ablation으로만 쓴다.

### 수학 (증명 가능한 것과 추측)

**명제 1 (대수적 항등식, 가정 없음).** 위 기록 규칙을 쓰고 u_0 = 0, p_{T+1} := 0으로 둔다. 그러면 다음이 성립한다.

- e_t = q_t − v_t = u_t − u_{t−1}
- e_o = Σ_t p_t e_t = Σ_t u_t (p_t − p_{t+1})

summation by parts일 뿐이다. 이 항등식은 Q와 p에 무관하다. eviction은 p_t = 0인 위치로 그대로 표현된다. 따라서 eviction은 telescoping을 "깨는" 것이 아니라 Δp에 점프를 만든다. 형식 자체는 고전적이고, KV 출력에 적용한 것이 새롭다.

**명제 2 (subtractive dither 가정).** shaped quantizer에 subtractive dither를 넣고 overload가 없으면, u_t는 신호와 독립이며 평균 0, 원소 분산 σ'²로 토큰 간 iid이다(Schuchman; Gray–Stockham, 교과서적). 그러면 다음이 성립한다.

- shaped: E‖e_o‖² = dσ'² Σ_t (p_t − p_{t+1})²
- unshaped: E‖e_o‖² = dσ²Σp_t²
- 비는 h·R이다.

p ≥ 0이면 (a−b)² ≤ a²+b²이므로 R ≤ 2이다. spike 헤드의 최악 손실은 2h배다. 이 조합과 헤드 선택 규칙은 새롭다. 단, per-token min-max에서는 Δ_t가 w_t에 의존한다. 그래서 iid는 "Δ_t가 주어졌을 때 조건부 균등"이라는 근사다. [S]

**따름정리 3 (iid logit jitter).** p_t ∝ exp(βz_t), z ~ N(0,1), N → ∞이면 R → 2(1 − e^{−β²})이다(lognormal 모멘트 계산). 손익분기점은 β* = √(−ln(1 − 1/(2h)))이다.

- 실제 행은 RoPE 진동과 내용 의존 logit 때문에 jitter가 클 수 있어, 이득 영역이 β < β* ≈ 0.75로 좁을 수 있다. 실제 β 분포는 측정된 바 없다. [S]

**h에 대한 정정.** kill test 노트는 2-bit per-token min-max에서 h ≈ 16/9 ≈ 1.78을 제시했다. 이는 양끝이 모두 Δ/2씩 넘치는 최악 상한이다. 합성 데이터(이질적 채널 Gaussian, d=128)의 측정값은 h = 1.175(2b), 1.029(3b), 1.006(4b)이다. h는 V outlier 구조에 의존하므로 모델에서 측정해야 한다.

**추측 (증명 없음).**

- 고차 shaping: L차로 바꾸면 Δ^L이 되고, 일반 수열에 대한 최악 배율은 C(2L,L)이다. 다만 2-bit 같은 작은 alphabet에서 L ≥ 2의 안정성(overload 없음)은 보장되지 않는다. 고전 sigma-delta의 알려진 난점이다.
- K 쪽: 1차 근사 δo ≈ Σ_t p_t δs_t (v_t − o)에서 가중치 p_t(v_t − o)는 v의 토큰 간 변동 때문에 매끄럽지 않다. 그래서 K shaping은 약 2h배 손해라고 예측한다(휴리스틱).

### 판정 실험 (약 3 GPU·일)

- **1단계(약 4 GPU·시간, kill gate)**
  - Llama-3.1-8B, Qwen3-8B, Mistral-7B에서 LongBench/RULER 프롬프트(4K/8K/16K/32K)를 대상으로 한다.
  - FP 어텐션 행을 저장하고, 헤드별 R, β 추정치, V outlier에서 나오는 h를 계산한다.
  - attention-weighted mean h·R > 0.7이면 중단한다.
  - h·R < 0.18인 헤드의 비율과, 그 헤드들이 RTN 출력 오차 총량에서 차지하는 비중을 보고한다. 비중이 작으면 이득도 작다.
- **2단계(약 2.5 GPU·일)**
  - K는 4-bit per-channel로 고정한다. sink 토큰과 128-token residual은 FP로 둔다.
  - V arm은 {RTN-2, RTN-2+dither, SD-2 전 헤드, SD-2 선택(h·R < 1), SD-2 선택+dither, RTN-3, SD-3}이다.
  - 각 arm을 Hadamard 회전 유무로 나누어 비교한다. 기준선은 KIVI-2와 KVarN이다.
  - K shaping arm을 별도로 둔다.
  - 지표
    - 헤드별 출력 MSE: log 예측(h·R)과 log 측정의 상관.
    - PPL(PG19 8–32K), LongBench, RULER 4–32K, GSM8K-CoT.
  - 구현
    - Q가 비선형이라 associative scan이 안 된다. prefill은 토큰 방향 순차, 헤드·채널 방향 병렬 루프로 하고 지연을 보고한다.
    - chunked write는 경계에서 carry를 넘기고, 마지막 carry는 FP residual로 흡수한다. [S]
- **성공 조건**
  - 상관 ρ ≥ 0.8.
  - SD-2 선택이 RTN-2→RTN-3 격차의 ≥ 50%를 닫는다.
  - 회전 2-bit 최강 기준선보다 ≥ 1pt 높다.
  - K shaping은 손해다.
- **실패 조건**
  - ρ < 0.5.
  - 격차의 25% 미만.
  - 이득이 dither arm에서만 나온다.

### 수치 확인

스크립트는 `/home/user/Find-New-Idea/scripts/new_ideas/idea01_check.py`이다. CPU에서 47초 걸린다.

- 설정: N=4096, d=128, 6 seeds, per-token min-max, sink FP.
- 13개 p 패밀리: uniform, local, recency, 매끄러운 logit, iid jitter β 0.25–2, sink+jitter, 4-spike 검색, last-token 0.5.

출력 수치는 다음과 같다.

- **법칙 h·R** (log 예측과 log 측정의 Pearson, 손익 부호 일치)

  | bits | dither 있음 | dither 없음 |
  |---|---|---|
  | 2 | 0.9997, 13/13 | 0.9880, 11/13 |
  | 3 | 0.9997 | 0.9995 |
  | 4 | 0.9991 | 0.9994 |

  2-bit dither 없음의 불일치 2건(β=0.75, 1.0)에서는 RTN 오차가 백색보다 coherent했다. 그래서 측정 이득(0.42, 0.72)이 예측(1.05, 1.42)보다 컸다. 이 합성 V에서는 2-bit RTN의 uniform 헤드 출력 MSE가 9.3e-2로, 디더를 넣은 2.7e-2보다 컸다. 이는 채널 평균에서 오는 coherent 오차다. WitCert의 INT8 null과는 비트 수가 달라 충돌하지 않는다.
- **예시 비** (2-bit, dither 없음, SD/RTN): uniform 7.5e-5, local256 7.0e-3, recency 2.7e-3, 매끄러운 logit 0.126, β=0.5 0.166, β=1.5 1.94, 4-spike 2.30, last-token 0.5 2.66.
- **손익분기점 β\*** = 0.745(2b, h=1.175), 0.829(4b)이다. 4b 측정에서 β=0.75는 0.875, β=1.0은 1.334였다. 예측 β*와 맞는다.
- **shaped 2-bit / RTN 3-bit**: uniform 1.5e-3, local 4.4e-2, recency 2.5e-2, 매끄러운 logit 0.70, β=0.25에서 0.84로 이긴다. β=0.5에서는 2.92, sink+jitter0.3에서는 1.23, spike 헤드에서는 11–12로 진다.
- **K shaping (V FP, 매끄러운 K로 R=0.39)**: 출력 MSE 비 SD/RTN이 2.44(4b), 4.64(2b)이다.

판정은 다음과 같다.

- 출력 오차 법칙 h·R, R ≤ 2, β* 공식, V 전용 예측은 합성 데이터에서 지지된다.
- 원 가설의 "h·R < 1 헤드에서 2-bit가 3-bit와 동등"은 반증되었다. 문턱은 h·R ≲ 0.18이다.
- 이득이 큰 패밀리(uniform, local)는 원래 RTN 오차가 작은 헤드다. spike/검색 헤드는 RTN 오차가 약 200배 크지만 shaping이 손해다. 모델 전체 이득은 작을 수 있다.
- h ≈ 1.78이라는 주장은 과대평가였고, 측정값은 1.18이다.

### 평가

**신규성: 3/5 (보통).**

- 디더와 1/N_eff 평균은 선점되었다. WitCert가 구현했고, 8비트 null도 있다.
- telescoping carry는 고전이다.
- 새로운 것은 "어텐션 확률을 재구성 필터로 보는 토큰 축 오차 피드백"과 h·R 선택 규칙뿐이다. 수학은 한 줄짜리 summation by parts라서, 리뷰어는 이론적 기여를 낮게 볼 것이다.

**메인트랙 가능성: 낮음에서 중간(약 20–30%) [S].**

- 출력 MSE만 있으면 워크숍 수준이다.
- 메인트랙이 되려면 실제 LLM에서 h·R < 0.18인 헤드가 오차 총량의 상당 부분을 차지해야 한다. 그리고 회전 2-bit 기준선 대비 0.5–1비트 등가 이득을 LongBench/RULER에서 보여야 한다.

**가장 큰 위험.**

- (i) 실제 어텐션 행의 고주파 jitter(RoPE 진동, 내용 의존 logit) 때문에 β > 0.75인 헤드가 대다수일 수 있다.
- (ii) 이득이 오차가 원래 작은 diffuse 헤드에 몰린다.
- (iii) decode의 최근 토큰 질량이 boundary 항을 만든다. FP residual로 완화할 수 있지만, 그러면 기여가 줄어든다.
- (iv) prefill에서 순차 루프 비용이 든다.
- (v) "디더 효과"와 구분하라는 리뷰 요구가 나올 것이다.

**선점 가능 주체.**

WitCert 저자(Philox dither kernel 보유), 2609.39185 저자(KV를 "not fed back"이라고 직접 언급), AATC 저자(토큰 상관을 공백으로 명시), KVarN 저자. 누구든 한 번의 후속작으로 쓸 수 있는 거리다. 진행한다면 1단계 R 측정(수 GPU·시간)을 먼저 끝내고 판단해야 한다.

---

## 새 아이디어 2. 데이터 병렬 replica 간 공유 시드 stochastic rounding: 부호 편향이 있을 때만 생기는 1/P 평균화 붕괴의 진단과 측정

### 가설

**원래 가설에서 버린 것.** 원래 T01의 두 주장, "공유 RNG state면 SR 오차가 rank 간 ρ=1/2로 결합된다"와 "stratified dither면 O(1/P²)"는 그대로는 성립하지 않는다.
- micro-batch 재사용은 일어나지 않는다. TE는 quantize 호출마다 Philox offset을 전진시킨다.
- stratified 이득은 최대 2배이다.
- FP4 SR은 sign-magnitude 방식(|x|를 0 쪽 또는 먼 쪽으로 반올림)이다. 그래서 같은 uniform을 공유해도 **gradient 부호가 rank 간에 독립이고 대칭이면 오차 상관은 0이다** (수치 확인 Part B). "공유 시드 = 완전 결합"은 틀렸다.

**좁힌 가설 (반증 가능).** 기본 설정의 Megatron-LM + Transformer Engine(TE) NVFP4 학습을 가정한다.
- 전제: gradient operand(dY)의 SR 난수가 모든 DP rank에서 같은 원소 위치에 같은 값으로 배정된다. 이것은 소스 코드를 읽어서 얻은 결론이며 아직 실행으로 확인하지 않았다.
- 결합 계수: 원소 위치 (t,i)에서 rank 간 SR 오차 상관은 ρ_ti ≈ ½·(E[sign dY_ti])²이다. uniform residue를 가정한 값이며, NVFP4의 비균일 residue에서는 상한이 약 0.35이다.
- 평균된 Wgrad의 SR 잡음: 입력 채널 k에 대해 Var ≈ T·v·[(μ_k²+σ_k²)/P + (1−1/P)·ρ_i·μ_k²]이다. 따라서 **부호가 편향된 출력 feature i와 평균이 0이 아닌 입력 채널 k가 만나는 칸에서만** 잡음이 P에 대해 plateau를 이룬다. post-SwiGLU FC2 입력이나 massive-activation 채널이 그런 k의 예이다.
- 교차점: plateau가 1/P 항을 넘는 지점은 P* ≈ (μ_k²+σ_k²)/(ρ_i μ_k²)이다.
- 실패 기준 1 (해롭지 않은 버그): 실제 1B급 모델의 모든 linear layer에서 측정한 max_{i,k} ρ_i·μ_k²/(μ_k²+σ_k²)가 1/1024보다 작으면, P≤1024에서는 공유 시드가 무해한 버그이다.
- 실패 기준 2 (전제 붕괴): 2-GPU 실측에서 SR 난수가 rank 간에 비트 단위로 일치하지 않으면 진단 자체가 무너진다.

### 문헌상 공백과 선행 연구 위치

- [Correlated quantization for distributed mean estimation and optimization 2203.04925](https://arxiv.org/abs/2203.04925)
  - l.492-503: "Ui = πi/n + γi", Qi(xi)=1{πi/n+γi<xi}. l.155: "Correlated quantization uses shared randomness to introduce correlation".
  - **rank-stratified dither 해법은 이미 출판되어 있다** [V]. 다만 [0,1] 입력 vector를 직접 평균하며, sign-magnitude FP SR이나 GEMM 후 all-reduce는 다루지 않는다.
- [Correlated Quantization for Faster Nonconvex Distributed Optimization 2401.05518](https://arxiv.org/abs/2401.05518)
  - l.1882: "the permutations require no extra communications, since they can be seeded" [V].
- [Training with Fewer Bits ... Stochastic Rounding 2511.00874](https://arxiv.org/abs/2511.00874)
  - l.578-584: "stochastic rounding is performed independently on each number", "SR variance ... diminishes as the mini-batch size b increases" (Lemma 3, 1/b).
  - 이 아이디어가 겨냥하는 바로 그 독립 가정이다 [V].
- [Pretraining LLMs with NVFP4 2509.25149](https://arxiv.org/abs/2509.25149)
  - l.571: "Use stochastic rounding for gradients and round-to-nearest-even for weights and activations".
  - App. E.4.3 l.1709-1713: RHT sign을 "seed per instance"로 바꿔도 "no improvements from inducing randomness at every transform instance".
  - 상관된 randomness가 무해할 수 있다는 반대 증거이다. 단, SR이 아니라 RHT 결과이다 [V].
- [Quartet II 2601.22813](https://arxiv.org/abs/2601.22813)
  - App. A l.1544-1545: "we re-randomize rotations per-tensor per-micro-batch". micro-batch 재사용 주장을 반박하며, DP rank는 논의하지 않는다 [V].
- [Low-Discrepancy Dither for Quantized Recurrent State Caches 2609.39185](https://arxiv.org/abs/2609.39185)
  - 시간 합산 축을 따라 dither 상관 구조를 설계한다 (Proposition 1, Weyl). l.1209 부근: "a dither correlated across entries"가 해롭다. 원리는 같고 축만 다르다 [V].
- [What is New in Stochastic Rounding (survey) 2603.06060](https://arxiv.org/abs/2603.06060)
  - l.328-340: NVIDIA 특허는 LFSR seed를 layer/block 위치로 정한다. 이렇게 하면 replica 간에 randomness가 공유될 수 있다. 상관 분석은 없다 [V].
- [Format-Aware Fusion for Fast FP4 Pretraining 2610.00053](https://arxiv.org/abs/2610.00053)
  - l.2070-2071: worker별 SR seed와 counter offset을 checkpoint한다. 시간 축 재사용만 다루고, rank 간 동일성은 다루지 않는다 [V].

**남는 공백 (이번 검색, arXiv 쿼리 약 27회와 WebSearch 4회에서 발견되지 않음):**
1. 기본 Megatron-LM + TE NVFP4에서 SR Philox state가 DP rank 간에 동일하다는 진단 [S, 소스 읽기].
   - Megatron은 `data_parallel_random_init=False`가 기본값이며, `seed += 10*dp_rank`는 이 옵션이 켜져 있을 때만 적용된다.
   - TE의 quantizer.cpp l.2656-2666과 cast.cpp l.173-185는 `getDefaultCUDAGenerator()`를 사용한다.
2. sign-magnitude SR에서 공유 난수의 효과가 (E[sign])²로 제곱 억제된다는 관찰과, 이것이 GEMM→all-reduce 경로에서 μ_k²와 곱해진다는 channel × feature 분해 [S].
3. stratified 이득도 같은 (E[sign])² 인자에 묶여, 부호에 편향이 없는 gradient에서는 이득이 1배라는 정정 [S].

### 수학 (증명 가능한 것과 추측)

**명제 1 (교과서 수준, 초등).** 가정: residue f_r ~ iid U(0,1), e_r = 1[u_r<f_r] − f_r.
- 독립 u: Var(ē) = 1/(6P).
- 공유 u: Cov = E[min(f_r,f_s)] − E[f_r f_s] = 1/12이므로 Var(ē) = 1/12 + 1/(12P).
- stratified (u_r=(π_r+γ_r)/P): Var(ē) = 1/(12P) + 1/(12P²).
- 원래 주장한 O(1/P²)는 거짓이다. 이 값은 residue가 rank 간에 같거나 집중된 경우, 즉 Suresh et al.의 small-deviation regime에서만 성립한다.
- 증명은 직접 계산이며, 새로운 결과가 아니라 DME 문헌의 특수한 경우이다.

**명제 2 (sign-magnitude 감쇠, 새 관찰이지만 초등).**
- 가정: FP SR은 e = s·δ·(1[u<f] − f)이다. 여기서 s는 부호, δ는 grid 간격, f는 크기 residue이다.
- PTX `cvt.rs`가 "toward zero / away from zero"를 가수 carry로 고른다는 문서 기술에 근거한다 [S, 웹 문서].
- 공유 u에서는 E_u[e_r e_s] = s_r s_s δ_r δ_s (min(f_r,f_s) − f_r f_s)이다.
- 부호가 크기와 독립이고 rank 간에 독립이면 Cov = (E s)²·E[δδ(min−ff)]이다. uniform f에서는 ρ = ½(E s)²이다.
- stratified의 음의 상관도 같은 인자 (E s)²에 비례한다.
- floor 기반 SR(정수 grid, f = x − ⌊x⌋)에서는 부호와 무관하게 ρ = ½이다. 따라서 INT형 gradient 압축이라면 원래 결론이 그대로 성립한다.

**명제 3 (Wgrad 전이).**
- 가정: rank r은 E_r[i,k] = Σ_t e^r_{t,i} x^r_{t,k}를 계산한다. 데이터는 rank 간에 독립이며, 계산 편의상 dY와 x도 독립으로 둔다.
- 결과: Var(mean_r E_r[i,k]) = T·v_i·[(μ_k²+σ_k²)/P + (1−1/P)·ρ_i·μ_k²].
- 고정 sign RHT(token 축 16점, 모든 rank에서 동일)를 쓰더라도, mean 에너지 Σ_j E[x̃_j]² = Σ_j μ²이 보존되므로 결합은 사라지지 않는다. 오히려 평균이 일부 위치에 집중되어 ρ가 볼록하게 커질 수 있다 [S, 수치로만 확인].

**추측 (미검증).**
- 실제 LLM의 dY는 출력 feature 단위의 부호 편향 E[sign dY_{:,i}]이 작다.
- 다만 BOS·attention-sink 위치나 FC2 출력의 일부 feature에서는 위치에 고정된 편향이 있을 수 있다.
- 이 편향과 massive-activation 채널(μ_k ≫ σ_k)이 겹치는 칸이 실제로 존재하는지가 이 아이디어의 사활을 가른다.

### 판정 실험 (Step 0-1 약 20 GPU-시간, Step 2 조건부 약 1-2 GPU-주)

- **Step 0: 전제 실측 (2×B200, 1시간 미만).**
  - Megatron-LM과 TE NVFP4 기본 recipe를 DP=2로 실행한다.
  - (a) 두 rank에 동일한 tensor를 SR quantize해서 출력이 비트 단위로 같은지 본다.
  - (b) 서로 다른 데이터에서 층별로 FC2·QKV·out-proj의 dY SR 오차 상관을 잰다.
  - `--data-parallel-random-init`를 켠 경우를 대조군으로 둔다.
  - (a)가 불일치하면 즉시 종료한다.
- **Step 1: channel × feature 통계 (단일 노드, 약 20 GPU-시간).**
  - 공개 1B·7B checkpoint와 자체 350M의 여러 시점에서 층별 ρ_i = ½(E[sign dY_{t,i}])², μ_k²/(μ_k²+σ_k²), 예측 P*_{ik}를 잰다.
  - DP는 micro-batch로 에뮬레이션한다. P개 micro-batch에 공유/독립/stratified uniform을 강제 배정하면 수학적으로 DP와 같다.
  - P ∈ {8, 64, 256, 1024}에서 Wgrad SR 잡음을 channel-resolved로 측정한다.
  - **성공:** 어떤 layer의 (i,k) 칸에서든 P=1024에서 공유/독립 잡음비가 2 이상이고, 그 비가 명제 3의 예측과 ±30% 안에서 맞는다.
  - **실패:** 모든 칸에서 비가 1.1 미만이다. 이 경우 "무해한 버그" 노트로 종료한다.
- **Step 2: 학습 (Step 1 성공 시에만).**
  - 350M·1B, NVFP4 backward, 에뮬레이션 DP 256·1024 (전역 batch 고정).
  - arm {공유, 독립, stratified} × 3 seed, 20B token.
  - 지표: 최종·후반부 loss gap, downstream 5종, Wgrad 잡음 중 sampling 잡음 비율.
  - **성공:** 공유와 독립 사이의 loss gap이 seed std의 2배를 넘는다.
  - **실패:** gap이 seed std 이내이다.
  - stratified는 독립 대비 이득을 따로 보고한다. 예상 이득은 최대 2배이며 실제로는 거의 1배이다.

### 수치 확인

스크립트는 `scripts/new_ideas/idea02_check.py`이다. CPU에서 약 40초가 걸리며 seed는 고정되어 있다.

- **Part A (명제 1).** P·Var를 이론값과 비교한다.

  | 방식 | P=8 (MC / 이론) | P=64 (MC / 이론) | P=512 (MC / 이론) |
  |---|---|---|---|
  | 독립 | 0.1678 / 0.1667 | 0.1659 / 0.1667 | 0.1667 / 0.1667 |
  | 공유 | 0.7493 / 0.7500 | 5.4200 / 5.4167 | 42.5728 / 42.7500 |
  | stratified | 0.0850 / 0.0938 | 0.0833 / 0.0846 | 0.0840 / 0.0835 |

  - 명제 1이 확인된다. stratified는 O(1/P²)가 아니라 1/(12P)에 머문다.
- **Part B (명제 2).** P=2에서 rank 간 원소별 오차 상관 ρ를 잰다.
  - floor-SR, 공유 u: ρ = +0.5000.
  - **NVFP4 sign-magnitude, 평균 0인 Gaussian gradient, 공유 u: ρ = +0.0004.** 원래 주장한 "완전 결합"을 반박한다.
  - gradient 평균/표준편차 비 m_g에 따른 공유 u 상관 (괄호 안은 ½(2Φ(m_g)−1)² 예측):

    | m_g | 0.25 | 0.5 | 1 | 2 | 모두 양수 |
    |---|---|---|---|---|---|
    | 공유 ρ | 0.0218 (0.0195) | 0.0771 (0.0733) | 0.2193 (0.2330) | 0.3683 (0.4555) | 0.3536 (0.5) |

  - stratified의 ρ도 −0.0009에서 −0.2620으로 같은 경향을 따른다.
- **Part C (명제 3).** 공유 u의 Wgrad 잡음 penalty(공유/독립)를 P=256에서 잰다.
  - (m_g=0, μ=3): 1.32
  - (m_g=0.5, μ=3): **14.2**
  - (m_g=2, μ=3): **96.8**
  - μ=0인 칸: 0.96-1.11
  - plateau 비 Var(256)/Var(16) (순수 1/P = 0.0625): 공유 u에서 (0.5, 3) 0.45, (2, 3) 0.72, 나머지 칸 약 0.06. 고정 RHT를 써도 0.68, 0.75로, RHT는 결합을 없애지 못한다.
  - stratified의 독립 대비 이득은 0.84-1.42이며 MC 잡음이 약 15%이다.
- **결론.** 원래 명제는 floor-SR에서만 맞는다. FP4 sign-magnitude SR에서는 **부호 편향이 없으면 효과도 없다.** 가설은 이번 결과에 맞춰 약화했다.

### 평가

- **신규성: 2/5 (낮음-보통).**
  - kill test는 3/5로 평가했다. 그러나 명제 2 때문에 효과가 (E sign)² × μ² 칸으로 제한되어 한 단계 낮췄다.
  - 진단(기본 설정에서 SR 난수가 rank 간에 동일)과 sign-magnitude 감쇠 분석은 이번 검색 범위에서 출판된 것이 없다.
  - 해법(per-rank seed offset, correlated quantization)은 이미 출판되어 있거나 한 줄 수정으로 끝난다.
- **main-track 가능성: 낮음 (10% 미만).**
  - 대규모 DP에서 loss gap이 나와야 하는데, 그러려면 큰 부호 편향과 massive channel이 겹쳐야 하며 사전적으로 그럴듯하지 않다. NVIDIA RHT seed ablation(2509.25149 E.4.3)도 무해 쪽을 가리킨다.
  - 현실적 결과물은 workshop 노트나 TE/Megatron 이슈 보고이다.
- **가장 큰 위험.**
  1. Step 0에서 rank 간 난수가 실제로는 정렬되지 않는 경우 (kernel launch 수 차이, fork context).
  2. 실제 dY의 부호 편향이 너무 작아, 측정 가능한 칸이 P*>1024에만 존재하는 경우.
  3. TE가 이미 rank offset을 넣었거나 조용히 수정해 전제가 사라지는 경우.
- **선점 위험.** NVIDIA TE·Megatron 유지보수자가 한 줄 수정으로 문제와 논문 가치를 함께 없앨 수 있다. SR 분산의 batch scaling을 이미 다루는 Quartet II·2511.00874 저자군도 독립 가정을 점검하다 같은 관찰에 도달할 수 있다.

---

## 새 아이디어 3. Haar 회전 GPTQ와 WaterSIC 사이 최악 간극의 정확한 점근식과 damping 상한

### 가설

고율(high-rate), entropy-coded 모델에서, 스펙트럼이 λ_min/λ_max ≥ ε를 만족하는 경우 전체에 대해 Haar 회전 GPTQ와 WaterSIC의 rate 간극 sup은 다음과 같다.

**sup gap(ε) = ½log₂(1/ε) − log₂ln(1/ε) + ½log₂(2/e²) + o(1) bit**

- gap의 정의는 ½log₂(AM(D)/GM(λ))이고, D는 QΛQᵀ의 LDL pivot이다.
- sup은 two-point 스펙트럼 {ε, 1}에서 달성되며, 이때 ε 쪽 질량은 f* = 1 − 2/ln(1/ε)이다.
- 따름정리 1: PCA 기저(Specht ratio)와 비교한 회전 이득(rotation dividend)은 정확히 ½log₂ln(1/ε) + O(1) bit이다.
- 따름정리 2: damping δ가 있을 때 기준이 되는 값은 ε_eff = (λ_min+δ)/(λ_max+δ)이다.

반증 기준은 세 가지다.
- (i) n ≥ 400에서 exact Haar pivot으로 구한 two-point sup이 ε ∈ [1e-6, 1e-3] 구간에서 극한식과 0.1 bit 넘게 어긋난다.
- (ii) two-point sup보다 0.05 bit 넘게 큰 일반 스펙트럼이 발견된다.
- (iii) 실제 층에서 예측 gap과 실측 gap의 Spearman 상관이 0.7 미만이다. 이 경우에도 이론은 남지만 진단 도구로서의 가치는 없다.

원안에서 철회한 내용은 다음과 같다.
- log-log 항의 계수 ½를 1로 고친다.
- ε = 1e-3에서의 1.23 bit는 최대화가 덜 된 값이었다. 실제 값은 1.38이다.
- "massive channel이 gap을 키운다"는 주장은 철회한다(수치 확인 F 참조).

### 문헌상 공백과 선행 연구 위치

- [High-Rate Quantized Matrix Multiplication II 2605.13768](https://arxiv.org/abs/2605.13768) (O&P, v2 2026-06-08)
  - l.1594-1597에서 "worst possible gap (over possible spectra λj ≥ ϵ) between GPTQ with rotation and WaterSIC"를 열린 문제로 제시한다 [V].
  - Eq. 33(l.1551-1559)은 pivot을 e_k/e_{k−1}로 근사한다. 기댓값은 footnote 3에 따라 정확하지만 concentration은 증명되지 않았다. AM에서 HM으로 이어지는 ramp는 서술만 되어 있다 [V].
  - Llama-3-8B에서는 "within 0.1 bit"를 보고한다(l.32). singular Σ_X 층은 "we skip"이라고 하여 다루지 않는다(Fig. 5, l.1438) [V].
- [QuIP 2307.13304](https://arxiv.org/abs/2307.13304)
  - Lemma 2(l.303-309)는 tr(D) ≤ (μ²/n)(tr H^{1/2})²이다. Haar를 쓰면 μ = polylog(n)이다(l.412, l.468) [V].
  - 이를 합치면 gap ≤ ½log₂μ² + ½log₂(AM(√λ)²/GM(λ))가 바로 나온다. 둘째 항은 이미 올바른 rate를 가진다 [S, 본 계산].
  - 따라서 "unbounded"라는 결론 자체는 새롭지 않다. O&P도 이 Lemma를 인용한다(l.1484) [V].
- [CurveTQ 2610.09212](https://arxiv.org/abs/2610.09212)
  - Prop. 2(l.1320): native-weighted 방식과 rotated 방식 사이에 어느 한쪽이 항상 우월하다는 dominance가 없다 [V].
  - App. A.3: 2×2 예에서 0.10 bit [V].
  - ramp는 "checked numerically, not proved"라고만 되어 있다(l.1399) [V].
- [WaterSIC 2603.04956](https://arxiv.org/abs/2603.04956)
  - 회전이 없을 때 gap이 무한히 커질 수 있음을 보였다(AM-GM, l.619) [V].
  - 실험 설정은 δ = 1e-4에 dead-feature erasure를 더한 것이다(l.1033-1040) [V].
- [Hegland 2011.14626](https://arxiv.org/abs/2011.14626): volume-sampling 소거 오차가 elementary symmetric 다항식의 비가 된다는 결과로, 증명 도구 후보다 [V].
- [WaterKron 2609.14706](https://arxiv.org/abs/2609.14706): 최악 gap 분석이 없다 [V].
- [서베이 2608.25188](https://arxiv.org/abs/2608.25188): l.4248-4250에서 이 문제를 "open"이라고 다시 확인한다 [V].

**남은 공백.** 이번 검색(arXiv API 쿼리 약 30회, 웹 3회, 2026-08 이후 포함)에서는 다음 세 가지가 발견되지 않았다.
- 회전 gap sup의 정확한 상수
- μ-free 상한
- damping을 ε_eff로 연결하는 분석

### 수학 (증명 가능한 것과 추측)

1. **교과서적 결과**
   - Newton 부등식에 따라 r_k = E_k/E_{k−1}은 단조 감소하고, GM(r) = GM(λ)이다.
   - [ε,1]에서 AM/GM의 최대는 Specht ratio다. ½log₂S = ½log₂(1/ε) − ½log₂ln(1/ε) − ½log₂e + o(1)이다.
2. **새 결과, 증명 가능성 높음 (하한)**
   - Eq. 33의 n→∞ 극한에서 two-point 스펙트럼은 r(t) = (1−f−t)/(1−t) (t < 1−f)를 준다.
   - 따라서 AM(r) → (1−f) + f ln f + O(ε ln(1/ε))이고, 이는 x = 1−f일 때 약 x²/2이다.
   - 이를 최적화하면 x* = 2/ln(1/ε)이고 상수는 −0.943이다.
   - Haar에 대해 엄밀하게 하려면, two-point 스펙트럼이 ε·I + (1−ε)P 꼴이므로 사영(projection)의 free compression(Jacobi 법칙) 극한이나 log-det concentration을 쓰면 된다 [S].
3. **추측 A (μ-free 기댓값형 Lemma 2)**
   - 주장: (1/n)Σ_k E_k/E_{k−1} ≤ AM(√λ)².
   - 이것이 성립하면 QuIP의 ½log₂μ² slack이 사라지고, sup gap ≤ ½log₂(1/ε) − log₂ln(1/ε) + (1 − log₂e) + o(1)을 얻는다. rate는 확정되고, 상수 차이 ½ bit만 남는다.
   - n = 2이면 차이가 √(ab)(√(ab)/(a+b) − ½) ≤ 0이므로 성립한다 [S].
4. **추측 B (two-point 극값성)**
   - 극한 범함수를 Schur-concavity로 다루면 정확한 상수가 확정된다.
5. **따름정리**
   - dividend는 ½log₂ln(1/ε) + c이고, c → +0.221이다.
   - damping은 ε를 ε_eff로 바꾸는 것뿐이다. 다만 damping은 목적함수 자체를 바꾸므로, 이 상한은 damped H에 대한 gap이다.

### 판정 실험 (CPU + 약 2 GPU-day)

- **모델**: Llama-3-8B, Qwen3-8B의 모든 linear 층(224개).
- **Hessian**: Wikitext-2에서 m ∈ {32, 512} 시퀀스로 구한다. 여기에 token-starved 조건(128/512 토큰)과 O&P가 건너뛴 singular 층을 더한다.
- **비교군(arm)**: GPTQ+Haar/RHT, WaterSIC, PCA 기저 GPTQ를 같은 damped H(δ ∈ {1e-4, 1e-2, 1e-1}·mean) 위에서 비교한다. 지표는 같은 distortion에서의 entropy-coded rate다.
- **측정**
  - 실측 gap
  - Eq. 33 예측 gap: log 영역 DP로 O(n²)이므로 n = 14336도 CPU에서 계산된다.
  - ε_eff
- **합성 단계**: two-point 스펙트럼, n ∈ {400, 1000}, Haar 100회.
- **성공 기준**
  - Spearman ≥ 0.7
  - 예측과 실측의 차이 ≤ 0.05 bit
  - 1% damping에서 일부 층이 0.3 bit를 넘음
- **실패 기준**
  - δ ≥ 1e-2인 모든 층에서 gap < 0.3 bit이면 실무적 의미가 없다.
  - 반증 기준 (i)이나 (ii)가 나오면 가설을 기각한다.

### 수치 확인

`scripts/new_ideas/idea03_check.py` (numpy, seed 고정, 24초)

- **A. Eq. 33 대 Haar MC** (n = 200)
  - ε = 1e-1 / 1e-2 / 1e-3 / 1e-4 / 1e-6에서 Eq. 33은 0.045 / 0.466 / 1.381 / 2.568 / 5.289, Haar MC는 0.048 / 0.472 / 1.384 / 2.567 / 5.284다.
  - 차이는 0.006 bit 이내다.
- **B. two-point sup**
  - Eq. 33(n = 1000)은 1.375 / 2.566 / 5.277 / 8.175다(ε = 1e-3 / 1e-4 / 1e-6 / 1e-8). 선도항 극한 1.332 / 2.556 / 5.272 / 8.169와 0.04 bit 이내로 맞는다.
  - 잔차는 −0.862 / −0.885 / −0.906 / −0.915로, −0.943을 향해 느리게 수렴한다.
  - 1−f*는 0.308 / 0.227 / 0.149 / 0.111이고, 예측값 2/ln(1/ε)는 0.290 / 0.217 / 0.145 / 0.109다.
  - ε = 1e-2에서는 극한(0.309)과 n = 1000 값(0.464)이 어긋난다. 따라서 점근식은 ε ≤ 1e-4에서만 정량적으로 쓸 만하다.
- **C. dividend − ½log₂ln(1/ε)**
  - ε = 1e-3에서 1e-20까지 0.145 / 0.164 / 0.184 / 0.194 / 0.203 / 0.211로, 이론값 +0.221로 수렴한다.
- **D. QuIP(μ = 1) sup − 회전 sup**
  - 0.536 / 0.495 / 0.471 / 0.474 / 0.482 / 0.489이고, 극한은 0.500이다.
  - 무작위 스펙트럼 1500개에서 max log₂(AM(D_Eq33)/AM(√λ)²) = +0.0000이었다(등호는 평탄한 스펙트럼).
  - 소규모 Haar MC에서 max(gap − QuIP) = −0.016이었다.
  - 추측 A의 수치적 근거일 뿐, 증명은 아니다.
- **E. 국소 탐색** (n = 40, ε = 1e-4)
  - 탐색 최댓값은 2.588로 two-point 2.630보다 작다. 끝점은 모두 {ε, 1}에 모였다.
  - 추측 B에 대한 약한 지지다.
- **F. 합성 near-singular 공분산** (n = 1024, 1% damping)
  - massive channel이 없을 때 gap은 0.743(128 토큰, Haar MC 0.766) / 0.497(512) / 0.186(16384)이다.
  - massive channel이 있을 때(λ_max/mean ≈ 940)는 0.095 / 0.072 / 0.057이다. δ가 mean 기준이라 bulk에 비해 커지기 때문이다.
  - ε_eff가 1e-5에서 1e-4 사이일 때 최악 상한은 2.5-3.9 bit로, 실제 gap보다 3-40배 느슨하다.
- **판정**
  - 지지: 점근식, dividend, ½ bit 차이, token 부족 시 gap 증가.
  - 반박: massive channel이 gap을 키운다는 주장.
  - 확인된 한계: damping 상한은 상한일 뿐이고 실제 gap을 예측하지 못한다.

### 평가

- **신규성: 3/5 (보통).** 명시된 열린 문제에 답하는 연구다. 그러나 "unbounded"와 rate 자체는 QuIP Lemma 2와 Specht ratio에서 거의 바로 나온다. 실제로 남는 기여는 다음 세 가지다.
  - μ-free 상한(추측 A)
  - 정확한 상수 −0.943
  - log-log dividend
- **본 트랙 가능성: 낮다.** 좁은 이론 노트로 읽힐 가능성이 크다. 또 결과가 entropy-coded 고율 rate에 관한 것이라 고정 비트폭 커널에는 그대로 이어지지 않는다. ISIT/TMLR이나 더 큰 논문의 한 절이 현실적이다.
- **위험**
  - (1) 추측 A와 B의 증명이 막히면 정확한 상수는 수치 결과로만 남는다.
  - (2) F에서 보듯 실제 층에서는 damping과 massive channel 때문에 gap이 0.1 bit 수준일 수 있고, 그러면 실무적 동기가 약하다.
  - (3) Eq. 33의 concentration을 엄밀하게 다루려면 free probability 기술이 필요하다.
- **선점 위험**: 가장 높은 쪽은 문제를 낸 O&P 본인들이다(Eq. 33과 footnote 3를 이미 보유). 그다음은 CurveTQ와 WaterKron 저자들이다.
- **실행 판단**: 착수한다면 추측 A(일반 n)를 먼저 검증해야 한다. 이것이 가장 깔끔한 새 lemma이다.

---

## 새 아이디어 4. FP4 역전파 RL의 Wgrad에서 토큰 축 공유 순열로 블록 스케일 결합 끊기

### 가설

**설정.** 기울기에 SR을 쓰는 FP4-backward RL(native NVFP4/MXFP4, TRIAGE·HiFloat4형)을 대상으로 한다. Wgrad = dYᵀX의 축소 축은 토큰이고, 두 operand 모두 토큰 축으로 1×16 또는 1×32 블록 양자화된다.

**방법.**
- 매 step마다 dY와 X에 같은 토큰 순열 P를 적용한다. (PdY)ᵀ(PX) = dYᵀX이므로 GEMM 결과는 정확히 같다.
- 정렬 키는 ‖dY_t‖이며, row-wise Dgrad 양자화가 이미 계산한 per-token amax를 재사용한다.
- gather는 transpose-quantize 커널에 fuse한다.

**수치 확인 후 수정.** kill test가 제안한 leverage 키 ‖dY_t‖‖X_t‖와 "꼬리 지수 α<2이면 정렬이 RHT보다 낫다"는 주장은 아래 확인에서 지지되지 않았다. 그래서 예측을 다음 셋으로 좁힌다.
- **(P1)** ‖dY_t‖ 하위 50% 토큰의 RTN zero 비율이 자연 순서 대비 3배 이상 줄어든다.
- **(P2)** X에 토큰 outlier가 있는 층(예: down_proj)에서 Wgrad 상대 MSE가 RHT16보다 낮다. 그 밖의 층에서는 RHT 대비 악화가 5% 이내다.
- **(P3)** 같은 처리량에서 BF16 대비 reward·정확도 격차가 SR+RHT보다 seed 잡음 이상으로 줄어든다.

**실패 기준.**
- 실제 GRPO 내부 층 어디에서도 RHT 대비 MSE 개선이 10% 이상인 층군이 없고, 동시에 (P3)이 3 seed 잡음 안에 머물면 기각한다.
- 주요 층군 중 하나에서라도 ‖dY‖ 정렬이 RHT보다 10% 넘게 나쁘면 "RHT 대체" 주장을 철회한다.

### 문헌상 공백과 선행 연구 위치

- [PermuQuant 2605.09503](https://arxiv.org/abs/2605.09503)
  - Prop. 3.2는 μ² 내림차순 정렬 뒤 연속 그룹으로 나누는 방식이 Σ_k max μ_i²를 최소화함을 보인다.
  - Thm. 3.3은 E(P_sort) ≤ C log₂(2g)·min_P E(P)를 준다.
  - 범위는 channel 축, 정적, PTQ, RTN이며, RHT와 비교하는 이론은 없다. [V]
- [Training Transformers with 4-bit Integers 2306.11987](https://arxiv.org/abs/2306.11987)
  - 61–62행: "gradients of a few tokens are extremely large".
  - 432행: leverage score ‖∇Y_i‖‖X_i‖로 token row를 sampling한다. 해법은 재배열이 아니라 선택이다. [V]
- [HALO 2501.02625](https://arxiv.org/abs/2501.02625): 토큰 축 Hadamard를 쓴다. [V]
- [Pretraining LLMs with NVFP4 2509.25149](https://arxiv.org/abs/2509.25149): 568행에서 Wgrad 입력에 16×16 RHT를 건다. [V]
- [Format-Aware Fusion 2610.00053](https://arxiv.org/abs/2610.00053)
  - 647–650행: "One outlier can set the scale for a whole block ... In Wgrad, this can keep smaller contributions from underflowing to zero".
  - 해법은 32-point Hadamard다. [V]
- [Transforms for LLM Quantization 2608.25188](https://arxiv.org/abs/2608.25188)
  - 3588–3591행: "What a permutation buys is control over grouping".
  - 5337–5360행: 기울기 row sparsity에는 "selection, or a rotation on the row axis"를 쓴다고 한다.
  - 토큰 순열은 제안하지 않는다. [V]
- [TRIAGE 2610.07043](https://arxiv.org/abs/2610.07043): 1406행에서 NVFP4 RL에 RHT와 기울기 SR을 쓴다. [V]
- [HiFloat4 RL 2607.26515](https://arxiv.org/abs/2607.26515): forward와 backward를 모두 4-bit로 하는 RL이다. [V]
- [MXFP4 RL error decomposition 2605.20402](https://arxiv.org/abs/2605.20402): 564–566행에서 deadzone을 forward에서만 분석한다. [V]
- [Quartet II 2601.22813](https://arxiv.org/abs/2601.22813): Table 1이 N(0,1) 입력만 다룬다. [V]
- 인접 연구:
  - [PAROAttention 2506.16054](https://arxiv.org/abs/2506.16054): 추론 단계의 정적 토큰 reorder.
  - [PRQuant 2609.22106](https://arxiv.org/abs/2609.22106): channel permutation.
  - [FP4 MoE on Hopper 2603.02731](https://arxiv.org/abs/2603.02731): row→column 변환을 하지만 reorder는 없다. [V]

**남은 것.** Wgrad 축소 축에 매 step 동적 공유 순열을 적용한 뒤 블록 양자화하는 학습 방법은 이번 검색(kill test 쿼리 30회 이상, 추가 쿼리 3회)에서 발견되지 않았다. 남은 기여는 셋이다.
- 토큰 축 SR 오차 분석
- operand 간 outlier 결합에서 RHT가 일으키는 악화와, 정렬을 통한 분리
- RL 내부 층 기울기 꼬리의 실측과 종단 결과

[2506.01939](https://arxiv.org/abs/2506.01939)는 엔트로피 이질성만 보였다. 따라서 RL에서 기울기 norm이 이질적이라는 전제는 [S]다.

### 수학 (증명 가능한 것과 추측)

**교과서 수준.**
- 공유 순열 불변성.
- SR 원소 분산은 (x−lo)(hi−x) ≤ Δ²/4이다.
- |x| < 0.5s(E2M1 첫 격자점 아래)에서는 분산이 ≈0.5s|x|로, |x|에 선형이다.
- 블록 최댓값 원소는 격자점 6s 위에 놓이므로, 스케일 반올림을 무시하면 오차가 0이다.

**이미 발표됨.** 분리 모델 |dY_{t,i}| = a_t c_i에서 상계 Σ_b n_b max a_t²를 최소화하는 것은 정렬 후 연속 분할이다. PermuQuant Prop. 3.2와 같은 교환 논증이므로 새 결과로 주장하지 않는다.

**새로 쉽게 증명 가능(미작성) — 명제 A.**
- E‖ΔW‖² ≈ Σ_b Σ_{t≠t*_b} v(a_t, M_b)‖x_t‖² + (X 항)이다. 여기서 v ≤ min(Δ_b²/4, c·a_t·Δ_b)이다.
- 블록 최댓값은 공짜이고, 작은 원소의 비용은 선형이다. 그래서 위 상계는 heavy tail에서 느슨하다.
- 정렬하면 최상위 블록에 a_(1)…a_(16)이 모인다. 이들의 비 ≈16^{1/α}가 여전히 크기 때문에(α=0.75에서 약 40), α≲1이면 무작위 배치가 더 나을 수 있다.

**새로 쉽게 증명 가능(미작성) — 명제 B.**
- X 오차 항은 Σ_t ‖dY_t‖²·v_X(b_t, max_b b)이다.
- 따라서 핵심은 큰 dY 토큰과 X-outlier 토큰을 같은 블록에 두지 않는 것이다.
- leverage 곱 키는 두 종류의 토큰을 같은 블록으로 모은다.
- 블록 내 RHT는 X-outlier를 16개 토큰 전체에 퍼뜨린다.

**추측.** RHT와 정렬의 우열을 가르는 것은 α 하나가 아니다. 두 operand 토큰 outlier 사이의 상관, 그리고 a_t가 이봉(bimodal)인지가 가른다. 원래의 α-crossover 정리는 철회한다.

### 판정 실험 (약 1–1.5 GPU-주)

**0단계: 측정 (1 GPU-일).**
- Qwen2.5-1.5B GRPO(MATH/GSM8K)에서 층 유형별 ‖dY_t‖와 ‖X_t‖의 꼬리 지수(Hill 추정)를 잰다.
- a_t의 이봉성, 그리고 두 operand outlier 사이의 순위 상관도 잰다.

**1단계: 오프라인 오차.**
- 고정 batch 32개, NVFP4 에뮬레이션(SR과 RTN).
- arm: 자연 순서, RHT16, sort_‖dY‖, sort_leverage, sort+RHT, leader 배치.
- 지표: 층별 상대 MSE, cosine, 하위 토큰 zero 비율.

**2단계: 종단.**
- 300 step, 3 seed.
- arm: BF16, SR+RHT, SR+sort_‖dY‖, 혼합(down_proj만 정렬).
- 대조군: 125M pretraining.
- argsort와 gather의 오버헤드가 3% 이하인지 microbenchmark로 확인한다.

**판정.** 성공은 (P1)–(P3)을 모두 충족하는 것이고, 실패는 가설 절의 기준을 따른다.

### 수치 확인

**스크립트와 설정.**
- 경로: `/home/user/Find-New-Idea/scripts/new_ideas/idea04_check.py` (약 25초).
- 크기: T=2048, M=K=32, 블록 16, 8 trial.
- 양자화: E2M1 + E4M3 ceil 스케일.
- 데이터: dY = a_t c_i g, X = b_t g'.
- 아래 표의 값은 RHT16 대비 상대 MSE 비율이다.

**X plain (b=1)**

| a_t | sort_dy/RHT, SR | sort_dy/RHT, RTN | 비고 |
|---|---|---|---|
| α=0.75 | 1.195 | 1.238 | sort_lev 1.323, sort+RHT 1.360 (SR), 로그비 표준오차 17.9% |
| α=1 | 1.083 | 1.072 | |
| α=1.5 | 0.956 | 0.914 | |
| α=2 | 0.984 | 0.908 | |
| α=3 | 1.074 | — | |
| α=5 | 0.997 | — | |
| flat | 1.066 | — | |
| mix (5%가 30배) | 0.895 | 0.804 | |

- sort+RHT/RHT 비율은 0.83–1.64다.
- 하위 50% 토큰의 RTN zero 비율(자연 순서 → sort_dy): α=0.75에서 0.693→0.080, α=1에서 0.504→0.071, mix에서 0.391→0.068, flat에서 0.077→0.069.
- leader 배치는 SR에서 0.98–1.05다.

**X massive (256 토큰마다 b=50), SR**
- 자연 순서 0.184, RHT16 1.242 (α=0.75).
- sort_dy/RHT: α=0.75에서 0.033, α=1.5에서 0.152, α=2에서 0.443.
- sort_lev/RHT: 1.4–11.8.
- sort_lev+RHT/RHT: 최대 47.9.

**해석.**
- **(a) 반박.** α≤1에서는 정렬이 SR에서 8–20%, RTN에서 7–24% 나쁘다(표준오차 약 18%로 경계선). α 1.5–5에서는 차이가 ±10% 안이다. mix에서만 10–20% 개선된다.
- **(b) 반박.**
- **(d) 강하게 지지.** zero 비율이 최대 8.7배 줄었다. 다만 이 감소는 MSE에 거의 반영되지 않는다.
- **leverage 키 반박.**
- **massive 조건에서 ‖dY‖ 정렬의 우위.** RHT 대비 6.6–30배 낫다(α≤1.5). 다만 합성 가정이 "행 전체 50배, dY와 독립"이다. 실제 massive activation은 channel에 국한되므로 이 효과는 과장되었을 수 있다 [S].

### 평가

**신규성.** 2.5/5(보통 이하). kill test의 3/5에서 낮췄다. lemma(PermuQuant)와 키(Xi et al.)는 이미 발표되었고, α-crossover는 합성 데이터에서 성립하지 않았다. 남은 것은 두 가지다.
- 명제 B: operand 간 outlier 결합으로 RHT가 악화되는 현상과 ‖dY‖ 정렬을 통한 분리
- 작은 기울기 토큰의 zeroing 보호

**main track 가능성.** 냉정하게 10–15%다. 경계선에 오르려면 실제 RL 모델에서 dY-heavy 토큰과 X-outlier 토큰이 분리되어 있고 RHT가 실제로 손해를 본다는 실측이 필요하고, 종단 이득도 나와야 한다. 그렇지 않으면 workshop이나 systems track 수준이다.

**위험.**
1. 실측 분포에서 MSE 이득이 ±5% 안에 그칠 수 있다.
2. massive 결과가 합성 가정의 산물일 수 있다.
3. FP4-RL 격차의 주원인이 rollout mismatch나 forward deadzone일 수 있다(2605.20402).
4. argsort 비용.

**선점 위험.** 한 줄짜리 확장이라 높다. 가능한 주체는 NVIDIA TE 팀(2509.25149), 2610.00053 저자, PermuQuant 저자, TRIAGE 저자, 2608.25188 저자다.

---

## 새 아이디어 5. KL 앵커·teacher 양자화: 노이즈가 아니라 길이·내용 의존 보상 shaping

### 가설
설정은 BF16 actor에 저정밀 KL anchor(reference) 또는 OPD teacher π_q를 붙인 경우로 한정한다. 앵커를 π_q로 바꾸면 목적함수는 FP 목적함수에 shaping −β·η가 더해진 것과 같다. 여기서 η = Σ_t[log π_ref − log π_q]이다. 효과의 형태는 KL을 어떻게 구현하느냐에 따라 갈린다.
- (H1, k1-in-reward와 OPD) shaping은 세 성분으로 나뉜다: 길이에 선형인 성분(평균 ≈ T·KL_tok), 시퀀스 내용에 의존하는 계통 성분, 토큰 수준 요동. 이 shaping은 W4/NVFP4 앵커에서 β ≥ 1e-2이면 GRPO advantage에서 이진 보상과 비슷한 크기가 된다. 그 결과 응답 길이가 줄고 엔트로피가 오른다. FP8 앵커는 β ≤ 1e-2에서 무해하다.
- (H2, GRPO k3-as-loss, verl `use_kl_loss`) 시퀀스 수준 길이 shaping이 없다. 남는 효과는 토큰별 forward KL로 π_q 쪽에 당기는 힘뿐이므로 엔트로피는 바뀌어도 길이 변화는 H1보다 작다.
- 기각 기준: (a) W4/NVFP4 앵커, β=4e-2, k1-in-reward에서 길이·엔트로피가 BF16 앵커 대비 시드 잡음(2σ) 이내이면 H1 기각. (b) FP8 앵커가 β ≤ 1e-2에서 시드 잡음 밖으로 벗어나면 "FP8 무해" 기각. (c) k3-loss arm의 길이 변화가 k1-in-reward arm의 절반을 넘으면 H2 기각.

### 문헌상 공백과 선행 연구 위치
- [Jet-RL 2601.14243](https://arxiv.org/abs/2601.14243): FP8 on-policy RL에서 KL loss 계수를 1e-3으로 둔다(l.1054). FP8 덕분에 "1.80× speedup in the reference model inference"를 얻었다고 보고한다(l.1313-1314) [V]. 즉 양자화 앵커와 활성 KL을 함께 쓰는 실무 설정은 이미 존재한다. 다만 그 선택에 대한 분석이나 ablation은 없다.
- [QeRL 2510.11696](https://arxiv.org/abs/2510.11696): "quantization noise increases policy entropy"(l.32, l.218)를 보인다. 양자화 분포가 더 평탄하다는 전제는 이 논문이 이미 제공한다. 학습은 "without using entropy or KL losses"로 한다(l.1739) [V].
- [Comedy of Estimators 2512.21852](https://arxiv.org/abs/2512.21852) (COLM 2026): k3-in-loss의 기울기가 "토큰별 forward KL 최소화와 동등"함을 보인다(§3.2 l.936-943, §F.4 l.2485-2491) [V]. 아래 명제 3은 이 결과에 π_ref → π_q를 대입한 것일 뿐이다. 이 논문은 reference 정밀도를 다루지 않는다 [V].
- [FP8-RL 2601.18150](https://arxiv.org/abs/2601.18150), [AIS 2605.13907](https://arxiv.org/abs/2605.13907), [CIS 2609.32444](https://arxiv.org/abs/2609.32444), [QUADS 2607.15810](https://arxiv.org/abs/2607.15810), [ACRL 2607.24062](https://arxiv.org/abs/2607.24062): 롤아웃과 trainer 사이의 불일치만 다룬다. AIS는 "the reference model ... remain in BF16"이라고 밝힌다(l.1269-1270) [V].
- [2609.22870](https://arxiv.org/abs/2609.22870): "We do not apply any KL divergence loss"(l.2506-2507) [V]. [QuRL 2602.13953](https://arxiv.org/abs/2602.13953)은 k3 KL 계수 1e-3(l.134-135)을 쓰지만 reference는 양자화하지 않는다 [V]. [QaRL 2604.07853](https://arxiv.org/abs/2604.07853)은 `actor.use_kl_loss` 설정만 나열하고(l.1468) reference 양자화 분석은 없다 [V].
- [OnlineQAT 2610.09346](https://arxiv.org/abs/2610.09346), [2609.26708](https://arxiv.org/abs/2609.26708), [QAD 2601.20088](https://arxiv.org/abs/2601.20088): FP teacher에서 양자화 student로 가는 반대 방향이다 [V]. [Nemotron 3 Ultra 2606.15007](https://arxiv.org/abs/2606.15007)의 MOPD는 teacher 정밀도를 언급하지 않는다 [V].
- 남은 것: 이번 검색(arXiv 쿼리 26회, WebSearch 4회)에서 다음은 발견되지 않았다. 앵커/teacher 양자화를 shaping으로 분해한 측정(형식 × β × KL 구현), 그로 인한 길이·엔트로피 drift의 실측, 양자화 teacher OPD의 형식별 정확도 손실, 그리고 값싼 보정법.

### 수학 (증명 가능한 것과 추측)
- 명제 1 (교과서): KL(π‖π_q) = KL(π‖π_ref) + E_π[η]이다. 따라서 max E[r] − βKL(π‖π_q)의 최적해는 π* ∝ π_q·e^{r/β}이다. 새로운 기여가 아니다.
- 명제 2 (표준적인 2차 근사): π = π_ref이고 섭동이 작으면 E[η_t | s] = KL(π_ref‖π_q)(s)이고 Var(η_t | s) ≈ 2KL(s)이다. 분산은 Var(η) = Tσ_w² + T²σ_b²로 쓸 수 있다. 여기서 σ_b²는 시퀀스 간 토큰 평균 KL의 분산이다. 따라서 sd(η)의 T 지수는 0.5에서 1 사이에 놓이고, "√T 법칙"은 σ_b = 0일 때만 성립한다.
- 명제 3 (2512.21852 §F.4의 직접 따름): on-policy에서 k3-in-loss의 기대 기울기는 β·Σ_t E_{y<t∼π_θ}[∇_θ KL(π_q(·|s_t)‖π_θ(·|s_t))]이다. prefix 분포에는 score-function 항이 없으므로 시퀀스 길이를 보상하거나 벌하는 경로가 없다. 보상이 없으면 문맥마다 π_θ = π_q가 고정점이다. 새것은 "양자화 앵커에서 k1-in-reward와 k3-loss가 서로 다른 편향을 낳는다"는 관찰뿐이다.
- 추측 1: RL이 정책을 확신도 높은 토큰 쪽으로 날카롭게 만들면, 그런 토큰에서는 log π_ref − log π_q > 0이다(양자화가 top-prob를 낮추기 때문). 이 경우 k1-in-reward와 OPD에서는 최적해에서도 길이 축소 압력이 남는다. 지금 있는 증거는 장난감 MDP의 정확해뿐이다(수치 확인 B).
- 추측 2: 양자화 teacher로 OPD를 하면 student 정확도의 상한은 π_T^q의 정확도가 된다. 수렴점이 π_T^q라는 것은 사실이지만, 실제 정확도 손실의 크기는 측정해야 한다.

### 판정 실험 (약 3–4 H100 GPU·주)
- 0단계(수 GPU·시간): Qwen3-1.7B 롤아웃에서 BF16/FP8/W4A16-GPTQ/NVFP4 앵커별로 KL_tok, E[η]/T, σ_b, 그리고 GRPO 그룹 내 shaping/보상 sd 비율을 실측한다. 이것으로 어느 arm이 움직일지 사전 예측한다.
- 1단계: Qwen3-1.7B GRPO를 MATH에서 최대 4k 토큰으로 150 step 돌린다.
  - 주 실험: 앵커 {BF16, FP8, NVFP4 또는 W4} × β {1e-2, 4e-2} × {k1-in-reward, k3-loss}, 시드 2개.
  - null 확인: β=1e-3에서 BF16 대 NVFP4.
  - 측정: 응답 길이, 토큰 엔트로피, 정확도, 실측 E[η]/T, π_q > π_ref인 토큰 빈도.
- 2단계: Qwen3-8B teacher(BF16 대 NVFP4/W4)로 Qwen3-1.7B student를 OPD하고 MATH/AIME 정확도를 비교한다.
- 보정 arm [S]: 길이 선형 성분만 제거한 shaping −β(η − κ̂T)를 쓴다. κ̂는 0단계 측정값이다.
- 성공 기준: 0단계 비율 ≥ 1인 arm만 움직이고 ≤ 0.1인 arm은 움직이지 않는다. 기각 기준은 "가설"의 (a)–(c)와 같다.

### 수치 확인
스크립트는 `scripts/new_ideas/idea05_check.py`이고 실행 시간은 약 42초다. KL_tok 값(FP8 1e-3, W4 2e-2)은 가정값이며 실측이 아니다.
- A1: Var(η_t)/(2KL)은 FP8에서 0.992, W4에서 0.945로, 2차 근사 법칙을 지지한다.
- A2: 평균의 T 지수는 0.97–1.00으로 선형이다. sd의 지수는 독립 모델에서 0.483이지만, 시퀀스 상관 모델에서는 FP8 0.572, W4 0.882다. T=8192에서 평균/sd는 W4 독립 12.1, W4 상관 2.5다. √T 법칙은 상관이 있으면 깨진다.
- A3: G=8 그룹, Bernoulli(0.5) 보상, 상관 모델에서 shaping sd/보상 sd는 다음과 같다.
  - W4: β=1e-3에서 0.129, β=1e-2에서 1.215, β=4e-2에서 4.878.
  - FP8: β=1e-3에서 0.009, β=1e-2에서 0.087, β=4e-2에서 0.327.
  - 길이로 설명되는 분산 비율은 W4 0.52–0.55, FP8 0.22–0.28뿐이다.
  - 따라서 kill test의 "길이 선형 성분이 주효과이고 요동은 부차적"이라는 주장은 지지되지 않는다. 내용 의존 계통 성분이 같은 크기로 존재하므로 가설을 세 성분으로 고쳤다. "FP8은 어떤 β에서도 무해"도 β=4e-2에서 0.33이 나와 β ≤ 1e-2로 좁혔다.
- B (정확해 장난감, 보상이 길이와 무관한 경우): EOS 확률을 그대로 두고 앵커만 평탄화했다(KL/tok 0 → 0.0052). 그러자 최적 길이가 41.1에서 25.0으로 줄고 엔트로피가 0.114에서 0.168로 올랐다. 메커니즘은 지지되지만, 길이가 β와 무관하게 정해지는 장난감의 특성이 섞여 있다.
- C: k3 손실의 Monte Carlo 기울기와 π_θ − π_q의 cosine은 1.0000, 최대 오차는 9.7e-5(스케일 6.3e-2)로 명제 3을 확인한다.

### 평가
- 참신성: 2/5 (낮음). 명제 1은 교과서 내용이고, 명제 3은 2512.21852의 대입이다. 기여는 순수하게 측정이다.
- main-track 가능성: 단독으로는 낮다(워크숍 수준). 실무 기본값(FP8 앵커, β=1e-3, Jet-RL)이 무해하다고 스스로 예측하기 때문이다. NVFP4 앵커나 teacher가 실제로 길이·정확도를 깎는다는 측정과 보정법이 함께 있으면 FP4-RL 시스템 논문의 한 절이 될 수 있다.
- 최대 위험:
  1. 실제 KL_tok가 가정보다 작아서 모든 arm이 null이 나오는 경우.
  2. 주류 레시피가 β=0(DAPO, QeRL, CIS)으로 가고 있어 적용 범위가 좁은 경우.
  3. k3-loss에서는 길이 효과가 없어서, 가장 많이 쓰는 스택에서 효과가 작은 경우.
- 선점 가능 주체: Jet-RL·QeRL 팀(FP8/NVFP4 ref ablation 한 줄로 끝낼 수 있음), verl/NeMo-RL의 FP4 RL 작업, OnlineQAT·2609.26708 저자(teacher 양자화 방향으로 뒤집기만 하면 됨), Comedy of Estimators 저자.

---

## 새 아이디어 6. 오차 피드백을 반영한 WUSH: 주변(marginal) 블록 대신 Schur 보수(pivot) 블록으로 만든 AbsMax 최적 블록 변환

### 가설

이 가설은 kill test를 통과한 좁은 형태만 다룬다.

**(H1) weight-only.** 설정은 다음과 같다.
- per-group AbsMax 양자화 (INT4, MXFP4/NVFP4)
- group 크기 = block 크기인 GPTQ식 block 오차 피드백(block SIC)

이 설정에서 WUSH Algorithm 2가 변환을 만들 때 쓰는 입력 `MX(i) ← diagonal blocks of H`(주변 블록 H_gg)를 pivot 블록 D_g = (L_gg L_ggᵀ)^{-1} = H_gg − H_{g,>g} H_{>g,>g}^{-1} H_{>g,g}로 바꾼다(pivot-WUSH). 바꾸면 layer proxy tr(ΔWᵀHΔW)가 WUSH(+GPTQ)보다 낮아진다. 개선 폭은 층별 inter-group 상관, 즉 ‖H_gg − D_g‖/‖H_gg‖가 클수록 커진다.

**(H2) W4A4.** 단순 치환은 W4A4에서 오히려 나쁠 것으로 예상한다(아래 수치 확인 참조). 그래서 활성화 오차는 주변 Σ_gg로, 가중치 오차는 D_g로 가중하는 비대칭 목적함수를 따로 풀어야 한다.

**실패 기준.**
- (H1): Llama-3-8B와 Qwen3-8B에서 q/k/v, gate/up 층의 proxy 개선이 WUSH+GPTQ 대비 평균 2% 미만이다.
- 또는 개선량과 층별 gap 지표의 Spearman 상관이 0.3 미만이다.
- 또는 MXFP4/INT4 g32 weight-only에서 Wiki2 PPL 변화가 seed 간 편차 안에 들어간다.

셋 중 하나라도 해당하면 H1은 기각한다. (H2)에서 비대칭 해가 WUSH-marginal보다 낫지 않으면 W4A4 확장은 포기한다.

### 문헌상 공백과 선행 연구 위치

- [WUSH: Near-Optimal Adaptive Transforms for LLM Quantization 2512.00956](https://arxiv.org/abs/2512.00956) (v2 2026-05-30, ICML 2026). Algorithm 2 "GPTQ case"(p.3)는 다음과 같다. [V]
  - `MX(i) ← diagonal blocks of H`로 변환을 만든다.
  - `LL⊤ ← Cholesky(H^{-1})`로 이미 pivot (L_ii L_iiᵀ)^{-1} = D_i를 계산한다. 이를 intra-block GPTQ Hessian H(i) = T(L_ii L_iiᵀ)^{-1}Tᵀ로 쓴다.
  - MW(i)는 피드백으로 갱신된 가중치의 2차 모멘트이므로, 변환은 처리 순서와 이미 결합되어 있다.
  - 그러므로 제안은 **이미 공개된 알고리즘에서 이미 계산된 양으로 한 줄을 바꾸는 것**이다.
  - Theorem 4.1(최적 변환)의 증명은 확률적 unbiased 양자화 모델 위에서만 이루어진다(Eq. 12–14). 논문 스스로 이 모델을 "tractable surrogate"라 부른다.
  - 모든 GPTQ 결과는 W4A4다. 초록 기준 GPTQ 개선은 MXFP4에서 Hadamard 대비 +0.7점이다.
- [SchurQuant 2608.15567](https://arxiv.org/abs/2608.15567) (2026-08-16). 이 논문은 "group을 raw block G_cc로 평가하지만 suffix가 오차를 흡수할 수 있으니, Schur complement가 정확한 곡률"이라고 말한다(본문 69–70행). 같은 통찰을 scale/zero-point refit과 integer code 선택에 적용했다. 변환에는 적용하지 않았다. [V]
- [CurveTQ 2610.09212](https://arxiv.org/abs/2610.09212). Viterbi metric에 e⊤He = Σ_b η_bᵀ D_b η_b를 쓰며(284–285행), Eq. 2에서는 diag(D_b)만 사용한다. 회전/변환은 쓰지 않는다. [V]
- [Ordentlich–Polyanskiy 2605.13768](https://arxiv.org/abs/2605.13768). 이 논문의 결과는 세 가지다. [V]
  - GPTQ = SIC다(Lemma 1).
  - WaterSIC 격차는 pivot들의 AM/GM이다.
  - 회전은 pivot의 AM을 GM 쪽으로 당긴다.
  - 단, 스칼라·엔트로피 부호화 설정이며 AbsMax가 아니다.
- [변환 서베이 2608.25188](https://arxiv.org/abs/2608.25188). 다음 두 문장이 공백을 정의한다. [V]
  - Thm 4.9 서술(2095–2104행): WUSH는 "unbiased noise"를 가정하고 GPTQ로는 "advantage transfers empirically"라고만 한다.
  - §13 open problem 1(6949–6957행): "jointly optimal (T, Q) pair" 이론이 없다.
- [Chen et al. 2507.18553](https://arxiv.org/abs/2507.18553). GPTQ = Babai이며, basis reduction은 future work로 남겼다. [V]
- 함께 확인했으나 겹치지 않은 것: WUSH-KV 2609.38121, Cherniuk 2609.11687, JARQ 2609.38599, OJBKQ 2602.08376, HeRo-Q, MR-GPTQ(block Hadamard, 웹으로만 확인). [V/S]
- 인접 고전 분야: MIMO의 GMD/UCD precoder + SIC 순서(V-BLAST)는 "변환과 순서의 공동 선택"이라는 틀 자체의 선행 연구다. [S]

**남은 것.** 이번 검색(arXiv 쿼리 17회 + 웹 검색 2회, 2026-10-11 재검색 1회)에서 발견되지 않은 것은 다음 넷이다.
- (a) block SIC 하에서 AbsMax 왜곡이 Σ_g E‖δ_g‖²_{D_g}로 분해된다는 사실을 WUSH 변환 최적화와 연결한 명제
- (b) D_g로 만든 WUSH가 weight-only 확률 모델 안에서 최적이라는 증명
- (c) W4A4 비대칭 bilateral 최적해
- (d) 순서와 T의 공동 선택, 그리고 개선량과 inter-group 상관의 연결에 대한 실측

vLLM, torchao, llm-compressor에도 구현이 없다(웹 확인). [S]

### 수학 (증명 가능한 것과 추측)

**설정.** 기호는 다음과 같다.
- W ∈ R^{din×dout}, H = Σ, loss = tr(ΔWᵀΣΔW)
- group g는 연속한 d개 입력 채널이고, 처리 순서는 1..G다.
- group g를 변환 T_g 아래에서 양자화하면 원좌표 오차는 δ_g = T_gᵀ q(T_g^{-ᵀ} w_g) − w_g다.
- 피드백은 W_{>g} ← W_{>g} − Σ_{>g,>g}^{-1} Σ_{>g,g} δ_g다.

**Lemma A (교과서).** 최종 loss는 정확히 Σ_g tr(δ_gᵀ D_g δ_g)다.
- block LDL 분해의 대수적 항등식이며, 근사가 아니다.
- QuIP/LDLQ와 O–P Lemma 1의 블록판이다.
- 수치로도 상대오차 1e-16 수준에서 확인했다(아래). 새롭지 않다.

**Lemma B (block 합동 동변성, 교과서).** block-diagonal T에 대해 변환 좌표의 pivot은 T_g D_g T_gᵀ다. WUSH Alg. 2의 H(i)가 바로 이 양이다. [V]

**Proposition C (새로움, 증명은 짧음).** 가정은 다음과 같다.
1. WUSH의 확률 모델: unbiased AbsMax 잡음, 열들을 i.i.d. 표본으로 취급한다.
2. group 내부 피드백이 없다(block 단위 SIC).
3. 고해상도 근사: δ_g가 후속 group의 2차 모멘트 M_W(h>g)를 바꾸는 효과를 무시한다.

결론: 각 단계의 기대 왜곡은 WUSH Eq. 13의 가중치 쪽 항에서 X′X′ᵀ = M_X를 D_g로 바꾼 것과 같다. 따라서 Theorem 4.1을 그대로 적용하면, D_g의 Cholesky로 만든 WUSH가 FP에서는 정확히 최적이다. INT에서는 Theorem 4.1이 주는 오차 인수 범위(Gaussian/Laplacian에서 d·o(1), 일반 분포에서 d배) 안에서 최적이다.

이 명제의 기여는 최적성 주장의 범위를 RTN에서 "AbsMax + block 피드백"으로 넓히는 데까지다. 가정 3을 빼면 단계별(greedy) 최적일 뿐 전역 최적은 아니다.

**비대칭 W4A4 (미해결, 추측).** 목적함수는 다음과 같다.

E‖C_gᵀ T_W^{-1} ε(T_W w)‖²(가중치 항, C_g C_gᵀ = D_g) + E‖W′ᵀ T_X^{-1} ε(T_X x)‖²(활성화 항, x의 분포는 주변 Σ_gg)

두 항이 서로 다른 행렬(D_g와 Σ_gg)을 쓰므로, WUSH의 "두 항 각각의 최적해가 T_W = T_X^{-ᵀ}로 일치한다"는 성질이 깨진다. 단일 SVD 닫힌 형태는 성립하지 않는다.

추측: 최적해는 D_g와 Σ_gg 사이의 일반화 고유문제(generalized eigenproblem) 해에 Hadamard를 붙인 형태다. 증명은 없다.

**순서 (추측).** D_g는 순서에 의존한다. 다음 두 기준은 휴리스틱이며 최적성을 주장하지 않는다.
- det(D_g) 기준 greedy 순서
- 변환 후 group 내 pivot의 AM/GM 기준

### 판정 실험 (약 4 GPU-일, H100 1장 기준)

**모델.** Llama-3-8B, Qwen3-8B. 보조로 WUSH 원 논문 설정인 Llama-3.1-8B-Instruct를 쓴다.

**형식.**
- weight-only: MXFP4(g32), NVFP4(g16), INT4 g32
- W4A4: MXFP4

**Arms.**
- GPTQ + {none, block Hadamard(MR-GPTQ식), WUSH-marginal(원 Alg. 2), WUSH-pivot}
- 순서 ablation: 자연 순서 / act-order / det(D_g) greedy. 이로써 "순서 기여"와 "변환 기여"를 2×2로 분리한다.
- W4A4에서는 위 arm에 비대칭 해 후보를 추가한다.

**측정.**
- 층별 proxy tr(ΔWᵀHΔW)
- 층별 gap 지표 ‖H_gg − D_g‖_F/‖H_gg‖_F
- Wiki2/C4 PPL, lm-eval 평균
- seed 3개

**성공 기준.**
- q/k/v, gate/up에서 proxy가 WUSH 대비 5% 이상 낮다.
- 개선량과 gap 지표의 Spearman 상관이 0.5 이상이다.
- MXFP4 weight-only에서 WUSH→BF16 PPL 격차의 20% 이상을 회수한다.

**실패 기준.** 가설 절의 기준과 같다.
- proxy 개선이 2% 미만이다.
- 상관이 0.3 미만이다.
- PPL 변화가 노이즈 수준이다.

**사전 점검(1시간).** 먼저 실제 층의 gap 지표 분포를 잰다. 대부분 층에서 0.3 미만이면 아래 수치 확인 기준으로 기대 이득이 2% 미만이다. 이 경우 실험 전에 접는다.

### 수치 확인

**스크립트.** `scripts/new_ideas/idea06_check.py`, CPU 29초, seed 0–2.

**합성 설정.**
- din = 512 (32×16 group), dout = 256
- Σ = (1−ρ)·비등방 block-diagonal + ρ·공유 rank-8 인자
- 가중치는 t(4) 분포에 채널 스케일을 곱했다.
- 실측 LLM 값이 아니다.

**Lemma A 확인.**
- full loss = 9.868934e+02, pivot 합 = 9.868934e+02, 상대차 1.15e-16 (blk)
- gptq 변형: 7.710276e+02 = 7.710276e+02, 상대차 1.47e-16

**weight-only 결과.** 값은 tr(W⊤ΣW)로 정규화한 proxy이며, 마지막 값이 pivot-WUSH의 marginal-WUSH 대비 개선율이다.

| 형식 | 변형 | ρ | gap | Had | WUSH-marg | WUSH-piv | 개선 |
|---|---|---|---|---|---|---|---|
| INT4 | blk | 0.0 | 0.000 | 7.970e-3 | 5.590e-3 | 5.590e-3 | +0.00% |
| INT4 | blk | 0.3 | 0.476 | 5.696e-3 | 4.197e-3 | 4.011e-3 | +4.48% |
| INT4 | blk | 0.6 | 0.823 | 3.426e-3 | 2.761e-3 | 2.367e-3 | +14.23% |
| INT4 | blk | 0.9 | 0.976 | 1.143e-3 | 1.039e-3 | 6.996e-4 | +32.70% |

같은 ρ = 0.0/0.3/0.6/0.9에서 다른 변형의 개선율:
- INT4, Alg. 2식 intra-block GPTQ: +0.00 / +1.60 / +8.46 / +25.00%
- FP4 E2M1, blk: +0.00 / +3.91 / +16.10 / +35.94%
- FP4 E2M1, gptq: +0.00 / +1.55 / +10.80 / +28.51%

ρ = 0.9인 INT4 gptq에서는 WUSH-marginal(6.809e-4)이 Hadamard(6.553e-4)보다도 나빴다. 주변 블록으로 만든 변환이 피드백 아래에서 해가 될 수 있다는 뜻이다.

**W4A4 결과.** 가중치는 blk-SIC, 활성화는 RTN이며 같은 변환 쌍을 썼다. 값은 출력 상대 MSE 기준 pivot-WUSH의 marginal-WUSH 대비 개선율이다.

| 형식 | ρ = 0.0 | ρ = 0.6 | ρ = 0.9 |
|---|---|---|---|
| INT4 | +0.00% | −19.82% | −89.71% |
| FP4 | +0.00% | −15.65% | −68.56% |

**해석.**
- H1의 방향은 지지된다. 이득이 0에서 단조 증가하고 gap 지표와 함께 커진다.
- 실제 피드백 구조(intra-block GPTQ)에서는 이득이 약 3분의 1에서 절반가량 줄어든다.
- 단순 치환은 W4A4에서 크게 나빠진다. kill test의 지적("W4A4에서 verbatim 전이 불가")이 수치로 확인되었고, H2는 반드시 별도 해가 필요하다.
- 이득의 크기는 전적으로 합성 ρ에 달려 있다. 실제 LLM 층의 gap이 이 범위 어디에 있는지는 모른다. 결론적으로 원 가설의 "verbatim 전이"는 weight-only로만 약화해 유지한다.

### 평가

**신규성: 3/5 (보통).**
- weight-only 부분은 WUSH Alg. 2의 한 줄 수정이다. 그 줄에 들어갈 양(D_i)을 Alg. 2가 이미 계산하고 있다.
- SchurQuant가 같은 "주변 대신 Schur 곡률" 원리를 scale/code 선택에 이미 적용했다.
- 남은 신규성은 세 가지뿐이다.
  - Proposition C의 명시와 증명
  - W4A4 비대칭 최적해
  - 순서–변환 공동 선택

**메인트랙 가능성: 낮음.**
- weight-only 결과만으로는 리뷰어가 "incremental refinement"로 볼 것이 거의 확실하다. WUSH 자체의 GPTQ 이득이 Hadamard 대비 +0.7점에 불과해, 남은 개선 여지가 작다.
- 비대칭 W4A4 해가 닫힌 형태나 증명 가능한 bound로 나오고, 실측에서 WUSH를 넘어야 한다. 그러면 서베이 open problem 1(공동 최적 (T, Q))에 대한 부분 답이 되어 중간 정도 가능성이 생긴다.
- 현재 W4A4에 대해서는 추측만 있고, 합성 실험은 단순판이 실패함을 보였다.

**최대 위험.**
1. 실제 LLM 층에서 gap 지표가 작아, 이득이 2% 미만으로 사라질 수 있다.
2. Proposition C의 가정 2(group 내부 피드백 없음)는 WUSH Alg. 2의 실제 구조가 아니다. 그 구조에서는 정리가 성립하지 않고 경험적 이득만 남는다.
3. 4-bit에서는 고해상도·unbiased 가정이 약하다.
4. W4A4 비대칭 문제가 닫힌 형태를 갖지 않을 수 있다.

**선점 위험.** 다음 순서로 높다.
- WUSH 저자군(IST Austria, Alistarh 그룹): 코드에 D_i가 이미 있고 WUSH-KV로 후속 연구를 이어 가는 중이다. 가장 유력하다.
- SchurQuant 저자들
- O–P(pivot과 회전의 관계를 이미 분석)
- 이 셋 모두 몇 주 안에 같은 수정을 시도할 수 있는 위치다.

---

## 새 아이디어 7. 저비트 키 프록시 기반 top-k 선택의 레짐 의존 사후 질량 랭킹: 크로스오버 분석과 비가우시안 경험적 베이즈 사전

### 가설

**설정.** 사후(post-hoc) 저비트 키 프록시로 토큰이나 페이지를 고르는 경우만 다룬다. 대상은 1–2 bit RaBitQ나 sign sketch, 저비트 Quest bound, 양자화 K 위의 H2O 축출, recompute 선택이다. 이 경우 프록시 점수 ŝ_i = s_i + δ_i의 토큰별 잡음 σ_i를 저장된 scale과 norm에서 계산할 수 있다. QAT로 학습한 indexer(DSA 계열)는 제외한다. 그 점수는 학습된 ranker 자체이고, 정의된 s_i를 잡음 섞어 추정한 값이 아니기 때문이다.

**주장.** 질의마다 경험적 베이즈(EB) 사전을 적합하고 사후 기대 질량 E[e^{s_i} | ŝ_i]로 순위를 매기면 점추정 ŝ 랭킹보다 낫다.
- 보정의 부호는 레짐에 따라 바뀐다. 가우시안 사전에서 잡음 큰 토큰은 크로스오버 ŝ = μ + τ²/2 위에서는 강등되고 아래에서는 승격된다.
- 따라서 이미 출판된 고정 보정 ŝ − σ²/2(Jensen debias)를 랭커로 그대로 쓰면, 예산 경계가 크로스오버 아래에 있는 레짐(날카로운 head, 큰 예산)에서는 오히려 해롭다.
- logit 상단 꼬리가 두꺼우면 가우시안 사전이 과수축한다. 이때는 NPMLE 같은 비가우시안 사전이 필요하다.

**정량 예측.** 수치 확인 결과에 맞춰 원래 주장보다 약하게 잡는다.
- 이득이 의미 있는 크기(질량 ≥0.3pp, 또는 gap 회복률 ≥0.3)가 되는 것은 σ²/τ² ≳ 0.2이고 예산이 ≤5%일 때다.
- σ²/τ² ≤ 0.05이거나 FP8·INT8 수준(≈1e-3)이면 이득은 0이다. 이것을 음성 대조로 쓴다.

**실패 기준.** 아래 셋 중 하나면 가설을 기각한다.
1. 1-bit 프록시, 1–5% 예산에서 posterior 랭킹이 point-estimate recall과 exact recall 사이 gap의 30% 미만만 회복하고, RULER나 LongBench 이득도 1점 미만이다.
2. 크로스오버 아래 레짐에서 EB가 Jensen-debias 랭커보다 낫지 않다.
3. 이득이 σ_i의 분산(dispersion)과 상관이 없다.

### 문헌상 공백과 선행 연구 위치

**선택 규칙과 보정 항.**
- [Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution 2510.00636](https://arxiv.org/abs/2510.00636): Eq. 7(l.325-343)이 z_i = exp(μ_qᵀk_i/√d + k_iᵀΣ_q k_i/(2d))를 "moment-generating function of a Gaussian"으로 유도하고, 이 값으로 KV를 순위 매겨 축출한다 [V]. 원 가설의 정리(top-k by m+v/2)와 순위 규칙이 같다. 불확실성의 원천만 다르다(미래 query vs 양자화 잡음). 수축(shrinkage)이나 사전 적합은 없다.
- [Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in Video Diffusion 2605.26266](https://arxiv.org/abs/2605.26266): l.286-300에서 키 양자화 잡음의 "Jensen bias" E[Ẑ_S] ≥ Z_S를 보이고, l.382-393에서 b_i ≈ σ_i²/2 보정을 CGF로 정당화한다 [V]. dense attention용이며 top-k 선택이나 수축은 다루지 않는다.

**통계학의 표준 결과.**
- [Making the cut: improved ranking and selection for large-scale inference 1312.5776](https://arxiv.org/abs/1312.5776): 초록(l.13)에 "local maximum likelihood (ML) approaches tend to favor units with high measurement error"라고 쓰고, EB 랭킹으로 이를 고친다 [V]. 원래의 "winner's curse" 주장은 여기서 이미 교과서적 결과다.
- [Invidious Comparisons: Ranking and Selection as Compound Decisions 2012.12550](https://arxiv.org/abs/2012.12550): 이질적 정밀도 아래 선택을 compound decision으로 놓고 Kiefer–Wolfowitz NPMLE 사전을 쓴다(l.17, l.222-224) [V]. 비가우시안 사전이라는 우리 쪽 요소도 통계학에는 이미 있다.

**top-k 및 KV 선택 쪽 문헌.**
- [UNIQUE: Universal Top-k Sparse Attention for Training-free Inference and Sparsity-aware Training 2605.27740](https://arxiv.org/abs/2605.27740): page score = q·mean + λ‖q‖·std (l.205, l.340) [V]. 휴리스틱 분산 보너스이고, 양자화 잡음과는 무관하다.
- [RaBitQCache: Rotated Binary Quantization for KVCache in Long Context LLM Inference 2606.31519](https://arxiv.org/abs/2606.31519): unbiased estimator를 강조하고(l.28, l.113) 점추정으로 순위와 threshold를 정한다 [V]. 선택 편향 분석은 없다.
- [Fathom 2609.17652](https://arxiv.org/abs/2609.17652)는 분산을 bit-plane 깊이에만 쓰고 추정치로 순위를 정한다 [V, note 기준]. [SAKI 2608.03228](https://arxiv.org/abs/2608.03228) Prop. 2(l.279)는 margin 조건(2ε < γ_k) 아래 top-k 정확성만 보인다 [V, note 기준]. [Self-Indexing KVCache 2603.14224](https://arxiv.org/abs/2603.14224), [AMEND 2609.09823](https://arxiv.org/abs/2609.09823), [Runtime-Certified Bounded-Error Quantized Attention 2605.20868](https://arxiv.org/abs/2605.20868), [WitCert 2607.28699](https://arxiv.org/abs/2607.28699)는 오차 상계·인증·margin audit이며 posterior/수축 랭킹은 없다 [S, note 요약 기준].
- [Spend Bytes on Breadth: Precision-Count Trade-offs for Decode-Time KV Compression in Long Chain-of-Thought Reasoning 2610.05685](https://arxiv.org/abs/2610.05685): 10% 보존 시 oracle도 미래 attention 질량의 약 13%를 잃는다(l.26-27, l.436-437). 저자들은 이를 "a floor that no scorer can remove"(l.434)라고 부른다 [V]. 이것은 선택 오류가 섞이지 않은 oracle 하한이므로 "선택 오류가 곧 질량 손실"이라는 근거로 쓸 수 없다.
- [DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression 2609.19969](https://arxiv.org/abs/2609.19969): l.1002에서 indexer의 FP4 query와 key에 QAT를 쓴다고 밝힌다 [V]. DSA 계열을 이 모델의 적용 대상에서 빼는 근거다. V3.2의 FP8 indexer 주장은 2차 블로그로만 확인된다 [S].

**남은 공백.** 다음 조합은 이번 검색(쿼리 약 23회: kill test 20회와 본 절 작성 중 arXiv 3회)에서 발견되지 않았다.
- 질의별로 적합한 EB 사전(가우시안 또는 NPMLE) 아래 E[e^{s}|ŝ]로 순위를 매긴다.
- 대상은 사후 양자화 키 프록시의 top-k 선택이다.
- 크로스오버 ŝ = μ + τ²/2를 기준으로 예산과 head 날카로움에 따른 부호 반전을 분석한다.

본 절 작성 중 추가 검색에서 "posterior bias"라는 표현을 쓰는 pre-hoc sparsity 논문(Gao et al., 2026)이 걸렸다. 초록을 보면 선택 시점을 앞당기는 방법이고 EB 랭킹과는 무관하다 [S, 초록만 확인].

### 수학 (증명 가능한 것과 추측)

**명제 1 (교과서, 자명).**
- 가정: s_i ~ N(μ, τ²), ŝ_i | s_i ~ N(s_i, σ_i²)이고 토큰끼리 독립이다.
- 사후분포: s_i | ŝ_i ~ N(m_i, v_i). 여기서 m_i = (τ²ŝ_i + σ_i²μ)/(τ²+σ_i²), v_i = τ²σ_i²/(τ²+σ_i²)이다.
- 결론: 기대 질량 E Σ_{i∈S} e^{s_i}를 최대화하는 크기 k 집합은 m_i + v_i/2 기준 top-k다. 기대값의 선형성과 lognormal 평균에서 바로 나온다.
- 이 순위 규칙은 2510.00636 Eq. 7과 같은 형태다. 따름정리 E[s−ŝ | ŝ] = −σ²(ŝ−μ)/(τ²+σ²)는 Tweedie 수축이다. **새로운 것 없음.**

**명제 2 (크로스오버; 대수적으로는 자명하지만 이 문맥에서 지적된 적은 없음).**
- (m_i + v_i/2) − ŝ_i = σ_i²/(τ²+σ_i²) · (μ − ŝ_i + τ²/2).
- 따라서 잡음이 큰 토큰은 ŝ_i > μ + τ²/2이면 강등되고 그 아래면 승격된다.
- 예산 분위 z_k에서 경계는 대략 μ + τ z_k에 있다. 그래서 z_k > τ/2이면(작은 예산이거나 평평한 head) winner's curse 방향이 되고, z_k < τ/2이면(날카로운 head이거나 큰 예산) 반대 방향이 된다.
- 따름정리: 고정 보정 ŝ − σ²/2(2605.26266의 dense 보정을 랭커로 전용한 것)는 경계가 크로스오버 아래에 있을 때 부호가 틀린다. **새로운 관찰은 이 부분이다.** 수학적 깊이는 얕다.

**추측 1.**
- 꼬리가 두꺼운 logit 분포(sink나 heavy hitter)에서는 가우시안 EB가 상단 꼬리를 과수축해 naive보다 나빠질 수 있다.
- NPMLE 사전은 이를 피하며, 이때 크로스오버는 점 하나가 아니라 ŝ에 대한 함수 log E[e^s|ŝ] − ŝ의 영점 집합이 된다.
- 증명은 없고 합성 실험으로만 지지된다.

**추측 2.**
- RaBitQ나 Hadamard처럼 공유 회전을 쓰면 토큰 간 오차가 상관되어 독립 가정이 깨진다.
- 이 상관이 이득을 얼마나 줄이는지는 분석하지 않았다.

### 판정 실험 (3–5 GPU-days)

**설정.**
- 모델: Llama-3.1-8B, Qwen3-8B.
- 과제: RULER와 LongBench, 32–128K.
- 선택 키:
  - RaBitQ 1-bit와 sign sketch: σ_i는 1-bit 추정량 분산식에서 얻는다.
  - 2-bit per-token INT: σ_i² ≈ ‖q‖²Δ_i²/12.
  - 2-bit KIVI per-channel.
  - 음성 대조로 FP8과 INT8.
- 선택 방식: Quest식 page top-k와 token top-k, 예산 1/2/5/10%. 추가로 양자화 K 위 H2O 축출.

**비교 arm.**
- (a) ŝ
- (b) ŝ − σ²/2 (Jensen debias 랭커)
- (c) ŝ + σ²/2 (flat prior)
- (d) 가우시안 EB m + v/2
- (e) NPMLE EB log E[e^s|ŝ]
- (f) exact top-k

**측정.**
- 지표: 포착한 exact 질량, oracle recall, 과제 정확도.
- 층과 head별로 σ²/τ²와 "경계 − 크로스오버" 값을 기록하고, 이득과의 상관을 본다.

**성공 기준.** 다음을 모두 만족해야 한다.
- 1-bit, 1–5% 예산에서 (e)가 (a) 대비 gap 회복률 ≥0.3, 과제 정확도 +1점 이상.
- 크로스오버 아래 레짐(경계 < μ + τ²/2)의 head에서 (e)가 (b)보다 낫다.
- FP8과 INT8에서 이득이 0이다.

**실패.** 위 조건 중 하나라도 어기거나, 이득이 σ 분산과 무상관이면 실패다.

### 수치 확인

**스크립트.** `scripts/new_ideas/idea07_check.py`. numpy, seed 20261011, 실행 56초.
- N = 4096, σ_i는 lognormal(disp 0.5), 질의 150개(NPMLE는 30개).
- 표기: Δ는 naive 대비 질량 변화, g는 gap 회복률.

**C1 크로스오버.** 수치 계산과 닫힌형이 모든 점에서 일치한다. τ = 3일 때 ŝ = 4.5에서 보정이 +0.0000이 되고, 7.0에서는 σ² = 4이면 −0.7692다. **지지.**

**C2 음성 대조.** σ²/τ² = 0.001에서 모든 arm의 Δ는 |Δ| ≤ 0.001pp다. 등분산 대조(disp = 0)에서도 gEB − naive = 0.000이다. **지지.**

**C3 방향.** τ = 3에서 "경계 − (μ+τ²/2)"와 "naive 선택 평균 σ − gEB 선택 평균 σ"는 1%: +3.24 / +0.427, 10%: −0.31 / −0.023, 20%: −1.78 / −0.089. 8개 설정 모두 부호가 예측과 일치했다. **지지.**

**C4 크기.** 원래 예측보다 약하다.
- σ²/τ² = 0.05에서 gEB의 Δ는 최대 +0.139pp(τ = 3, 1%)이고 g는 ≤ 0.14다. "0.05 이상이면 non-negligible"이라는 원래 문턱은 **기각**한다.
- g ≥ 0.3을 넘는 설정은 다음 넷뿐이다. 모두 1% 예산이다. 10% 예산에서 g는 ≤ 0.16이다.
  - τ = 1, r = 0.2: g = 0.377
  - τ = 1, r = 1.0: g = 0.479
  - τ = 3, r = 1.0: g = 0.351 (Δ = +15.2pp)
  - heavy 사전, r = 1.0, npEB: g = 0.320
- gEB가 recall을 오히려 낮추는 경우도 있다(τ = 3, r = 1, 20%: 0.591 vs naive 0.614). 질량 최적과 recall 최적은 다르다.

**Jensen-debias 랭커와의 비교 (핵심 위협).**
- 작은 예산에서는 ŝ − σ²/2가 gEB 이득의 대부분을 이미 얻는다: τ = 3, r = 1, 1%에서 +14.69 vs +15.25pp; τ = 1, r = 1, 5%에서 +2.10 vs +2.51pp.
- 경계가 크로스오버 아래인 레짐에서는 (b)가 크게 해롭고 gEB는 손해가 없다: τ = 3, r = 1, 10%에서 −9.38 vs +0.08pp; τ = 3, r = 0.2, 10%에서 −1.54 vs −0.02pp.
- flat prior (c)는 일관되게 파국적이다(최대 −46.7pp).

**C5 꼬리가 두꺼운 경우.** 사전은 bulk N(0,1)에 2% spike N(5, 1.5²)를 섞었다.
- 가우시안 EB가 naive보다 나쁘다: r = 0.2, 1%에서 −2.10pp, r = 1, 1%에서 −2.21pp.
- NPMLE는 r = 0.2, 1%에서 −0.016pp(본전), r = 1, 1%에서 +3.10pp(g = 0.32)다.
- 가우시안 데이터에서는 NPMLE가 gEB와 거의 같다(+2.44 vs +2.47pp). **추측 1을 지지.**

**결론.**
- 크로스오버, 음성 대조, 비가우시안 사전의 필요성은 지지된다.
- "이득이 크다"는 주장은 σ²/τ² ≳ 0.2이면서 예산이 ≤5%인 경우로 좁혀야 한다.
- 크로스오버 위 레짐에서 EB가 출판된 Jensen 보정을 넘어서 더하는 몫은 0.4–0.6pp 수준이다.

### 평가

**신규성: 2/5 (낮음).**
- 순위 규칙은 2510.00636, σ²/2 보정은 2605.26266, EB 랭킹과 NPMLE는 1312.5776과 2012.12550에 이미 있다.
- 남는 기여는 세 가지다. (i) 크로스오버와, 그로부터 고정 Jensen 보정이 레짐에 따라 해롭다는 진단. (ii) 사후 저비트 프록시 선택에 NPMLE EB를 적용한 것. (iii) 1–2 bit에서 실제로 얻는 이득의 측정.

**메인 트랙 가능성: 낮음 (대략 10–15%) [S].**
- 이론은 두 줄 결합이라 이론 기여로는 통과하지 못한다.
- 합성 결과에서 큰 이득은 1% 예산과 σ²/τ² ≈ 1이라는 극단에 몰려 있다. 실제 1-bit RaBitQ의 σ²/τ²가 이 범위인지는 측정되지 않았다 [S].
- 10% 예산에서는 이득이 거의 없고, oracle 하한(약 13% 손실) 자체가 이득의 상한을 누른다.
- 현실적인 경로는 둘이다. 크로스오버 진단과 1-bit 엔드투엔드 +1점을 모두 보이면 workshop이나 short paper. 아니면 RaBitQCache나 Self-Indexing 계열 논문의 한 절(ablation)로 흡수되는 것이다.

**가장 큰 위험.**
1. 실제 모델의 σ²/τ²가 0.05 미만이면 이득이 0이다.
2. 공유 회전에서 오는 토큰 간 오차 상관을 아직 분석하지 않았다.
3. 실제 head들이 대부분 크로스오버 근처에 있으면 Jensen 랭커와 EB의 차이가 사라진다.
4. 질량 이득이 RULER 정확도로 이어지지 않을 수 있다. recall이 오히려 떨어지는 사례가 있다.

**선점할 수 있는 쪽.**
- Expected Attention 저자들(Devoto, Jeblick, Jegou): MGF 점수에 양자화 분산 항만 더하면 된다.
- Quantized Keys Steal Attention 저자들(Tuncer et al.): Jensen 보정을 top-k 선택으로 확장하기만 하면 된다.
- RaBitQCache나 Fathom 팀: 이미 분산 추정치를 가지고 있다.

셋 모두 한 번의 후속 작업으로 이 아이디어를 흡수할 수 있는 위치에 있다.

---

## 새 아이디어 8. QeRL형 양자화 RL의 그룹 공유 섭동: 측정, 그리고 양자화기 무작위화를 롤아웃마다 다시 뽑기

### 가설

**설정.** QeRL형 NVFP4-LoRA GRPO에서는 두 섭동 모두 한 그룹의 G개 rollout 전부에 똑같이 걸린다.
- 고정된 quantization error.
- AQN noise. 논문은 "resampled for each forward pass"라고 쓰지만, 공식 코드는 optimizer step마다 한 번만 뽑는다.

**예측.** 같은 섭동 분포에서 rollout chunk마다 draw를 새로 뽑으면 zero-advantage group 비율이 떨어진다. 다시 뽑을 대상은 AQN draw, 또는 양자화기 자체의 무작위성(random Hadamard seed, stochastic-rounding seed, block-scale dither)이다.
- 감소폭은 측정한 across-draw Var(p')에 2605.07689 Prop. 2의 convexity bound(인용할 뿐, 우리 기여가 아님)를 적용한 값과 일치해야 한다.
- 같은 섭동 분포에서 다시 뽑으므로 marginal p̄과 평균 token entropy는 자동으로 맞춰진다. 이 조건에서 shared AQN보다 최종 정확도가 높아야 한다.
- 별도 예측: QeRL과 entropy를 맞춘 temperature-only 대조군이 BF16 LoRA 대비 GSM8K +2.7 이득의 대부분(≥70%)을 재현한다.

**실패 기준.** 다음 중 하나라도 해당하면 가설을 기각한다.
- (i) 실측 Var(p')로 예측한 제거 가능 gap이 0.5pp 미만이다.
- (ii) chunk 재추출 arm의 zero-advantage 감소가 예측의 95% CI 밖에 있다.
- (iii) 3 seed에서 정확도 차이가 seed std 이내다.

### 문헌상 공백과 선행 연구 위치

- [Gradient Starvation in Binary-Reward GRPO 2605.07689](https://arxiv.org/abs/2605.07689), Proposition 2 (본문 205–224행).
  - D(p,G)=p^G+(1−p)^G가 볼록함을 보이고 Jensen을 적용해 D_real ≥ D_iid를 얻는다. 2차 Taylor로 하한 D_iid + G(G−1)/2·Var(p_x)·min_p[p^{G−2}+(1−p)^{G−2}]도 준다.
  - 실측: G=2에서 D_real=0.90, D_iid=0.56 [V].
  - 원래 T09의 "수학 핵심"은 이 정리에서 확률변수 이름만 prompt에서 섭동 draw로 바꾼 것이다.
- [Parameter Exploration for RLVR via Variational Learning 2608.09805](https://arxiv.org/abs/2608.09805) (3PO).
  - B3PO는 "samples a single perturbation θ̂ per batch and reuses it across all prompts and groups"(312행)이다. 이것이 정확히 group-shared 체제다.
  - C3PO는 "divides each GRPO group across N independently sampled weights"(102–103행)이다.
  - 초록은 "fewer zero-advantage groups"라고 보고한다. §5.3은 N∈{1,2,4,8,16}을 ablation해 "all runs with N ≥ 2 outperform the single-perturbation baseline"을 얻는다 [V].
  - Polaris, EntReg, KL-Cov와도 비교한다. 다만 wall-clock 오버헤드가 약 1.5×이고(1086·1425행), 양자화는 언급하지 않는다 [V].
  - 따라서 "shared보다 per-chunk가 낫다"는 이미 실증된 결과다.
- [PSN-RLVR 2602.02555](https://arxiv.org/abs/2602.02555): "θ̃ is held fixed for the entire rollout"(419–428행). per-rollout이 아니라 group-shared 설계다 [V].
- [Are Verifier Errors Independent Within a GRPO Group? 2609.06386](https://arxiv.org/abs/2609.06386): Table 3에서 degenerate 비율이 p^k+(1−p)^k 기준의 6.3–6.5×다. 원인으로 within-group 상관과 prompt 이질성을 든다. 7B는 AWQ 모델이지만 양자화 효과는 분리하지 않는다 [V].
- [QeRL 2510.11696](https://arxiv.org/abs/2510.11696).
  - 563행: "A key limitation of quantization errors is their deterministic nature". 585행: "resampled for each forward pass". σ_start=1e-2이고 exponential decay를 쓴다 [V].
  - OpenReview 재반론 표(Qwen2.5-7B GSM8K): BF16 LoRA 88.1, +AQN 89.7, NVFP4 LoRA 88.5, +AQN 90.8. 모두 single run이고 std는 ±0.9–1.2다 [V].
  - 공식 코드: `noise_scheduler.py::generate_gaussian_noise`가 vLLM RMSNorm weight에 `add_`로 draw를 한 번 더하고, step이 바뀔 때만 호출된다(`grpo_trainer.py` 1364–1369, 1542–1544행). layernorm이 다시 로드되지 않으면 noise가 random walk처럼 누적될 가능성도 있다 [V, kill test의 코드 열람].
- [MXFP4 error decomposition 2605.20402](https://arxiv.org/abs/2605.20402): "grid noise raises the policy's entropy". shared와 per-sample을 구분하지 않는다 [V].

**남은 것**은 모두 양자화에 특화된 경험적 항목이다.
- (a) QeRL의 논문과 코드 불일치(group-shared이고 누적될 수 있음)를 문서화하는 것.
- (b) AQN draw 간 Var(p')를 실측하고 bound의 예측력을 검증하는 것.
- (c) NVFP4-LoRA 파이프라인 안에서 shared AQN, per-chunk AQN, temperature를 대조하는 것.
- (d) 양자화기 자체의 무작위화(rotation/SR seed, scale dither)를 rollout마다 다시 뽑는 것.

(d)는 이번 검색(arXiv API 쿼리 약 40회, WebSearch 3회, OpenReview 검색 포함)에서 발견되지 않았다. 이 문장은 "선행 연구 없음"의 증명이 아니다.

### 수학 (증명 가능한 것과 추측)

prompt x와 섭동 draw ξ에 대해 p′=p_x(ξ), p̄_x=E_ξ p′라 두고 D(p)=p^G+(1−p)^G로 쓴다.

- **명제 1 (교과서적 결과, 2605.07689 Prop. 2를 재적용).**
  - 가정: G개 보상이 draw를 조건으로 독립인 Bernoulli(p′)이고, 재추출은 같은 분포에서 iid로 한다.
  - 결론: shared에서는 E_ξ D(p′) ≥ D(p̄_x)이다. D″ ≥ G(G−1)2^{3−G}이므로 엄밀한 하한은 gap ≥ G(G−1)2^{2−G}·Var(p′)이다.
  - ½G(G−1)Var(p′)(p̄^{G−2}+(1−p̄)^{G−2})는 근사이지 bound가 아니다.
- **분해 (자명하지만 실험 설계에 핵심).** E_xE_ξD ≥ E_xD(p̄_x) ≥ D(E_x p̄_x)이다.
  - rollout별 재추출로 없앨 수 있는 것은 첫 번째 gap뿐이다.
  - prompt 이질성에서 오는 두 번째 gap(2609.06386의 6.3×)은 그대로 남는다.
- **따름정리 2 (chunking 단조성, power-mean 부등식 적용).** 한 그룹을 N개 chunk(크기 m=G/N, chunk마다 독립 draw)로 나누면 D_N = E_x[(E p′^m)^N + (E(1−p′)^m)^N]이다.
  - (E p′^m)^{1/m}은 m에 대해 비감소이므로 D_N은 N(G의 약수)에 대해 비증가한다.
  - 3PO는 이를 실험으로만 보였다. 정리 형태로 쓴 것은 노트에서 찾지 못했다 [S]. 다만 수준은 교과서적이므로 기여로 내세우지 않는다.
- **원 가설의 오류 정정.** deterministic error는 Var(p′)=0이므로 Jensen gap도 0이다.
  - D(p_Q)<D(p)인지는 p_Q가 p보다 ½에 가까운지에 달려 있다. 따라서 "deterministic error는 탐색 기제일 수 없다"는 성립하지 않는다.
  - "matched KL이면 strict 감소"도 정리가 아니다. KL을 맞춰도 p̄은 맞춰지지 않기 때문이다.
- **추측(검증 대상).**
  - (C1) Hadamard/SR seed를 바꿔 다시 뽑을 때의 Var(p′)가 AQN과 같은 차수다.
  - (C2) zero-advantage 감소가 최종 정확도로 이어진다.
  - (C3) QeRL 이득의 대부분은 entropy 효과다.

### 판정 실험 (약 250 GPU-시간, H100/B200 1대 기준 약 1.5주)

- **0단계, 코드 감사 (수 시간).** 공식 QeRL 코드에서 step별 RMSNorm weight와 trainer 사본의 차이를 기록해 shared 여부와 누적 여부를 확정한다.
- **1단계, Var(p′) 측정 (약 30 GPU-시간).**
  - 모델: Qwen2.5-1.5B-Instruct NVFP4-LoRA. 데이터: GSM8K 512 prompt.
  - 3개 checkpoint에서 K=32 draw × M=64 rollout을 돌린다. 이항 잡음을 보정한 추정량 s²−mean(p̂(1−p̂))/(M−1)을 쓴다.
  - AQN의 각 σ, Hadamard seed, SR seed, scale dither를 따로 측정한다.
- **2단계, GRPO 학습.** G=8, step당 128 prompt, 300 step, arm마다 3 seed.
  - Arm A: BF16 LoRA.
  - Arm B: NVFP4 LoRA.
  - Arm C: NVFP4+AQN shared(공식 코드).
  - Arm D: AQN per-chunk, N=2, 4.
  - Arm E: 양자화기 무작위화 per-chunk, N=4.
  - Arm F: BF16 LoRA + temperature. C의 평균 token entropy에 ±5% 이내로 맞춘다.
- **지표.** zero-advantage group 비율과 예측치(1단계의 Var(p′)로 계산), GSM8K/MATH500 정확도, 평균 token entropy, 메모리, throughput.
- **성공 기준.**
  - D−C의 zero-advantage 차이가 예측의 95% CI 안에 있다.
  - E ≥ C이고 차이가 2 seed-std를 넘는다.
  - E의 추가 메모리가 NVFP4 가중치 사본(N−1)개분 이하이고, throughput 손실이 C3PO의 1.5×보다 작다.
  - F가 C−A 이득의 70% 이상을 재현하면 entropy 귀속을 지지하는 것으로 본다.
- **실패 기준.** 위 '가설'의 (i)–(iii)에 하나라도 해당하면 실패다. 또 E가 per-sequence로 다른 가중치를 쓰느라 vLLM batching을 깨서 C3PO보다 느려지면 (d)의 "저비용" 주장을 철회한다.

### 수치 확인

스크립트는 `scripts/new_ideas/idea08_check.py`(numpy, seed 고정, 약 24초)다. 합성 데이터는 다음과 같다: prompt 4000개(logit은 bimodal), 섭동은 logit shift τ·z, G=8.

**1. 정확한 degeneracy와 단조성 (τ별).**

| τ | D_shared (N=1) | N=2 | N=4 | N=8 (iid) | 제거 가능 gap | Taylor 예측 | 엄밀 하한 |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.4312 | 0.4307 | 0.4305 | 0.4304 | 0.083pp | 0.083pp | 0.0185pp |
| 0.3 | 0.4327 | 0.4285 | 0.4263 | 0.4253 | 0.744pp | 0.748pp | 0.1646pp |
| 0.6 | 0.4376 | 0.4215 | 0.4130 | 0.4087 | 2.899pp | 2.952pp | 0.6316pp |
| 1.0 | 0.4494 | 0.4082 | 0.3854 | 0.3732 | 7.612pp | 7.930pp | 1.6148pp |

- 모든 τ에서 monotone=True다. 엄밀 하한은 항상 실제 gap 이하였다(LB≤gap: True).
- Taylor 예측 대비 비율은 1.00 / 1.01 / 1.02 / 1.04다.
- N=2가 제거 가능 gap의 0.54–0.57을 회수한다.
- 제거 가능 gap이 전체 초과 degeneracy에서 차지하는 비율은 0.002 / 0.018 / 0.067 / 0.172다. 전역 D_iid는 0.0078이다.

**2. Var(p′) 추정 (τ=0.3, 실제 gap 0.703pp).**

| 설정 | 보정 없는 예측 | 보정한 예측 |
|---|---|---|
| K=8, M=16 | 4.815pp | 1.238pp |
| K=16, M=32 | 2.787pp | 0.783pp |
| K=32, M=64 | 1.805pp | 0.744pp |

**3. 검정력 (2σ 기준으로 필요한 arm당 group 수).**
- τ=0.1: 약 282만 group(22,050 step).
- τ=0.3: 35,362 group(276 step).
- τ=0.6: 2,315 group.

**4. 전제 점검 (deterministic error).** p=0.95에서 logit을 −0.5 이동하면 D는 0.6634→0.5139로 낮아진다. p=0.50에서는 어느 방향으로 이동해도 0.0078→0.0229로 높아진다.

**판단.**
- 방향, 단조성, Taylor 예측력은 가설을 지지한다.
- 그러나 효과 크기는 τ²에 비례한다. 제거 가능한 몫도 prompt 이질성 대비 작다(τ≤0.3에서 2% 미만).
- 보정하지 않은 Var 추정은 gap을 최대 7배 과대평가한다. 보정해도 prompt당 512 rollout 이상이 필요하다.
- 따라서 실제 AQN 섭동의 τ가 0.1 수준이면 (b)와 (d)는 측정 불가능하거나 무의미하다. 이에 맞춰 주장을 "τ≥0.3일 때만 검출 가능"으로 약화한다. 1단계가 사실상 go/no-go 관문이다.

### 평가

- **신규성:** 2/5(낮음).
  - 정리는 2605.07689의 재적용이다.
  - shared 대 per-chunk 비교는 3PO의 B3PO 대 C3PO 결과다.
  - 신규로 남는 것은 (a) QeRL의 논문/코드 불일치와 entropy를 맞춘 대조, (d) 양자화기 seed 재추출뿐이다.
- **Main-track 가능성:** 낮음(대략 10–15%) [S].
  - (a)–(c)만으로는 분석 또는 workshop 논문 수준이다.
  - (d)가 C3PO와 동등 이상의 정확도를 더 낮은 비용으로 multi-seed에서 보여야 main track을 노릴 수 있다.
- **최대 위험.**
  - ① AQN 규모의 섭동은 τ가 작아 gap이 검출 한계 아래일 수 있다(수치 확인 3).
  - ② rollout마다 가중치가 다르면 vLLM batching이 깨지거나 NVFP4 사본이 N개 필요해져 "free"가 아니게 된다.
  - ③ activation SR처럼 token 단위 잡음은 parameter 수준의 일관된 탐색이 아니라 사실상 temperature와 같아질 수 있다.
  - ④ single-run 재반론 표에 기댄 entropy 귀속은 seed 분산에 묻힐 수 있다.
- **선점 가능 주체.**
  - 3PO 저자(Venkatkrishna, Daheim, Gurevych)가 양자화 실험을 추가하는 경우.
  - QeRL 저자(NVlabs)가 코드나 논문을 정정하면서 per-forward 재추출을 구현하는 경우.
  - 2605.20402와 2605.07689 저자 그룹.
  - 셋 다 몇 주 안에 할 수 있는 작업이다.

---

## 새 아이디어 9. PTQ 층간 오차 상쇄는 어느 부분공간에서 일어나는가: RMSNorm 반경 방향을 통제한 고유대역별 상쇄 계수와 오차 셰이핑

### 가설

원래 후보의 Wiener-가중 GPTQ는 2610.07522가, 선형-가우시안 대수는 교과서가 선점했다. 살아남은 좁은 주장만 다룬다.

- **H1 (스펙트럼 국소화)**: 사전학습된 pre-norm(RMSNorm) LLM에서 2609.11716의 counteraction ⟨Δh^(ℓ−1), Δu^(ℓ)⟩ < 0은 스펙트럼상 국소화되어 있다.
  - 은닉 오차에서 RMSNorm 반경 방향(ĥ = h/‖h‖)을 먼저 제거한다.
  - 대역별 상쇄 계수는 κ_i = −⟨P_iΔh, P_iΔu⟩ / ‖P_iΔh‖²로 정의한다. P_i는 은닉상태 공분산의 i번째 고유대역 사영이다.
  - 예측: κ_i는 저분산(다양체 법선) 대역에서 크고, 상위 PC 대역과 massive 채널 대역에서는 작다.
- **H2 (Jacobian 정합)**: 대역별 κ_i는 블록 Jacobian 대칭부 J_s를 대역에 제한한 Rayleigh 몫 −tr(P_i J_s P_i)/rank P_i와 일치한다. Pythia/OLMo 체크포인트를 따라 둘은 함께 나타난다.
- **H3 (방법)**: 회전과 비트 할당으로 가중치 양자화 오차를 자기상쇄 대역으로 밀어 넣는 error-shaping PTQ를 설계한다. 이 방법은 같은 평균 비트에서 GPTQ, QEP, GPTAQ, Activation Denoising(2610.07522)보다 종단 KL을 더 줄인다.

**실패 기준** (하나라도 해당하면 해당 가설을 기각한다)

- 반경 방향을 제거한 뒤 하위/상위 대역 κ 비율이 1.5 미만인 모델 계열이 3개 중 2개 이상이면 H1을 기각한다.
- 체크포인트 전반에서 측정 κ_i와 J_s 예측값의 Spearman 상관이 0.5 미만이면 H2를 기각한다.
- 위 셋 중 최강 기준선 대비 W3 평균 비트에서 KL 감소가 3% 미만이거나 PPL 개선이 0.05 미만이면 H3를 기각한다.
- τ² 가설은 부차적이다. 잡음 주입 arm의 하위 대역 κ 변화가 시드 표준편차 이내이면 기각하며, 이 결과는 2609.32060의 결과와 일치하게 된다.

### 문헌상 공백과 선행 연구 위치

- [Why Does Post-Training Quantization Work? 2609.11716](https://arxiv.org/abs/2609.11716) [V]
  - counteraction 음수 비율: Qwen3-32B 블록의 82.5% (l.1302)
  - App. E.6 (l.1374–1377): 음의 누적 상호작용 중 99.9%는 블록이 입력 오차에 응답한 결과이고, 가중치의 직접 효과는 0.1% 미만이다. 즉 원인이 Jacobian 수준이라는 점은 이미 이 논문에 있다.
  - App. E.5 / Fig. 14 (l.4453, l.4539): 사전학습이 진행되면서 counteraction이 생겨나는 모습을 Pythia와 OLMo 체크포인트로 보였다.
  - Limitations (l.1398–1400): "whether training stochasticity produces counteraction"을 열린 질문으로 남겼다. τ² 질문 자체는 이 논문이 먼저 던진 것이다.
  - PCA, 부분공간, 고유분해, 채널 분해는 본문에 없다(grep 결과 참고문헌에만 등장).
- [Activation Denoising 2610.07522](https://arxiv.org/abs/2610.07522) [V]
  - Thm. 1 (l.303–336): 디노이징 목적을 W̃ = WF로 바꾼 뒤 계량 G = H + αI + λ²N + λ(C+Cᵀ)에서 반올림하는 문제와 같음을 보였다.
  - ρ = 0이면 F는 "generalized Wiener filter"다. N = τ²I, C = 0을 넣으면 F = H(H+τ²I)⁻¹이 되어 후보의 연산자와 같다.
  - App. C에서 GPTQ, GPTAQ, QEP를 특수경우로 회수한다. 따라서 **Wiener-가중 GPTQ는 이미 출판된 방법이다.**
  - 방향별 분석·오차 셰이핑은 없다.
- [Depth Laws for the Precision Floor 2609.32060](https://arxiv.org/abs/2609.32060) [V]
  - GPT-2 12–48L: ρ = −0.57, α_PTQ = −0.53 (l.631)
  - l.1726–1740: Chen et al.의 상쇄 지표를 직접 측정했다. MLP에서 c = −0.01 ± 0.11, CNN 중앙값 −0.05로 "add rather than cancel"이었다.
  - 결정적 문장: "In residual MLPs QAT does not change this gain." τ² 가설에 대한 직접 반례다.
- [LM Activations Inhabit Privileged Error-Correcting Basins 2610.04183](https://arxiv.org/abs/2610.04183) [V]: forward pass가 활성을 "funnel"한다는 attractor-basin 관점. 양자화는 다루지 않는다.
- [Evidence for feature-specific error correction in LLMs 2606.24964](https://arxiv.org/abs/2606.24964) [V]: 오차 교정은 feature 방향(p > 2)에서 우선적으로 일어나고, PCA 방향은 p ≈ 2로 특권이 없다(l.143, l.522; 6개 LLM). H1의 "PCA 대역" 기저가 잘못된 기저일 수 있다는 직접 증거다.
- [Quantization Degradation: A Signal-Noise Perspective 2608.08188](https://arxiv.org/abs/2608.08188), [Training Dynamics Impact PTQ Robustness 2510.06213](https://arxiv.org/abs/2510.06213) [S]: 집계 이득 또는 학습 동역학만 다루며 방향 분해는 없다.
- 이번 점검에서 추가로 발견한 논문: PulseQuant 2609.33384 (video DiT, 초록만 확인, 노트 미수록) [V-초록]. "response-subspace correction"으로 지배적 활성 방향의 잔여 오차 성분을 줄인다. 즉 *지배 방향에서 오차를 빼내는* 셰이핑의 한 형태가 다른 도메인에 이미 있다. H3의 참신성을 더 깎는 요소다.

**남는 것**: 사전학습 LLM에서 counteraction을 고유대역과 massive 채널별로 분해하고, RMSNorm 반경 방향을 통제하고, 이를 J_s 스펙트럼과 대조한 실증 분석이다. 이 분석은 선행 연구 노트(arXiv 16회, WebSearch 2회)와 이번 추가 arXiv 검색 5회에서 발견되지 않았다. 방법 기여는 2610.07522의 Wiener 전변환과 다를 때에만 성립한다. 예를 들어 PCA 기저로 잔차 스트림을 직교 회전한 뒤 출력 행별로 비트를 할당하는 방식이다.

### 수학 (증명 가능한 것과 추측)

- **명제 A (교과서: Wiener 1949; Alain & Bengio 2014)**
  - 가정: h = x + n, x ~ N(0, Σ), n ~ N(0, τ²I)이고, 잔차 블록을 제곱오차 최소화로 학습한다.
  - 결론: h + f(h) = Σ(Σ+τ²I)⁻¹h이고 J_f = −τ²(Σ+τ²I)⁻¹ ≺ 0이다.
  - 고유방향 i에서 κ_i = τ²/(λ_i+τ²)이다.
  - 새로운 내용은 없다.
- **명제 B (자명하지만 핵심 통제 변수)**
  - 블록이 f(h) = g(h/rms(h))이면 f(ch) = f(h)이므로 J_f h = 0이다.
  - 따라서 반경 방향 오차의 κ는 블록이 무엇을 학습했든 정확히 0이다.
  - LLM의 massive 토큰에서는 ĥ ≈ e_massive이므로, "massive 방향에서는 상쇄가 없다"는 관측은 Wiener 가설과 무관하게 정규화만으로도 나온다.
- **관찰 C (전역 Wiener 모형의 결함)**
  - L개의 Wiener 사상을 합성하면 정상적인 저분산 신호가 Π λ/(λ+τ²)만큼 소멸한다.
  - 이는 LLM에서 feature가 보존되고 노름이 커지는 사실과 모순이다.
  - 따라서 정합적인 모형은 다양체의 법선 방향으로만 수축하는 모형(contractive/DAE형)이어야 한다. 이 모형의 κ 예측을 정리 형태로 쓰는 일은 **추측**이며 증명하지 않았다.
- **추측 D**: 사전학습 LLM의 J_s는 반경 방향을 제거한 저분산 대역에서 음의 정부호에 가깝고, 상위 PC 대역에서는 부호가 정해지지 않는다. 경험적 명제이며 증명 대상이 아니다.

### 판정 실험 (≈1.5 GPU·주)

1. **측정** (약 3 GPU·일)
   - 모델: Qwen3-8B, Llama-3.1-8B, OLMo-2-1B (중간 체크포인트 6개), Pythia-1.4B/2.8B (체크포인트 6개).
   - 양자화: W4/W3, RTN과 GPTQ.
   - 고유대역: 상위 1%, 1–10%, 10–50%, 나머지. 각 대역에서 반경 방향을 제거한 경우와 제거하지 않은 경우의 κ_i를 모두 잰다. massive 채널은 별도로 잰다.
   - 대조 기저: 같은 차원의 SAE feature 부분공간과 무작위 부분공간 (2606.24964 대응).
2. **Jacobian 대조** (약 2 GPU·일): JVP와 Hutchinson 추정으로 대역별 −tr(P_i J_s P_i)를 구하고, 체크포인트별로 κ_i와의 Spearman 상관을 계산한다.
3. **경계 사례** (약 1 GPU·일): 2609.32060의 잔차 MLP에서 PTQ와 QAT를 비교하는 설정을 재현한다. 대역별 κ로 집계 c ≈ 0 아래에 대역 간 상쇄가 숨어 있는지 본다.
4. **τ² arm (선택)**: 125M 모델을 2.5B 토큰으로 학습한다. 은닉 잡음 τ² ∈ {0, 0.05·RMS²} × 시드 3개.
5. **방법** (약 3 GPU·일)
   - 잔차 스트림 PCA 직교 회전을 RMSNorm과 호환되게 융합한다.
   - o_proj와 down_proj 출력 행에 비트를 할당한다: 상위 대역은 4bit, 하위 대역은 2bit, 평균 3bit와 2.5bit.
   - 비교 대상: GPTQ+QuaRot, QEP, GPTAQ, Activation Denoising.
   - 지표: 종단 KL, WikiText PPL, 0-shot 5종.
   - 성공과 실패 판정은 가설 절의 기준을 따른다.

### 수치 확인

스크립트 `/home/user/Find-New-Idea/scripts/new_ideas/idea09_check.py`는 합성 데이터를 쓰며 CPU에서 약 59초 걸린다.

- **설정**: d = 48, 멱법칙 스펙트럼(2.0·j^−1.5), massive 토큰 5%(e₀ = 30), RMSNorm 뒤 random-feature ReLU 블록(readout ridge 적합).
- **C1 (교과서)**: 적합한 κ = [0.01, 0.019, 0.039, 0.091, 0.166, 0.286, 0.5, 0.801], 이론값과의 최대 오차 0.0010. 명제 A를 확인한다.
- **C2 (대역별 κ, 반경 방향 제거)**: 시드 두 개의 값을 '/'로 나눠 적었다.

  | 학습 잡음 | top(1–4) | mid | bot(17–48) | bot/top |
  |---|---|---|---|---|
  | τ² = 0 | 0.089/0.090 | 0.035/0.036 | 0.000/0.000 | – |
  | τ² = 0.02 | 0.164/0.162 | 0.256/0.258 | 0.606/0.605 | 3.7 |
  | τ² = 0.08 | 0.344/0.341 | 0.614/0.614 | 0.976/0.974 | 2.8 |

  - J_s로 예측한 값은 0.18 / 0.269 / 0.638이고 측정값은 0.164 / 0.256 / 0.606이다.
  - 전역 Wiener 평균은 0.022 / 0.224 / 0.617이다.
- **C3 (교란 변수)**: 블록의 학습 방식과 무관하게 반경 방향의 κ는 0이다.

  | 블록 | massive 토큰의 e₀ 방향 κ | 일반 토큰의 반경 방향 κ |
  |---|---|---|
  | 잡음 학습 | −0.0012 | +0.0000 |
  | clean 학습 | −0.0000 | +0.0000 |
  | 무작위 W2 | +0.0001 | +0.0000 |

  - massive 토큰의 |cos(h, e₀)|는 0.995다.
  - 일반 토큰에서 평균 방향 에너지의 72.5%가 하위 대역에 있다. 반경 방향을 제거하지 않으면 하위 대역 κ가 과소평가된다.
- **C4 (깊이)**: 전역 Wiener를 합성했을 때 하위 대역에 남는 신호 에너지는 L = 1에서 1.59e−1, L = 8에서 2.54e−5, L = 24에서 1.70e−12다.
- **C5 (셰이핑 여유)**: 같은 에너지의 오차를 넣었을 때 ‖Δh+Δu‖²/‖Δh‖² 값은 등방 0.435, 상위 대역만 1.047, 하위 대역만 0.233이다.

**해석**

- 이 결과가 지지하는 것
  - 국소화 비율이 실패 기준 1.5를 넘는다 (3.7, 2.8).
  - J_s 예측과 측정값이 0.03 이내로 맞는다. 다만 작은 ε의 선형화이므로 이 일치는 거의 동어반복이다.
  - 반경 방향 통제가 필수라는 점(C3), 전역 Wiener 모형이 깊이에서 깨진다는 점(C4).
- 이 결과가 약화시키는 것
  - **상위 PC 대역의 κ는 0에 가깝지 않다** (0.16, 전역 Wiener의 0.022보다 7배 크다). 블록이 접공간에서 실제로 하는 계산의 Jacobian이 κ에 섞이기 때문이다. 따라서 "상위 대역 κ ≈ 0" 대신 "**하위/상위 비율 ≥ 1.5**"로 주장을 약화했다.
  - τ² = 0이면 하위 대역 κ가 0인 것은 모형의 구성상 당연한 결과다. 이것은 τ² 가설의 증거가 아니다.
  - C5의 1.9배 여유는 블록 하나, 합성 데이터 기준의 상한이다. 실제 LLM에서 가중치 오차를 하위 대역으로 보낼 수 있는지, 하위 대역 오차가 출력 KL에 얼마나 영향을 주는지는 검증하지 않았다.

### 평가

- **참신성: 2/5 (낮음).**
  - 방법(Wiener 전변환)은 2610.07522가 선점했다.
  - Jacobian 수준의 원인은 2609.11716 App. E.6에 이미 있고, τ² 질문은 같은 논문의 Limitations에 있다.
  - 오차 교정 기하는 2610.04183과 2606.24964에 있다.
  - 지배 방향 오차를 제거하는 셰이핑은 다른 도메인에 있다 (2609.33384).
  - 남는 것은 "방향 분해 + 반경 통제 + 셰이핑 PTQ" 조합뿐이다.
- **메인트랙 가능성: 낮음 (약 10–15%) [S].**
  - 측정만으로는 워크숍 수준이다. 세 계열에서의 깨끗한 국소화, Tarawneh 영(null) 결과의 설명, W3/W2에서 Activation Denoising·QEP·GPTAQ 대비 측정 가능한 이득이 모두 있어야 한다.
- **최대 위험**
  - 2606.24964가 PCA 방향은 특권이 없다고 보고했다. 교정이 고유대역이 아니라 feature 단위로 일어난다면 H1은 기저 선택 단계에서 틀린다.
  - 합성에서도 상위 대역 κ가 계산 Jacobian에 오염되었다. 실제 모델에서 비율이 1.5 근처로 흐려질 수 있다.
  - PCA 회전·행별 비트 할당은 QuaRot류 등방화와 충돌하고, 하위 대역 오차는 원래 출력 영향이 작아 KL 이득이 작을 수 있다.
- **선점 가능성이 큰 쪽**: Chen et al. (2609.11716 저자; 이미 질문을 공개했다), Scholten et al. (2610.07522; 방향별 G 해석은 바로 다음 단계다), Finlayson et al., Tarawneh.

---

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

---

## 부록 A. 선점 검증에서 상위 10개에 들지 못한 후보

| 후보 | 제목(원문) | 판정 | novelty | 가장 가까운 선행 연구(요약) |
|---|---|---|---|---|
| P-LOOP | Loop gain of error-feedback rounding: low-rate overload of GPTQ/LDLQ under fixed | 좁혀서 생존 | 2 | 2307.13304 QuIP (Jul 2023, NeurIPS'23), Sec. 5.2 + Thm 7 + Algorithm 5 + Supplement C. 2508.04853 Zhang/Zhang/Colbert/Saab (Aug 2025, v2 2026-04-09), Remark 3.7 |
| P06 | Second-moment linearity theorem: rotation-seed variance is first order in the er | 좁혀서 생존 | 2 | 2606.10929 "Recoverable but Not Stationary: Local Linear Structures in Weights and Activations" (2026-06-09): Theorem 1, Gaussian best-of-N in a local-linear ne |
| S01 | Posterior-quantile drafts: the acceptance-optimal draft under known quantization | 좁혀서 생존 | 2 | 2609.24150 Acceptance-Aware Draft Model Training (WTV/EAL losses, 2026-09-21); 2610.10411 Training Parallel Speculative Draft Models by Directly Minimizing Expe |
| S03 | Fan-out-aware precision: a bit's value scales with coherent read fan-out (prefix | 좁혀서 생존 | 2 | - RDKV, 2605.08317 (2026-05-08): the token weight is the attention summed over every query that reads it, w_t = sum_tau a_{tau,t} (Prop 3.1, l.211-219). Bits th |
| S04 | Common-mode quantization error defeats self-consistency; lazy prompt-KV quantiza | 전제 오류 | 2 | 2403.02419 Chen et al. 'Are More LLM Calls All You Need?' (Mar 2024). 2510.10964 Kim et al. 'Not All Bits Are Equal' (Oct 2025). 2411.14860 Nam et al. 'Ex Uno P |
| S07 | Dispatch vs combine quantization in expert-parallel MoE: common-mode vs independ | 좁혀서 생존 | 2 |  - **2505.09343** Insights into DeepSeek-V3 (2025-05-14, ISCA'25). Section 3.2 LogFMT: they tested combine quantization in a simulation, quantizing the residual |
| S08 | Evict or quantize? A closed-form head-wise rule from the attention tail exponent | 좁혀서 생존 | 2 | - 2605.08317, RDKV: Rate-Distortion Bit Allocation for Joint Eviction and Quantization of the KV Cache (2026-05-08). Main preemption of the theory. - 2505.19586 |
| S10 | Why route-flip harm is at chance: a tie-indifference theorem for quantized MoE s | 좁혀서 생존 | 2 | 2608.11212 Gu, 'Detecting a Route Flip Is Easier Than Knowing Whether to Fix It' (2026-07-14). 2609.12550 Huang, 'Quality-Constrained Routing over a Fixed Pool  |
| T06 | Rotation-randomness reuse in unbiased FP4 backward estimators: a persistent bias | 전제 오류 | 2 | - 2509.25149, NVIDIA, "Pretraining Large Language Models with NVFP4" (2025-09-29). App. E.4.3 and Fig. 13 ablate a new sign vector per transform vs one fixed se |
| T08 | Quantized rollouts with truncated importance sampling induce a predictable respo | 전제 오류 | 2 | 2606.00206 Lotfi et al., "Quantized Reasoning Models Think They Need to Think Longer, but They Do Not" (2026-05-29). 2609.20807 Marek & Ryabinin, "Score Centeri |
| T10 | Quantization-variance-corrected Adam preconditioner for stochastic-rounding back | 좁혀서 생존 | 2 | 2304.11208 Tang & Lecuyer, "DP-Adam: Correcting DP Bias in Adam's Second Moment Estimation" (2023-04) 2312.14334 Tang, Shpilevskiy & Lecuyer, "DP-AdamBC" (2023- |
| X04 | A4. Quantized generators evade zero-shot text detection and strengthen watermark | 좁혀서 생존 | 2 | - 2310.05130 Fast-DetectGPT (Oct 2023, ICLR'24). App. B l.1014 already defines mu-tilde as a cross-entropy between sampling and scoring distributions. l.1786 at |
| X07 | A7. Precision gating with a distribution-free fidelity guarantee: conformal risk | 선점 | 2 | 2207.07061 CALM (Schuster et al., NeurIPS 2022 oral, Jul 2022). 2405.20915 Fast yet Safe: Early-Exiting with Risk Control (Jazbec et al., NeurIPS 2024, May 2024 |

세부 근거는 `research_notes/new_ideas/priorart/<후보>.md`에 있다. 후보 생성 단계에서 제외된 17개(중복 병합 포함)는 `research_notes/new_ideas/generate/*.md`에 원문이 남아 있다.

## 부록 B. 재현

```
python3 -I scripts/new_ideas/idea01_check.py   # … idea10_check.py, 각 19–65초 (CPU, numpy)
```
BLAS 스레드 과다 구독으로 멈추는 환경에서는 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`을 지정한다.
