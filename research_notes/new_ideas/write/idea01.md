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
