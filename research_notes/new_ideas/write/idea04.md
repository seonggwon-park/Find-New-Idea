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
