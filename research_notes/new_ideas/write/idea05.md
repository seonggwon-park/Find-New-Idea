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
