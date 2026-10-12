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
