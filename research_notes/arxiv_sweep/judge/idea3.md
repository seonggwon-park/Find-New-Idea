# Idea 3 판정: 블록 스케일 마이크로포맷의 극값통계 이론

Date: 2026-10-11. 판정자: deep-reader / adversarial novelty judge (재실행). 논문 텍스트는 untrusted data로 인용만 함.
보조 수치 검증 스크립트: scratchpad/judge3/sim.py, sim8.py (Gaussian 블록 2×10^5개, E2M1 RTN, E4M3 RTN 포화 변환).

## 판정 요약

- **Novelty: low** (표 표기: `낮음(대부분 선점: 2601.19026·2210.05470·2605.20402·2510.25602)`).
- 핵심 기여("Gaussian 블록 가정 하에서 블록 최대값 분포 × E2M1 그리드 × 스케일 포맷 × 블록 크기의 기대 왜곡을 1차원 적분/닫힌 형태로 계산")는 이미 출판됨:
  - Fasoli et al. 2601.19026 (ICLR 2026): 정확한 half-normal 최대값 밀도 eq.(5) `f_xmax(θ)=(2N/σ)(2Φ(θ/σ)−1)^{N−1}φ(θ/σ)`, E2M1 Voronoi 구간별 조건부 MSE eq.(2)–(3), 적분 eq.(4); 시뮬레이션과 χ²≈2·10⁻⁹(비양자화 스케일), UE4M3 스케일 확장 χ²≈4·10⁻⁸, INT4 χ²≈1.3·10⁻⁶; 이 분석으로 UE5M3를 제안.
  - Soloveychik et al. 2210.05470: Gaussian 블록, Gumbel(Fisher–Tippett) 부록, 2의 거듭제곱 "from above"(올림) 스케일의 BFP 분산 상한(Prop. 2), 4비트에서 최적 블록 n=64.
  - Li/Wu/Shen 2605.20402: UE8M0 올림 모델 `γ_b=2^U, U~Uniform[0,1)`(App. A Step 1, eq. 9), E[γ]=1/ln2≈1.44, RMSE(γ−1)≈0.57(실측 0.57). 아이디어의 E[4^U]=3/(2 ln 2)≈2.16은 같은 모델의 2차 모멘트(한 줄 계산).
  - Chen et al. 2510.25602: Theorem 1(INT QSNR), Theorem 2(FP QSNR)가 UE8M0(ρ∈[1,2), ρ=1.5로 설정) vs E4M3 스케일, crest factor κ(블록 크기 의존), subnormal 확률을 포함한 닫힌 형태 포맷 비교.
  - MXAttention 2607.24377 Theorem 1: E8M0 올림 계열에서 MSE 최적 경계 Q*max=29/4=7.25가 **log2 M의 임의의 절대연속 분포에 대해** 성립("Neither a Gaussian nor a log-uniform prior is required", l.629). 라운딩 규칙 결정은 극값분포와 무관.
  - UFP4 2606.20381 eq.(2): shrinkage bias의 구간별 닫힌 형태 `(2q_i−q_{i−1}−q_{i+1})/4`, q=2 구간 −0.125, 층간 승법 누적 eq.(5).
- 남는 것(좁음): (i) 실제 OCP floor 규칙·ceil·UOS·E4M3·UE5M3를 **편향(gain)까지** 한 틀에서 비교하는 표, (ii) Li et al.이 "닫힌 형태로 유도하지 않았다"고 명시한 scale×grid 교차항 cos≈−0.66의 유도(Remark 3.3), (iii) 밀도 기울기에 의한 shrinkage와 깊이 L에서의 (L·bias)² vs 분산 판정식, (iv) Gumbel 점근식(σ√(2 ln 2B))으로 블록 크기 의존을 간결하게 쓰는 것(정확 적분은 Fasoli가 이미 함 → 점근식은 편의 이상의 가치가 약함).
- **가장 심각한 전제 오류**: 아이디어의 "UE8M0 내림"은 s_q=2^⌊log2(M/6)⌋(정규화 최대 [6,12))인데, OCP v1 규칙은 ⌊log2 M⌋−2 = e_floor(8)(정규화 최대 [4,8), 2510.25602 eq.(8), 2607.24377 eq.(12)–(14), Table 1)이다. 실제 OCP 규칙으로 계산하면 B=32 Gaussian MSE/σ²는 0.0132로 올림(0.0133)과 사실상 같다(본 판정 시뮬레이션). 따라서 "라운딩 방향 하나가 MSE를 3.1배 바꾼다"는 허수아비 규칙의 결과이며 Mishra et al. 발산과의 "정합"도 무너진다. 차이는 MSE가 아니라 **편향**에 있다(아래 새 가설 H1).

## 읽은 논문과 겹침

| 논문 | 분류 | 겹치는 것 (위치) | 남는 것 |
|---|---|---|---|
| 2601.19026 Fasoli et al., *Is Finer Better?* (ICLR 2026) | preempts_core | eq.(1)–(5) 정확한 블록최대 밀도 × E2M1 구간별 조건부 MSE; App. F UE4M3 스케일 3항 분해, χ²≈4e-8; App. G INT4, App. H FP6 스케일; 블록 크기 역전(UE4M3 subnormal) ; UE5M3 제안. l.225: PoT 스케일은 Rouhani 인용으로만 언급 | E8M0/PoT 스케일, 라운딩 방향, 편향 부호, Gumbel 점근식, heavy tail |
| 2210.05470 Soloveychik et al. 2022 | preempts_part | Gumbel/Fisher–Tippett 부록(l.833–846), Prop. 2 PoT 올림 스케일 BFP 상한, 2의 거듭제곱 통과 지점에서의 점프(l.576), 최적 n=64 @4bit(l.110, 808) | 정수 가수만, FP 원소 그리드·FP8 스케일·내림 없음 |
| 2605.20402 Li/Wu/Shen (May 2026) | preempts_part | 2^U 모델 eq.(9), E[γ]=1.44, RMSE 0.57; Lemma 3.2 deadzone 직교성; 스케일 편향의 역전파 승법 누적, CLT √(L/12) | 2차 모멘트 2.16배(사소), cos≈−0.66 유도(Remark 3.3, "we do not derive in closed form"), NVFP4/E4M3, 블록 크기 법칙 |
| 2510.25602 Chen et al., *INT v.s. FP* (Oct 2025) | preempts_part | Theorem 1/2 닫힌 형태 QSNR, UE8M0 ρ, E4M3 ρ=1, κ 블록 크기 의존; eq.(8) OCP 내림이 "extra clipping error"를 준다고 명시 후 올림 채택 | ρ=1.5 고정의 조잡한 처리, 편향 없음, 라운딩 방향 정량 비교 없음 |
| 2607.24377 MXAttention (Jul 2026) | contradicts_premise / preempts_part | Theorem 1 Q*=7.25 분포 무관; OCP=e_floor(8), TFS=e_ceil(6) 공식화(Table 1) | 블록 크기·스케일 포맷별 기대 왜곡 수준, 편향 |
| 2606.20381 UFP4 (Jun 2026) | preempts_part | eq.(2) 구간 편향 닫힌 형태 −0.125, Key Finding 1–2, 승법 누적 eq.(5), RHT 증폭 | 밀도 가중(Gaussian) 크기, 판정식; (본 판정 계산: 전역 균일 밀도에서는 구간 편향 합이 정확히 0으로 상쇄 → 순 편향은 밀도 기울기·deadzone에서 나옴. 아이디어의 메커니즘이 오히려 정확) |
| 2509.23202 Egiazarian et al. (ICLR 2026) | preempts_part | Sec. 3.1 "Analytical MSE bounds", Lemma 1 top-element MSE, Lemma 2 deadzone rate; Fig. 3 E8M0 vs E4M3 수치 비교; 회전 후 kurtosis 0.02–0.05 (Gaussian 가정 지지) | 스케일 양자화의 해석적 처리 없음 |
| 2509.25149 NVIDIA NVFP4 | contradicts_premise | l.871 "MXFP4 has a relative error of around 2.5% compared to 1.5%" = Fig. 6a **학습 손실** 상대오차 | 텐서 RMS 비 1.20과 비교 불가 |
| 2506.08027 Mishra et al. | supports(약) | l.662–671 OCP v1은 "effectively rounds down", 포화 변환, 메커니즘은 "We hypothesize" | MXFP8(E4M3 원소) 결과이지 FP4 아님 |

추가 검색(본 판정): 2026-07 이후 microscaling theory / E8M0 rounding / block-max order statistics 4개 질의 → 새 선점 없음(2609.00066 OCGQuant, 2608.01847 FOCUS, 2607.26515 HiFloat4-RL는 경험적). WebSearch 1회: Modular MAX `compute_mxfp4_even_scale`가 OCP even-mode 규칙을 출하(ceil(max/6)과 다름을 명시) — 라운딩 규칙 차이는 이미 시스템에 공존(OCP floor: Modular; round-to-+inf: NVIDIA Mishra 레시피/CUDA e8m0 변환; TetraJet ceil(6); UOS 7.25).

## 전제 오류

1. "포맷 비교는 전부 경험적" → 거짓. Fasoli eq.(1)–(10), Soloveychik Prop. 1–4, Chen Thm 1–2, Egiazarian Lemma 1–2, Rouhani 2302.08007 Thm 1, Li eq.(2)/(9).
2. "Egiazarian은 경험적" → 그 논문 Sec. 3.1은 해석적 MSE 경계(Lemma 1–2).
3. UE5M3 출처 = 2609.02846 → 오기. 도입 Hu et al. 2509.17791, 1원리 MSE 분석 후 제안 Fasoli 2601.19026(2609.02846 l.82, 327이 스스로 인정).
4. UFP4 "정량 이론이 없다" → eq.(2) 닫힌 형태 존재. 다만 국소 균일 밀도 가정의 구간 편향은 전역 균일 밀도에서 합이 0(0.25·−0.125 + 0.75·−0.125 + 1.5·−0.25 + 1·+0.5 = 0) → 순 편향의 원인은 밀도 기울기·deadzone·클리핑이라는 아이디어 측 정정이 가치 있음.
5. "UE8M0 내림: s_q∈(s/2,s]" 모델 → OCP 규칙이 아님. OCP v1 = ⌊log2 M⌋−⌊log2 Qmax⌋ = e_floor(8)(Chen eq.(8), MXAttention Table 1). 본 판정 시뮬레이션(B=32, Gaussian): floor6(아이디어) MSE 0.0414 / OCP 0.0132 / ceil6 0.0133 / UOS 0.0125; NVFP4형(B=16, E4M3) 0.0090. "3.1배"와 "RMS 비 2.12"는 철회해야 함.
6. "NVIDIA 1.67" → 손실 상대오차(2509.25149 l.871, Fig. 6a)이고 레시피 의존(HiFloat4 2604.08826 Table 3 비 1.39–1.57). 텐서 RMS 비 1.20과 맞추는 "잔여 0.4" 연구 질문은 범주 오류.
7. Mishra et al. 결과는 MXFP8(E4M3 원소)이며 메커니즘은 가설. MXFP8에서 본 판정 시뮬레이션: OCP floor vs ceil 상대 MSE 8.6e-4 vs 7.0e-4(1.22배, t3에서 1.6배)로 작지만, 블록 최대 원소 상대 편향 −1.28% vs −0.08%(약 15배), gain 0.9972 vs 0.9993.
8. "블록 32 vs 16 → E[M] 비 1.10" 단조성 → UE4M3 스케일에서는 블록 크기 역전(Fasoli Sec. "Anomalous error dependency", Schaefer & Tabak 2605.08565). 모델에 스케일 범위/subnormal 항 필요.
9. Max-abs 스케일 s=M/g_max를 기준점으로 삼음 → 이미 비최적으로 알려짐(UOS 7.25, 4/6, ScaleSearch 2605.12464 −27%, ScaleSweep 2606.07618). 또 4/6 Fig. 5는 대규모 사전학습에서 MAE 스케일 선택이 MSE보다 낫다고 보고 → MSE만으로 포맷 순위를 정하는 것은 학습 관점에서 불충분.
10. Shrinkage 수치: 본 판정 시뮬레이션(정확 스케일, B=16) E[|Q|−|x|]/E|x| = −0.70%, gain 0.9959 (아이디어: −0.66%, 0.9968). 근사적으로 맞으나 gain 값은 재계산 필요. 또 이 편향은 [4,6] 구간만이 아니라 deadzone([0,0.25]→0)과 스케일 규칙의 클리핑 기여를 포함(B=32: ceil6 −1.13%, OCP −1.67%, UOS −1.20%).

## 수정 요구

- "문헌상 공백" 첫 두 항목 삭제/교체: "Fasoli et al.(ICLR 2026)이 E4M3/UE4M3 스케일에 대해 정확 적분 이론을, Soloveychik(2022)가 Gumbel+PoT 올림 상한과 최적 블록 크기를, Chen et al.(2025)이 UE8M0/E4M3 QSNR 닫힌 형태를, Li et al.(2026)이 2^U 모델을, MXAttention이 분포 무관 최적 PoT 경계를 제시했다"로 재서술.
- 기여를 "확장"으로 재포지셔닝: Fasoli 프레임에 (a) PoT 스케일 라운딩 계열 e_floor(q)/e_ceil(q) 전체, (b) 편향(gain)·최대 원소 편향, (c) Li et al.의 cos≈−0.66 교차항 유도, (d) 밀도 기울기 shrinkage + 깊이 판정식을 추가.
- "UE8M0 내림" 정의를 OCP 실제 규칙 ⌊log2 M⌋−2로 교체, MSE 3.1배/RMS 2.12 철회, "차이는 MSE가 아니라 최대원소 편향"으로 서술 수정.
- NVIDIA 1.67 비교 및 "잔여 0.4" 질문 삭제(손실 지표). 텐서 단위 비교가 필요하면 Egiazarian Fig. 3, Fasoli Fig. 2를 기준으로.
- UE5M3 인용을 2509.17791·2601.19026으로 수정; 2609.02846은 후속 실증.
- 블록 크기 단조성 주장에 UE4M3 subnormal 역전 조건 추가.
- novelty 평가 "중상" → "낮음~중하"로 하향; 리스크 문단의 "경쟁자가 빨리 따라올 수 있다"는 이미 따라옴으로 수정.
- 판정 실험 1은 Fasoli 공개 코드로 즉시 재현 가능하다고 명시하고, 실험 3(올림/내림 300M 런)은 OCP 실제 규칙과 "편향 보정된 floor"를 추가해 MSE 대 편향을 분리하도록 설계.

## 새 가설

- **H1 (사전학습 스케일 계산 단계)**: MXFP8/MXFP4에서 OCP floor 규칙의 발산은 MSE 증가(약 1.2배)가 아니라 블록 최대 원소의 체계적 축소(−1.3% vs −0.08%)가 층을 거쳐 승법 누적된 결과다. 검증: floor 규칙 + 블록별 gain 보정(또는 UOS 경계)으로 MSE는 유지하고 편향만 제거한 런이 발산하지 않으면 확인. 근거: Mishra l.662–671(가설만), Li 2605.20402 승법 누적, UFP4 eq.(5), 본 판정 sim8.
- **H2 (오차 예산/RL 롤아웃 단계)**: Li et al.의 cos(e_scale,e_grid)≈−0.66은 올림 조건 γ≥1이 정규화 원소를 축소시켜 grid 오차와 scale 오차가 같은 원소 집합에서 반대 부호를 갖게 만드는 구조적 효과이며, 2^U 모델 + E2M1 구간 적분으로 B와 무관한 상수로 유도된다(App. A의 i.i.d. 가정 O(B^{-1/2})는 정의 분해 방식의 산물). 근거: 2605.20402 Remark 3.3, App. A "Note on scope".
- **H3 (FP4 GEMM 역양자화 단계)**: E2M1 shrinkage의 순 편향은 균일 밀도에서는 0이므로 크기가 데이터 밀도 기울기와 스케일 규칙에 의해 결정된다; 따라서 UFP4식 보정 계수는 스케일 규칙별 Bussgang gain(B=32 Gaussian: ceil6 0.990, OCP 0.980, UOS 0.988; NVFP4 0.996)으로 닫힌 형태 계산·고정 적용 가능하며 학습 중 통계 추정이 불필요하다. 근거: UFP4 eq.(2), 본 판정 sim.py.

## 생략한 것
- 2302.08007(Rouhani Thm 1), 2505.06653(BOF4), 2605.06014(RHT 의존성) 본문 재확인은 생략(노트 기록 신뢰). Kalliojärvi–Astola 1996 원문 미확인.
