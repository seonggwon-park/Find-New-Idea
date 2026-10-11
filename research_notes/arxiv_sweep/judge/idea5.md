# 아이디어 5 판정: PTQ 캘리브레이션 이론 (표본복잡도·댐핑·공변량 이동·VQ/회전 과적합)

작성일 2026-10-11. 이번 판정에서 전문을 읽은 논문은 2609.31560, 2610.11226, 2604.13806, 2508.04853, 2605.05693, 2607.00908, 2405.16406, 2406.11235/2402.04396(일부), 2610.07522(일부)이다. 그 밖의 수치는 discover 노트(idea5_citations/contrarian/keywords)에서 가져왔고 해당 항목에 표시했다. 논문 텍스트는 신뢰하지 않는 데이터로 다뤘다.

## 판정 요약

- **novelty: 낮음 (low)** — 핵심인 증명 가능한 부분 (1)~(4)는 모두 2025~2026년 논문에 같은 형태나 더 강한 형태로 이미 나와 있다.
- **주된 선점 연구: George & Saab, "Generalization behavior of OPTQ and the role of regularization" (arXiv 2609.31560, 2026-09-25).**
  - Thm 3.1: m개 i.i.d. 표본이 있고 ‖x‖≤R, λ > 2R² log(N/ε)이면, 확률 1−ε로 모든 (데이터 의존) r에 대해 동시에 다음이 성립한다: E|xᵀr|² ≤ (1 − √(2R² log(N/ε)/λ))⁻¹ · ((1/m)‖Xr‖² + λ‖r‖²). Matrix Chernoff로 증명한다.
  - 이 정리는 아이디어의 가산형 상한 proxŷ + ‖H−Ĥ‖_op‖ΔW‖²보다 강하다. 곱셈형이고, 댐핑 λ가 정리 안에 들어가 있다.
  - Cor 3.2: OPTQ의 모집단 오차 상한.
  - Remark 4.1: 표본수에 의존하는 λ = β‖X‖_F²/(m^{1/3} min{m,N}^{2/3})를 제시하고, GPTQ의 0.01‖X‖_F²/N과 Qronos의 10⁻³‖X‖_op보다 표본이 적은 영역에서 우월하다고 보고한다(Figs. 1–3).
  - 저자들 스스로 "calibration error vs generalization error", "spurious correlations when the calibration set is small … regularization … to prevent this form of overfitting"이라고 쓴다(서론 l.65–69).
  - 한계: 실험은 합성 분포(Hadamard, ReLU 네트워크)뿐이고 LLM 실험은 없다. 표본은 i.i.d.를 가정해 n_eff가 없다. MP/LW 형태도 아니다.
- **부분 선점 연구**
  - (1) 분해식: 2508.04853 Remark 3.8에 그대로 있다.
  - (1)·(2) 측정과 수축: DASH-Q 2604.13806이 Ĥ_n 대 Ĥ_4096을 측정하고(n = 8…2048), LW를 인용한 수축족 H̃(ρ)=D+ρO를 쓴다. 결론은 ρ≈0이다.
  - (3) 균일수렴: SARQC 2605.05693 Thm 3.1, R² M_X² √(log(2|Q_R|/δ)/2n).
  - (4) 공변량 이동: DRQ 2610.11226 Thm 3.1/3.2(tr(CΔ)≤α 모호집합)와 TASA 2607.00908 Thm 1(tr(H_test H_α⁻¹)).
- **아직 남은 것**
  - 상관된 토큰의 n_eff를 추정하는 것(어느 논문에도 없다).
  - LLM 규모에서 λ 스케일링 지수를 검증하는 것. George–Saab는 (d/n)^{1/3}을 예측하고 아이디어는 √(d/n_eff)를 예측한다.
  - 방법 간 "보류 격차 대 P" 비교(GPTQ/AWQ/SpinQuant/AQLM/QTIP).
  - ‖H_T−H_C‖_op 예측량을 TASA/DRQ의 대안과 비교하는 것.
  - 이것들은 이론적 기여라기보다 경험적 정리·검증 작업이다.
- **이론 렌즈**: 도구 조합(Weyl, Vershynin, 균일수렴)은 위 논문들에 이미 있다. MP/LW를 GPTQ Ĥ에 직접 적용한 논문은 없다. 하지만 그 결과는 George–Saab Thm 3.1의 변형 정도에 그칠 것이다.
- **시스템 렌즈**: 실무 구현에서 damp는 수치 안정용 손잡이로 쓰인다.
  - GPTQModel은 damp_percent와 damp_auto_increment를 둔다. 웹 결과로 확인한 체크포인트 설정은 0.01/0.0025, 0.05/0.01, 0.01/0.01이다.
  - Transformers GPTQConfig 문서는 버전에 따라 0.01과 0.1로 서로 다르게 적혀 있다(웹 결과).
  - 표본 적응형 또는 LW 댐핑을 출하한 엔진은 찾지 못했다(검색 1회, 확인 불충분).

## 읽은 논문과 겹침

| 논문 | 겹치는 것 (정확한 위치) | 남는 것 |
|---|---|---|
| 2609.31560 George & Saab (2026-09) | Thm 3.1 (l.293–312) 정규화된 경험→모집단 곱셈형 상한, 데이터 의존 r에 균일. Cor 3.2 OPTQ 특화. Remark 4.1 eq.(35) λ ∝ ‖X‖_F²/(m^{1/3}min{m,N}^{2/3}). l.731–746 유도. 합성 실험 Figs.1–3(30 trials). 결론 l.942–946 "formula for choosing λ that outperforms previous recommendations in a low-sampling regime". | i.i.d. 가정(n_eff 없음). LLM 검증 없음. MSE-최적(LW) λ가 아니라 상한을 최소화하는 λ임. R²(최대 노름)이 outlier 채널에 지배되는 문제를 다루지 않음. |
| 2508.04853 Zhang–Zhang–Colbert–Saab | Remark 3.8 (l.995–1023): E_z\|zᵀ(w−q_X)\|² = (1/m)‖X(w−q_X)‖² + (w−q_X)ᵀ(E zzᵀ − XᵀX/m)(w−q_X). 아이디어 (1)·(4)의 분해와 동일함. 결론 l.2202에서 일반화를 future work로 남김. | 수렴 속도 없음. 2609.31560이 이 공백을 채움. |
| 2604.13806 DASH-Q (EuroMLSys'26) | l.229–242: n=8…2048 Ĥ 대 4096 기준 ℓ1 오차. off-diagonal은 큰 n에서도 불안정. 수축족 D+ρO(LW [26] 인용). Eq.(7) SNR. 결론 ρ≈0(l.307–311). Sec 7.2(l.1222–1232): GPTQ는 n≤4에서 PPL>10², n>2⁸에서 "fluctuate or even slightly degrades". +7.01% 평균, +14.01% 최대 zero-shot. | 이론 없음. 원소별 ℓ1·SNR일 뿐 연산자/백색화 노름 분석이 아님. 단위행렬 방향 수축(ridge) λ 최적화 없음. |
| 2610.11226 DRQ (2026-10-08) | Thm 3.1 (l.411–435): Δ⪰0, tr(CΔ)≤α 하에서 순서 역전이 일어날 필요충분조건. Thm 3.2: 최악 손실 = L + α S_C. Thm B.2 (l.1693): "More calibration data need not improve the selected weights". 수치: 560개 수정 중 73.8%(AWQ), 47.1%(GPTQ)가 보류 PPL을 올림(l.160–190). Fig.4: rec-only 정제는 n이 클수록 PPL이 "orders of magnitude" 상승(l.1247–1253). | 연산자 노름 대신 trace 예산을 씀. 표본복잡도 없음. 격차의 주원인을 표본 분산이 아니라 목적 불일치로 봄. 아이디어의 "과적합=표본 분산" 프레이밍과 충돌함. |
| 2605.05693 SARQC (2026-05) | Thm 3.1 (l.378–405): 유한 클래스 Q_R에서 R(Ŵ)−R̂_cal ≤ R²M_X²√(log(2\|Q_R\|/δ)/2n), i.i.d. | P(회전·코드북 파라미터 수)로 매개화하지 않음. n_eff 없음. 방법 간 비교 없음. |
| 2607.00908 TASA (2026-07) | Thm 1: f(α)=tr(H_test H_α⁻¹)는 엄격 볼록이고 유일한 α*∈(0,1)을 가짐. 단일 도메인 캘리브레이션에서 κ가 2배 이상 커지고 유효랭크가 6→3으로 붕괴(l.377–378). Table 19 κ 2143 vs 950, ER 3.3 vs 5.8(LLaMA-3-8B). | 공변량 이동의 다른 예측량. 아이디어의 ‖H_T−H_C‖_op는 이것과 비교해야 함. |
| 2405.16406 SpinQuant | A.3 (l.1504–1510) "rotation optimization is resilient to modifications in the number of samples … reducing [800] to 128 samples does not lead to a significant change". Table 13 C4 대 Wiki 결과 일관. | 분포 내 과적합은 null. 분포 간 격차(DartQuant Table 1, PTB 38.24 vs 45.13; discover 노트 기준)는 이동 문제임. |
| 2610.07522 Activation Denoising (MSR, 2026-10) | ridge α와 상류 노이즈 정규화. G = H + αI + λ²N + …. Thm 2 "noise injection acts as a magnitude-reducing ridge"(l.468). 댐핑의 또 다른(강건성) 해석임. | 표본 추정 해석이 아님. 댐핑을 LW로 정당화하려면 이 해석과 분리해야 함. |
| 2406.11235 QTIP / 2402.04396 QuIP# | Hessian을 6144 시퀀스로 생성함(l.2148 / l.2878). | VQ가 "같은 캘리브레이션"을 쓴다는 전제를 반박함. |
| 2602.05902 CoreQ (노트 기준) | Thm 3.1 α* = μ²/(μ²+σ²): 유한 캘리브레이션 추정량의 최적 수축. | 대상이 Ĥ+λI가 아님. |

## 전제 오류

1. **"전부 경험적이다 / RMT·표본복잡도 논문 없음"은 틀렸다.**
   - 2508.04853 Remark 3.8에 분해식이 있다.
   - 2609.31560 Thm 3.1/Cor 3.2/Remark 4.1에 경험→모집단 상한과 표본수 의존 λ가 있다.
   - 2605.05693 Thm 3.1에 균일수렴이 있다.
   - 2610.11226 Thm 3.1/B.2에 이동과 n→∞ 한계가 있다.
   - 2607.00908 Thm 1에 tr(H_T H_C⁻¹)가 있다.
   - 정확히 맞는 문장은 "MP/LW를 Ĥ에 적용하거나 n_eff를 다룬 논문은 없다"뿐이다.
2. **2608.01078 "새 모델은 덜 민감"은 오인용이다.** 이 문장은 2608.01078의 관련연구 요약([69])이고, 원출처는 Paglieri et al. 2405.20835다. 2608.01078 자체는 토큰을 늘릴수록 1.58-bit 정확도가 오른다고 보고한다(discover 노트).
3. **Williams & Aletras 2311.09755의 "GPTQ는 과적합"은 그들 자신의 결과가 아니다.** 그들은 이 주장을 Wu et al. 2312.08583(ZeroQuant(4+2))에 귀속시킨다(discover 노트 l.171 인용). 그들이 실제로 측정한 것은 결과가 소수 예시에서 정체(plateau)되고, 양자화가 pruning보다 덜 민감하다는 것이다.
4. **OpenReview pfw3saHzGU는 확인되지 않았다.** 같은 문장이 2504.04823("Quantization Hurts Reasoning?", COLM 2025) Sec.5에 있다(discover 노트). arXiv id로 인용해야 한다.
5. **"GPTQ 1% 댐핑"은 보편 기본값이 아니다.**
   - SparseGPT 부록은 λ 0.001–0.1이 "quite similar"하다고 한다.
   - QEP 2504.09629는 평균 대각(100×)을 쓴다.
   - GPTQModel은 auto-increment를 두고, 체크포인트에서 0.05도 쓰인다.
   - 댐핑에는 수치 안정성 해석과 상류 노이즈 강건성 해석(2610.07522 Thm 2)도 있다.
   - "ad hoc LW 수축"은 여러 해석 중 하나일 뿐이다.
6. **"최적 λ ∝ √(d/n_eff)·평균 고유값"은 이미 경쟁 예측이 있는 주장이다.** George–Saab eq.(35)를 정규화(H=XᵀX/m)하면 m≥N일 때 λ/m ≈ E‖x‖²·(N/m)^{1/3}/N, 즉 평균 고유값 × (d/n)^{1/3}이다.
   - 반면 고전 LW 수축 강도는 대략 (d/n)에 비례한다(MP bulk의 퍼짐 √(d/n)과는 구분해야 함).
   - 세 지수 1/3, 1/2, 1 중 무엇인지가 실험으로 판별할 질문이다.
7. **가산형 상한 proxŷ + ‖H−Ĥ‖_op‖ΔW‖_F²은 거의 공허하다.**
   - LLM 활성화는 massive-activation 채널 때문에 ‖Σ‖_op ≫ 평균 고유값이다. TASA Table 19의 층별 유효랭크는 3.3–5.8이다.
   - GPTQ는 오차를 작은 고유값 방향으로 밀어 넣는다. 그래서 의미 있는 양은 백색화된 상대 오차 ‖(Σ+λI)^{-1/2}(Ĥ−Σ)(Σ+λI)^{-1/2}‖다.
   - George–Saab는 이 형태를 λ를 넣은 곱셈형으로 이미 증명했다.
8. **SpinQuant/VQ 과적합 산술(P≈8×10⁶ 대 1.6×10⁶ 토큰 → O(1) 격차)에는 오류가 있다.**
   - (i) 토큰 하나가 d차원 제약을 주므로, 스칼라 제약 수는 n·d ≈ 6.6×10⁹이다.
   - (ii) R은 Stiefel 제약 아래 100 iteration만 학습된다(암묵적 정규화).
   - (iii) SpinQuant Table 11에서 128과 800 표본의 PPL이 같다.
   - (iv) QuIP#/QTIP Hessian은 6144 시퀀스로, "같은 캘리브레이션"이 아니다. AQLM도 4096 길이의 RedPajama를 쓴다.
   - 실제로 관찰된 회전 과적합은 분포 간 이동이다(DartQuant Table 1, PTB 약 7 PPL).
9. **"격차가 1/n_eff에 단조 증가"라는 성공 기준은 반례가 있다.**
   - DASH-Q Sec 7.2: GPTQ PPL이 n>2⁸에서 흔들리거나 악화된다.
   - 2510.10618(노트 기준): 코드 정확도가 57.67%(16개)에서 34.03%(128개)로 떨어진다.
   - DRQ Thm B.2: 격차가 n→∞에서도 남는다.
10. **"n_eff가 시퀀스 수에 가까울 가능성"은 근거가 약하다.**
    - n_eff≈128 < d=4096이면 상대 오차가 O(1)을 넘어야 한다. 그런데 GPTQ 성능은 128 시퀀스에서 정체되고, DASH-Q에서는 대각이 빠르게 안정된다.
    - MaCa/RSQ는 부분적 상관만 시사한다. 전제가 아니라 측정할 가설로 써야 한다.
11. **"2-bit SOTA 증분 ≈0.3 PPL"은 출처 없는 수치다.** 방법마다 캘리브레이션 예산(128 대 6144 시퀀스, 미세조정 유무)이 달라, 같은 축에서 비교할 수 없다.

## 수정 요구

- **"문헌상 공백" 절**
  - "전부 경험적이다"와 "RMT 논문 없음"을 철회한다.
  - 2508.04853 Remark 3.8, 2609.31560 Thm 3.1/Remark 4.1, 2605.05693 Thm 3.1, 2610.11226 Thm 3.1/B.2, 2607.00908 Thm 1, 2604.13806을 선행 연구로 명시한다.
  - 2608.01078 인용은 2405.20835로 바꾼다.
  - 2311.09755의 과적합 문장은 2312.08583로 귀속한다.
  - pfw3saHzGU는 2504.04823으로 바꾼다.
- **수학 (1)**
  - 가산형 Vershynin 상한을 빼고, George–Saab Thm 3.1을 출발점으로 삼는다.
  - 새로 기여할 것은 의존 토큰(β-mixing 또는 블록 부트스트랩)의 n_eff 버전, 그리고 R²(최대 노름) 대신 유효랭크 r(Σ)나 백색화 노름으로 바꾼 상한이다.
- **수학 (2)**
  - "1%는 ad hoc"이라는 주장 대신 세 예측을 판별하는 질문으로 바꾼다: George–Saab의 (d/n)^{1/3}, MP/LW, 그리고 DASH-Q의 대각 수축(ρ≈0).
  - 실험 (c)에 George–Saab λ, DASH-Q, QEP의 평균 대각 λ를 기준선으로 추가한다.
- **수학 (3)**
  - "SpinQuant P≈8×10⁶ 대 1.6×10⁶ 토큰 → O(1)" 산술을 삭제한다.
  - SARQC Thm 3.1을 인용하고, 기여를 "|Q|를 P로 매개화하고 방법 간 격차를 측정하는 것"으로 좁힌다.
  - SpinQuant Table 11의 null 결과와 VQ의 6144 시퀀스 사실을 반영한다.
- **수학 (4)**
  - DRQ와 TASA를 인용한다.
  - ‖H_T−H_C‖_op, tr(H_T H_C⁻¹), DRQ의 S_C 중 무엇이 도메인 열화를 가장 잘 예측하는지 비교하는 실험으로 바꾼다.
- **성공 기준**: 격차를 세 항으로 분해하도록 바꾼다: 목적 불일치(n→∞에서도 남음), 분포 이동, 표본 분산. 단조성 기준은 삭제한다.
- **평가**: novelty를 "중"에서 "낮음"으로 내린다. 기여를 "LLM 규모의 경험적 검증 + n_eff"로 재정의한다.

## 새 가설

1. **이상치 채널이 지배하는 댐핑 척도.** (단계: GPTQ/LDLQ Hessian 댐핑)
   - 근거: TASA의 층별 유효랭크 3–6, George–Saab 조건 λ > 2R² log(N/ε)(R은 최대 노름), QuIP의 고유값 통계.
   - 가설: GPTQ 댐핑을 mean(diag H)에 비례시키면 massive-activation 채널이 λ를 부풀려 bulk 방향을 과수축한다. λ를 bulk 고유값(median diag 또는 trimmed mean)에 비례시키고 (d/n)^p로 스케일하면 W2/W3에서 보류 KL이 개선된다.
   - 판별: George–Saab, LW, DASH-Q, QEP와 함께 split-half 교차검증으로 λ를 선택해 비교한다.
2. **회전 과적합은 표본 분산이 아니라 이동이고, 예측량은 백색화된 Hessian 차이다.** (단계: 학습 회전/코드북 캘리브레이션)
   - 근거: SpinQuant A.3(128≈800 표본), DartQuant Table 1(분포 간 PTB 격차 약 7 PPL), DRQ Thm B.2.
   - 가설: 분포 간 열화는 ‖H_C^{-1/2}(H_T−H_C)H_C^{-1/2}‖ 또는 tr(H_T H_C⁻¹)에 비례하고 P에는 거의 무관하다. 혼합 도메인 캘리브레이션(TASA α*)이 회전 학습에서도 최적 혼합비를 가진다.
3. **시퀀스 블록 부트스트랩으로 재는 n_eff와 위치 의존성.** (단계: 캘리브레이션 데이터 구성)
   - 근거: RSQ(앞쪽 1024 토큰만 써도 더 좋음; 노트 기준), MaCa(길이마다 다른 Hessian), DASH-Q Fig.1b.
   - 가설: 토큰 위치 블록 간 Ĥ 분산을 쟀을 때 n_eff/n이 0.01–0.1 수준이다. 같은 토큰 예산이면 짧은 시퀀스 다수가 긴 시퀀스 소수보다 Ĥ 상대 오차가 작다.

## 생략한 것

- OpenReview pfw3saHzGU, 2510.10618, DartQuant, 2312.08583은 이번에 직접 다시 읽지 않았다. discover 노트의 인용 위치를 신뢰했다.
- 시스템 검색은 1회만 했다. llm-compressor의 기본 dampening_frac은 확인하지 못했다.

Sources: [HF quantize_config 예시](https://huggingface.co/Matt300209/8B-instruct-int4/blob/main/quantize_config.json), [HF quantize_config 예시 2](https://huggingface.co/irish-quant/HuggingFaceTB-SmolLM-360M-Instruct-3bit/blob/main/quantize_config.json)
