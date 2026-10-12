# Citation verification — batch 16

IDs: 2401.18079, 2506.19505, 2610.03027, 2604.08426
Report context (line 321 of the report): the pre-RoPE vs post-RoPE key-quantization discussion.

> "- pre-RoPE 지지: KVQuant([2401.18079, NeurIPS 2024] [V]), AnTKV(pre-RoPE 클러스터가 더 조밀, [2506.19505] [S]), TaSQ([2610.03027] [S]). 반대: YAKV는 Hadamard 기반 HIGGS 양자화기에서 pre-RoPE 이득이 없다고 보고 ([2604.08426] [U: 이 세션에서 미확인]); ... 양자화기 의존성을 설명한 논문이 없다."

Also line 393: "2604.08426(YAKV)은 존재·내용을 PDF로 확인하기 전까지 인용하지 말 것."

All four papers exist on arXiv and their full text was fetched. Summary of verdicts:

| id | paper | venue | verdict |
|---|---|---|---|
| 2401.18079 | KVQuant (Hooper et al.) | NeurIPS 2024 (arXiv comment) | confirmed; one new fact changes the report's closing sentence |
| 2506.19505 | AnTKV (Li et al.) | preprint/unknown (v2 Oct 2025) | confirmed (qualitative Fig. 8 support only) |
| 2610.03027 | TaSQ (Cheong, Son, Yoo) | preprint (header "Preprint.", v1 2 Oct 2026) | confirmed; can move [S] -> [V] |
| 2604.08426 | KV Cache Offloading for Context-Intensive Tasks / YAKV (Bocharnikov et al., Yandex) | preprint (arXiv comment "Preprint", v5 1 Sep 2026) | confirmed; [U] can be lifted, with scope caveats |

---

## 2401.18079 — KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization

- First author: Coleman Hooper. Published 2024-01-31 (v updated 2025-05-28). arXiv comment: "NeurIPS 2024".
- Claim in report: KVQuant supports pre-RoPE key quantization; cited as NeurIPS 2024. **Confirmed.**

Quotes (text file lines 362-366, Sec. 3.2):
> "This motivated our investigation into whether we could perform pre-RoPE Key quantization (meaning that we quantize Kn) and then efficiently apply the positional embeddings on-the-fly after dequantization. The benefits of pre-RoPE Key quantization are highlighted in Appendix H, yielding 0.82 perplexity improvement on Wikitext-2 for 3-bit LLaMA-7B quantization."

Table 10 (Appendix H, lines 2180-2210; LLaMA-7B, per-channel Key / per-token Value, int3): fp16 5.68; int3 post-RoPE 7.05; int3 pre-RoPE 6.23.

**New relevant fact (bears on the report's last sentence "양자화기 의존성을 설명한 논문이 없다"):** KVQuant itself documents a quantizer-dependent reversal. Appendix "Post-RoPE Per-Token Quantization Ablation", Table 20 (lines 3826-3872):
> "These results show that post-RoPE Key quantization is superior to pre-RoPE Key quantization when quantizing Keys per-token. This is because when rotating an outlier channel with large average magnitude and another channel with smaller average magnitude together, at some positions in the sequence, part of the magnitude from the outlier channel will be shifted to the smaller channel. This partially mitigates the impact of the outlier channel on skewing the quantization range for some of the tokens in the sequence."

Table 20 numbers (LLaMA-7B, Wikitext-2, uniform, per-token Key): int4 pre 6.02 / post 5.98; int4-gs128 5.76 / 5.77; int3 14.68 / 10.87; int3-gs128 6.28 / 6.17.

So pre-RoPE helps per-channel key quantization and hurts per-token key quantization, and KVQuant gives a (qualitative) mechanism for both. The report's sentence should be softened to "정량적·이론적으로 설명한 논문이 없다" or should cite KVQuant Table 20 as the one existing empirical demonstration of the dependency. This also strengthens the report's own derivation (line 318) rather than contradicting it.

---

## 2506.19505 — AnTKV: Anchor Token-Aware Sub-Bit Vector Quantization for KV Cache in Large Language Models

- First author: Zeyu Li (HKUST-GZ / Xiangtan / Microsoft). Published 2025-06-24, v2 2025-10-18. No arXiv comment/journal_ref; a web search found no acceptance notice. Venue: preprint/unknown.
- Claim in report: AnTKV supports pre-RoPE, "pre-RoPE 클러스터가 더 조밀". **Confirmed**, but the evidence is a single visualization, not an ablation.

Quotes:
- Sec. 3 (lines 555-557): "since the application of RoPE disrupts the channel-wise magnitude distribution of K (see Appendix A), which otherwise exhibits large inter-cluster distances and small intra-cluster variances, the pre-RoPE strategy, consistent with [21], is adopted in AnTKV." ([21] = KVQuant)
- Appendix A (lines 1328-1331): "We observe that, compared to the post-RoPE K, the pre-RoPE K exhibits smaller inter-cluster distances and lower intra-cluster variance, which contributes to reduced quantization error."
- Fig. 8 caption (lines 1773-1774): "We sampled a 2048-length sentence from WikiText2 and generated pre- and post- RoPE Key on the LLaMA-3-8B model." (Layer 0, Head 0.)

Caveats for the report:
1. The paper does not run a pre- vs post-RoPE ablation; it adopts pre-RoPE by appeal to KVQuant plus the Fig. 8 picture (one sample, one layer/head). "더 조밀" (lower intra-cluster variance) is a fair paraphrase; "tighter clusters" is the safe wording.
2. Internal inconsistency: Sec. 3 says pre-RoPE K has "large inter-cluster distances" while Appendix A says "smaller inter-cluster distances". Only the intra-cluster-variance statement is consistent; cite that.
3. Headline numbers (abstract, lines 33-34; Sec. 1 lines 185-186): 1-bit Mistral-7B WikiText-2 PPL 6.32 vs CQ 7.25 vs KVQuant-1% 15.36; LLaMA3-8B to 840K tokens on one 80GB A100 at 0.375-bit; up to 3.5x decoding throughput. Theorem 3.1 is an error-propagation bound for per-row K/V perturbations (proof in Appendix B); it is used to define the anchor score, not to prove anything about RoPE.
4. Own Limitation section (line 1147): "more accurate AnS for tokens and higher performance implementations for its computation may be possible ... further empirical validation is required."

Suggested tag: keep [S] or move to [V] with wording "AnTKV는 pre-RoPE K의 intra-cluster 분산이 더 작다는 시각화(Fig. 8)를 근거로 KVQuant를 따라 pre-RoPE를 채택(ablation 없음)".

---

## 2610.03027 — Tailoring the Quantization Space for 1-Bit KV Cache Compression (TaSQ)

- First author: Minsoo Cheong (SNU; co-first Donghyun Son, Stanford; Sungjoo Yoo). Published 2026-10-02 (v1). Header of the PDF: "Preprint." No comment/journal_ref; web search found nothing further. Venue: preprint.
- Claim in report: TaSQ supports pre-RoPE key VQ. **Confirmed**, with a quantitative reconstruction-error argument.

Quotes (Sec. 3.3, lines 232-238):
> "However, RoPE rotates keys by position-dependent angles, substantially spreading their distribution. As shown in Figure 1c, the relatively compact pre-RoPE key distribution becomes widely dispersed after RoPE, making it more difficult for a shared codebook to cover the space efficiently. Figure 1d confirms this effect: pre-RoPE VQ achieves lower reconstruction error across all 32 layers, with a 35% lower total reconstruction error. Consequently, we design TaSQ to apply VQ to pre-RoPE keys."

Fig. 1 caption (lines 146-150): "(d) K-means reconstruction MSE on the fitting samples across all 32 layers, with matched group layouts and codebook sizes. All panels use a single 2,048-token GPQA window from Llama-3.1-8B; (a–c) show layer 15, KV head 0."

Additional supporting evidence (Appendix, lines 1957-1970, Fig. 7): pre-RoPE VQ methods (CQ, TaSQ) degrade only gradually beyond the calibration length, while post-RoPE NovaKV "exhibits a sharp increase in error beyond the calibration range" because "its transforms and codebooks are fitted directly to post-RoPE activations". NSNQuant (post-RoPE, calibration-free) stays stable.

Caveats: the 35% figure is k-means reconstruction MSE on fitting samples (one 2048-token GPQA window, Llama-3.1-8B), not an end-task ablation of pre vs post-RoPE inside TaSQ. Headline serving numbers: up to 14x larger batch (6 -> 84) and 1.87x peak throughput (220.8 -> 412.6 tok/s) on one RTX 6000 Ada in SGLang vs BF16 (lines 1506, 1541-1542). TaSQ's transforms are "RoPE-compatible" only because grouping preserves RoPE channel pairs (Alg. 1, "RoPE-pair-constrained hierarchical partitioning"). No Limitations section.

Suggested tag: [S] -> [V]; add "pre-RoPE VQ 재구성 MSE 35% 감소 (Fig. 1d, Llama-3.1-8B)" as the concrete number.

---

## 2604.08426 — KV Cache Offloading for Context-Intensive Tasks (defines YAKV)

- First author: Andrey Bocharnikov (HSE / Yandex; with Ermakov, Kuznedelev, Zhdanovskiy, Yershov). Published 2026-04-09, v5 2026-09-01. arXiv comment: "Preprint". Code: github.com/yandex-research/context-intensive-kv-offloading.
- Existence: **confirmed**. "YAKV" is not the paper's title; it is the algorithm introduced in it ("Yet Another KV Offloading", line 75; "YAKV Offloading", lines 403-412). The report's shorthand "2604.08426(YAKV)" is therefore acceptable but should be expanded on first use.
- Claim in report: "YAKV는 Hadamard 기반 HIGGS 양자화기에서 pre-RoPE 이득이 없다고 보고". **Confirmed** (Appendix L, Table 10), with important scope caveats.

Quotes (Appendix L "Comparison of pre-RoPE and post-RoPE", lines 2676-2698):
> "Our main YAKV setup and inference experiments in Section 4.5 employ post-RoPE quantization to enable efficient inference by avoiding the need to rotate past keys at each step. However, some prior work on KV compression suggests that pre-RoPE quantization may yield higher accuracy. ... In the pre-RoPE variant, keys are quantized before RoPE and rotated after retrieval. Note that landmarks are always quantized post-RoPE to maintain inference efficiency. The results indicate that pre-RoPE key quantization does not significantly improve model accuracy for YAKV. We attribute this to the randomized Hadamard transformation used in the HIGGS quantizer [85]."

Table 10: Llama 3.1 8B MultiNeedle — pre-RoPE 93.36, post-RoPE 93.73; Qwen3 4B Text2JSON — pre-RoPE 90.83, post-RoPE 91.05. (Post-RoPE is marginally higher in both.)

Scope caveats the report must carry:
1. Setting is a KV *offloading* system, not a pure KV-quantization method: YAKV = "offloading with 4-bit HIGGS compression and 2-bit HIGGS for selection" (Sec. 4.4, line 899). The 4-bit HIGGS grid is d=2, n=256, "which averages 4.02 bits per value" (line 499-500); selection proxies use 1-2-bit grids (d=8/n=256 = 1.02 bit, d=4/n=256 = 2.02 bit, line 651). The pre/post comparison applies only to the 4-bit offloaded keys; the 2-bit selection landmarks stay post-RoPE in both arms.
2. Evidence is two (model, task) pairs on retrieval-style accuracy, no perplexity, no sub-4-bit pre/post comparison. The "Hadamard" explanation is a one-sentence attribution with no ablation (e.g., HIGGS without the Hadamard transform is not tested).
3. Main paper topic: Text2JSON benchmark; KV offloading (ShadowKV, ArkVale, LRQK, InfiniGen) degrades on context-intensive tasks because of low-rank key SVD and unreliable landmarks; HIGGS 4-bit gives a better memory-accuracy trade-off than SVD (Sec. 4.1); 2-bit HIGGS per-key selection beats ShadowKV landmarks at equal GPU memory (Sec. 4.3).
4. Own Limitations (line 1215): "The main focus of our work is to better understand the effectiveness of KV offloading, not design the highest performance algorithm ... Our analysis is also limited to models trained with full attention".

Suggested edit to report line 321: replace "[2604.08426] [U: 이 세션에서 미확인]" with "[2604.08426] [V]" and reword to "YAKV(KV offloading, 4-bit HIGGS=d2/n256 + 랜덤 Hadamard)는 Appendix L Table 10에서 pre/post-RoPE 차이가 없다고 보고(Llama-3.1-8B MultiNeedle 93.36 vs 93.73; Qwen3-4B Text2JSON 90.83 vs 91.05); Hadamard 때문이라는 설명은 ablation 없는 추정". Line 393's "인용하지 말 것" entry for 2604.08426 can be removed.

---

## Cross-cutting note on "양자화기 의존성을 설명한 논문이 없다"

Partially incorrect as stated. Two of the four cited papers do report the dependency empirically and give (qualitative) explanations:
- KVQuant Table 10 vs Table 20: pre-RoPE wins for per-channel key quantization (int3 7.05 -> 6.23) but loses for per-token key quantization (int3 14.68 pre vs 10.87 post), with a channel-mixing explanation in both directions.
- YAKV Appendix L: no gain with Hadamard+VQ (HIGGS) at 4-bit, attributed to the random Hadamard transform.
What is missing is a quantitative/theoretical account that predicts the sign and size of the pre-RoPE gain as a function of the quantizer (per-channel vs per-token vs rotation/VQ) — which is what the report's line 318 derivation attempts. Suggested rewording: "의존성을 보고한 논문은 있으나(KVQuant Tab. 20, YAKV App. L) 정량적으로 설명한 논문은 없다."
