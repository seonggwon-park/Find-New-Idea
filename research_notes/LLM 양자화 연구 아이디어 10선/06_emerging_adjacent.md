# Emerging and Adjacent Quantization Niches in the LLM Pipeline (survey as of 2026-10-10)

*Scope: everything that is NOT plain weight PTQ, NOT plain KV-cache quantization, NOT FP8 pretraining. Fourteen areas plus a cross-area summary. Method: ~37 WebSearch (extended) sweeps over arXiv 2023–2026 and venue pages; direct fetches of arxiv.org / huggingface.co / alphaxiv were refused by the session's egress policy, so quantitative numbers come from search excerpts of abstracts/summaries. Items tagged **[ID unverified]** are papers I am confident exist but whose arXiv ID I could not confirm in this session; items tagged **[secondary]** are numbers that came from a third-party summary rather than the paper text. Crowdedness counts are "papers surfaced in this sweep", a lower bound.*

---

## Area 1 — Non-Transformer / hybrid architectures (Mamba, SSD, RWKV, hybrids, recurrent-state quantization)

### Takeaway
Mamba-LLM PTQ is now a small but established line (Quamba → Quamba2 → Quamba-SE, Mamba-PTQ, MambaQuant, Ternary Mamba, SSDi8), vision-SSM PTQ has ~4 papers, RWKV has exactly one dedicated PTQ paper, and quantization of the *recurrent SSM state itself* (the KV-cache analogue) is essentially unstudied academically — it is currently being worked out in a vLLM RFC with self-reported numbers. No theory exists for error accumulation through the quantized recurrence.

### Cited Findings
- Quamba (static 8-bit per-tensor PTQ for selective SSMs) suppresses SSM-input outliers with percentile clipping (p = 99.999 used) and moves SSM *output* outliers into an outlier-free space with a Walsh–Hadamard transform whose inverse is fused into the output projection; 8-bit Mamba-2.8B gets 1.72× lower generation latency on an Nvidia Orin Nano 8G with 0.9% average zero-shot accuracy drop — [arXiv 2410.13229](https://arxiv.org/abs/2410.13229); OpenReview listing [mnna9LUg7P](https://openreview.net/forum?id=mnna9LUg7P) (ICLR 2025 venue not confirmed in this sweep).
- Quamba2 extends to W8A8 / W4A8 / W4A16 for both Mamba1 and Mamba2 backbones; it sorts-and-clusters the input x for 8-bit, applies *per-state-group* quantization to the input-dependent B and C parameters, and rearranges weights offline so SSM output is unchanged; the 8B model reports ≈1.3× faster prefill, 3× faster generation, 4× memory reduction, 1.6% average accuracy drop; latest version v4 revised 6 Nov 2025; code at enyac-group/Quamba — [arXiv 2503.22879](https://arxiv.org/abs/2503.22879).
- Mamba-PTQ (workshop-stage, July 2024) finds Mamba models exhibit the same outlier-channel pattern as attention LLMs and reports baseline PTQ results — [arXiv 2407.12397](https://arxiv.org/abs/2407.12397).
- MambaQuant (rotation-based 8-bit PTQ) appears as a W8A8/W4A8 baseline in Quamba2's comparisons; I could not confirm its own arXiv ID this session — [Quamba2 html](https://arxiv.org/html/2503.22879v2) **[MambaQuant ID unverified; believed 2501.13229, ICLR 2025]**.
- Quamba-SE ("Late Breaking Results") replaces hard clipping with a soft-edge activation quantizer; evaluated only on Mamba-130M across six zero-shot benchmarks, up to +0.83% average over Quamba — [arXiv 2601.09451](https://arxiv.org/pdf/2601.09451).
- Ternary Mamba: "Grouped Quantization-Aware Training of W1.58A16 State Space Models" — title surfaced, contents not retrieved — [arXiv 2606.18114](https://arxiv.org/html/2606.18114v1).
- SSDi8: "Accurate and Efficient 8-bit Quantization for State Space Duality" — title surfaced, contents not retrieved — [arXiv 2608.21952](https://arxiv.org/pdf/2608.21952).
- Vision-SSM PTQ line: QMamba [arXiv 2501.13624](https://arxiv.org/abs/2501.13624); PTQ4VM [arXiv 2412.20386](https://arxiv.org/pdf/2412.20386); k-scaled quantization + reparameterization for Vision Mamba [arXiv 2501.16738](https://arxiv.org/html/2501.16738v1); Channel-wise and token-aware PTQ for visual SSD [arXiv 2609.16656](https://arxiv.org/pdf/2609.16656).
- eMamba (edge acceleration) states the core state-quantization difficulty: when the state is quantized the recurrence repeats over the whole sequence and "the bit width can blow up", while re-quantizing the state each step keeps values stable but loses state information — [arXiv 2508.10370](https://arxiv.org/html/2508.10370).
- RWKVQuant is the only RWKV-specific PTQ framework found: a coarse-to-fine proxy adaptively selects scalar vs. vector quantization per layer plus codebook optimization, targeting ≈3-bit weights — [arXiv 2505.03803](https://arxiv.org/abs/2505.03803).
- Hybrid recipes from NVIDIA: Nemotron-H trained with FP8 per-tensor current scaling, keeping the first and last four GEMMs in BF16; an FP8 PTQ checkpoint of Nemotron-H-8B-Reasoning-128K quantizes linear layers with minimal loss — [arXiv 2504.03624](https://arxiv.org/abs/2504.03624). Nemotron 3 Super and Nemotron-Labs-3-Puzzle-75B-A9B quantize MoE GEMMs and Mamba linear layers to FP8 but keep attention projections and embeddings in BF16 and the Mamba SSM state in FP16 — [arXiv 2604.12374](https://arxiv.org/pdf/2604.12374); [arXiv 2607.04371](https://arxiv.org/pdf/2607.04371).
- vLLM RFC #55196 (2026): fp8 KV cache gives 2.00× capacity on Qwen2.5-3B but only 1.00× at 512/1024 tokens and 1.84× at 32k on the hybrid Falcon-H1-1.5B, because the Mamba state page (~1.59 MB, bf16 regardless of `--kv-cache-dtype`) pins the shared page size; it proposes an fp8/int8 `--mamba-cache-dtype` and notes SGLang and Quamba2 already quantize SSM state to int8. A contributor reports uniform INT8 state dropping Qwen3.5-4B GSM8K from 81% to 41%, restored to 81% with error feedback; INT8 state + INT4 residual used 1.28× bf16 memory with no detectable loss in a 50-question run (self-reported, no PR) — [vllm issue 55196](https://github.com/vllm-project/vllm/issues/55196).
- "Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM" — reports that the recurrent (GDN) layers of a hybrid 27B model tolerate NVFP4 W4A4 — [arXiv 2609.04098](https://arxiv.org/pdf/2609.04098) (details beyond abstract-level not retrieved).
- Adjacent hybrid work (not quantization): Component-Aware Self-Speculative Decoding in Hybrid LMs [arXiv 2605.01106](https://arxiv.org/pdf/2605.01106); RAD redundancy-aware distillation for hybrids [arXiv 2505.22135](https://arxiv.org/pdf/2505.22135); COREY Hadamard reparameterization for selective-scan kernels [arXiv 2604.10597](https://arxiv.org/html/2604.10597v3).

### Inferences
- **Crowdedness:** Mamba-LLM weight/activation PTQ ≈7 papers (Quamba, Quamba2, Quamba-SE, Mamba-PTQ, MambaQuant, Ternary Mamba, SSDi8); vision-SSM ≈4; RWKV = 1; hybrid-specific quantization studies ≈1 academic paper (2609.04098) + vendor recipes; **recurrent-state quantization ≈0 academic papers** (one RFC, one int8 implementation in Quamba2/SGLang).
- **Strongest result:** Quamba2 W4A8 on Mamba2-8B (3× generation speedup, 1.6% drop). 
- **What makes SSM quantization hard (from the sources):** input-dependent B/C/Δ carry outliers that are not channel-stable; SSM output has large outliers requiring Hadamard; the state recurrence compounds quantization error over the sequence (eMamba, vLLM RFC accuracy collapse); hybrid page-size coupling makes attention-only KV quantization nearly useless at short context.
- **Theory:** none found — no bound on how per-step state quantization error propagates through a selective (input-dependent) recurrence, and no analysis of why GDN tolerates 4-bit while Mamba state collapses at INT8 without error feedback.
- **What would be new:** (a) a formal error-propagation model for quantized selective recurrences (contractivity of the A-decay vs. injected noise, with Δ-dependence) and a derived per-step bit allocation; (b) a principled error-feedback / residual scheme for SSM state (the RFC's 81%→41%→81% anecdote is exactly the signal that this is a real, unpublished problem); (c) a hybrid-aware mixed-precision scheme that treats Mamba state and attention KV as one memory budget.

### Gaps
- Could not retrieve full text of 2609.04098, 2606.18114, 2608.21952 (egress policy blocks arxiv.org); their quantitative claims are not recorded here.
- No paper found on quantizing Jamba or Zamba specifically; no linear-attention (Lightning/RetNet-style) quantization paper surfaced.
- MambaQuant's method details (KLT-enhanced rotation) are from memory and could not be confirmed.

---

## Area 2 — Diffusion language models (LLaDA, Dream, CoDA) and quantization

### Takeaway
Five PTQ papers (Aug 2025 – Jun 2026) exist and agree that dLLMs have *more spatially dispersed* massive activation outliers than AR models, making W4A4 smoothing methods collapse while rotation methods survive; one 2026 study finds a diffusion coder degrades *less* than an AR counterpart under 2–4-bit weight quantization. No QAT, no theory, nothing on closed models (Mercury, Gemini Diffusion).

### Cited Findings
- "Quantization Meets dLLMs" (first systematic PTQ study) identifies two outlier types in LLaDA-Base, LLaDA-Instruct and Dream: Normal Outliers (large across tokens) and Massive Outliers (extreme on few tokens); W4A16 is the best weight-only point, GPTQ beats AWQ, degradation is concentrated on math reasoning and code, and the instruction-tuned LLaDA is more robust than base — [arXiv 2508.14896](https://arxiv.org/abs/2508.14896). Secondary summaries add: 3-bit causes severe degradation on logic tasks; W8A8 <1% loss; at W4A4 SmoothQuant collapses while DuQuant/QuaRot retain accuracy; dLLM massive outliers are spread over more tokens than AR models' first-token outliers **[secondary]** — [Emergent Mind summary](https://www.emergentmind.com/papers/2508.14896); [Moonlight review](https://www.themoonlight.io/en/review/quantization-meets-dllms-a-systematic-study-of-post-training-quantization-for-diffusion-llms).
- DLLMQuant reports AWQ loses ≈16% accuracy on LLaDA at W4A4 and attributes dLLM-specific failure to (i) token distributions shifting across denoising steps, (ii) error accumulation over iterations, (iii) mixture of stable (unmasked) and still-masked tokens in each forward — [arXiv 2508.14090](https://arxiv.org/abs/2508.14090).
- Quant-dLLM targets 2-bit weights with Masked Calibration Simulation (calibration mimics timestep-dependent masking) and a data-aware any-order quantizer; plain PTQ transferred to dLLMs performs poorly at 2 bits — [arXiv 2510.03274](https://arxiv.org/abs/2510.03274).
- STaR-Quant ("State-Time Consistent PTQ") evaluates LLaDA-8B, LLaDA-1.5-8B and Dream-7B under weight-activation quantization; up to 1.69× speedup and 3.14× memory saving over FP16 — [arXiv 2606.04945](https://arxiv.org/html/2606.04945v1).
- "On the Quantization Robustness of Diffusion Language Models" applies GPTQ and a modified HAWQ to the diffusion coder CoDA-1.7B vs. AR Qwen3-1.7B and finds the AR model degrades more under 2–4-bit — [arXiv 2604.20079](https://arxiv.org/pdf/2604.20079).

### Inferences
- **Crowdedness:** ≈5 papers; all PTQ; all open-weight 7–8B models; one is a robustness comparison.
- **Strongest result:** W4A16 near-lossless on general QA for LLaDA/Dream (2508.14896); W4A4 feasible only with rotation (2508.14896) or step-aware calibration (DLLMQuant / STaR-Quant).
- **Open problems stated by the papers:** calibration–inference mismatch due to masking schedules; activation statistics that vary with denoising step (no step-conditional scales exist); error accumulation across iterations; low-bit (≤3b) collapse on reasoning/code.
- **Theory:** none. The "bidirectional denoising changes sensitivity" question is answered only empirically and in opposite directions (more dispersed outliers → harder activation quantization; but a coder dLLM → more robust weight quantization).
- **What would be new:** timestep-/mask-ratio-conditioned activation quantization (analogue of timestep-aware PTQ in image diffusion, which none of these cite as transferred); QAT for dLLMs; quantization of dLLM KV/feature caches (block-wise caching schemes exist for dLLMs but no quantized variant surfaced); a formal comparison of AR vs. masked-diffusion sensitivity via the different error-propagation structure (sequential vs. parallel refinement).

### Gaps
- No results for Mercury or Gemini Diffusion quantization (closed models).
- Could not read the 2604.20079 or STaR-Quant full text; the "AR degrades more" claim rests on the abstract.
- No dLLM KV-cache / cache quantization paper found; could not verify absence beyond this sweep.

---

## Area 3 — Multimodal LLMs: modality-dependent sensitivity, video/omni KV, audio LLMs

### Takeaway
VLM weight/activation PTQ is moderately crowded (≈8 papers since late 2024) and has a reproducible core finding — vision and language tokens have very different quantization sensitivity, and the ViT itself tolerates W4A8 — but the field disagrees on *which* side is more fragile at extreme bit-widths. Video/omni KV-cache quantization has ≈4 papers; audio-LLM weight quantization has none found.

### Cited Findings
- MBQ (CVPR 2025): a "significant sensitivity gap between language and vision tokens"; uses loss gradients w.r.t. vision and language token features as sensitivity indicators to reweight calibration; W4A8 on the ViT does not hurt and even improves some benchmarks (attributed to vision-token redundancy); up to +4.4% (W3) and +11.6% (W4A8) over SOTA for 7B–70B VLMs — [arXiv 2412.19509](https://arxiv.org/abs/2412.19509); [CVPR 2025 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_MBQ_Modality-Balanced_Quantization_for_Large_Vision-Language_Models_CVPR_2025_paper.pdf).
- Q-VLM (NeurIPS 2024): mines cross-layer dependency via an entropy proxy to assign blocks for rounding search; 2.78× memory compression and 1.44× generation speedup for 13B LLaVA without degradation; targets W4A4; LLaVA and MoE-LLaVA — [arXiv 2410.08119](https://arxiv.org/abs/2410.08119); [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/cffbaf4f47546ece96bb42c0edda40ee-Paper-Conference.pdf).
- SalQ-VLM: smoothing-based AWQ and MBQ "largely collapse" on Qwen2-VL at INT3 while Hessian-based methods stay stable — [arXiv 2508.03351](https://arxiv.org/html/2508.03351v3).
- A Llama-3.2-Vision study at ≈1–1.1-bit weights reports the vision model highly sensitive, the projector less so, the LM considerably sensitive — partly conflicting with MBQ's W4A8 finding (different bit regime) — surfaced via [Bi-VLM, arXiv 2509.18763](https://arxiv.org/pdf/2509.18763) **[attribution to Bi-VLM unverified]**.
- Other VLM PTQ: RGSQ (Riemannian geometry-sensitive) [arXiv 2609.25492](https://arxiv.org/html/2609.25492v1); Quant Experts (token-aware MoE error reconstruction) [arXiv 2602.24059](https://arxiv.org/pdf/2602.24059); survey [arXiv 2504.09724](https://arxiv.org/pdf/2504.09724).
- Video/omni KV quantization: VidKV compresses video-LLM KV to 1.5-bit / 1.58-bit with almost no drop on LLaVA-OV-7B and Qwen2.5-VL-7B — [HF paper page 2503.16257](https://huggingface.co/papers/2503.16257); OmniKVQuant (training-free, Qwen2.5-Omni / Qwen3-Omni) beats TurboQuant at 2-bit across seven audio-visual benchmarks — [arXiv 2609.11582](https://arxiv.org/pdf/2609.11582); WindowQuant mixed-precision KV via window-level similarity for VLMs — [arXiv 2605.02262](https://arxiv.org/pdf/2605.02262); QuantWM 2-bit temporally consistent KV for world models / video generation — [arXiv 2609.26425](https://arxiv.org/pdf/2609.26425); AccKV (audio-video KV eviction, not quantization) notes video+audio extend the temporal dimension and inflate KV — [arXiv 2511.11106](https://arxiv.org/html/2511.11106).

### Inferences
- **Crowdedness:** VLM weight/act PTQ ≈8 (moderate); video/omni KV quantization ≈4 (growing fast in 2026); audio-only LLM weight PTQ = 0 found.
- **Strongest result:** Q-VLM W4A4 on LLaVA-13B lossless; VidKV 1.5-bit KV on video LLMs near-lossless.
- **Open problems:** modality-sensitivity ordering is bit-regime dependent and unresolved; vision-token redundancy is exploited heuristically (gradient weights) not quantified; no joint token-pruning × quantization budget; audio encoders (Whisper-style) + LLM backbones unstudied.
- **Theory:** none.
- **What would be new:** a sensitivity model tying token redundancy (e.g., attention entropy or effective rank of vision tokens) to achievable bit-width; audio-LLM PTQ (first paper); unified compression budget across vision-token dropping and KV bit-width for video.

### Gaps
- No audio/speech LLM (Qwen-Audio, SALMONN) quantization paper surfaced; search budget ran out before a second targeted attempt.
- Could not confirm which paper contains the Llama-3.2-Vision 1-bit sensitivity analysis.

---

## Area 4 — Embedding / retrieval side: binary & int8 embeddings, QAT for embeddings, PQ in vector DBs, ColBERT token quantization, inner-product theory

### Takeaway
Industrial practice (Cohere, mixedbread, Vespa, HF) is far ahead of academic theory: binary/int8 embeddings with ~96–97% retained quality are standard, and SIGMOD-level theory exists for *data-oblivious* binary quantization (RaBitQ's O(1/√D) bound), but theory for *learned contrastive* embeddings under quantization is a single 2026 paper, and ColBERT token-level quantization beyond ColBERTv2's 2-bit residuals has 1–2 papers.

### Cited Findings
- Hugging Face embedding-quantization write-up: int8 retains ≈97% and binary ≈96.45% of default retrieval performance, with binary at 3.125% of the size (post-hoc quantization, not QAT) — [HF blog](https://huggingface.co/blog/embedding-quantization); [Medium summary](https://medium.com/@abletobetable/two-methods-for-optimizing-embedding-models-1d9f43d8a7a6) **[secondary]**.
- Vespa: mixedbread's model trained with MRL + a binary-quantization loss retains 90% MTEB-Retrieval accuracy using 64 int8 dims (512 bits) binarized from 512 float dims; Cohere and mixedbread both ship binary embeddings — [Vespa blog](https://blog.vespa.ai/combining-matryoshka-with-binary-quantization-using-embedder/); model card [mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1).
- Bekko Embedding gives a reference QAT recipe: an auxiliary contrastive loss on embeddings fake-quantized to int8 and unsigned-binary, with a straight-through estimator for binarization — [arXiv 2607.25180](https://arxiv.org/pdf/2607.25180).
- Qwen3-VL-Embedding integrates MRL and QAT into the training pipeline — [arXiv 2601.04720](https://arxiv.org/pdf/2601.04720).
- "Covariance Structure and Coordinate Heterogeneity Govern Binary Quantization of Contrastive Embeddings" analyzes the 1–2-bit regime with Stein's lemma, Hoeffding and high-dimensional concentration — [arXiv 2605.17524](https://arxiv.org/pdf/2605.17524) (theorem statements not retrieved).
- "Quantization Beyond Uniform Bit Allocation" partitions an embedding into contiguous buckets with variable bits; on MRL embeddings it improves recall up to +8% (PQ) and +18% (SQ) at equal compression — [arXiv 2608.19388](https://arxiv.org/abs/2608.19388).
- QAMA pairs MRL with bitwise (XOR/NOT/POPCOUNT) retrieval at 0.5–2 bits/dimension — [ACM DL 10.1145/3746252.3761077](https://dl.acm.org/doi/pdf/10.1145/3746252.3761077).
- SMEC rethinks MRL for retrieval embedding compression — [arXiv 2510.12474](https://arxiv.org/pdf/2510.12474).
- RaBitQ (SIGMOD 2024): quantizes D-dim vectors to D bits via random orthogonal rotation + sign; unbiased estimator with a sharp probabilistic error bound (reviewer: O(1/√D)); argues PQ has no bound and "fails disastrously" on some datasets; the bound is used at query time to decide re-ranking — [ACM DL 10.1145/3654970](https://dl.acm.org/doi/pdf/10.1145/3654970); [RaBitQ Library](https://vectordb-ntu.github.io/RaBitQ-Library/); Extended RaBitQ (scalar) — [DEV post](https://dev.to/gaoj0017/extended-rabitq-an-optimized-scalar-quantization-method-83m). "Provable Quantization with Randomized Hadamard Transform" is a 2026 follow-on — [arXiv 2605.13810](https://arxiv.org/pdf/2605.13810).
- Classical binary-embedding theory: sign random projections — bit-disagreement probability equals angular distance; fundamental limits in Yi et al. (ICML 2015) — [PMLR v37](https://proceedings.mlr.press/v37/yi15.pdf); Circulant Binary Embedding — [arXiv 1405.3162](https://arxiv.org/pdf/1405.3162); structured hashed projections — [arXiv 1511.05212](https://arxiv.org/abs/1511.05212).
- ColBERTv2 stores each token as nearest-centroid ID + residual quantized to 2 bits/dim by default — [arXiv 2112.01488](https://arxiv.org/pdf/2112.01488) / [NAACL 2022](https://aclanthology.org/2022.naacl-main.272.pdf); PLAID adds centroid pruning and optimized decompression kernels — [arXiv 2205.09707](https://arxiv.org/pdf/2205.09707); WARP — [arXiv 2501.17788](https://arxiv.org/html/2501.17788v2); a 2026 paper estimates default ColBERTv2 at ≈8 B/token and proposes CrossQ, task-aligned cross-token conditional quantization for late interaction — [arXiv 2608.19204](https://arxiv.org/pdf/2608.19204); query-aware token budgeting for late-interaction visual document retrieval — [arXiv 2609.07262](https://arxiv.org/pdf/2609.07262).
- HAKARI-Bench traces the binary-candidate-then-rescore pattern to BPR (Yamada et al., 2021) — [arXiv 2606.22778](https://arxiv.org/pdf/2606.22778).

### Inferences
- **Crowdedness:** practitioner material abundant; academic QAT-for-embeddings ≈3; theory for learned embeddings = 1 (2605.17524); ColBERT token quantization ≈2 beyond ColBERTv2; vector-DB quantization theory ≈2 (RaBitQ line).
- **Strongest result:** RaBitQ's provable O(1/√D) unbiased inner-product estimate (data-oblivious); mixedbread's 90% retention at 32× compression (MRL+BQL).
- **Open problems:** no theory connecting contrastive training geometry (anisotropy, coordinate heterogeneity) to optimal post-hoc vs. QAT bit allocation; no inner-product-preservation guarantee for *trained* binary heads; late-interaction MaxSim under token-level quantization lacks an error analysis.
- **Theory:** exists for random projections and RaBitQ; one paper for contrastive embeddings.
- **What would be new:** a RaBitQ-style bound that is *data-aware* (uses the learned covariance) and a matching QAT objective; MaxSim error bounds for quantized token embeddings; joint MRL-dimension × bit-width optimization with a provable recall guarantee.

### Gaps
- Cohere's training recipe for int8/binary embeddings is not public in the sources found.
- Theorem statements of 2605.17524 not retrieved.

---

## Area 5 — Speculative decoding × quantization

### Takeaway
"Quantization as the draft" is a real, ~7-paper niche (QSpec, QuantSpec, ML-SpecQD, Quasar, SpecDec-meets-Quant, QSV, component-aware hybrid SSD) with strong empirical acceptance rates (71–91% for MXFP4 self-drafts), but the theory is generic (acceptance = 1 − TV; Pinsker-type KL bound) and no paper derives acceptance as a function of quantization error, nor analyzes what a *quantized verifier* does to exactness.

### Cited Findings
- QSpec: a W4A4 joint weight-activation scheme drafts, a W4A16 weight-only scheme verifies, sharing weights and KV for near-zero-cost switching; motivated by W4A4's degradation on multi-step reasoning — [arXiv 2410.11305](https://arxiv.org/abs/2410.11305).
- QuantSpec: self-speculative draft with 4-bit weights and a hierarchical 4-bit KV cache; >90% acceptance, up to ≈2.5× end-to-end speedup, long-context focus — [arXiv 2502.10424](https://arxiv.org/abs/2502.10424).
- ML-SpecQD: MXFP4 draft of the same model; Llama2-7B QA 71.2% acceptance vs 41.7% for a 68M draft; Qwen2.5-Coder-7B code 91.1% vs 61.7% for a 0.5B draft; multi-level drafting up to 2.72× over BF16; gains largest when memory-bandwidth-bound — [arXiv 2503.13565](https://arxiv.org/pdf/2503.13565) (speedups partly **[secondary]** via [alphaxiv](https://www.alphaxiv.org/abs/2503.13565)).
- "Speculative Decoding Meets Quantization" evaluates compatibility of lightweight drafts with quantized targets and proposes a hierarchical framework; notes self-speculation inherently limits speedup — [arXiv 2505.22179](https://arxiv.org/pdf/2505.22179).
- Quasar replaces the BF16 verifier with a W8A8 one; comparable acceptance length on OpenPangu and Qwen3, 1.28× throughput; asserts the KL between quantized and original distributions is negligible (no derivation seen) — [arXiv 2603.01399](https://arxiv.org/pdf/2603.01399).
- Quantize-Sample-and-Verify: adaptive edge-cloud speculative decoding with a quantized edge model — [arXiv 2507.00605](https://arxiv.org/pdf/2507.00605).
- Theory: expected per-position acceptance equals distributional overlap = 1 − TV (AngelSpec) — [arXiv 2607.25852](https://arxiv.org/pdf/2607.25852); Pinsker gives acceptance ≥ 1 − √(KL/2), observed rates far exceed it (distillation context) — [arXiv 2605.30580](https://arxiv.org/pdf/2605.30580); "When Is a Draft Accepted?" characterizes exact KL thresholds for greedy and relaxed acceptance — [arXiv 2606.30265](https://arxiv.org/html/2606.30265); exponential-race formulation — [arXiv 2504.15475](https://arxiv.org/pdf/2504.15475); speculative verification via information gain — [arXiv 2509.24328](https://arxiv.org/html/2509.24328v2).
- Hardware anatomy of SD on consumer GPUs — [arXiv 2607.17283](https://arxiv.org/html/2607.17283).

### Inferences
- **Crowdedness:** ≈7 systems papers (2024-10 → 2026-03); theory papers ≈4 but none quantization-specific.
- **Strongest result:** ML-SpecQD 91.1% acceptance / 2.72× with an MXFP4 self-draft; QuantSpec >90% at long context.
- **Open problems:** (i) a quantized verifier (Quasar) is no longer lossless w.r.t. the FP model, so "exactness" is redefined relative to the W8A8 model — nobody quantifies the drift; (ii) acceptance vs. bit-width curves are reported, not modeled; (iii) self-draft speedup ceiling.
- **Theory:** generic TV/KL only. There is an obvious missing link: per-token quantization error → logit perturbation → TV distance → acceptance rate.
- **What would be new:** a predictive model of acceptance from layer-wise quantization error (e.g., via logit-perturbation bounds), enabling bit-width selection for the draft without running SD; an "ε-exact" verification notion for quantized verifiers with a certified bound on output distribution shift; hybrid-architecture self-speculation using quantized SSM halves (combining Areas 1 and 5, only 2605.01106 touches it).

### Gaps
- Quasar's per-draft acceptance numbers not retrieved.
- No paper found on verifying with a quantized *KV cache* alone vs. quantized weights alone (QuantSpec mixes both).

---

## Area 6 — LM head / logits / sampling: vocabulary projection, softmax over quantized logits, temperature, logit storage for distillation

### Takeaway
Three to four 2026 papers single out the LM head as structurally special (its error hits softmax with no normalization or residual after it); three papers study quantization × temperature/calibration/entropy; three papers study sparse logit storage for distillation — but none quantize logit *values*, and no softmax-perturbation theory for quantized logits exists in the LLM setting.

### Cited Findings
- ARCHead: the LM head is a special case because its error "is not followed by normalization, a residual path, or another learned transformation; it reaches the softmax directly"; a small error along a frequently activated direction alters many vocabulary logits; compresses the head with quantized low-rank factors plus a group-wise INT4 residual — [arXiv 2608.02703](https://arxiv.org/html/2608.02703).
- LFQ: block-wise PTQ skips the unembedding and uses MSE, letting token probabilities drift from FP; proposes quantizing the final block by minimizing cross-entropy between quantized and FP logits — [arXiv 2605.29756](https://arxiv.org/html/2605.29756).
- A layer-importance metric paper measures how far LM-head weights can be compressed before generation quality degrades — [arXiv 2608.26926](https://arxiv.org/pdf/2608.26926).
- Variable Bit-width Quantization reports the LM head averaging 1.09 bits and uses KL and argmax-flip rate as output-distribution metrics — [arXiv 2607.02893](https://arxiv.org/abs/2607.02893).
- "The Joint Effect of Quantization and Sampling Temperature": factorial study, nine models from six families, INT4/INT8, six temperatures from greedy to 1.0, measuring safety (attack success rate) — [arXiv 2606.29581](https://arxiv.org/pdf/2606.29581).
- ReSET: quantization (NVFP4) shifts token-level entropy, which can distort inference-time sampling; proposes step-aware temperature scaling; states temperature × quantized inference is underexplored — [arXiv 2606.13233](https://arxiv.org/pdf/2606.13233).
- GPTQ 4-bit lowers confidence on true labels and raises calibration error, worst on examples the FP model was already unsure about — [arXiv 2604.08976](https://arxiv.org/html/2604.08976v1); related: Tethered Reasoning (entropy vs hallucination in quantized LLMs) — [arXiv 2602.17691](https://arxiv.org/pdf/2602.17691).
- Logit storage: full teacher distributions for Llama 3 on 1T tokens ≈128 PB; Top-K storage is biased and needs ≈6400 logits; importance-sampled sparse logits are unbiased with <10% training overhead (300M–3B students) — [arXiv 2503.16870](https://arxiv.org/abs/2503.16870) (ACL 2025); offline top-100 logit caching matches online KD at ≈29% faster iterations with a fused chunked KL — [arXiv 2608.03796](https://arxiv.org/abs/2608.03796); TopKD — [arXiv 2508.04539](https://arxiv.org/pdf/2508.04539).

### Inferences
- **Crowdedness:** LM-head-specific quantization ≈4; quantization × sampling/temperature/calibration ≈4; sparse-logit storage ≈3 (none quantize values).
- **Strongest result:** ARCHead's structural argument + LFQ's CE-objective fix; the 128 PB storage number quantifies why logit compression matters.
- **Open problems:** no bit-width for stored logits is studied (sparsity only); the temperature-equivalence question (does W4 ≈ FP16 at a different T?) is only hinted (ReSET); calibration shift under quantization is measured, not corrected in a principled way.
- **Theory:** none for softmax under quantized logits in LLMs.
- **What would be new:** (a) a perturbation bound: logit error from head quantization → KL/TV of next-token distribution, as a function of hidden-state statistics, giving a head bit-allocation rule; (b) lossy logit codecs (quantized sparse logits) with unbiasedness guarantees for distillation; (c) temperature re-calibration as a zero-cost post-quantization fix with a derived mapping.

### Gaps
- 2606.29581's quantitative findings are fragmentary in the excerpts.
- No source found on softmax over quantized logits in the LLM setting beyond the above.

---

## Area 7 — Communication / distributed-inference quantization (TP all-reduce, EP all-to-all, PD KV transfer, offload/prefix stores, pipeline activations)

### Takeaway
Industrial FP8 MoE dispatch (DeepSeek-V3, MegaScale-MoE, FP8-Flow-MoE) and low-bit TP all-reduce (Flash Communication v1/v2, Apple/CMU 4.2-bit) are established; prefill–decode KV *transfer* compression became a six-paper burst in 2025–26 (SpectrumKV, SplitZip, KVServe, Lynx, HACK); precision of *stored* prefix caches (CacheGen/LMCache) is handled by one 2023 system. No theory anywhere; the striking empirical fact is that INT4 KV transfer tolerance is model-specific (Qwen2.5-7B collapses, Mistral/Gemma fine).

### Cited Findings
- DeepSeek-V3 quantizes activations to FP8 before the MoE up-projection and dispatches them in FP8; scales are powers of two; all-to-all runs over InfiniBand then NVLink — [arXiv 2412.19437](https://arxiv.org/abs/2412.19437).
- MegaScale-MoE compresses communication to FP8, replacing BF16 TP reduce-scatter with FP8 all-to-all (FP32 reduction) and per-token activation quantization to limit loss misalignment — [arXiv 2505.11432](https://arxiv.org/abs/2505.11432).
- FP8-Flow-MoE: Transformer Engine's blockwise FP8 recipe leaves dispatch all-to-all in higher precision; proposes a casting-free FP8 dataflow without double-quantization error — [arXiv 2511.02302](https://arxiv.org/abs/2511.02302).
- UCCL-EP: a per-token FP8 activation at hidden size 7168 is ≈7 KB, so EP traffic is many small transfers — [arXiv 2512.19849](https://arxiv.org/pdf/2512.19849); CloudMatrix384 reports dispatch/combine ≥25% of MoE execution time — [arXiv 2508.02520](https://arxiv.org/pdf/2508.02520).
- Flash Communication: low-bit fine-grained activation quantization + two-step all-reduce (quant/dequant steps from N to 2); >3× intra-node comm speed, 2× TTFT; up to 2× on L40 and 1.19× on A100 for LLaMA-3 — [arXiv 2412.04964](https://arxiv.org/html/2412.04964v1); V2 adds bit splitting and spike reserving for arbitrary bit widths — [arXiv 2508.03760](https://arxiv.org/pdf/2508.03760).
- Apple/CMU "Towards Low-bit Communication for TP LLM Inference": 16 → 4.2 bits average with nearly all performance preserved — [arXiv 2411.07942](https://arxiv.org/pdf/2411.07942). Adjacent: FlexLink (+27% NVLink bandwidth) [arXiv 2510.15882](https://arxiv.org/pdf/2510.15882); SiFAR synchronization-free all-reduce [arXiv 2607.08973](https://arxiv.org/pdf/2607.08973).
- SpectrumKV: per-token FP16/INT8/INT4 tiers for PD KV transfer with a deployment-time probe; Qwen2.5-7B "catastrophically fails" under INT4 KV while Mistral-7B and Gemma-2-9B are stable — [arXiv 2606.08635](https://arxiv.org/abs/2606.08635).
- SplitZip: lossless, bit-exact KV compressor exploiting BF16 exponent redundancy — [arXiv 2605.01708](https://arxiv.org/abs/2605.01708); KVServe: service-aware compression, cross-cluster bandwidth often <100 Gbps — [arXiv 2605.13734](https://arxiv.org/pdf/2605.13734); Lynx: progressive speculative quantization for KV transfer — [arXiv 2607.01831](https://arxiv.org/html/2607.01831v1); HACK: compute directly on compressed KV; PD overhead up to 42.2% of JCT — [arXiv 2502.03589](https://arxiv.org/pdf/2502.03589).
- "Disaggregated Quantization": different formats for prefill vs decode; removing activation quantization during decode improves decode-heavy tasks on Qwen3 and Gemma3 — [arXiv 2609.26333](https://arxiv.org/html/2609.26333v1); Mix-Quant: quantized prefill, precise decode for agentic LLMs — [arXiv 2605.20315](https://arxiv.org/html/2605.20315v1).
- CacheGen encodes precomputed KV into bitstreams with quantization + arithmetic coding, 3.5–4.3× smaller, bandwidth-adaptive, no retraining — [arXiv 2310.07240](https://arxiv.org/abs/2310.07240); now part of LMCache (offloading + PD) — [arXiv 2510.09665](https://arxiv.org/pdf/2510.09665); disk/S3 storage — [LMCache blog](https://blog.lmcache.ai/en/2025/07/31/cachegen-store-your-kv-cache-on-disk-or-s3-load-blazingly-fast/); ShadowServe interference-free fetching — [arXiv 2509.16857](https://arxiv.org/html/2509.16857v1); KV-management survey — [arXiv 2607.02574](https://arxiv.org/pdf/2607.02574).
- Pipeline parallel: TAH-Quant, activation quantization for PP over slow networks — [arXiv 2506.01352](https://arxiv.org/html/2506.01352v1).

### Inferences
- **Crowdedness:** FP8 EP comm ≈4 (industrial); TP all-reduce quantization ≈4; PD KV transfer compression ≈6 (hot in 2026); stored-prefix precision ≈1–2; PP activation quant ≈1; "phase-dependent precision" (prefill vs decode) ≈2 (new axis, 2025-26).
- **Strongest result:** Flash Communication 2× TTFT; CacheGen 3.5–4.3× with arithmetic coding; SpectrumKV's model-specific INT4 cliff.
- **Open problems:** no principled bit allocation across *transfer* vs *compute* precision; INT4-KV-transfer tolerance is unexplained and model-dependent; lossless vs lossy split for KV (SplitZip vs SpectrumKV) not unified; prefix-store precision tiers (hot GPU / CPU / SSD / S3) never optimized jointly.
- **Theory:** none.
- **What would be new:** rate–distortion treatment of KV transfer with per-token importance (bridging SpectrumKV's tiers and CacheGen's entropy coding); an explanation/predictor of the per-model INT4 cliff (likely tied to key-channel outlier structure, cf. Area 8); precision-tiered prefix caches with storage-aware error budgets.

### Gaps
- No INT8 all-reduce for MoE *training* found; no quantified data on SSD-tier prefix precision choices.

---

## Area 8 — Positional-encoding and normalization numerics (RoPE under low precision, RoPE scaling × quantization, RMSNorm/epsilon, residual stream)

### Takeaway
Pre- vs post-RoPE key quantization is a contested ~6-paper topic (KVQuant, AnTKV, TaSQ favor pre-RoPE; YAKV finds no gain with Hadamard quantizers), one 2026 paper (Block-GTQ) gives the first *frequency-block* bit-allocation formulation, and a 2025 pair (Q-ROAR, "Rethinking RoPE scaling") shows position interpolation × weight quantization interacts destructively. RMSNorm-epsilon/FP8 and residual-stream quantization have no dedicated papers — only massive-activation analyses and an engineering bug report.

### Cited Findings
- KVQuant (NeurIPS 2024) quantizes keys per-channel *before* RoPE because outlier-channel magnitudes become inconsistent after RoPE; applies RoPE post-dequant in a fused kernel — [KVQuant PDF](https://people.eecs.berkeley.edu/~ysshao/assets/papers/kvquant-neurips2024.pdf).
- AnTKV: pre-RoPE K has smaller inter-cluster distance and lower intra-cluster variance → lower VQ error — [arXiv 2506.19505](https://arxiv.org/pdf/2506.19505); TaSQ: VQ on pre-RoPE keys with covariance-aware grouping preserving RoPE pairs; less degradation as context grows — [arXiv 2610.03027](https://arxiv.org/html/2610.03027).
- Block-GTQ / "RoPE-Aware Bit Allocation": a key's contribution to future logits decomposes into a position-dependent sum over 2-D frequency blocks; allocates bits per block by energy, built on TurboQuant-MSE; RoPE is an orthogonal 2×2 rotation so block norms are computable pre-RoPE — [arXiv 2606.24033](https://arxiv.org/abs/2606.24033).
- Counter-evidence: YAKV finds pre-RoPE key quantization gives no significant gain with the Hadamard-based HIGGS quantizer (Llama-3.1-8B MultiNeedle 93.36 vs 93.73; Qwen3-4B Text2JSON 90.83 vs 91.05) — [arXiv 2604.08426](https://arxiv.org/pdf/2604.08426).
- ReQAT's Q-FIT applies channel-wise scaling before RoPE and a shift after; notes post-RoPE key magnitudes oscillate across tokens — [arXiv 2606.15682](https://arxiv.org/pdf/2606.15682); RAP: RoPE-aligned KV pruning — [arXiv 2602.02599](https://arxiv.org/pdf/2602.02599).
- Q-ROAR: under YaRN 32K/64K, RTN and AWQ show clear accuracy/perplexity drops; LLaMA-2-7B at 32K (YaRN s=8) WikiText2 ppl FP16 6.09 / RTN-W4 6.41 / Q-ROAR-W4 6.19 — [arXiv 2509.14391](https://arxiv.org/pdf/2509.14391).
- "Rethinking RoPE scaling in quantized LLMs": quantized models degrade much more sharply even with YaRN; NTK/linear scaling fail entirely (ppl >100); attributes this to long-context aliasing and dynamic-range dilation; fixes with per-band scales on W_Q/W_K — [arXiv 2510.00028](https://arxiv.org/pdf/2510.00028).
- WitCert: Qwen2.5-1.5B + YaRN loses coverage at 128k, but Llama-3.1-8B (native 128k, no extrapolation) does not — [arXiv 2607.28699](https://arxiv.org/pdf/2607.28699).
- Normalization numerics: massive activations (>100 magnitude, ≥1000× median) skew the RMSNorm denominator — [arXiv 2402.17762](https://arxiv.org/pdf/2402.17762); the FFN generates/sustains them and the pre-FFN RMSNorm controls their magnitude — [arXiv 2605.08504](https://arxiv.org/html/2605.08504); "Scaling FP8 Training to Trillion-Token LLMs" (ICLR 2025) concludes RMSNorm is *not* the cause of FP8 instability in Llama (weight-norm/correlation effects are) — [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/f48b5133e89854a9e97cc22a6db83f25-Paper-Conference.pdf); "Towards Fully FP8 GEMM LLM Training at Scale" — [arXiv 2505.20524](https://arxiv.org/pdf/2505.20524).
- Engineering evidence of epsilon fragility: coremltools PR #2882 fixes RMSNorm by rescaling epsilon with the max-abs input scaling introduced to avoid FP16 overflow in the square — [coremltools PR 2882](https://github.com/apple/coremltools/pull/2882).

### Inferences
- **Crowdedness:** pre/post-RoPE ≈6 (contested); RoPE-scaling × quantization ≈3; frequency-resolved error theory = 1 (Block-GTQ); RMSNorm/epsilon/FP8 = 0 dedicated papers; residual-stream quantization = 0 found.
- **Strongest result:** Block-GTQ's per-frequency-block decomposition; Q-ROAR's 6.41→6.19 recovery at 32K.
- **Open problems:** whether pre-RoPE helps depends on the quantizer (per-channel yes, Hadamard no) — unresolved; the "aliasing + dynamic-range dilation" mechanism is qualitative; no analysis of RoPE frequency bands' differential sensitivity to *weight* (W_Q, W_K) quantization vs *KV* quantization.
- **Theory:** partial (Block-GTQ decomposition). Nothing for norms.
- **What would be new:** a frequency-domain error analysis unifying W_Q/W_K quantization, KV quantization and position interpolation (all three perturb the same q·k rotation structure); a study of the residual stream itself as a quantization target (massive-activation-aware); a numerics audit of RMSNorm epsilon/scale under FP8/FP4 storage of the residual.

### Gaps
- No paper found that quantizes the residual stream between blocks for inference memory/bandwidth.
- Block-GTQ's quantitative gains not retrieved.

---

## Area 9 — Hardware / co-design 2025–26: LUT GEMM, bit-serial, any-precision, NVFP4/MXFP4, FP6, NPUs, and where kernels lag algorithms

### Takeaway
LUT-based mpGEMM is a moderately crowded hardware-venue topic (~10 papers: T-MAC, FLUTE, LUT Tensor Core, Vec-LUT, FluxBin, TENET, T-MAN, LUT-LLM, XtraMAC, TileFuse); Blackwell NVFP4 (E2M1 + E4M3 block-16 scales) is now the format algorithms target; FP6 has one established line; mobile NPUs (Hexagon, Core ML) support only per-tensor/per-channel quantization, which is the clearest documented algorithm–kernel gap. Nobody quantifies the cost of non-uniform/VQ decode or per-token scaling systematically.

### Cited Findings
- LUT Tensor Core (ISCA 2025): software–hardware co-design for LUT-based mixed-precision GEMM; 1.44× compute density and energy efficiency over a SOTA LUT accelerator; code in the T-MAC repo — [arXiv 2408.06003](https://arxiv.org/pdf/2408.06003); [ACM DL 10.1145/3695053.3731057](https://dl.acm.org/doi/10.1145/3695053.3731057).
- FLUTE (EMNLP 2024 Findings): LUT engine for non-uniform lookup-table-quantized weights — [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.724.pdf). T-MAC is described as the SOTA LUT mpGEMM kernel for ternary LLMs on CPU — [Vec-LUT, arXiv 2512.06443](https://arxiv.org/html/2512.06443v2). Others: FluxBin — [arXiv 2608.15602](https://arxiv.org/pdf/2608.15602); TENET sparsity-aware LUT for ternary — [arXiv 2509.13765](https://arxiv.org/pdf/2509.13765); LUT-LLM on FPGA — [arXiv 2511.06174](https://arxiv.org/pdf/2511.06174); XtraMAC mixed-precision MAC on FPGA — [arXiv 2605.06052](https://arxiv.org/pdf/2605.06052).
- NPUs: T-MAN enables end-to-end low-bit LLM inference on NPUs via unified table lookup — [arXiv 2511.11248](https://arxiv.org/pdf/2511.11248); TileFuse fused mixed-precision kernels for AMD NPUs — [arXiv 2606.11357](https://arxiv.org/pdf/2606.11357); "Fast On-device LLM Inference with NPUs" (ASPLOS 2025) — [PDF](https://xumengwei.github.io/files/ASPLOS25-NPU.pdf); scaling test-time compute on mobile NPUs — [arXiv 2509.23324](https://arxiv.org/pdf/2509.23324). Documented gaps: Hexagon NPUs lack native support for fine-grained (group) quantization; QNN supports only per-tensor/per-channel; dequantizing unsupported formats on the NPU is ≈10× slower, making one kernel 3.8× slower than CPU — sources: [arXiv 2509.23324](https://arxiv.org/pdf/2509.23324), [ASPLOS 2025](https://xumengwei.github.io/files/ASPLOS25-NPU.pdf); Core ML-style stacks support restricted subsets (per-tensor, per-channel, symmetric) — [arXiv 2606.11357](https://arxiv.org/pdf/2606.11357).
- NVFP4 pretraining (NVIDIA): E2M1 elements with E4M3 scales over blocks of 16 (MXFP4: UE8M0 over 32); Blackwell Tensor Cores consume block scales in hardware; FP4 throughput 4× BF16 on GB200, 6× on GB300; MXFP4 relative error ≈2.5% vs 1.5% NVFP4; stability via Random Hadamard transforms, 2-D block scaling, stochastic rounding and keeping sensitive layers (≈15% of linears, mostly late **[secondary]**) in BF16; 12B model over 10T tokens matches higher-precision baseline — [arXiv 2509.25149](https://arxiv.org/abs/2509.25149); [NVIDIA blog](https://research.nvidia.com/labs/eai/blogs/pushing-intelligence-to-4-bit/). MXFP4 on native FP4 hardware — [arXiv 2605.09825](https://arxiv.org/pdf/2605.09825); FP4 pretraining exploration — [arXiv 2502.11458](https://arxiv.org/pdf/2502.11458); Quartet fully-MXFP4 training — [arXiv 2505.14669](https://arxiv.org/pdf/2505.14669).
- FP6-LLM / Quant-LLM (USENIX ATC 2024): TC-FPx kernel runs arbitrary-bit FP weights (FP6, FP5) on Tensor Cores; first FP6 weight-only Tensor Core support — [arXiv 2401.14112](https://arxiv.org/abs/2401.14112); [ATC24 PDF](https://www.usenix.org/system/files/atc24-xia.pdf).
- Any-Precision LM: store an n-bit model and derive (n−1), (n−2)-bit versions from MSBs — concept confirmed via [arXiv 2510.16805 survey](https://arxiv.org/html/2510.16805v1) **[original ID unverified; believed 2402.10517, ICML 2024]**.
- New hardware-friendly formats: SQ-format unified sparse-quantized format — [arXiv 2512.05409](https://arxiv.org/pdf/2512.05409); SPARQLe sub-precision activation representation — [arXiv 2606.00365](https://arxiv.org/pdf/2606.00365); TriGen NPU co-design — [arXiv 2602.12962](https://arxiv.org/pdf/2602.12962).

### Inferences
- **Crowdedness:** LUT/co-design ≈10 (moderately crowded, mostly ISCA/MICRO/FPGA venues); NPU-format-gap papers ≈4 (growing); FP6 ≈1–2; "cost of non-uniform grids / VQ decode / per-token scales" quantified systematically = 0 found.
- **Strongest result:** NVFP4 pretraining of 12B/10T tokens matching BF16; LUT Tensor Core 1.44× over prior LUT accelerators.
- **Where kernels lag algorithms (documented):** group-wise/fine-grained scales on Hexagon; non-uniform (codebook) weights need LUT paths (FLUTE/T-MAC) rather than native tensor cores; VQ decode cost and per-token activation scaling overhead are asserted, not measured, in these sources.
- **Theory:** none (engineering).
- **What would be new:** a cost model mapping algorithmic choices (grid type, group size, VQ codebook size, per-token vs per-tensor scales) to measured latency on Blackwell/NPUs, used to co-optimize accuracy and speed; "NPU-native" PTQ that reaches group-wise accuracy under per-channel-only constraints (e.g., via rotations folded into weights).

### Gaps
- QTIP / vector-quantization decode-cost studies did not surface; Apple ANE format documentation not found in this sweep.

---

## Area 10 — Activation sparsity × quantization; pruning × quantization; low-rank × quantization frontier

### Takeaway
Joint weight-sparsity × quantization is moderately crowded (~8 papers) and has a clear stated tension (quantization wants a narrow range, pruning wants high magnitude variance — OBR); activation-sparsity × quantization is newer (~4 papers) with the sharpest insight from SharQ: sparsification and FP4 block scaling errors are *coupled*, not additive. No formal theory of the interaction exists.

### Cited Findings
- TEAL: training-free 40–50% model-wide activation sparsity across Llama-2/3 and Mistral 7B–70B; 1.53×/1.8× decode speedups; compatible with 8-bit RTN, 4-bit AWQ and 2/3-bit weights; CATS sparsifies only part of SwiGLU blocks — [arXiv 2408.14690](https://arxiv.org/abs/2408.14690).
- "Enabling Dynamic Sparsity in Quantized LLM Inference": works with GPTQ at any bitwidth; up to 1.55× decode throughput; Q4_K perplexity 5.28 → 5.27 at 25% sparsity, 5.65 at 50% — [arXiv 2511.04477](https://arxiv.org/pdf/2511.04477).
- SharQ: directly applying N:M masks before FP4 couples sparsification loss with quantization error because the FP4 block scale is chosen from the altered distribution; extracts an outlier-dominated sparse backbone (sparse FP4 GEMM) and a dense FP4 residual defined relative to the *quantized* backbone; 2.2–2.4× latency vs FP16 and 1.2–1.4× throughput vs FP8 on RTX 5090 (authors' benchmarks) — [arXiv 2606.26587](https://arxiv.org/abs/2606.26587); code [actypedef/SharQ](https://github.com/actypedef/SharQ).
- OBR ("Optimal Brain Restoration"): quantization favors a narrow weight range while pruning benefits from large magnitude variation — the objectives conflict; up to 4.72× speedup and 6.4× memory vs FP16-dense with INT4 sparse GEMM — [arXiv 2509.11177](https://arxiv.org/html/2509.11177).
- Shared inverse-Hessian pruning + quantization (2:4 + 4-bit) for specialized domains — [arXiv 2502.18424](https://arxiv.org/pdf/2502.18424); BESA + OmniQuant 4-bit outperforms Wanda — [arXiv 2402.16880](https://arxiv.org/pdf/2402.16880); joint structural pruning + mixed-precision beats best 2:4 by 30% prefill / 10% memory — [arXiv 2606.07819](https://arxiv.org/html/2606.07819v1); 2:4 → 8:16 patterns with variance correction — [arXiv 2507.03052](https://arxiv.org/pdf/2507.03052); pruning vs quantization for small LMs (EMNLP 2025 Findings) — [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.645.pdf).
- Adjacent: SparseInfer — [arXiv 2411.12692](https://arxiv.org/pdf/2411.12692); Prox FFN sparsity — [arXiv 2607.27591](https://arxiv.org/pdf/2607.27591); PyTorch sparse-inference blog — [pytorch.org](https://pytorch.org/blog/beyond-quantization-bringing-sparse-inference-to-pytorch/).

### Inferences
- **Crowdedness:** weight-sparsity × quantization ≈8; activation-sparsity × quantization ≈4; low-rank × quantization: covered elsewhere (only note: SALT/ultra-low-rank serving in Area 13 and D-QReLO in Area 13 are the 2026 frontier).
- **Strongest result:** SharQ's coupled-error formulation with Blackwell FP4 kernels; OBR's 4.72×.
- **Open problems (stated):** error coupling between mask and block scale; conflicting objectives of pruning vs quantization; whether activation sparsity removes or concentrates outliers is argued both ways (SharQ: outliers *are* the backbone).
- **Theory:** none formal — OBR's tension and SharQ's coupling are heuristic/empirical.
- **What would be new:** a joint rate–distortion model for (sparsity ratio, bit-width, group size) showing when sparsity *helps* quantization (outlier removal lowers scale) vs *hurts* (distribution shift); activation-sparsity-aware calibration for FP4.

### Gaps
- JSQ (joint sparsification and quantization, ICML 2024) not confirmed in this sweep **[ID unverified]**.

---

## Area 11 — On-device / continual / federated learning with quantization; quantized agent memory

### Takeaway
Federated LLM fine-tuning × quantization has ~7 papers but nearly all quantize the *base model or adapters* for memory/communication (FedFit, FedQuad, HAFLQ, FAH-QLoRA, FwdLLM), not gradients per se; continual learning × quantization has two 2025–26 papers with *opposite* conclusions (8-bit helps retention vs bitsandbytes amplifies interference); quantized long-term agent memory has zero papers found.

### Cited Findings
- FedFit: PTQ of vector-bank adapter parameterizations with client-side error feedback; communication ≈1–3% of standard federated LoRA on Qwen2.5 — [arXiv 2610.01537](https://arxiv.org/html/2610.01537).
- FedQuad: layer-wise LoRA deployment + activation quantization for federated fine-tuning, cutting transmitted data on constrained devices — [arXiv 2506.01001](https://arxiv.org/pdf/2506.01001).
- HAFLQ: heterogeneous adaptive federated LoRA with quantization — [arXiv 2411.06581](https://arxiv.org/pdf/2411.06581); FAH-QLoRA: heterogeneous base-model quantization + per-device LoRA ranks — [NSF PAR](https://par.nsf.gov/servlets/purl/10635831).
- FwdLLM: forward-gradient federated training keeping INT8/INT4 base weights with FP32 LoRA — [arXiv 2308.13894](https://arxiv.org/pdf/2308.13894); FedSODA (NF4/QLoRA preliminaries) — [arXiv 2508.12727](https://arxiv.org/pdf/2508.12727).
- Non-federated quantized fine-tuning frontier: LoTA-QAF lossless ternary adaptation — [arXiv 2505.18724](https://arxiv.org/pdf/2505.18724); AQLoRA zero-search quantized LoRA — [arXiv 2608.23816](https://arxiv.org/pdf/2608.23816); configuration-aware LoRA for quantized LLMs — [arXiv 2509.25214](https://arxiv.org/pdf/2509.25214).
- Continual learning: "When Less is More: 8-bit Quantization Improves Continual Learning" argues quantization noise is implicit regularization and quantized models need less replay — [arXiv 2512.18934](https://arxiv.org/abs/2512.18934); "Compress and Forget: bitsandbytes Quantization Amplifies Proactive Interference" — title only — [arXiv 2608.18578](https://arxiv.org/pdf/2608.18578); DurableUn: quantization-induced recovery attacks on unlearning — [arXiv 2605.02196](https://arxiv.org/pdf/2605.02196); survey — [arXiv 2506.13045](https://arxiv.org/pdf/2506.13045).
- On-device deployment: 4-bit BitsAndBytes PTQ of Llama-3.2-3B for mobile — [arXiv 2512.06490](https://arxiv.org/abs/2512.06490); ZipServ lossless hardware-aware compression — [arXiv 2603.17435](https://arxiv.org/pdf/2603.17435).

### Inferences
- **Crowdedness:** FedLLM × quantization ≈7 (adapter/base/activation, little on gradient quantization specifically); continual × quantization = 2 (contradictory); agent long-term memory × quantization = 0.
- **Strongest result:** FedFit's 30–100× communication reduction; the 8-bit-helps-CL claim (2512.18934) is the most provocative and least verified.
- **Open problems:** gradient/update quantization with convergence guarantees for LoRA under heterogeneous client bit-widths; reconciling the two CL papers (which quantizer? which forgetting metric?); unlearning durability under re-quantization.
- **Theory:** none LLM-specific; classical compressed-SGD theory not applied in the found papers.
- **What would be new:** quantization-as-regularizer analysis for continual fine-tuning (noise scale vs forgetting); quantized retrieval memories for agents (combining Area 4 binary embeddings with episodic memory — no paper does this); error-feedback quantized LoRA updates with heterogeneous bit-widths and a convergence bound.

### Gaps
- 2608.18578's abstract not retrieved; its claim direction is from the title only.
- No "agent memory quantization" literature found; search budget exhausted before a second phrasing.

---

## Area 12 — Agentic / tool-use / structured output under quantization; quantization-conditioned backdoors; fingerprinting

### Takeaway
A fresh 2025–26 cluster (~5 papers) shows quantization damage is *masked* by aggregate agent benchmarks — JSON/tool-call formatting and out-of-list tool calls degrade far more than task-success scores — while a parallel security cluster (~5 papers) exploits the FP↔quantized gap as a backdoor trigger. Watermark/fingerprint papers routinely test robustness *to* quantization, but "quantization as a fingerprint" has only one 2023 watermark paper.

### Cited Findings
- "Can Compressed LLMs Truly Act?": compressed models lose more when producing JSON than plain strings; 4-bit keeps tool use within 1–3% but costs 10–15% on real-world application accuracy — [arXiv 2505.19433](https://arxiv.org/abs/2505.19433).
- "Flat Score, Amplified Failures": on τ²-bench 4-bit looks nearly free on task success, but out-of-allowed-list tool calls roughly double (19.5% → 38.3%) — [arXiv 2607.27275](https://arxiv.org/html/2607.27275v1).
- AgentQ: quantization-conditioned backdoor against agents, dormant at FP and active after quantization; tool-call formatting depends on a narrow band of layers; a quantized endpoint failed to emit valid tool calls at all; weight-space audits do not detect it — [arXiv 2609.14060](https://arxiv.org/html/2609.14060).
- Mix-Quant: uniform NVFP4 lowers BFCL-v4 and τ²-bench across families; proposes quantized prefill + precise decode — [arXiv 2605.20315](https://arxiv.org/html/2605.20315v1); adjacent: uncertainty quantification for function calling — [arXiv 2604.22985](https://arxiv.org/pdf/2604.22985).
- Defenses/attacks on the rounding gap: QuantGuard "Breaking the Rounding Trap" uses rounding control variables — [arXiv 2606.29239](https://arxiv.org/abs/2606.29239); QVec treats FP−quantized difference as a malicious task vector — [arXiv 2606.20254](https://arxiv.org/abs/2606.20254); outlier injection extends attacks to advanced quantizers — [arXiv 2605.15152](https://arxiv.org/abs/2605.15152v1); the founding attack paper (Egashira et al., NeurIPS 2024) **[ID unverified; believed 2405.18137]**.
- "Watermarking LLMs with Weight Quantization" (2023): the watermark lives in the quantization gap — invisible when quantized, visible at FP; GPT-Neo and LLaMA — [arXiv 2310.11237](https://arxiv.org/pdf/2310.11237).
- Fingerprint robustness to quantization: semantically conditioned watermarks survive 8-bit float and 4-bit int (BitsAndBytes) while the SF baseline does not — [arXiv 2505.16723](https://arxiv.org/pdf/2505.16723); Prompt2Fingerprint persists under quantization (single table) — [arXiv 2605.18474](https://arxiv.org/pdf/2605.18474); SELF (singular/eigenvalue fingerprints) robust to quantization, pruning, fine-tuning — [arXiv 2512.03620](https://www.arxiv.org/pdf/2512.03620); RoFL black-box fingerprint tested 16→4 bit, high TPR at 16/8-bit — [arXiv 2505.12682](https://arxiv.org/html/2505.12682v1); watermark-teacher rethinking lists quantization among deployment transforms — [arXiv 2610.04169](https://arxiv.org/html/2610.04169); surveys/related: copyright-protection survey — [arXiv 2508.11548](https://arxiv.org/html/2508.11548v2); robustness to model merging — [arXiv 2404.05188](https://arxiv.org/pdf/2404.05188); SEAL subspace watermarks — [arXiv 2511.11356](https://arxiv.org/pdf/2511.11356).

### Inferences
- **Crowdedness:** agent-benchmark × quantization ≈5 (2025-26, new); quantization-conditioned backdoors/defenses ≈5; fingerprint-robustness-to-quantization: many as a side experiment; "quantization gap as the fingerprint" = 1 (2023).
- **Strongest result:** the 19.5%→38.3% out-of-list tool-call doubling at flat task score; AgentQ's undetectable backdoor.
- **Open problems (stated):** aggregate metrics mask formatting/constraint failures; which layers carry formatting ability; multi-turn error compounding under quantization is not measured; structured-output (grammar-constrained) decoding under quantization is not studied directly.
- **Theory:** none.
- **What would be new:** constrained-decoding-aware quantization (protect the layers/heads governing format tokens, measured via grammar-violation rate rather than accuracy); a positive use of the rounding gap — deterministic quantization *signatures* (which quantizer/config produced a checkpoint) as provenance evidence; multi-turn compounding analysis of quantized agents.

### Gaps
- No paper found on JSON-schema / grammar-constrained decoding × quantization specifically; the structured-output evidence is indirect (JSON outputs in 2505.19433).

---

## Area 13 — Model merging / weight averaging and quantization; delta compression; multi-tenant serving of many fine-tunes

### Takeaway
Delta compression is a moderately crowded line (~10 papers: BitDelta, Delta-CoMe, DeltaDQ, ImPart, PrinMix, D-QReLO, SineLoRAΔ, …) with Delta-CoMe's mixed-precision-by-singular-vector still the reference result; merging × quantization is sparse (Task Vector Quantization, E-PMQ) but has the cleanest empirical fact — INT4 *task vectors* merge losslessly while INT4 *checkpoints* collapse; multi-tenant adapter serving (S-LoRA lineage) has almost no low-bit-adapter work.

### Cited Findings
- BitDelta (NeurIPS 2024): 1-bit deltas (sign + per-matrix scale refined by distillation); >10× GPU-memory and >10× per-user latency reduction in multi-tenant serving — [NeurIPS 2024](https://proceedings.neurips.cc//paper_files/paper/2024/hash/187d94b3c93343f0e925b5cf729eadd5-Abstract-Conference.html); code [FasterDecoding/BitDelta](https://github.com/fasterdecoding/bitdelta).
- Delta-CoMe: training-free mixed-precision delta compression (hybrid low-rank + low-bit); beats low-rank and BitDelta; 7B suite average 53.2 vs 53.5 uncompressed, BitDelta 49.3, SVD 47.8; 50 fine-tunes on one 80 GB A100 vs 4 **[secondary]** — [arXiv 2406.08903](https://arxiv.org/html/2406.08903v2); [Lacuna summary](https://lacuna.tiptreesystems.com/work/delta-come-training-free-delta-compression-with-mixed-precision-for-large/wrk_9a47f2b044db74681d4adc6072d2c480).
- DeltaDQ: group-wise dropout + separate quantization; 16× compression with improved accuracy on WizardMath/WizardCoder — [arXiv 2410.08666](https://arxiv.org/abs/2410.08666); ImPart importance-aware delta sparsification, 2× over baselines, also boosts delta quantization and merging — [arXiv 2504.13237](https://arxiv.org/pdf/2504.13237); D-QReLO training/data-free 1-bit + compensated residual low-rank — [arXiv 2604.16940](https://arxiv.org/abs/2604.16940); PrinMix SVD-based quantization-error minimization — [arXiv 2506.11087](https://arxiv.org/pdf/2506.11087); SineLoRAΔ — [arXiv 2505.21895](https://arxiv.org/pdf/2505.21895); "Quantized Delta Weight Is Safety Keeper" — [arXiv 2411.19530](https://arxiv.org/html/2411.19530); Skill Weaving modular skillpacks — [arXiv 2605.22205](https://arxiv.org/pdf/2605.22205).
- Task Vector Quantization (ICCV 2025): task vectors have a narrow weight range enabling ≤4-bit; INT4 task vectors keep 8-task Task-Arithmetic/TIES accuracy flat while INT4 fine-tuned checkpoints collapse; Residual TVQ splits base + offset — [arXiv 2503.06921](https://arxiv.org/pdf/2503.06921); [ICCV page](https://mlanthology.org/iccv/2025/kim2025iccv-task/).
- E-PMQ (post-merge quantization with merged-weight anchoring): CLIP-ViT-B/32 8-task 4-bit GPTQ improves 65.0% → 73.6% (Task Arithmetic) and 69.1% → 74.8% (TIES); also FLAN-T5 and LLMs — [arXiv 2605.16882](https://arxiv.org/html/2605.16882).
- Merging background/surveys: reversible merging for low-rank weights — [arXiv 2510.14163](https://arxiv.org/pdf/2510.14163); surveys — [arXiv 2408.07666](https://arxiv.org/html/2408.07666v5), [arXiv 2511.21437](https://arxiv.org/html/2511.21437v1).
- Serving: S-LoRA keeps adapters in host memory with a unified GPU pool for adapters + KV — [S-LoRA PDF](https://www.cl.cam.ac.uk/~ey204/teaching/ACS/R244_2024_2025/papers/SO-LORA_ARXIV_2024.pdf); Compress-then-Serve for thousands of adapters — [arXiv 2407.00066](https://arxiv.org/pdf/2407.00066); SALT ("Pin Once, Swap Light"): r<2 adapters degrade drastically; claims up to 16× PCIe/VRAM reduction — [arXiv 2608.03579](https://arxiv.org/pdf/2608.03579); InfiniLoRA disaggregated multi-LoRA — [arXiv 2604.07173](https://arxiv.org/pdf/2604.07173); heterogeneous adapter serving — [arXiv 2511.22880](https://arxiv.org/pdf/2511.22880); EdgeLoRA — [ACM DL](https://dl.acm.org/doi/10.1145/3711875.3729141); LoRA-Inlaid (LoRA + quantization multi-adapter serving) cited by a 2026 survey — [arXiv 2604.21905](https://arxiv.org/pdf/2604.21905) **[LoRA-Inlaid ID unverified]**.

### Inferences
- **Crowdedness:** delta compression ≈10 (moderately crowded); merging × quantization ≈2–3 (sparse); low-bit adapter serving ≈1–2 (sparse).
- **Strongest result:** Delta-CoMe (near-lossless, 12× more fine-tunes per GPU); TVQ's INT4-task-vector vs INT4-checkpoint dichotomy.
- **Open problems:** no theory for why deltas/task vectors are so compressible (narrow range + low effective rank is observed, not derived); merging of *already-quantized* models (quantized base + quantized deltas) and interference under rounding is open; adapter bit-width vs. swap bandwidth trade-off unstudied.
- **Theory:** none found.
- **What would be new:** a quantization-aware merging rule (TIES-style sign election under rounding noise) with an interference bound; serving many 1-bit deltas with fused kernels and cache-aware scheduling; delta compression for *reasoning-RL* fine-tunes (all evaluated deltas are SFT/chat deltas — Delta-CoMe itself notes BitDelta covered only chat models).

### Gaps
- DeltaZip and Delta-DCT could not be confirmed in this sweep **[IDs unverified]**.

---

## Area 14 — Training-time activation (not weight) quantization for the backward pass

### Takeaway
Below 8 bits for *stored activations* in LLM training is a 3–4-paper niche (AGoQ 4-bit activations + 8-bit gradients, Quartet MXFP4 fwd+bwd, HALO/LUQ as baselines) sitting on the 2021–22 ActNN/GACT 2-bit CNN lineage; INT8/FP8 (Jetfire, COAT) is established. The only formal tool is ActNN's gradient-variance-minimizing quantizer; no comparison against gradient checkpointing exists in the sources found.

### Cited Findings
- ActNN (ICML 2021): randomly quantized 2-bit stored activations, <0.5% loss on vision tasks; per-group quantizer and mixed precision approximately minimize gradient variance under a memory budget — [arXiv 2104.14129](https://arxiv.org/pdf/2104.14129); described by COAT as achieving 12× activation-memory reduction — [arXiv 2410.19313](https://arxiv.org/html/2410.19313v2). GACT (ICML 2022) generalizes via PyTorch hooks to arbitrary operators — [arXiv 2206.11357](https://arxiv.org/pdf/2206.11357).
- Jetfire: INT8 data flow with per-block quantization; 1.49× activation memory reduction and 1.42× training speedup vs FP16 — [arXiv 2403.12422](https://arxiv.org/abs/2403.12422).
- COAT (ICLR 2025): FP8 optimizer states + activations with mixed-granularity quantization; targets non-linear layers (~half of activation memory in Llama) — [arXiv 2410.19313](https://arxiv.org/abs/2410.19313); [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/6ac807c9b296964409b277369e55621a-Paper-Conference.pdf).
- AGoQ (2026): 4-bit activations and 8-bit gradients for memory-efficient distributed LLM training; states Jetfire/COAT dynamic block schemes are not applicable at 4 bits and prior approaches show slow convergence or accuracy loss there — [arXiv 2605.00539](https://arxiv.org/html/2605.00539v1).
- Quartet: fully 4-bit (MXFP4) forward and backward on Blackwell; compares against Jetfire, LUQ and HALO and argues they lose precision/stability at 4 bits — [arXiv 2505.14669](https://arxiv.org/pdf/2505.14669). HALO **[ID unverified; believed 2501.02625]**.
- Surveys/adjacent: low-precision training survey — [arXiv 2505.01043](https://arxiv.org/html/2505.01043v1); approximate and memory-sharing backprop — [arXiv 2406.16282](https://arxiv.org/pdf/2406.16282); TAH-Quant activation quantization for pipeline parallelism — [arXiv 2506.01352](https://arxiv.org/html/2506.01352v1).

### Inferences
- **Crowdedness:** ≤8-bit stored-activation quantization for LLM backprop ≈3–4 (sparse); INT8/FP8 flows ≈crowded (excluded by scope).
- **Strongest result:** AGoQ's A4/G8 (details not retrieved); Quartet's end-to-end MXFP4 training.
- **Open problems:** sub-4-bit stored activations for Transformers; non-linear-layer (SwiGLU, softmax, norm) activation memory which COAT identifies as ~50%; interaction with gradient checkpointing (recompute vs store-quantized) has no published trade-off curve for LLMs in these sources.
- **Theory:** ActNN's variance-minimization framework (2021) only; nothing LLM-specific or for outlier-heavy activations.
- **What would be new:** an outlier-aware extension of ActNN's gradient-variance bound for Transformer activations (massive activations, Area 8) yielding layer-wise bit allocation; a Pareto study of checkpointing × quantized-activation storage × speed on Blackwell; 2–3-bit stored activations for fine-tuning (where accuracy tolerance is higher than pretraining).

### Gaps
- AGoQ's memory/accuracy numbers not retrieved; no 2025 "4-bit activation backprop" paper other than AGoQ/Quartet surfaced.

---

## Cross-area summary — crowdedness and whitespace (for idea generation)

### Takeaway
The sparsest, most defensible niches found are: recurrent-state quantization for SSM/hybrids (0 academic papers, live production pain), audio-LLM quantization (0), quantized agent memory (0), quantization-aware structured/constrained decoding (0 direct), softmax/logit perturbation theory for the LM head (0), acceptance-rate theory for quantized drafts (0), frequency-domain unification of RoPE × quantization (1), data-aware inner-product bounds for learned binary embeddings (1), and quantization-aware merging rules (2–3).

### Cited Findings
- Count table (papers surfaced in this sweep; lower bounds) — all sources are the ones listed in the respective areas above:

| Area | Papers found | Theory? | Strongest result (source) |
|---|---|---|---|
| 1 SSM/hybrid weights | ~7 Mamba-LLM, ~4 vision-SSM, 1 RWKV, 1 hybrid | none | Quamba2 W4A8 8B 3× gen, −1.6% ([2503.22879](https://arxiv.org/abs/2503.22879)) |
| 1′ Recurrent state quant | 0 academic (1 RFC) | none | INT8 state + error feedback restores GSM8K 41→81% ([vLLM #55196](https://github.com/vllm-project/vllm/issues/55196)) |
| 2 dLLM PTQ | ~5 | none | W4A16 near-lossless; rotation needed at W4A4 ([2508.14896](https://arxiv.org/abs/2508.14896)) |
| 3 VLM / omni | ~8 weight, ~4 KV, 0 audio | none | Q-VLM W4A4 LLaVA-13B lossless ([2410.08119](https://arxiv.org/abs/2410.08119)) |
| 4 Embeddings/retrieval | industrial many; theory 1–2; ColBERT 2 | RaBitQ bound (data-oblivious) | O(1/√D) unbiased estimator ([10.1145/3654970](https://dl.acm.org/doi/pdf/10.1145/3654970)) |
| 5 SpecDec × quant | ~7 systems, 0 quant-specific theory | generic TV/KL | MXFP4 self-draft 91% acceptance, 2.72× ([2503.13565](https://arxiv.org/pdf/2503.13565)) |
| 6 LM head / logits / sampling | ~4 + ~4 + 3 | none | 128 PB logit-storage motivation ([2503.16870](https://arxiv.org/abs/2503.16870)) |
| 7 Communication | ~4 EP, ~4 TP, ~6 PD, ~2 store | none | Flash Comm 2× TTFT ([2412.04964](https://arxiv.org/html/2412.04964v1)); INT4 KV cliff is model-specific ([2606.08635](https://arxiv.org/abs/2606.08635)) |
| 8 RoPE / norm | ~6 pre/post-RoPE, 3 scaling, 0 norm | partial (Block-GTQ) | per-frequency-block bit allocation ([2606.24033](https://arxiv.org/abs/2606.24033)) |
| 9 Hardware co-design | ~10 LUT, ~4 NPU gap | none | NVFP4 12B/10T tokens matches BF16 ([2509.25149](https://arxiv.org/abs/2509.25149)) |
| 10 Sparsity × quant | ~8 weight, ~4 activation | none | SharQ coupled-error FP4 ([2606.26587](https://arxiv.org/abs/2606.26587)) |
| 11 Fed / continual / memory | ~7 fed, 2 CL (contradictory), 0 agent memory | none | FedFit 1–3% comm ([2610.01537](https://arxiv.org/html/2610.01537)) |
| 12 Agents / backdoors / fingerprint | ~5 + ~5 + 1 | none | out-of-list tool calls 19.5→38.3% at flat score ([2607.27275](https://arxiv.org/html/2607.27275v1)) |
| 13 Merging / delta / multi-tenant | ~10 delta, 2–3 merge, 1–2 serving | none | INT4 task vectors lossless vs INT4 checkpoints collapse ([2503.06921](https://arxiv.org/pdf/2503.06921)) |
| 14 Training activations | 3–4 sub-8-bit | ActNN variance bound (2021) | AGoQ A4/G8 ([2605.00539](https://arxiv.org/html/2605.00539v1)) |

### Inferences
- Two cross-cutting *new axes* appeared in 2025–26 and are still thin: **phase-dependent precision** (prefill vs decode: 2609.26333, 2605.20315) and **transfer vs compute precision** (PD KV tiers). Both lack any error model.
- Several areas share one unexplained phenomenon — model-specific INT4 cliffs (Qwen2.5 in SpectrumKV; Qwen2-VL in SalQ-VLM; Qwen3.5 SSM state in the vLLM RFC) — which suggests an architecture-level predictor of 4-bit fragility would have impact across Areas 1, 3, 7.
- The cheapest "clear contribution" papers, judged coldly: (i) first SSM-state quantization paper with error-feedback and a recurrence error bound; (ii) first audio-LLM PTQ study; (iii) a logit-perturbation → acceptance-rate model for quantized self-drafts; (iv) constrained-decoding-aware quantization evaluated by grammar-violation rate; (v) quantization-aware TIES with an interference bound.

### Gaps
- Search budget exhausted before verifying: MambaQuant, Any-Precision LLM, HALO, JSQ, DeltaZip, Delta-DCT, LoRA-Inlaid, Egashira et al. IDs; the 2609.04098 (GDN NVFP4), 2606.18114 (Ternary Mamba), 2608.21952 (SSDi8), 2605.17524 (binary-embedding theory) and 2605.00539 (AGoQ) full results.
- All venue attributions beyond those explicitly surfaced (CVPR 2025 MBQ, NeurIPS 2024 Q-VLM/BitDelta, ICLR 2025 COAT, ISCA 2025 LUT TC, ICCV 2025 TVQ, ACL 2025 sparse-logit KD, EMNLP 2024 FLUTE, SIGMOD 2024 RaBitQ, ICML 2021/22 ActNN/GACT, USENIX ATC 2024 Quant-LLM, NAACL 2022 ColBERTv2) should be treated as arXiv preprints.
