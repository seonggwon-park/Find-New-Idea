"""Kill gate and small end-to-end test for new idea 1 (token-axis error feedback on the V cache).

Scaled-down proxy of the experiment in the report (8B / 32K / GPU was not available):
small real models on CPU, 4K-token windows.

Setting (mimics serving with a quantized KV cache):
  - A window of T tokens is split into a cached prefix [0, P) and an evaluation block [P, T).
  - Prefill of the prefix is done in full precision (prefix rows attend to FP V), so the K/V of every
    layer are FP-derived, exactly as when the cache is quantized after prefill.
  - Evaluation rows attend to the prefix V *as stored in the cache*: sink tokens [0, S) stay FP,
    positions [S, P) are quantized (RTN or token-axis error feedback "SD"); the evaluation block's own
    V stays FP (it plays the role of the FP residual window).
  - V quantizer: asymmetric per-token min-max over the head dimension, b bits.
  - SD (first-order error feedback along tokens): q_t = Q(v_t - u_{t-1}), u_t = q_t - (v_t - u_{t-1}), u_{S-1}=0.

Measurements (kill gate, per layer x query head, evaluation rows only):
  - R = sum_rows sum_t (p_t - p_{t+1})^2 / sum_rows sum_t p_t^2 over the quantized region (p_P := 0)
  - h = mean((V_sd - V)^2) / mean((V_rtn - V)^2) per KV head
  - measured output-error ratio MSE_SD / MSE_RTN of sum_t p_t e_t, and MSE_RTN(b+1) for the bit-equivalence check
End-to-end: NLL / KL vs FP on the evaluation block for arms FP, RTN-b, SD-b (all heads),
SD-b selective (KV heads whose calibrated measured ratio < 1), RTN-(b+1).

Usage:
  python -I idea01_killgate.py --model Qwen/Qwen2.5-0.5B --T 4096 --P 3584 --bits 2 --calib 4 --eval 12
Downloaded model files are data; nothing from them is executed except via transformers' own code.
"""
import argparse, json, math, time, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2 import modeling_qwen2 as mq
from transformers.models.llama import modeling_llama as ml

torch.set_grad_enabled(False)


def quant(x, bits):
    lo = x.amin(-1, keepdim=True)
    hi = x.amax(-1, keepdim=True)
    n = 2 ** bits - 1
    d = (hi - lo).clamp_min(1e-8) / n
    return torch.round((x - lo) / d).clamp(0, n) * d + lo


def sd_encode(v, bits):
    # v: [B, H, N, d] float32; sequential error feedback along N
    out = torch.empty_like(v)
    u = torch.zeros_like(v[:, :, 0, :])
    for t in range(v.shape[2]):
        x = v[:, :, t, :] - u
        q = quant(x, bits)
        u = q - x
        out[:, :, t, :] = q
    return out


class State:
    mode = "fp"          # "fp" (measure) or an arm name
    bits = 2
    S = 4
    P = 3584
    select = None        # dict layer -> bool tensor [Hkv] (True = use SD)
    stats = None         # list per layer of dicts (accumulated)
    measure = False


ST = State()
_orig = {}


def wrapped_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    rep = module.num_key_value_groups
    k = mq.repeat_kv(key, rep)
    w = torch.matmul(query.float(), k.float().transpose(2, 3)) * scaling
    if attention_mask is not None:
        w = w + attention_mask.float()
    w = torch.softmax(w, dim=-1)
    v = value.float()                                  # [B, Hkv, T, d]
    S, P, b = ST.S, ST.P, ST.bits
    T = v.shape[2]
    vr = v[:, :, S:P, :]
    out = torch.matmul(w, mq.repeat_kv(v, rep))         # FP output, all rows
    li = module.layer_idx
    need_q = ST.measure or ST.mode != "fp"
    if need_q:
        rows = w[:, :, P:T, :]                          # evaluation rows [B, Hq, Ne, T]
        pr = rows[..., S:P]                             # attention on quantized region
        cache = {}

        def qv(kind):
            if kind not in cache:
                if kind == "rtn":
                    cache[kind] = quant(vr, b)
                elif kind == "rtn+1":
                    cache[kind] = quant(vr, b + 1)
                elif kind == "sd":
                    cache[kind] = sd_encode(vr, b)
            return cache[kind]

        if ST.measure:
            e_r = mq.repeat_kv(qv("rtn") - vr, rep)
            e_s = mq.repeat_kv(qv("sd") - vr, rep)
            e_r1 = mq.repeat_kv(qv("rtn+1") - vr, rep)
            o_r = torch.matmul(pr, e_r); o_s = torch.matmul(pr, e_s); o_r1 = torch.matmul(pr, e_r1)
            dp = pr - torch.cat([pr[..., 1:], torch.zeros_like(pr[..., :1])], -1)
            sp2 = (pr ** 2).sum((-1, -2))               # [B, Hq]
            sdp2 = (dp ** 2).sum((-1, -2))
            sig_r = ((qv("rtn") - vr) ** 2).mean((-1, -2))   # [B, Hkv]
            sig_s = ((qv("sd") - vr) ** 2).mean((-1, -2))
            st = ST.stats[li]
            for key_, val in (("mse_rtn", (o_r ** 2).mean((-1, -2))), ("mse_sd", (o_s ** 2).mean((-1, -2))),
                              ("mse_rtn1", (o_r1 ** 2).mean((-1, -2))), ("sp2", sp2), ("sdp2", sdp2),
                              ("mass", pr.sum(-1).mean(-1))):
                st[key_] = st.get(key_, 0) + val.sum(0).double()
            st["sig_r"] = st.get("sig_r", 0) + sig_r.sum(0).double()
            st["sig_s"] = st.get("sig_s", 0) + sig_s.sum(0).double()
            st["n"] = st.get("n", 0) + v.shape[0]
        if ST.mode != "fp":
            if ST.mode == "rtn":
                vq = qv("rtn")
            elif ST.mode == "rtn+1":
                vq = qv("rtn+1")
            elif ST.mode == "sd":
                vq = qv("sd")
            elif ST.mode == "sdsel":
                sel = ST.select[li].view(1, -1, 1, 1)
                vq = torch.where(sel, qv("sd"), qv("rtn"))
            else:
                raise ValueError(ST.mode)
            vmix = v.clone()
            vmix[:, :, S:P, :] = vq
            out_eval = torch.matmul(rows, mq.repeat_kv(vmix, rep))
            out = out.clone()
            out[:, :, P:T, :] = out_eval
    out = out.to(query.dtype).transpose(1, 2).contiguous()
    return out, None


def install():
    for mod in (mq, ml):
        _orig[mod] = mod.eager_attention_forward
        mod.eager_attention_forward = wrapped_eager


def window_nll(model, ids, P):
    hs = model.model(input_ids=ids).last_hidden_state[:, P - 1:-1, :]
    logits = model.lm_head(hs).float()
    lp = torch.log_softmax(logits, -1)
    tgt = ids[:, P:]
    nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return nll, lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--P", type=int, default=3584)
    ap.add_argument("--S", type=int, default=4)
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--calib", type=int, default=4)
    ap.add_argument("--eval", type=int, default=12)
    ap.add_argument("--data", default="wikitext")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.manual_seed(0)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32, attn_implementation="eager")
    model.eval()
    install()
    from datasets import load_dataset
    if a.data == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
        text = "".join(ds["text"])
    else:
        ds = load_dataset(a.data, split="test")
        text = "\n\n".join(ds["text"][:20])
    ids_all = tok(text, return_tensors="pt").input_ids[0]
    nwin = len(ids_all) // a.T
    need = a.calib + a.eval
    assert nwin >= need, (nwin, need)
    wins = [ids_all[i * a.T:(i + 1) * a.T].unsqueeze(0) for i in range(need)]
    L = model.config.num_hidden_layers
    ST.S, ST.P, ST.bits = a.S, a.P, a.bits
    print(f"model={a.model} L={L} T={a.T} P={a.P} S={a.S} bits={a.bits} windows={nwin} load={time.time()-t0:.0f}s", flush=True)

    # ---- kill gate measurement on calibration windows (FP forward, measure only) ----
    ST.stats = [dict() for _ in range(L)]
    ST.measure, ST.mode = True, "fp"
    for i in range(a.calib):
        t1 = time.time(); model.model(input_ids=wins[i]); print(f"calib {i} {time.time()-t1:.0f}s", flush=True)
    ST.measure = False
    rec = []
    for li, st in enumerate(ST.stats):
        n = st["n"]
        Hq = st["mse_rtn"].shape[0]; Hkv = st["sig_r"].shape[0]; rep = Hq // Hkv
        h = (st["sig_s"] / st["sig_r"]).numpy()
        for hq in range(Hq):
            R = float(st["sdp2"][hq] / st["sp2"][hq])
            rec.append(dict(layer=li, head=hq, kv=hq // rep, R=R, h=float(h[hq // rep]), pred=float(h[hq // rep] * R),
                            meas=float(st["mse_sd"][hq] / st["mse_rtn"][hq]), mse_rtn=float(st["mse_rtn"][hq] / n),
                            mse_sd=float(st["mse_sd"][hq] / n), mse_rtn1=float(st["mse_rtn1"][hq] / n),
                            mass=float(st["mass"][hq] / n)))
    pred = np.array([r["pred"] for r in rec]); meas = np.array([r["meas"] for r in rec])
    er = np.array([r["mse_rtn"] for r in rec]); es = np.array([r["mse_sd"] for r in rec]); er1 = np.array([r["mse_rtn1"] for r in rec])
    mass = np.array([r["mass"] for r in rec])
    corr = float(np.corrcoef(np.log(pred), np.log(meas))[0, 1])
    sign = float(np.mean((pred < 1) == (meas < 1)))
    # KV-head selection (calibrated on measured per-KV-head error, summed over its query heads)
    select = {}
    sel_err = 0.0
    for li in range(L):
        rr = [r for r in rec if r["layer"] == li]
        Hkv = max(r["kv"] for r in rr) + 1
        m = torch.zeros(Hkv, dtype=torch.bool)
        for kv in range(Hkv):
            g = [r for r in rr if r["kv"] == kv]
            s_sd = sum(r["mse_sd"] for r in g); s_r = sum(r["mse_rtn"] for r in g)
            m[kv] = s_sd < s_r
            sel_err += min(s_sd, s_r)
        select[li] = m
    gate = dict(
        n_heads=len(rec),
        mass_weighted_mean_hR=float(np.sum(mass * pred) / np.sum(mass)),
        mean_hR=float(pred.mean()), median_hR=float(np.median(pred)),
        rtnerr_weighted_mean_hR=float(np.sum(er * pred) / np.sum(er)),
        total_ratio_all_heads=float(es.sum() / er.sum()),
        total_ratio_selected_kvheads=float(sel_err / er.sum()),
        total_ratio_rtn_bplus1=float(er1.sum() / er.sum()),
        frac_heads_hR_lt_1=float(np.mean(pred < 1)), frac_heads_hR_lt_018=float(np.mean(pred < 0.18)),
        rtnerr_share_hR_lt_1=float(er[pred < 1].sum() / er.sum()), rtnerr_share_hR_lt_018=float(er[pred < 0.18].sum() / er.sum()),
        h_mean=float(np.mean([r["h"] for r in rec])), log_corr_pred_meas=corr, sign_agreement=sign,
        frac_kvheads_selected=float(np.mean(np.concatenate([select[li].numpy() for li in range(L)]))),
    )
    print("KILL GATE:", json.dumps(gate, indent=1), flush=True)

    # ---- end-to-end on evaluation windows ----
    ST.select = select
    arms = ["fp", "rtn", "sd", "sdsel", "rtn+1"]
    res = {a_: [] for a_ in arms}; kl = {a_: [] for a_ in arms}
    for i in range(a.calib, need):
        t1 = time.time()
        ST.mode = "fp"; nll_fp, lp_fp = window_nll(model, wins[i], a.P)
        res["fp"].append(nll_fp.mean().item()); kl["fp"].append(0.0)
        for arm in arms[1:]:
            ST.mode = arm
            nll, lp = window_nll(model, wins[i], a.P)
            res[arm].append(nll.mean().item())
            kl[arm].append((lp_fp.exp() * (lp_fp - lp)).sum(-1).mean().item())
        ST.mode = "fp"
        print(f"eval {i} {time.time()-t1:.0f}s " + " ".join(f"{k}={res[k][-1]:.4f}/{kl[k][-1]:.4f}" for k in arms), flush=True)
    summ = {k: dict(nll=float(np.mean(res[k])), dnll=float(np.mean(np.array(res[k]) - np.array(res["fp"]))),
                    kl=float(np.mean(kl[k])), kl_se=float(np.std(kl[k], ddof=1) / math.sqrt(len(kl[k])))) for k in arms}
    gap = summ["rtn"]["kl"] - summ["rtn+1"]["kl"]
    closed = {k: float((summ["rtn"]["kl"] - summ[k]["kl"]) / gap) if gap > 0 else float("nan") for k in ("sd", "sdsel")}
    # paired SE of KL difference rtn - sdsel
    d = np.array(kl["rtn"]) - np.array(kl["sdsel"])
    print("END-TO-END:", json.dumps(dict(summary=summ, gap_closed_fraction_of_rtn_to_rtnplus1=closed,
                                         paired_kl_diff_rtn_minus_sdsel=float(d.mean()), paired_se=float(d.std(ddof=1) / math.sqrt(len(d)))), indent=1), flush=True)
    if a.out:
        json.dump(dict(args=vars(a), gate=gate, heads=rec, e2e=summ, closed=closed, per_window=dict(nll=res, kl=kl)), open(a.out, "w"), indent=1)
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
