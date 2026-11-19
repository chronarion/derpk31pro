"""Label every position of the bm-eval cuts by the message role / channel of the token being predicted, then split the
KL of the runner universes (and the engine's exact per-position KL where present) by label. CPU, read-only on /s."""
import torch, sys, json, re, glob, numpy as np
from collections import Counter, defaultdict
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/m", trust_remote_code=True)
def load_corpus(p):
    c = torch.load(p, map_location="cpu", weights_only=False)
    return c["tokens"], c.get("kinds"), c.get("cut_index")
def label_cut(ids, show=False):
    """labels per token: 'hdr' (message framing), role[/type] of the enclosing message, 'pre' before the first header, 'text' if no header at all"""
    toks = tok.convert_ids_to_tokens(ids)
    T = len(toks)
    lab = ["pre"] * T
    cur = "pre"; i = 0; headers = []
    while i < T:
        if toks[i] == "<|open|>":
            j = i + 1
            while j < T and toks[j] != "<|sep|>" and j - i < 48:
                j += 1
            head = tok.decode(ids[i + 1:j])
            headers.append(head)
            m = re.search(r'role="([^"]+)"', head); ty = re.search(r'type="([^"]+)"', head)
            if m:
                cur = m.group(1) + ("/" + ty.group(1) if ty else "")
            else:
                cur = "chan:" + head.strip()[:24]
            for k in range(i, min(j + 1, T)):
                lab[k] = "hdr"
            i = j + 1
            continue
        if toks[i] in ("<|close|>", "<|end_of_msg|>"):
            lab[i] = "hdr"
            # the tokens between <|close|> and <|end_of_msg|> are framing too
            i += 1
            continue
        lab[i] = cur
        i += 1
    return lab, headers

def analyse(tag_corpus, corpus_path, kl_sets):
    tokens, kinds, cut_index = load_corpus(corpus_path)
    N, T = tokens.shape
    labs = []; allh = Counter()
    for n in range(N):
        lab, headers = label_cut(tokens[n].tolist())
        labs.append(lab); allh.update(headers)
    nohdr = [n for n in range(N) if all(l == "pre" for l in labs[n])]
    for n in nohdr:
        labs[n] = ["text"] * T
    print(f"===== {tag_corpus}: {N} cuts; cuts without any message header (plain text): {len(nohdr)}")
    print("message headers seen:", allh.most_common(16))
    # label of the PREDICTED token for row t is labs[t+1]
    def group(l):
        if l in ("hdr", "pre", "text"): return l
        if l.startswith("assistant"): return "assistant"
        if l.startswith("chan:"): return "assistant"          # channels inside an assistant message
        if l.startswith("system"): return "system"
        if l.startswith("user"): return "user"
        if l.startswith("tool"): return "tool"
        return "other:" + l
    G = [[group(l) for l in lab] for lab in labs]
    cnt = Counter(g for row in G for g in row)
    print("token share by label:", {k: round(v / (N * T), 4) for k, v in cnt.most_common()})
    fine = Counter(l for lab in labs for l in lab)
    print("fine labels:", {k: round(v / (N * T), 4) for k, v in fine.most_common(14)})
    names_g = sorted(cnt.keys())
    masks = {g: torch.tensor([[G[n][t + 1] == g for t in range(T - 1)] + [False] for n in range(N)]) for g in names_g}
    out = {}
    for tag, kl, names, nll_ref in kl_sets:
        print(f"--- {tag}: mean KL by label of the predicted token (share of rows; mean KL; share of the total KL)")
        hdr = "   universe".ljust(28) + " ".join(f"{g:>22s}" for g in names_g) + "   | all   gen-only(assistant+text)"
        print(hdr)
        if nll_ref is not None:
            row = "   ref ppl".ljust(28)
            for g in names_g:
                m = masks[g][:, :T - 1]
                row += f"{float(nll_ref[m].mean().exp()) if m.any() else float('nan'):22.2f} "
            print(row)
        for u in range(len(names)):
            x = kl[u]
            if float(torch.nan_to_num(x).abs().sum()) == 0: continue
            valid = ~torch.isnan(x)
            tot = float(x[valid].sum())
            row = f"   {names[u][:24]:24s} "
            for g in names_g:
                m = masks[g] & valid
                row += f"{100*float(m.float().mean()):5.1f}% {float(x[m].mean()) if m.any() else float('nan'):7.4f} {100*float(x[m].sum())/tot if m.any() else 0:5.1f}% "
            gen = (masks.get("assistant", torch.zeros_like(valid)) | masks.get("text", torch.zeros_like(valid))) & valid
            row += f"  | {tot/float(valid.sum()):.4f}  {float(x[gen].mean()) if gen.any() else float('nan'):.4f} (rows {100*float(gen.float().mean()):.1f}%)"
            print(row)
            out.setdefault(tag, {})[names[u]] = {"all": tot / float(valid.sum()), "gen": float(x[gen].mean()) if gen.any() else None,
                                                 "gen_rows": float(gen.float().mean()),
                                                 "by": {g: (float(x[masks[g] & valid].mean()) if (masks[g] & valid).any() else None) for g in names_g}}
            # robust stats on generated rows
            xg = x[gen].float()
            if xg.numel():
                print(f"       gen rows: median {float(xg.median()):.4f} p90 {float(torch.quantile(xg[:2000000],0.9)):.3f} p99 {float(torch.quantile(xg[:2000000],0.99)):.2f} >1: {100*float((xg>1).float().mean()):.2f}%")
    return masks, kinds, out

res = {}
# rot32
d = torch.load("/s/bmdense/kld.pt", map_location="cpu", weights_only=False)
sets = [("bmdense (vs official W4A8)", d["kl"], d["names"], d["nll"][0])]
d2 = torch.load("/s/bmarith/kld.pt", map_location="cpu", weights_only=False)
sets.append(("bmarith (vs BF16-input)", d2["kl"], d2["names"], d2["nll"][0]))
try:
    d3 = torch.load("/s/noise/kld.pt", map_location="cpu", weights_only=False)
    sets.append(("noise (vs official W4A8)", d3["kl"], d3["names"], d3["nll"][0]))
except Exception as e:
    print("noise/kld.pt not there yet:", e)
# engine exact per-position files
fs = sorted(glob.glob("/s/kl/out/engine-bm-x-dec2-b400s-s3-p0.1-*.npy"))
if fs:
    T = d["kl"].shape[-1]
    ek = torch.full((1, 32, T), float("nan"), dtype=torch.float64)
    for f in fs:
        n = int(f[-7:-4]); v = torch.from_numpy(np.load(f)).double()
        ek[0, n, :min(T, v.numel())] = v[:T]
    print(f"engine exact files: {len(fs)}; rows with a value: {int((~torch.isnan(ek)).sum())}")
    sets.append((f"engine x-dec2-b400s-s3-p0.1 exact, {len(fs)} cuts so far", ek, ["engine"], None))
m32, k32, r32 = analyse("rot32", "/s/corpus/k31-bm-eval-rot32.pt", sets)
res["rot32"] = r32
d4 = torch.load("/s/bma/kld.pt", map_location="cpu", weights_only=False)
m128, k128, r128 = analyse("rot128", "/s/corpus/k31-bm-eval-rot128.pt", [("bma (vs BF16-input teacher)", d4["kl"], d4["names"], d4["nll"][0])])
res["rot128"] = r128
json.dump(res, open("/r6/out/roles.json", "w"), indent=1)
