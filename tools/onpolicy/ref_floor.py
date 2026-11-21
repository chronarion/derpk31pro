"""The reference's own floor on on-policy rows: exact KL between two reference passes over the same bucket.

    ref_floor.py MODEL_DIR RUNNER_DIR BUCKET.pt HIDDEN_A.npy HIDDEN_B.npy OUT.json [--device cuda]

The reference runner is bit-reproducible for one batch layout, but another layout (batch size, block sizes) rounds
differently and the model amplifies it, most on rows inside user / tool text. HIDDEN_A / HIDDEN_B: the final hidden
states of two passes over BUCKET.pt (build_sets.py) that differ only in layout (e.g. the runner's --batch, --attn-block
or --row-block). Both are turned into full-vocabulary log-probabilities with the checkpoint's LM head (fp32 logits,
float64 log-softmax, as the runner scores) and compared row by row: KL(A || B) and KL(B || A) on the on-policy rows
(one per generated token) and on the prompt rows. An engine KLD measured against A cannot be read finer than this.
"""
import argparse
import json
import sys

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model"); p.add_argument("runner"); p.add_argument("bucket")
    p.add_argument("hidden_a"); p.add_argument("hidden_b"); p.add_argument("out")
    p.add_argument("--device", default="cuda"); p.add_argument("--rows", type=int, default=1024)
    a = p.parse_args()
    sys.path.insert(0, a.runner)
    from k31.stream import Shards
    name = "language_model.lm_head.weight"
    head = Shards(a.model).read([name], 1)[name].to(a.device).float()
    c = torch.load(a.bucket, weights_only=False)
    A, B = np.load(a.hidden_a, mmap_mode="r"), np.load(a.hidden_b, mmap_mode="r")
    assert A.shape == B.shape == (c["tokens"].shape[0], c["tokens"].shape[1], head.shape[1]), (A.shape, B.shape)
    fwd, bwd, top, kind, onp = [], [], [], [], []

    def logp(x):
        h = torch.from_numpy(np.array(x).view(np.int16)).to(a.device).view(torch.bfloat16).float()
        return torch.log_softmax((h @ head.T).double(), -1)

    for n in range(A.shape[0]):
        L, G = int(c["lens"][n]), int(c["prompt"][n])
        for r0 in range(0, L - 1, a.rows):
            r1 = min(L - 1, r0 + a.rows)
            la, lb = logp(A[n, r0:r1]), logp(B[n, r0:r1])
            fwd.append((la.exp() * (la - lb)).sum(-1).cpu().numpy())
            bwd.append((lb.exp() * (lb - la)).sum(-1).cpu().numpy())
            top.append((la.argmax(-1) == lb.argmax(-1)).cpu().numpy())
            onp.append(np.arange(r0, r1) >= G - 1)
            kind += [c["kinds"][n]] * (r1 - r0)
    fwd, bwd, top, onp, kind = map(np.concatenate, (fwd, bwd, top, onp, [np.array(kind)]))

    def st(m):
        if not m.any():
            return None
        x = fwd[m]
        q = np.quantile(x, [0.5, 0.9, 0.99, 0.999])
        return {"rows": int(m.sum()), "kl_a_b": float(x.mean()), "kl_b_a": float(bwd[m].mean()), "median": float(q[0]),
                "p90": float(q[1]), "p99": float(q[2]), "p999": float(q[3]), "top1_same": float(top[m].mean())}
    res = {"bucket": a.bucket, "a": a.hidden_a, "b": a.hidden_b, "onpolicy": st(onp), "prompt_rows": st(~onp),
           "by_kind": {k: st(onp & (kind == k)) for k in sorted(set(kind.tolist()))}}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
