"""Self-test of k31/stream.py on a tiny random Kimi-K3-architecture checkpoint written in K3.1's on-disk layout
(MXFP4 experts with UE8M0 scales, KDA A_log padded with zeros, a DSA indexer tensor to skip, F32 extras):

  1. universe "native" against the reference's own full forward (KimiLinearModel.forward): final hidden states within
     2e-2 relative L2, top-1 agreement of the logits > 0.97, every pick of every MoE layer routed and recorded;
  2. "planes depth 4" and "hotcut with every expert hot" reproduce "native" exactly (KLD 0, same hidden states);
  3. planes depth 2, a small hot set with cuts, MXFP8 expert activations and FP6 dense weights give finite, positive
     KLD, and the cut universe actually cuts.

usage: selftest.py REF_SRC MODEL_META_DIR [cpu|cuda] [bf16|fp32]   (MODEL_META_DIR: the real K3.1 config.json, scaled down here)
On the CPU it uses plain-PyTorch stand-ins for fla's kernels (tests/cpu_standins.py).
"""
import json
import os
import sys
import tempfile

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from k31 import stream  # noqa: E402
from k31.formats import FP4, decode_mxfp4  # noqa: E402
from k31.policy import CUT, DEFER, NATIVE, PLANES, WDEFER, Universe  # noqa: E402

PREFIX = "language_model."
F32_NAMES = ("e_score_correction_bias", "A_log", "dt_bias", "conv1d.weight", "o_norm.weight")


def tiny_config(cfgmod, meta_dir):
    c = json.load(open(os.path.join(meta_dir, "config.json")))["text_config"]
    lac = dict(c["linear_attn_config"], num_heads=4, head_dim=32, kda_layers=[1, 2, 3, 5], full_attn_layers=[4, 6])
    dsa = dict(c["dsa_config"], dsa_layers=[4, 6])
    c.update(hidden_size=256, intermediate_size=512, num_hidden_layers=6, attn_res_block_size=2, vocab_size=512,
             num_attention_heads=4, num_key_value_heads=4, qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
             kv_lora_rank=64, num_experts=16, num_experts_per_token=4, moe_intermediate_size=64,
             routed_expert_hidden_size=128, first_k_dense_replace=1, linear_attn_config=lac, dsa_config=dsa,
             pad_token_id=None, bos_token_id=1, eos_token_id=2)
    if c.get("q_lora_rank"):
        c["q_lora_rank"] = 96
    return c, cfgmod.KimiLinearConfig(**c)


def mxfp4_encode(w):
    """[out, in] float -> packed [out, in/2] uint8, scale [out, in/32] uint8 (kstream.py's packing)."""
    out, cin = w.shape
    blk = w.float().view(out, cin // 32, 32)
    ex = torch.round(torch.log2(blk.abs().amax(-1).clamp_min(1e-30) / 6)).clamp(-126, 127)
    nrm = (blk / torch.pow(2.0, ex).unsqueeze(-1)).clamp(-6, 6)
    codes = torch.bucketize(nrm.abs(), torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])) + nrm.signbit() * 8
    codes = codes.view(out, cin).to(torch.uint8)
    return (codes[:, ::2] | (codes[:, 1::2] << 4)).to(torch.uint8), (ex + 127).to(torch.uint8)


def init_(model):
    g = torch.Generator().manual_seed(0)
    for n, p in model.named_parameters():
        if n.endswith("A_log"):
            p.data = torch.log(torch.empty(p.shape).uniform_(1, 16, generator=g))
        elif n.endswith("dt_bias"):
            p.data = torch.empty(p.shape).uniform_(-0.5, 0.5, generator=g)
        elif n.endswith("e_score_correction_bias"):
            p.data = torch.randn(p.shape, generator=g) * 0.01
        elif p.ndim == 1:
            p.data = 1 + 0.1 * torch.randn(p.shape, generator=g)
        elif "conv1d" in n:
            p.data = torch.randn(p.shape, generator=g) * 0.3
        elif "embed_tokens" in n:
            p.data = torch.randn(p.shape, generator=g) * 0.5
        else:
            p.data = torch.randn(p.shape, generator=g) * (1.0 / p.shape[-1] ** 0.5)


def write_checkpoint(model, config, cfg_dict, path):
    """K3.1's layout: experts MXFP4 (U8), A_log padded to 2x heads, an indexer tensor per MLA layer, F32 extras."""
    sd = {}
    for n, p in model.state_dict().items():
        name = PREFIX + n
        if ".block_sparse_moe.experts." in n:
            packed, scale = mxfp4_encode(p)
            base = name[: -len(".weight")]
            sd[base + ".weight_packed"], sd[base + ".weight_scale"] = packed, scale
        elif n.endswith("A_log"):
            sd[name] = torch.cat([p.float(), torch.zeros_like(p.float())])
        elif n.endswith(F32_NAMES):
            sd[name] = p.float().contiguous()
        else:
            sd[name] = p.to(torch.bfloat16).contiguous()
    for i in range(config.num_hidden_layers):
        if not config.is_kda_layer(i):
            sd[f"{PREFIX}model.layers.{i}.self_attn.indexer.wk.weight"] = torch.zeros(128, config.hidden_size,
                                                                                   dtype=torch.bfloat16)
    os.makedirs(path, exist_ok=True)
    save_file(sd, os.path.join(path, "model.safetensors"))
    json.dump({"weight_map": {k: "model.safetensors" for k in sd}},
              open(os.path.join(path, "model.safetensors.index.json"), "w"))
    json.dump({"text_config": cfg_dict}, open(os.path.join(path, "config.json"), "w"))
    return sd


def load_back(model, sd, dtype):
    """The checkpoint into the reference model the way the runner reads it (experts decoded, A_log un-padded,
    router bias / A_log / dt_bias float32, the rest bfloat16)."""
    new = {}
    for name, t in sd.items():
        n = name[len(PREFIX):]
        if ".indexer." in n or n.endswith(".weight_scale"):
            continue
        if n.endswith(".weight_packed"):
            new[n[: -len("_packed")]] = decode_mxfp4(t, sd[name[: -len("_packed")] + "_scale"], dtype)
            continue
        if n.endswith("A_log"):
            t = t[: t.shape[0] // 2]
        keep32 = n.split(".")[-1] in stream.KEEP_F32 and t.dtype == torch.float32
        new[n] = t if keep32 else t.to(dtype)
    res = model.load_state_dict(new, strict=False, assign=True)
    assert not res.missing_keys and not res.unexpected_keys, res


def main(ref_src, meta_dir, device="cpu", precision="bf16"):
    patch = None
    if device == "cpu":
        import cpu_standins
        patch = cpu_standins.patch
    ref, cfgmod = stream.load_reference(ref_src, patch)
    cfg_dict, config = tiny_config(cfgmod, meta_dir)
    torch.manual_seed(0)
    model = ref.KimiLinearForCausalLM(config)
    config._attn_implementation = "eager"          # the model's constructor forces flash_attention_2
    init_(model)
    work = tempfile.mkdtemp(prefix="k31-selftest-")
    ckpt = os.path.join(work, "ckpt")
    sd = write_checkpoint(model, config, cfg_dict, ckpt)
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    load_back(model, sd, dtype)
    model = model.to(device).eval()
    tokens = torch.randint(0, config.vocab_size, (6, 96), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        want = model.model(input_ids=tokens.to(device), use_cache=False).last_hidden_state.float().cpu()
        head = model.lm_head.weight
        want_top = (want.to(device, dtype) @ head.T).float().argmax(-1).cpu()

    E = config.num_experts
    rank_path = os.path.join(work, "rank.json")       # a fixed usage ranking per layer (static universes)
    g = torch.Generator().manual_seed(1)
    json.dump({"layers": {str(i): torch.randperm(E, generator=g).tolist() for i in range(config.num_hidden_layers)}},
              open(rank_path, "w"))
    hot_layers_path = os.path.join(work, "hot-layers.json")
    json.dump({"1": 2, "2": 8, "3": 4, "4": 12, "5": 4}, open(hot_layers_path, "w"))
    cover_layers_path = os.path.join(work, "cover-layers.json")
    json.dump({"1": 4, "2": 16, "3": 8, "4": 24, "5": 8}, open(cover_layers_path, "w"))
    us = [Universe(name="native"),
          Universe(name="static-all", experts="static", hot=E, rank=rank_path),
          Universe(name="static-8", experts="static", hot=8, strong=0, rank=rank_path),
          Universe(name="budget0-8", experts="hotcut", hot=8, strong=0, rank=rank_path, promote=0),
          Universe(name="budget1-8", experts="hotcut", hot=8, strong=0, rank=rank_path, promote=1, delay=1,
                   halflife=16),
          Universe(name="planes4", experts="planes", depth=4),
          Universe(name="hot-all", experts="hotcut", hot=E, depth=2),
          Universe(name="planes2", experts="planes", depth=2),
          Universe(name="hotcut", experts="hotcut", hot=4, strong=2, depth=2, halflife=16),
          Universe(name="act-mxfp8", act="mxfp8"),
          Universe(name="dense-fp6", dense="fp6"),
          Universe(name="drop2", drop=2),
          Universe(name="wave", experts="hotcut", hot=4, strong=2, depth=2, halflife=16, miss="planes"),
          Universe(name="defer", experts="hotcut", hot=4, strong=2, depth=2, halflife=16, miss="defer"),
          Universe(name="defer-allhot", experts="hotcut", hot=E, miss="defer"),
          Universe(name="cutmax1", experts="hotcut", hot=4, strong=0, depth=2, halflife=16, cut_max=1),
          Universe(name="wd-allhot", experts="hotcut", hot=E, strong_mode="wdefer", miss="wdefer"),
          Universe(name="wd-all", experts="hotcut", hot=4, strong=2, depth=2, halflife=16, strong_mode="wdefer",
                   miss="wdefer"),
          Universe(name="wd-strong", experts="hotcut", hot=4, strong=2, depth=2, halflife=16, strong_mode="wdefer"),
          Universe(name="defer-late3", experts="hotcut", hot=4, strong=2, depth=2, halflife=16, miss="defer", late=3),
          Universe(name="wd-all-late3", experts="hotcut", hot=4, strong=2, depth=2, halflife=16, strong_mode="wdefer",
                   miss="wdefer", late=3),
          Universe(name="mid-defer", experts="hotcut", hot=4, strong=1, mid=3, mid_mode="defer", depth=2, halflife=16),
          Universe(name="defer-half", experts="hotcut", hot=4, strong=2, depth=2, halflife=16, miss="defer", land="mlp"),
          Universe(name="wd-all-half", experts="hotcut", hot=4, strong=2, depth=2, halflife=16, strong_mode="wdefer",
                   miss="wdefer", land="mlp"),
          Universe(name="wd-allhot-half", experts="hotcut", hot=E, strong_mode="wdefer", miss="wdefer", land="mlp"),
          Universe(name="cover8", experts="hotcut", hot=4, strong=4, depth=2, halflife=16, cover=8),
          Universe(name="link1", experts="hotcut", hot=4, strong=4, depth=2, halflife=16, link_max=1),
          Universe(name="defer-late3-half", experts="hotcut", hot=4, strong=2, depth=2, halflife=16, miss="defer",
                   late=3, land="mlp"),
          Universe(name="ladder", experts="hotcut", hot=4, strong=0, depth=2, halflife=16, ontime_max=1, defer_max=1),
          Universe(name="dfrom", defer_from=1, skip_from=3, layers="2-3"),
          Universe(name="tier-pl", experts="hotcut", hot=4, strong=0, rank=rank_path, promote=0.5, delay=1, halflife=16,
                   hot_layers=hot_layers_path),
          Universe(name="tier-2bit", experts="hotcut", hot=6, strong=0, halflife=16, hot_mode="planes"),
          Universe(name="tier-warm", experts="hotcut", hot=3, warm=4, warm_mode="planes", strong=0, rank=rank_path,
                   promote=0.5, warm_promote=1.0, delay=1, halflife=16),
          Universe(name="bank-st", experts="hotcut", hot=3, strong=0, rank=rank_path, promote=0.5, delay=1,
                   halflife=16, cover=8, ontime_max=16),
          Universe(name="bank-ad", experts="hotcut", hot=3, strong=0, rank=rank_path, promote=0.5, delay=1,
                   halflife=16, cover=8, ontime_max=16, cover_promote=1.0),
          Universe(name="bank-pl", experts="hotcut", hot=3, strong=0, rank=rank_path, promote=0.5, delay=1,
                   halflife=16, cover=8, ontime_max=16, cover_layers=cover_layers_path),
          Universe(name="bank-ad0", experts="hotcut", hot=3, strong=0, rank=rank_path, promote=0.5, delay=1,
                   halflife=16, cover=8, ontime_max=16, cover_promote=0.0),
          Universe(name="bank-ex", experts="hotcut", hot=3, strong=0, rank=rank_path, promote=0.5, delay=1,
                   halflife=16, cover=8, ontime_max=16, cover_promote=1.0, cover_excl=True)]
    final, res = stream.run(ckpt, tokens, os.path.join(work, "out"), ref_src, us, batch=4, device=device,
                            dtype=dtype, patch=patch, log=lambda s: print("  " + s, flush=True),
                            state_dir=os.path.join(work, "state"),     # the memmap store, as real passes use
                            save_means=os.path.join(work, "means"))
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + msg, flush=True)

    got = final[0].float()
    rel = ((got - want).norm() / want.norm()).item()
    agree = ((got.to(device, dtype) @ head.T).float().argmax(-1).cpu() == want_top).float().mean().item()
    # fp32 is the structural check (same modules, so only summation order differs: ~1e-6); in bf16 the different
    # batch shapes and combine order give rounding differences that a random, untrained 6-layer model amplifies
    tol, top = (1e-4, 0.999) if dtype == torch.float32 else (5e-2, 0.97)
    check(rel < tol and agree > top, f"native vs the reference forward ({precision}): relative L2 {rel:.2e} "
          f"(< {tol:g}), top-1 {agree:.4f}")
    route = torch.load(os.path.join(work, "out", "route.pt"))
    stats = torch.load(os.path.join(work, "out", "stats.pt"))["stats"]
    picks = tokens.numel() * config.num_experts_per_token * (config.num_hidden_layers - config.first_k_dense_replace)
    check(int(stats[0].sum()) == picks and route["ids"].shape == (config.num_hidden_layers, tokens.numel(),
                                                                   config.num_experts_per_token),
          f"routing: {int(stats[0].sum())} picks counted of {picks}, record {tuple(route['ids'].shape)}")
    w = route["w"][1:].float()
    check(bool((w[..., :-1] >= w[..., 1:]).all()), "routing record sorted by weight")
    names = res["names"]
    kl = {n: res["kl"][i] for i, n in enumerate(names)}
    for n in ("planes4", "hot-all", "defer-allhot", "wd-allhot", "wd-allhot-half"):
        i = names.index(n)
        check(torch.equal(final[i], final[0]) and float(kl[n].abs().max()) == 0.0, f"{n} == native exactly")
    for n in ("planes2", "hotcut", "act-mxfp8", "dense-fp6", "drop2", "wave", "defer", "cutmax1", "wd-all",
              "wd-strong", "defer-late3", "wd-all-late3"):
        m = float(kl[n].mean())
        check(m > 0 and torch.isfinite(kl[n]).all(), f"{n}: KLD {m:.5f} > 0 and finite")
    st = {n: stats[names.index(n)].sum(dim=(0, 2)) for n in names}             # picks per treatment
    moe_tokens = tokens.numel() * (config.num_hidden_layers - config.first_k_dense_replace)
    check(int(st["hotcut"][CUT]) > 0, f"hotcut cut {int(st['hotcut'][CUT])} picks of {picks}")
    check(int(st["drop2"][CUT]) == 2 * moe_tokens, f"drop2 cut {int(st['drop2'][CUT])} = 2 per token and layer")
    check(int(st["wave"][CUT]) == 0 and int(st["wave"][DEFER]) == 0 and int(st["wave"][PLANES]) > 0,
          f"wave: no cuts, {int(st['wave'][PLANES])} misses at the base")
    check(int(st["defer"][DEFER]) > 0 and int(st["defer"][CUT]) == 0, f"defer deferred {int(st['defer'][DEFER])}")
    check(0 < int(st["cutmax1"][CUT]) <= moe_tokens, f"cutmax1 cut {int(st['cutmax1'][CUT])} <= 1 per token and layer")
    check(int(st["wd-all"][WDEFER]) > 0 and int(st["wd-all"][PLANES]) == 0 and int(st["wd-all"][CUT]) == 0,
          f"wd-all: {int(st['wd-all'][WDEFER])} picks at the base now with the correction later")
    km = {n: float(kl[n].mean()) for n in names}
    check(km["wd-all"] < km["wave"], f"the deferred wavefront ({km['wd-all']:.4f}) beats the wavefront ({km['wave']:.4f})")
    check(km["wd-strong"] < km["hotcut"], f"correcting the strong misses later ({km['wd-strong']:.4f}) beats leaving "
          f"them at the base ({km['hotcut']:.4f})")
    check(int(st["mid-defer"][DEFER]) > 0 and int(st["mid-defer"][CUT]) > 0 and int(st["mid-defer"][PLANES]) > 0,
          f"mid-defer: strong at the base {int(st['mid-defer'][PLANES])}, middle deferred "
          f"{int(st['mid-defer'][DEFER])}, weakest cut {int(st['mid-defer'][CUT])}")
    check(km["defer-half"] < km["defer"], f"deferral landing before the next MoE ({km['defer-half']:.4f}) beats "
          f"landing after the next layer ({km['defer']:.4f})")
    check(km["wd-all-half"] < km["wd-all"], f"deferred wavefront, half layer ({km['wd-all-half']:.4f}) vs one layer "
          f"({km['wd-all']:.4f})")
    check(torch.isfinite(kl["defer-late3-half"]).all(), f"defer late 3, half: {km['defer-late3-half']:.4f}")
    from k31.policy import PDEFER
    sl = stats[names.index("tier-pl")]                                        # [L, NTREAT, K]
    res_by_layer = [int(sl[l, NATIVE].sum()) for l in range(1, 6)]
    check(res_by_layer[3] > res_by_layer[0] and res_by_layer[1] > res_by_layer[2],
          f"per-layer hot counts (2, 8, 4, 12, 4): resident picks by layer {res_by_layer}")
    s2 = stats[names.index("tier-2bit")].sum(dim=(0, 2))
    check(int(s2[NATIVE]) == 0 and int(s2[PLANES]) > 0, f"2-bit hot set: {int(s2[PLANES])} picks at 2 bits, none native")
    bs = stats[names.index("bank-st")].sum(dim=(0, 2))
    ba = stats[names.index("bank-ad")].sum(dim=(0, 2))
    b0 = stats[names.index("bank-ad0")].sum(dim=(0, 2))
    be = stats[names.index("bank-ex")].sum(dim=(0, 2))
    check(torch.equal(b0, bs), f"adaptive bank with no refills = the static bank ({b0.tolist()} vs {bs.tolist()})")
    # (routing differs between universes, so only the cut counts are compared; on this near-uniform router an
    # adaptive bank that includes the hot set spends ~3 of its 8 places on resident experts)
    check(int(ba[CUT]) != int(bs[CUT]) and int(be[CUT]) < min(int(ba[CUT]), int(bs[CUT])),
          f"adaptive bank: cut static {int(bs[CUT])}, adaptive {int(ba[CUT])}, adaptive besides the hot set "
          f"{int(be[CUT])}")
    bl = stats[names.index("bank-pl")]
    cut_by_layer = [int(bl[l, CUT].sum()) for l in range(1, 6)]
    check(cut_by_layer[0] > cut_by_layer[1] and cut_by_layer[2] > cut_by_layer[3],
          f"per-layer bank (4, 16, 8, 24, 8): cut picks by layer {cut_by_layer}")
    sw = stats[names.index("tier-warm")].sum(dim=(0, 2))
    check(int(sw[NATIVE]) > 0 and int(sw[PLANES]) > 0 and int(sw[CUT]) > 0,
          f"adaptive warm band: native {int(sw[NATIVE])}, 2-bit warm {int(sw[PLANES])}, cut {int(sw[CUT])}")
    check(int(st["ladder"][PLANES]) > 0 and int(st["ladder"][PDEFER]) > 0 and int(st["ladder"][CUT]) > 0
          and int(st["ladder"][PLANES]) <= moe_tokens and int(st["ladder"][PDEFER]) <= moe_tokens,
          f"ladder: {int(st['ladder'][PLANES])} on time, {int(st['ladder'][PDEFER])} 2-bit late (<= 1 per token each), "
          f"{int(st['ladder'][CUT])} cut")
    sd = stats[names.index("dfrom")]                                            # [L, NTREAT, K]
    in_l = [l for l in range(sd.shape[0]) if int(sd[l, DEFER].sum()) > 0]
    check(in_l == [2, 3] and int(sd[2:4, CUT].sum()) > 0 and int(sd[:, CUT].sum()) == int(sd[2:4, CUT].sum()),
          f"defer_from / skip_from only in layers 2-3: deferred in {in_l}")
    mfiles = sorted(os.listdir(os.path.join(work, "means")))
    m1 = torch.load(os.path.join(work, "means", mfiles[0]), weights_only=False)
    check(len(mfiles) == config.num_hidden_layers - config.first_k_dense_replace and int(m1["count"].sum()) > 0,
          f"mean tables: {len(mfiles)} layers, {int(m1['count'].sum())} picks in the first")
    us2 = [Universe(name="native"), Universe(name="cut4", experts="hotcut", hot=4, strong=0, halflife=16),
           Universe(name="cut4-mean", experts="hotcut", hot=4, strong=0, halflife=16, cut_sub="mean"),
           Universe(name="native-mean", cut_sub="mean")]
    _, res2 = stream.run(ckpt, tokens, os.path.join(work, "out2"), ref_src, us2, batch=4, device=device, dtype=dtype,
                         patch=patch, log=lambda s: None, means=os.path.join(work, "means"))
    k2 = {n: float(res2["kl"][i].mean()) for i, n in enumerate(res2["names"])}
    check(k2["native-mean"] == 0.0 and k2["cut4-mean"] != k2["cut4"] and k2["cut4-mean"] == k2["cut4-mean"],
          f"cut substitute: cut4 {k2['cut4']:.4f}, cut4 + mean outputs {k2['cut4-mean']:.4f}, native unchanged")
    check(int(st["cover8"][CUT]) > 0 and int(st["link1"][PLANES]) <= moe_tokens,
          f"cover: {int(st['cover8'][CUT])} picks cut outside memory; link_max 1: {int(st['link1'][PLANES])} link reads "
          f"<= {moe_tokens} (one per token and layer)")
    st_all, st8 = st["static-all"], st["static-8"]
    check(int(st_all[NATIVE]) == picks and int(st8[CUT]) > 0 and int(st8[NATIVE]) + int(st8[CUT]) == picks
          and km["static-all"] < 1e-9 and km["static-8"] > 0,
          f"static: all {E} in the set = native (KLD {km['static-all']:.1e}); 8 in the set: "
          f"{int(st8[CUT])} of {picks} picks cut, KLD {km['static-8']:.4f}")
    b0, b1 = st["budget0-8"], st["budget1-8"]
    check(torch.equal(final[names.index("budget0-8")], final[names.index("static-8")]) and int(b1[NATIVE]) != int(b0[NATIVE]),
          f"budgeted promotion: none = the static set exactly; 1 per token (delay 1) moves {int(b1[NATIVE]) - int(b0[NATIVE])} "
          f"picks to native (KLD {km['budget1-8']:.4f} vs static {km['static-8']:.4f})")
    pg = torch.load(os.path.join(work, "out", "pregate.pt"))
    r16 = pg["recall"][16][1:]
    check(bool(((r16 >= 0) & (r16 <= 1)).all()) and int(pg["rows"][1:].min()) == tokens.numel(),
          f"pre-gating recall recorded (top-16 covers every expert here: ranks 1-4 recall {float(r16.mean()):.3f})")
    print(f"INFO late by 1 vs 3 layers: defer {km['defer']:.4f} / {km['defer-late3']:.4f}, deferred wavefront "
          f"{km['wd-all']:.4f} / {km['wd-all-late3']:.4f}", flush=True)
    check(float(kl["defer"].mean()) < float(kl["hotcut"].mean()),
          f"deferring the weak misses ({float(kl['defer'].mean()):.4f}) beats cutting them "
          f"({float(kl['hotcut'].mean()):.4f})")
    print("SELFTEST " + ("PASS" if ok else "FAIL"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "cpu",
                  sys.argv[4] if len(sys.argv) > 4 else "bf16"))
