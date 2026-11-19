"""Instruction mix of the hot loop of a built library (from cuobjdump -sass output on stdin)."""
import collections, re, sys
txt = sys.stdin.read()
for f in re.split(r"\n\s*Function : ", txt)[1:]:
    name = f.split("\n")[0]
    short = next((k for k in ("band_l1", "band_l2", "band_fused") if k in name), None)
    if short is None:
        continue
    ins = [(int(m.group(1), 16), m.group(2).strip()) for m in re.finditer(r"/\*([0-9a-f]{4})\*/\s+(.*?);", f)]
    qm = [i for i, (a, t) in enumerate(ins) if "QMMA" in t]
    done = set()
    for q in qm:
        for i in range(q, len(ins)):
            m = re.search(r"BRA.*0x([0-9a-f]+)", ins[i][1])
            if m and int(m.group(1), 16) < ins[q][0]:
                if i in done:
                    break
                done.add(i)
                head = [k for k, (a, t) in enumerate(ins) if a == int(m.group(1), 16)][0]
                body = ins[head:i + 1]
                n = sum("QMMA" in t for a, t in body)
                c = collections.Counter(re.sub(r"^@\S+\s+", "", t).split()[0] for a, t in body)
                print(f"{short}: loop of {len(body)} instructions for {n} MMAs ({len(body) / n:.1f} per MMA): "
                      + ", ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])[:8]))
                break
