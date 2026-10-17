# Exact dominance filter: evaluate each candidate with no repeat key and with a repeat key on each
# thumb; report regressions vs HD-Promethium (no repeat key) beyond 0.02, ignoring the home-row
# share (repeat presses move keystrokes to the thumbs, which lowers it without being worse).
import re, subprocess, sys
from collections import OrderedDict

cands = OrderedDict()
for line in open(sys.argv[1]):
    name, lay = line.rstrip("\n").split("\t")
    cands.setdefault(lay[:31], name)

def evaluate(mode, layouts):
    out = subprocess.run(["./target/release/kb", "eval", mode, *layouts], capture_output=True, text=True).stdout
    blocks = out.strip().split("\n\n")
    res = []
    for b in blocks:
        m = re.search(r"vs PM \((\d+) worse beyond 0.02\): (.*)", b)
        regs = [] if "dominates" in m.group(2) else [r.strip() for r in m.group(2).split(",")]
        regs = [r for r in regs if not r.startswith("home row")]
        stats = {k: float(v) for k, v in re.findall(r"(SFB|SFS|LSB|Alt|RollIn|RingPinky) ([0-9.]+)%", b)}
        res.append((regs, stats))
    return res

lays = list(cands)
rows = []
for mode in ("none", "both"):
    for lay, (regs, st) in zip(lays, evaluate(mode, lays)):
        rows.append((len(regs), mode, cands[lay], lay, regs, st))
rows.sort(key=lambda r: (r[0], -r[5].get("Alt", 0) - r[5].get("RollIn", 0)))
for n, mode, name, lay, regs, st in rows[: int(sys.argv[2]) if len(sys.argv) > 2 else 40]:
    print(f"{n} {mode:<4} {name:<20} {lay}  SFB {st['SFB']:.2f} LSB {st['LSB']:.2f} Alt {st['Alt']:.1f} In {st['RollIn']:.1f}  | {'; '.join(regs) or 'DOMINATES PM'}")
