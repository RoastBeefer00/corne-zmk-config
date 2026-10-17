# Score candidate layouts under several corpora x effort grids; report weighted cost and rank.
import os, re, subprocess, sys
cands = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
corpora = [("corpus", "wordfreq"), ("corpus_go", "Go docs"), ("corpus_py", "Py docs"), ("corpus_top5k", "top-5k")]
grids = ["default", "flat", "steep"]
names = [n for n, _ in cands]
table = {}
for cdir, cname in corpora:
    for g in grids:
        env = dict(os.environ, CORPUS=cdir, EFFORT=g)
        out = subprocess.run(["./target/release/kb", "analyze", *[l for _, l in cands]], capture_output=True, text=True, env=env).stdout
        costs = [float(x) for x in re.findall(r"cost\(eng\) ([0-9.]+)", out)]
        sfb = [float(x) for x in re.findall(r"\n  SFB ([0-9.]+)%", out)]
        table[(cname, g)] = list(zip(costs, sfb))
w = max(len(n) for n in names)
print(f"{'':{w}}  " + "  ".join(f"{c[:8]:>8}/{g[:4]:<4}" for c, g in table))
for i, n in enumerate(names):
    row = []
    for key, vals in table.items():
        rank = sorted(v[0] for v in vals).index(vals[i][0]) + 1
        row.append(f"{vals[i][0]:7.1f} #{rank:<2}  ")
    print(f"{n:{w}}  " + " ".join(row))
