# Sweep fixed j/k and h/l pair placements; optimize the rest of the layout around each.
import subprocess, sys, itertools
BASE = "vwgmj;.'/=snthk,aeicfpdlx-uoyb" + "r"
COLS = (1, 2, 3, 5, 6, 7)  # pair start columns that keep both keys off the pinkies
jk = [(r*10+c, r*10+c+1) for r in range(3) for c in COLS]           # j left of k
jk += [(r*10+c, (r-1)*10+c) for r in (1, 2) for c in range(1, 9)]    # k stacked above j
hl = [(r*10+c, r*10+c+1) for r in range(3) for c in COLS]            # h left of l

def place(h, j, k, l):
    for order in (('h', 'j', 'k', 'l'), ('l', 'k', 'j', 'h')):
        s = list(BASE)
        for ch in order:
            slot = {'h': h, 'j': j, 'k': k, 'l': l}[ch]
            i = s.index(ch); s[i], s[slot] = s[slot], s[i]
        if s[h] == 'h' and s[j] == 'j' and s[k] == 'k' and s[l] == 'l':
            return "".join(s)
    raise RuntimeError((h, j, k, l))

res = []
for (j, k), (h, l) in itertools.product(jk, hl):
    if len({j, k, h, l}) < 4:
        continue
    out = subprocess.run(["./target/release/kb", "opt", "1.5", sys.argv[1], "4", place(h, j, k, l), "0.05", "hjkl"],
                         capture_output=True, text=True, env={"HINGE": "200"}).stdout.splitlines()[0]
    cost, lay = out.split()[1], out.split("   ", 1)[1]
    res.append((float(cost), lay))
    print(cost, lay, flush=True)
res.sort()
print("BEST")
for c, l in res[:20]:
    print(c, l)
