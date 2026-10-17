# Expected n-gram frequencies of English prose built from wordfreq's word list.
# Spaces break n-grams; punctuation is attached at realistic rates.
from collections import Counter
import os
import re
import sys

# --repeat: the second letter of every double ("ll", "ee") is typed with a repeat key, written '@'
REPEAT = "--repeat" in sys.argv
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "."
os.makedirs(OUT, exist_ok=True)

def with_repeat(w):
    out = []
    for i, c in enumerate(w):
        out.append("@" if REPEAT and i > 0 and c == w[i - 1] and out[-1] != "@" and c.isalpha() else c)
    return "".join(out)
from wordfreq import top_n_list, word_frequency

words = top_n_list('en', 80000)
ok = re.compile(r"^[a-z']+$")
P_COMMA, P_PERIOD, P_SEMI, P_HYPH, P_SLASH = 0.055, 0.050, 0.002, 0.004, 0.0008

mono, bi, skip, tri = Counter(), Counter(), Counter(), Counter()
first, last = Counter(), Counter()
tot = 0.0
rep_space = 0.0  # repeat press directly followed by a space (word ends in a double)
for w in words:
    if not ok.match(w):
        continue
    f = word_frequency(w, 'en')
    w = with_repeat(w)
    if w.endswith("@"):
        rep_space += f * (1 - P_COMMA - P_PERIOD - P_SEMI)
    tot += f
    first[w[0]] += f
    last[w[-1]] += f
    for tail, p in ((',', P_COMMA), ('.', P_PERIOD), (';', P_SEMI), ('', 1 - P_COMMA - P_PERIOD - P_SEMI)):
        s = w + tail
        g = f * p
        for c in s: mono[c] += g
        for i in range(len(s) - 1): bi[s[i:i+2]] += g
        for i in range(len(s) - 2):
            tri[s[i:i+3]] += g
            skip[s[i] + s[i+2]] += g

# hyphen / slash joins: last char of one word, symbol, first char of next
for sym, p in (('-', P_HYPH), ('/', P_SLASH)):
    mono[sym] += tot * p
    for a, fa in last.items():
        bi[a + sym] += tot * p * fa / tot
    for b, fb in first.items():
        bi[sym + b] += tot * p * fb / tot
    for a, fa in last.items():
        for b, fb in first.items():
            g = tot * p * (fa / tot) * (fb / tot)
            tri[a + sym + b] += g
            skip[a + b] += g

def dump(name, c):
    s = sum(c.values())
    with open(os.path.join(OUT, name), 'w') as fh:
        for k, v in c.most_common():
            fh.write(f"{k}\t{v/s:.9g}\n")
dump('mono.tsv', mono); dump('bi.tsv', bi); dump('skip.tsv', skip); dump('tri.tsv', tri)
if REPEAT:
    # same denominator as bi.tsv: share of all (space-free) bigrams
    with open(os.path.join(OUT, 'meta.tsv'), 'w') as fh:
        fh.write(f"rep_space\t{rep_space / sum(bi.values()):.9g}\n")
print(len(mono), len(bi), len(skip), len(tri))
print(sorted(((k, round(v/sum(mono.values())*100, 2)) for k, v in mono.items()), key=lambda x: -x[1]))
