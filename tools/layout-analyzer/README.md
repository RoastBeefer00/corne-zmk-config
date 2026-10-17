# layout-analyzer

Cyanophage-style analyzer and optimizer for the 3x10 + thumb-letter layouts in `config/corne.keymap`.

```sh
cargo build --release
export CORPUS=corpus
./target/release/kb analyze "xpdgv=,'/-lntsw;iaehjbmcf.uoyk r"   # 30 grid keys, row-major, then the thumb letter
./target/release/kb top     "xpdgv=,'/-lntsw;iaehjbmcf.uoyk r"   # worst bigrams per metric
./target/release/kb opt 0 2500000 24 "<baseline>" 0.05           # English layout, no regressions vs baseline beyond slack
LOCAL=1 HINGE=200 ./target/release/kb sweep 1.5 400000 2 "<baseline>" 0.05 "<seed>"   # vim: pin every jk/hl pair placement
./target/release/kb check "<layout>"                             # incremental vs full scoring agree
./target/release/kb eval both "<layout>"                         # exact eval with a repeat key per thumb (none|left|right|both)
./candidates.sh > cands.txt && python3 filter.py cands.txt       # dominance filter vs HD-Promethium
```

Env knobs: `HINGE` (no-regression penalty weight), `LOCAL=1` (start cool from the seed),
`VOWELS=left|right` (force a/i/o/u to one hand), `REPEAT=1` (thumb doubles cost nothing),
`TOPN=n` (list n results), `CORPUS` (directory with the tsv files).

## Adaptives, corpora, robustness

```sh
./target/release/kb adapt "<layout>" 6                         # greedy adaptive-key search, scored exactly
ADAPT_K=6 ./target/release/kb opt 0 4000000 64 - 0              # search assuming 6 bad bigrams get adaptives
python3 robust.py cands.tsv                                     # rank layouts across 4 corpora x 3 effort grids
```

An adaptive "after A, press Z -> types B" turns A->B into A->Z; typing A then Z's own letter
collides with the rule and is counted as `AdaptiveClash` (scored like an SFB, since you must pause).
Letter pairs across a space count as skipgrams (SFS), assuming independent words; `NOSPACE=1` turns
that off. `WeakDoubles` weights same-key doubles by finger (pinky/thumb 1, ring 0.5).
`EFFORT=flat|steep` swaps the effort grid. `corpus_go/` and `corpus_py/` are word lists from Go
doc comments and Python docstrings (`mkwords.py`); `corpus_top5k/` is wordfreq's top 5000 words.

## Repeat keys

`kb eval` rebuilds the n-grams from `corpus/words.tsv` for each layout. The second letter of a
double is typed with a repeat key: on the left thumb, on the right (space) thumb, or with one on
each thumb, where the typist picks whichever avoids a same-thumb conflict with the letter before
or the key after (space included).

## Metrics

All as % of bigrams/skipgrams/trigrams in the corpus (spaces break n-grams).

| Metric | Definition |
|---|---|
| SFB / SFS | same finger, different key, consecutive / one letter apart |
| LSB | index on the inner column + middle or ring on the same hand |
| FSB | full scissor: top <-> bottom row on any two fingers of one hand |
| HSB | half scissor: adjacent fingers one row apart, shorter finger higher |
| Pinky scissors | FSB/HSB involving a pinky |
| Alt, rolls, redirects | trigram hand/direction patterns; weak redirect = no index |
| Effort, finger/row load | per-key effort grid for a column-staggered Corne |
| Pinky off-home, same-finger travel, thumb doubles | as named |

Vim mode adds a normal-mode keystroke model (`VIM` in `src/main.rs`) and requires j/k and h/l
to sit in adjacent pairs.

## Corpus

`corpus/*.tsv` are expected n-gram frequencies of English built from
[wordfreq](https://pypi.org/project/wordfreq/)'s top 80k words with punctuation attached at
realistic rates (`gen_corpus.py`). Regenerate with `python gen_corpus.py` in an environment with
`wordfreq` installed. The numbers will not match cyanophage's site exactly (different corpus and
definitions) but rank layouts the same way.

## Speed

Swaps are scored incrementally (only n-grams containing the two swapped characters are
re-scored) and restarts / sweep placements run in parallel with rayon.
