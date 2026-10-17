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
```

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
