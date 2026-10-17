// Layout analyzer + simulated-annealing optimizer for a 3x10 + 1 thumb-letter Corne layout.
// Metrics mirror cyanophage's: SFB, SFS, LSB, scissors, alternation, rolls, redirects.
use std::collections::HashMap;
use std::fs;
use rayon::prelude::*;

const NSLOT: usize = 31; // permutable slots: 30 grid keys + the left-thumb letter
const NITEM: usize = 33; // the 31 placeable characters + a left- and a right-thumb repeat key
const NKEY: usize = 33; // NSLOT + a second left-thumb key + the right (space) thumb
const THUMB: usize = 30;
const LTHUMB2: usize = 31;
const RTHUMB: usize = 32;
const REP: usize = 31; // repeat key on the left thumb ('@'); never permuted
const REP_R: usize = 32; // repeat key on the right (space) thumb ('#'); never permuted
const ITEMS: &str = "abcdefghijklmnoprstuvwxy,.';-/=@#";

// ---------- geometry ----------
#[derive(Clone, Copy)]
struct Key {
    hand: u8,   // 0 left, 1 right
    finger: u8, // 0..7 = LP LR LM LI RI RM RR RP, 8 = left thumb
    rank: i8,   // pinky0 ring1 middle2 index3 thumb4 (inward = increasing)
    row: i8,
    col: i8,
    effort: f64,
}

fn keys() -> [Key; NKEY] {
    // Effort for a column-staggered Corne, left hand pinky..inner; mirrored for right.
    // EFFORT=flat|steep swaps in an alternative grid (robustness checks)
    let eff = match std::env::var("EFFORT").as_deref() {
        Ok("flat") => [[1.8, 1.6, 1.5, 1.5, 2.0], [1.2, 1.1, 1.0, 1.0, 1.6], [1.8, 1.6, 1.5, 1.4, 2.0]],
        Ok("steep") => [[4.5, 3.2, 2.0, 2.2, 3.2], [2.4, 1.6, 1.1, 1.0, 2.4], [4.8, 3.4, 2.3, 1.6, 3.2]],
        _ => [[3.0, 2.4, 2.0, 2.2, 3.0], [1.6, 1.3, 1.1, 1.0, 2.4], [3.2, 2.6, 2.3, 1.6, 3.0]],
    };
    let mut k = [Key { hand: 0, finger: 0, rank: 0, row: 0, col: 0, effort: 0.0 }; NKEY];
    for r in 0..3 {
        for c in 0..10 {
            let (hand, finger, rank, e) = if c < 5 {
                let f = [0, 1, 2, 3, 3][c];
                (0, f, f as i8, eff[r][c])
            } else {
                let f = [4, 4, 5, 6, 7][c - 5];
                let rk = [3, 3, 2, 1, 0][c - 5];
                (1, f, rk, eff[r][9 - c])
            };
            k[r * 10 + c] = Key { hand, finger, rank, row: r as i8, col: c as i8, effort: e };
        }
    }
    k[THUMB] = Key { hand: 0, finger: 8, rank: 4, row: 3, col: 4, effort: 1.0 };
    // a second left-thumb key (same thumb as the letter) and the right (space) thumb
    k[LTHUMB2] = Key { hand: 0, finger: 8, rank: 4, row: 3, col: 5, effort: 1.2 };
    k[RTHUMB] = Key { hand: 1, finger: 9, rank: 4, row: 3, col: 5, effort: 1.0 };
    k
}

const F_SFB: u8 = 1;
const F_LSB: u8 = 2;
const F_FSB: u8 = 4;
const F_HSB: u8 = 8;
const F_PSC: u8 = 16; // any scissor that involves a pinky
const F_RP: u8 = 32; // adjacent ring + pinky on one hand (shared tendons), any rows

fn bigram_flags(k: &[Key; NKEY], a: usize, b: usize) -> u8 {
    if a == b {
        return 0;
    }
    let (x, y) = (k[a], k[b]);
    if x.hand != y.hand {
        return 0;
    }
    if x.finger == y.finger {
        return F_SFB;
    }
    if x.finger >= 8 || y.finger >= 8 {
        return 0;
    }
    let mut f = 0;
    let inner = |q: Key| q.col == 4 || q.col == 5;
    let midring = |q: Key| q.rank == 1 || q.rank == 2;
    if (inner(x) && midring(y)) || (inner(y) && midring(x)) {
        f |= F_LSB;
    }
    let dr = (x.row - y.row).abs();
    if dr == 2 {
        // full scissor: top<->bottom on any two fingers of one hand (e.g. pinky-bottom -> middle-top)
        f |= F_FSB;
    } else if dr == 1 && (x.rank - y.rank).abs() == 1 {
        {
            // finger length: middle > ring > index > pinky; shorter finger higher = half scissor
            let len = |q: Key| [0, 2, 3, 1][q.rank as usize];
            let (hi, lo) = if x.row < y.row { (x, y) } else { (y, x) };
            if len(hi) < len(lo) {
                f |= F_HSB;
            }
        }
    }
    if x.rank + y.rank == 1 {
        f |= F_RP;
    }
    if f & (F_FSB | F_HSB) != 0 && (x.rank == 0 || y.rank == 0) {
        f |= F_PSC;
    }
    f
}

#[derive(Clone, Copy, PartialEq)]
#[repr(u8)]
enum Tri {
    Alt,
    RollIn,
    RollOut,
    OneIn,
    OneOut,
    Redirect,
    WeakRedirect,
    Other,
}

fn tri_class(k: &[Key; NKEY], a: usize, b: usize, c: usize) -> Tri {
    let (x, y, z) = (k[a], k[b], k[c]);
    if x.hand != y.hand && y.hand != z.hand {
        return Tri::Alt;
    }
    if x.hand == y.hand && y.hand == z.hand {
        if x.finger == y.finger || y.finger == z.finger || x.finger == z.finger {
            return Tri::Other;
        }
        let d1 = y.rank - x.rank;
        let d2 = z.rank - y.rank;
        if d1 > 0 && d2 > 0 {
            return Tri::OneIn;
        }
        if d1 < 0 && d2 < 0 {
            return Tri::OneOut;
        }
        if x.rank != 3 && y.rank != 3 && z.rank != 3 && x.finger < 8 && y.finger < 8 && z.finger < 8 {
            return Tri::WeakRedirect;
        }
        return Tri::Redirect;
    }
    let (p, q) = if x.hand == y.hand { (x, y) } else { (y, z) };
    if p.finger == q.finger {
        return Tri::Other;
    }
    if q.rank > p.rank { Tri::RollIn } else { Tri::RollOut }
}

// ---------- corpus ----------
struct Corpus {
    rep_space: f64, // repeat press directly followed by space (% of bigrams); 0 without a repeat key
    rep_space_exact: bool, // true when rep_space already counts only presses of the space-thumb repeat key
    ad_conflict: f64, // bigrams that collide with an adaptive rule (% of bigrams)
    rep: Vec<f64>, // same-letter double frequency (% of bigrams) per item
    mono: Vec<(usize, f64)>,
    bi: Vec<(usize, usize, f64)>,
    skip: Vec<(usize, usize, f64)>,
    tri: Vec<(usize, usize, usize, f64)>,
}

fn item_id(c: char) -> Option<usize> {
    ITEMS.chars().position(|x| x == c)
}

#[allow(dead_code)] // superseded by corpus_from_words; kept for the tsv tables
fn load(dir: &str, tri_cap: usize) -> Corpus {
    let read = |name: &str| -> Vec<(Vec<usize>, f64)> {
        let s = fs::read_to_string(format!("{dir}/{name}")).expect(name);
        let mut v = vec![];
        for line in s.lines() {
            let (g, f) = line.split_once('\t').unwrap();
            let ids: Option<Vec<usize>> = g.chars().map(item_id).collect();
            if let Some(ids) = ids {
                v.push((ids, f.parse::<f64>().unwrap()));
            }
        }
        let tot: f64 = v.iter().map(|x| x.1).sum();
        v.into_iter().map(|(g, f)| (g, f / tot * 100.0)).collect() // percent
    };
    let mono = read("mono.tsv").into_iter().map(|(g, f)| (g[0], f)).collect();
    let bi_all = read("bi.tsv");
    let mut rep = vec![0.0; NITEM];
    for (g, f) in &bi_all {
        if g[0] == g[1] {
            rep[g[0]] += f;
        }
    }
    let bi = bi_all.into_iter().filter(|(g, _)| g[0] != g[1]).map(|(g, f)| (g[0], g[1], f)).collect();
    let skip = read("skip.tsv").into_iter().filter(|(g, _)| g[0] != g[1]).map(|(g, f)| (g[0], g[1], f)).collect();
    let mut tri: Vec<_> = read("tri.tsv").into_iter().map(|(g, f)| (g[0], g[1], g[2], f)).collect();
    tri.truncate(tri_cap);
    let tt: f64 = tri.iter().map(|x| x.3).sum();
    let tri = tri.into_iter().map(|(a, b, c, f)| (a, b, c, f / tt * 100.0)).collect();
    let rep_space = fs::read_to_string(format!("{dir}/meta.tsv"))
        .ok()
        .and_then(|s| s.lines().find_map(|l| l.strip_prefix("rep_space\t").map(|v| v.parse::<f64>().unwrap() * 100.0)))
        .unwrap_or(0.0);
    Corpus { rep_space, rep_space_exact: false, ad_conflict: 0.0, rep, mono, bi, skip, tri }
}

// Vim normal-mode keystroke model: (sequence, relative weight). Uppercase uses the same key
// (sticky shift on thumb); ':' and '"' map to ';' and '\''. Sym-layer keys are dropped.
const VIM: &[(&str, f64)] = &[
    ("j", 300.), ("k", 260.), ("h", 60.), ("l", 70.), ("w", 120.), ("b", 90.), ("e", 50.),
    ("W", 15.), ("B", 12.), ("gg", 20.), ("G", 20.), ("n", 40.), ("N", 12.), ("/", 35.),
    ("f", 25.), ("t", 12.), (";", 12.), (",", 6.), ("d", 30.), ("u", 25.),
    ("dd", 40.), ("yy", 25.), ("p", 40.), ("P", 12.), ("u", 45.), ("x", 30.), (".", 35.),
    ("o", 35.), ("O", 18.), ("i", 55.), ("a", 30.), ("A", 25.), ("I", 10.), ("cw", 15.),
    ("ce", 6.), ("ciw", 22.), ("diw", 15.), ("dw", 18.), ("db", 5.), ("yiw", 12.), ("viw", 10.),
    ("ci'", 10.), ("di'", 4.), ("dap", 4.), ("vip", 5.), ("yip", 4.), ("v", 25.), ("V", 25.),
    ("y", 15.), ("d", 12.), ("c", 8.), ("r", 12.), ("s", 6.), ("C", 8.), ("D", 8.), ("J", 5.),
    ("gd", 15.), ("gr", 8.), ("gcc", 8.), ("K", 8.), (":w", 45.),
    ("dt", 6.), ("ct", 6.), ("df", 4.), ("cf", 4.), ("gv", 3.), ("yw", 6.), ("ve", 4.),
    ("de", 5.), ("cc", 6.), ("S", 3.), ("ea", 3.), ("bi", 3.),
];

struct VimSeq {
    keys: Vec<usize>,
    w: f64,
}

fn vim_seqs() -> Vec<VimSeq> {
    let tot: f64 = VIM.iter().map(|x| x.1).sum();
    VIM.iter()
        .map(|(s, w)| VimSeq {
            keys: s
                .chars()
                .map(|c| match c {
                    ':' => ';',
                    '"' => '\'',
                    c => c.to_ascii_lowercase(),
                })
                .map(|c| item_id(c).expect("vim char"))
                .collect(),
            w: w / tot * 100.0,
        })
        .collect()
}

// ---------- metrics ----------
#[derive(Default, Clone, Copy, Debug)]
struct Metrics {
    sfb: f64,
    sfs: f64,
    lsb: f64,
    fsb: f64,
    hsb: f64,
    psc: f64,
    ring_pinky: f64,
    vim_rp: f64,
    alt: f64,
    roll_in: f64,
    roll_out: f64,
    one_in: f64,
    one_out: f64,
    redirect: f64,
    weak_red: f64,
    effort: f64,
    fingers: [f64; 10], // 8 = left thumb, 9 = right thumb
    rows: [f64; 4], // top, home, bottom, thumb
    pinky_off: f64,
    thumb_rep: f64, // double taps of the thumb letter (% of bigrams)
    rep_space_sfb: f64, // repeat key then space on the same (right) thumb, counted into SFB
    ad_conflict: f64, // adaptive-rule collisions (typist pauses), % of bigrams
    weak_dbl: f64, // same-key doubles weighted by finger: pinky/thumb 1, ring 0.5 (% of bigrams)
    fixable: f64, // ADAPT_K=k: weighted cost of the k worst bigrams (what k adaptives could remove)
    sf_dist: f64, // same-finger travel (key units) per 100 bigrams, SFB + 0.5*SFS
    vim_effort: f64,
    vim_sfb: f64,
    vim_scis: f64,
    hjkl: f64,
    vowel_viol: f64, // VOWELS=left|right: a/i/o/u not on that hand
    moved: f64, // NEAR=<layout>: keys placed differently from that layout
    pref_viol: f64, // PREFS=1: bottom-heavy, no h/i or c/i ring-pinky, w off the pinkies, e middle-home + a index-home
    tri: [f64; 8], // trigram class totals, indexed by Tri
}

struct Model {
    k: [Key; NKEY],
    bflag: [[u8; NKEY]; NKEY],
    dist: [[f64; NKEY]; NKEY],
    tcls: Vec<Tri>,
    c: Corpus,
    // vim keystroke model, pre-normalized: per-item key share and per-pair share (percent)
    vim_mono: Vec<f64>,
    vim_pairs: Vec<(usize, usize, f64)>,
    // per item: indices of the n-grams that contain it (each n-gram listed once per item)
    // stored inline (not as indices) so the swap loop walks contiguous memory
    idx_bi: Vec<Vec<([u8; 2], f64)>>,
    idx_skip: Vec<Vec<([u8; 2], f64)>>,
    idx_tri: Vec<Vec<([u8; 3], f64)>>,
    idx_vim: Vec<Vec<([u8; 2], f64)>>,
    vowel_hand: Option<u8>,
    prefs: bool,
    near: Option<[usize; NITEM]>,
    adapt_k: usize,
}

fn index_by_item<const N: usize>(grams: impl Iterator<Item = ([usize; N], f64)>) -> Vec<Vec<([u8; N], f64)>> {
    let mut idx = vec![vec![]; NITEM];
    for (g, f) in grams {
        for (n, &c) in g.iter().enumerate() {
            if !g[..n].contains(&c) {
                idx[c].push((g.map(|x| x as u8), f));
            }
        }
    }
    idx
}

impl Model {
    fn new(c: Corpus) -> Self {
        let k = keys();
        let mut bflag = [[0u8; NKEY]; NKEY];
        let mut dist = [[0.0f64; NKEY]; NKEY];
        let mut tcls = vec![Tri::Other; NKEY * NKEY * NKEY];
        for a in 0..NKEY {
            for b in 0..NKEY {
                bflag[a][b] = bigram_flags(&k, a, b);
                dist[a][b] = (((k[a].row - k[b].row) as f64).powi(2) + ((k[a].col - k[b].col) as f64).powi(2)).sqrt();
                for d in 0..NKEY {
                    tcls[(a * NKEY + b) * NKEY + d] = tri_class(&k, a, b, d);
                }
            }
        }
        let vim = vim_seqs();
        let keys_total: f64 = vim.iter().map(|v| v.w * v.keys.len() as f64).sum();
        let pairs_total: f64 = vim.iter().map(|v| v.w * (v.keys.len() - 1) as f64).sum();
        let mut vim_mono = vec![0.0; NITEM];
        let mut pairs: HashMap<(usize, usize), f64> = HashMap::new();
        for v in &vim {
            for (i, &c) in v.keys.iter().enumerate() {
                vim_mono[c] += v.w / keys_total;
                if i > 0 && v.keys[i - 1] != c {
                    *pairs.entry((v.keys[i - 1], c)).or_default() += v.w * 100.0 / pairs_total;
                }
            }
        }
        let mut vim_pairs: Vec<_> = pairs.into_iter().map(|((a, b), w)| (a, b, w)).collect();
        vim_pairs.sort_by(|x, y| (x.0, x.1).cmp(&(y.0, y.1)));
        Model {
            idx_bi: index_by_item(c.bi.iter().map(|g| ([g.0, g.1], g.2))),
            idx_skip: index_by_item(c.skip.iter().map(|g| ([g.0, g.1], g.2))),
            idx_tri: index_by_item(c.tri.iter().map(|g| ([g.0, g.1, g.2], g.3))),
            idx_vim: index_by_item(vim_pairs.iter().map(|g| ([g.0, g.1], g.2))),
            k, bflag, dist, tcls, c, vim_mono, vim_pairs,
            prefs: std::env::var("PREFS").is_ok(),
            near: std::env::var("NEAR").ok().map(|l| parse(&l)),
            adapt_k: std::env::var("ADAPT_K").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
            vowel_hand: match std::env::var("VOWELS").as_deref() {
                Ok("left") => Some(0),
                Ok("right") => Some(1),
                _ => None,
            },
        }
    }

    // ---- per-n-gram contributions; `sg` is +1 to add, -1 to remove ----
    #[inline]
    fn add_mono(&self, m: &mut Metrics, s: usize, f: f64, sg: f64) {
        let k = self.k[s];
        let f = f * sg;
        m.effort += f * k.effort / 100.0;
        m.fingers[k.finger as usize] += f;
        m.rows[k.row as usize] += f;
        if k.rank == 0 && k.row != 1 {
            m.pinky_off += f;
        }
    }
    #[inline]
    fn add_bi(&self, m: &mut Metrics, sa: usize, sb: usize, f: f64, sg: f64) {
        let fl = self.bflag[sa][sb];
        if fl == 0 {
            return;
        }
        let f = f * sg;
        if fl & F_SFB != 0 {
            m.sfb += f;
            m.sf_dist += f * self.dist[sa][sb];
        }
        if fl & F_LSB != 0 { m.lsb += f; }
        if fl & F_FSB != 0 { m.fsb += f; }
        if fl & F_HSB != 0 { m.hsb += f; }
        if fl & F_PSC != 0 { m.psc += f; }
        if fl & F_RP != 0 { m.ring_pinky += f; }
    }
    #[inline]
    fn add_skip(&self, m: &mut Metrics, sa: usize, sb: usize, f: f64, sg: f64) {
        if self.bflag[sa][sb] & F_SFB != 0 {
            m.sfs += f * sg;
            m.sf_dist += 0.5 * f * sg * self.dist[sa][sb];
        }
    }
    #[inline]
    fn add_tri(&self, m: &mut Metrics, sa: usize, sb: usize, sc: usize, f: f64, sg: f64) {
        // branch-free: an indexed add is much faster than matching on the class
        m.tri[self.tcls[(sa * NKEY + sb) * NKEY + sc] as usize] += f * sg;
    }
    #[inline]
    fn add_vim_pair(&self, m: &mut Metrics, sa: usize, sb: usize, w: f64, sg: f64) {
        let fl = self.bflag[sa][sb];
        if fl & F_SFB != 0 { m.vim_sfb += w * sg; }
        if fl & (F_FSB | F_HSB | F_LSB) != 0 { m.vim_scis += w * sg; }
        if fl & F_RP != 0 { m.vim_rp += w * sg; }
    }

    fn finish(&self, m: &mut Metrics, pos: &[usize; NITEM], with_vim: bool) {
        m.alt = m.tri[Tri::Alt as usize];
        m.roll_in = m.tri[Tri::RollIn as usize];
        m.roll_out = m.tri[Tri::RollOut as usize];
        m.one_in = m.tri[Tri::OneIn as usize];
        m.one_out = m.tri[Tri::OneOut as usize];
        m.redirect = m.tri[Tri::Redirect as usize];
        m.weak_red = m.tri[Tri::WeakRedirect as usize];
        m.thumb_rep = (0..NITEM).find(|&i| pos[i] == THUMB).map_or(0.0, |i| self.c.rep[i]);
        m.weak_dbl = (0..NITEM)
            .map(|i| self.c.rep[i] * match self.k[pos[i]].rank { 0 | 4 => 1.0, 1 => 0.5, _ => 0.0 })
            .sum();
        if with_vim {
            m.hjkl = self.hjkl_penalty(pos);
        }
        if self.adapt_k > 0 {
            let mut worst = [0.0f64; 16];
            let k = self.adapt_k.min(16);
            for &(a, b, f) in &self.c.bi {
                let fl = self.bflag[pos[a]][pos[b]];
                if fl == 0 {
                    continue;
                }
                let w = if fl & F_SFB != 0 { 12.0 } else { 0.0 } + if fl & F_FSB != 0 { 6.0 } else { 0.0 }
                    + if fl & F_HSB != 0 { 3.0 } else { 0.0 } + if fl & F_LSB != 0 { 5.0 } else { 0.0 }
                    + if fl & F_PSC != 0 { 6.0 } else { 0.0 } + if fl & F_RP != 0 { 3.0 } else { 0.0 };
                let v = f * w;
                if v > worst[k - 1] {
                    let mut i = k - 1;
                    while i > 0 && worst[i - 1] < v {
                        worst[i] = worst[i - 1];
                        i -= 1;
                    }
                    worst[i] = v;
                }
            }
            m.fixable = worst[..k].iter().sum();
        }
        if let Some(r) = &self.near {
            m.moved = (0..NSLOT).filter(|&i| pos[i] != r[i]).count() as f64;
        }
        if self.prefs {
            let key = |c| self.k[pos[item_id(c).unwrap()]];
            let (h, i, w, e, a, c) = (key('h'), key('i'), key('w'), key('e'), key('a'), key('c'));
            let ring_pinky = |x: Key, y: Key| x.hand == y.hand && x.rank + y.rank == 1 && x.finger < 8 && y.finger < 8;
            let ea_roll = e.row == 1 && e.rank == 2 && a.row == 1 && a.rank == 3 && a.col != 4 && a.col != 5 && a.hand == e.hand;
            m.pref_viol = (m.rows[0] - m.rows[2]).max(0.0)
                + ring_pinky(h, i) as u8 as f64
                + ring_pinky(c, i) as u8 as f64
                + (w.rank == 0) as u8 as f64
                + (!ea_roll) as u8 as f64;
        }
        if let Some(hand) = self.vowel_hand {
            m.vowel_viol = "aiou".chars().filter(|&c| self.k[pos[item_id(c).unwrap()]].hand != hand).count() as f64;
        }
    }

    fn metrics(&self, pos: &[usize; NITEM], with_vim: bool) -> Metrics {
        let mut m = Metrics::default();
        for &(c, f) in &self.c.mono {
            self.add_mono(&mut m, pos[c], f, 1.0);
        }
        for &(a, b, f) in &self.c.bi {
            self.add_bi(&mut m, pos[a], pos[b], f, 1.0);
        }
        for &(a, b, f) in &self.c.skip {
            self.add_skip(&mut m, pos[a], pos[b], f, 1.0);
        }
        for &(a, b, c, f) in &self.c.tri {
            self.add_tri(&mut m, pos[a], pos[b], pos[c], f, 1.0);
        }
        if with_vim {
            for (c, &w) in self.vim_mono.iter().enumerate() {
                m.vim_effort += w * self.k[pos[c]].effort;
            }
            for &(a, b, w) in &self.vim_pairs {
                self.add_vim_pair(&mut m, pos[a], pos[b], w, 1.0);
            }
        }
        m.ad_conflict = self.c.ad_conflict;
        if self.c.rep_space_exact || self.k[pos[REP]].finger == self.k[RTHUMB].finger {
            m.rep_space_sfb = self.c.rep_space;
            m.sfb += m.rep_space_sfb;
        }
        self.finish(&mut m, pos, with_vim);
        m
    }

    // Swap items x and y in `pos`, updating `m` incrementally: every n-gram containing x or y is
    // removed at its old slots and re-added at its new slots in one pass.
    fn swap(&self, m: &mut Metrics, pos: &mut [usize; NITEM], x: usize, y: usize, with_vim: bool) {
        let (xb, yb) = (x as u8, y as u8);
        let (sx, sy) = (pos[x], pos[y]);
        let old = |c: u8| pos[c as usize];
        let new = |c: u8| if c == xb { sy } else if c == yb { sx } else { pos[c as usize] };
        for &(c, f) in self.c.mono.iter().filter(|g| g.0 == x || g.0 == y) {
            self.add_mono(m, pos[c], f, -1.0);
            self.add_mono(m, if c == x { sy } else { sx }, f, 1.0);
        }
        // n-grams containing both x and y are visited through x's list only
        for (i, list) in [(xb, &self.idx_bi[x]), (yb, &self.idx_bi[y])] {
            for &([a, b], f) in list.iter() {
                if i == yb && (a == xb || b == xb) { continue; }
                self.add_bi(m, old(a), old(b), f, -1.0);
                self.add_bi(m, new(a), new(b), f, 1.0);
            }
        }
        for (i, list) in [(xb, &self.idx_skip[x]), (yb, &self.idx_skip[y])] {
            for &([a, b], f) in list.iter() {
                if i == yb && (a == xb || b == xb) { continue; }
                self.add_skip(m, old(a), old(b), f, -1.0);
                self.add_skip(m, new(a), new(b), f, 1.0);
            }
        }
        for (i, list) in [(xb, &self.idx_tri[x]), (yb, &self.idx_tri[y])] {
            for &([a, b, c], f) in list.iter() {
                if i == yb && (a == xb || b == xb || c == xb) { continue; }
                self.add_tri(m, old(a), old(b), old(c), f, -1.0);
                self.add_tri(m, new(a), new(b), new(c), f, 1.0);
            }
        }
        if with_vim {
            m.vim_effort += self.vim_mono[x] * (self.k[sy].effort - self.k[sx].effort)
                + self.vim_mono[y] * (self.k[sx].effort - self.k[sy].effort);
            for (i, list) in [(xb, &self.idx_vim[x]), (yb, &self.idx_vim[y])] {
                for &([a, b], w) in list.iter() {
                    if i == yb && (a == xb || b == xb) { continue; }
                    self.add_vim_pair(m, old(a), old(b), w, -1.0);
                    self.add_vim_pair(m, new(a), new(b), w, 1.0);
                }
            }
        }
        pos.swap(x, y);
        self.finish(m, pos, with_vim);
    }

    // Pairs, Dvorak/Engram-style: j/k side by side on different adjacent fingers (j left of k), and
    // h/l adjacent (h left of l on one row, or stacked in either order). Neither pair on the thumb.
    // Pairs on opposite hands (like Engram) get a small bonus.
    fn hjkl_penalty(&self, pos: &[usize; NITEM]) -> f64 {
        let id = |c| self.k[pos[item_id(c).unwrap()]];
        let (h, j, k, l) = (id('h'), id('j'), id('k'), id('l'));
        if [h, j, k, l].iter().any(|q| q.finger == 8) {
            return 10.0;
        }
        let pair = |a: Key, b: Key, either_stack: bool| {
            let horiz = ((b.row - a.row).abs() + (b.col - a.col - 1).abs()) as f64
                + if a.hand != b.hand { 2.0 } else { 0.0 };
            let stack = |lo: Key, hi: Key| ((hi.col - lo.col).abs() + (lo.row - hi.row - 1).abs()) as f64;
            let vert = if either_stack { stack(a, b).min(stack(b, a)) } else { stack(a, b) };
            horiz.min(vert)
        };
        let same_hand = if j.hand == h.hand { 0.15 } else { 0.0 };
        // j and k must be on different, non-pinky fingers (index/middle or middle/ring)
        let spam = if j.finger == k.finger || j.rank == 0 || k.rank == 0 || (j.rank - k.rank).abs() != 1 || j.rank + k.rank == 1 { 2.0 } else { 0.0 };
        // NOHL=1: only j/k have to be grouped
        if std::env::var("NOHL").is_ok() {
            return pair(j, k, false) + spam;
        }
        pair(j, k, false) + pair(h, l, true) + same_hand + spam
    }
}

// ---------- objective ----------
#[derive(Clone)]
struct Obj {
    vim_w: f64,
    base: Option<Metrics>, // hinge: never worse than this on the headline metrics
    slack: f64,
    hinge: f64,
    repeat_key: bool, // REPEAT=1: doubles go to a repeat key, so thumb-letter doubles cost nothing
    near_w: f64, // NEAR_W: cost per key moved away from the NEAR layout
}

impl Obj {
    fn new(vim_w: f64, base: Option<Metrics>, slack: f64) -> Self {
        let hinge = std::env::var("HINGE").ok().and_then(|v| v.parse().ok()).unwrap_or(100.0);
        let repeat_key = std::env::var("REPEAT").is_ok();
        let near_w = std::env::var("NEAR_W").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
        Obj { vim_w, base, slack, hinge, repeat_key, near_w }
    }
}

fn cost(m: &Metrics, o: &Obj) -> f64 {
    let pinky = m.fingers[0] + m.fingers[7];
    let mut c = 12.0 * m.sfb + 3.0 * m.sfs + 5.0 * m.lsb + 6.0 * m.fsb + 3.0 * m.hsb + 6.0 * m.psc + 3.0 * m.ring_pinky
        + 2.0 * m.redirect + 5.0 * m.weak_red
        - 0.4 * m.alt - 0.4 * m.roll_in - 0.2 * m.roll_out
        + 4.0 * m.effort
        + if o.repeat_key { 0.0 } else { 6.0 * m.thumb_rep }
        + 4.0 * m.weak_dbl
        + 100.0 * m.vowel_viol
        + 100.0 * m.pref_viol
        + 12.0 * m.ad_conflict
        - 0.9 * m.fixable
        + o.near_w * m.moved
        + 1.5 * pinky + 2.0 * m.pinky_off;
    for f in m.fingers.iter().take(8) {
        c += 1.0 * (f - 16.0).max(0.0);
    }
    if let Some(b) = &o.base {
        let s = o.slack;
        let h = |x: f64| o.hinge * x.max(0.0);
        c += h(m.sfb - b.sfb - s) + h(m.sfs - b.sfs - s) + h(m.lsb - b.lsb - s)
            + h((m.fsb + m.hsb) - (b.fsb + b.hsb) - s) + h(m.fsb - b.fsb - s) + h(m.psc - b.psc - s) + h(m.ring_pinky - b.ring_pinky - s) + h(b.alt - m.alt - 4.0 * s) + h(b.roll_in - m.roll_in - 4.0 * s)
            + h(m.redirect + m.weak_red - b.redirect - b.weak_red - 4.0 * s)
            + h(10.0 * (m.effort - b.effort) - s)
            + h(b.rows[1] - m.rows[1] - 4.0 * s)
            + h(m.pinky_off - b.pinky_off - s)
            + h(m.sf_dist - b.sf_dist - s)
            + h(m.roll_out / m.roll_in.max(1e-9) - b.roll_out / b.roll_in.max(1e-9) - s)
            + h(pinky - (b.fingers[0] + b.fingers[7]) - 4.0 * s);
    }
    if o.vim_w > 0.0 {
        c += o.vim_w * (2.0 * m.vim_effort + 0.3 * m.vim_sfb + 0.1 * m.vim_scis + 0.3 * m.vim_rp) + 30.0 * m.hjkl;
    }
    c
}

// ---------- layout I/O ----------
// REPKEY=right puts the repeat key on the space thumb, REPKEY=grid lets it be placed on the grid
// in place of '=' (which is parked off-board); otherwise it sits on the left thumb.
fn rep_slot() -> usize {
    if std::env::var("REPKEY").is_ok_and(|v| v == "right") { RTHUMB } else { LTHUMB2 }
}

fn rep_grid() -> bool {
    std::env::var("REPKEY").is_ok_and(|v| v == "grid")
}

// The one item that never moves: the repeat key, or '=' when the repeat key is on the grid.
fn parked() -> usize {
    if rep_grid() { item_id('=').unwrap() } else { REP }
}

fn parse(s: &str) -> [usize; NITEM] {
    let s: Vec<char> = s.chars().filter(|c| !c.is_whitespace()).collect();
    assert_eq!(s.len(), NSLOT, "need 30 grid chars + thumb");
    let mut pos = [usize::MAX; NITEM];
    for (slot, &c) in s.iter().enumerate() {
        pos[item_id(c).unwrap_or_else(|| panic!("bad char {c}"))] = slot;
    }
    if rep_grid() {
        // the layout string carries '@' on the grid; '=' goes off-board
        let eq = item_id('=').unwrap();
        if pos[REP] == usize::MAX {
            pos[REP] = pos[eq];
        }
        pos[eq] = LTHUMB2;
    } else {
        pos[REP] = rep_slot();
    }
    pos[REP_R] = RTHUMB;
    assert!(pos.iter().all(|&p| p != usize::MAX), "missing chars");
    pos
}

fn show(pos: &[usize; NITEM]) -> String {
    let mut g = [' '; NKEY];
    for (i, c) in ITEMS.chars().enumerate() {
        g[pos[i]] = c;
    }
    let mut out = String::new();
    for r in 0..3 {
        let row: String = g[r * 10..r * 10 + 10].iter().map(|c| format!("{c} ")).collect();
        out += &format!("  {} {}\n", &row[..10], &row[10..]);
    }
    out + &format!("        [{}]\n", g[THUMB])
}

fn compact(pos: &[usize; NITEM]) -> String {
    let mut g = [' '; NKEY];
    for (i, c) in ITEMS.chars().enumerate() {
        g[pos[i]] = c;
    }
    let s: String = g.iter().collect();
    format!("{} {} {} {}", &s[0..10], &s[10..20], &s[20..30], &s[30..])
}

fn report(m: &Metrics) {
    println!(
        "  SFB {:.2}%  SFS {:.2}%  LSB {:.2}%  FSB {:.2}%  HSB {:.2}%  (scissors {:.2}%, pinky scissors {:.2}%)  RingPinky {:.2}%",
        m.sfb, m.sfs, m.lsb, m.fsb, m.hsb, m.fsb + m.hsb, m.psc, m.ring_pinky
    );
    println!(
        "  Alt {:.1}%  RollIn {:.1}%  RollOut {:.1}%  1hIn {:.1}%  1hOut {:.1}%  Redir {:.2}%  WeakRedir {:.2}%",
        m.alt, m.roll_in, m.roll_out, m.one_in, m.one_out, m.redirect, m.weak_red
    );
    let f = m.fingers;
    println!(
        "  Effort {:.3}  Fingers LP {:.1} LR {:.1} LM {:.1} LI {:.1} | LT {:.1} RT {:.1} | RI {:.1} RM {:.1} RR {:.1} RP {:.1}  (L {:.1} / R {:.1})",
        m.effort, f[0], f[1], f[2], f[3], f[8], f[9], f[4], f[5], f[6], f[7],
        f[0] + f[1] + f[2] + f[3] + f[8], f[4] + f[5] + f[6] + f[7]
    );
    println!(
        "  Rows top {:.1}% home {:.1}% bottom {:.1}% thumb {:.1}%  PinkyOffHome {:.2}%  SameFingerDist {:.3}  In/Out {:.2}  ThumbDoubles {:.2}%  Rep->Space SFB {:.2}%  AdaptiveClash {:.2}%  WeakDoubles {:.2}%",
        m.rows[0], m.rows[1], m.rows[2], m.rows[3], m.pinky_off, m.sf_dist, m.roll_in / m.roll_out, m.thumb_rep, m.rep_space_sfb, m.ad_conflict, m.weak_dbl
    );
    println!(
        "  Vim: effort/key {:.3}  SFB {:.2}%  scissor+LSB {:.2}%  ring-pinky {:.2}%  hjkl-geometry {:.2}",
        m.vim_effort, m.vim_sfb, m.vim_scis, m.vim_rp, m.hjkl
    );
}

// ---------- optimizer ----------
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn n(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn is_letter(i: usize) -> bool {
    ITEMS.as_bytes()[i].is_ascii_lowercase()
}

fn legal(pos: &[usize; NITEM], a: usize, b: usize) -> bool {
    a != b && !((pos[a] == THUMB && !is_letter(b)) || (pos[b] == THUMB && !is_letter(a)))
}

// Simulated annealing over item swaps with incremental scoring, then a steepest-descent polish.
// `local` starts from `start` at a low temperature instead of from a shuffle.
fn anneal(md: &Model, o: &Obj, start: [usize; NITEM], iters: usize, seed: u64, fixed: &[usize], local: bool) -> ([usize; NITEM], f64) {
    let mut rng = Rng(seed | 1);
    let vim = o.vim_w > 0.0;
    let mut pos = start;
    let free: Vec<usize> = (0..NITEM).filter(|&i| !fixed.contains(&i) && i != parked() && i != REP_R).collect();
    if !local {
        for _ in 0..200 {
            let (a, b) = (free[rng.n(free.len())], free[rng.n(free.len())]);
            if legal(&pos, a, b) {
                pos.swap(a, b);
            }
        }
    }
    let mut m = md.metrics(&pos, vim);
    let mut cur = cost(&m, o);
    let (mut best, mut best_c) = (pos, cur);
    let (t0, t1) = (if local { 0.6 } else { 8.0f64 }, 0.002f64);
    for it in 0..iters {
        if it % 50_000 == 0 {
            m = md.metrics(&pos, vim); // resync floating-point drift
            cur = cost(&m, o);
        }
        let t = t0 * (t1 / t0).powf(it as f64 / iters as f64);
        let (a, b) = (free[rng.n(free.len())], free[rng.n(free.len())]);
        if !legal(&pos, a, b) {
            continue;
        }
        let saved = m;
        md.swap(&mut m, &mut pos, a, b, vim);
        let c = cost(&m, o);
        if c <= cur || rng.f() < ((cur - c) / t).exp() {
            cur = c;
            if c < best_c {
                best_c = c;
                best = pos;
            }
        } else {
            pos.swap(a, b);
            m = saved;
        }
    }
    let mut pos = best;
    let mut m = md.metrics(&pos, vim);
    let mut best_c = cost(&m, o);
    loop {
        let mut improved = false;
        for i in 0..free.len() {
            for j in i + 1..free.len() {
                let (a, b) = (free[i], free[j]);
                if !legal(&pos, a, b) {
                    continue;
                }
                let saved = m;
                md.swap(&mut m, &mut pos, a, b, vim);
                let c = cost(&m, o);
                if c < best_c - 1e-9 {
                    best_c = c;
                    improved = true;
                } else {
                    pos.swap(a, b);
                    m = saved;
                }
            }
        }
        if !improved {
            break;
        }
    }
    let exact = cost(&md.metrics(&pos, vim), o);
    (pos, exact)
}

fn seed_for(i: usize) -> u64 {
    0x9E3779B97F4A7C15u64.wrapping_mul(i as u64 + 1 + std::process::id() as u64)
}

// Run `restarts` independent anneals in parallel and return them best-first.
fn multi_anneal(md: &Model, o: &Obj, start: [usize; NITEM], iters: usize, restarts: usize, fixed: &[usize], local: bool) -> Vec<([usize; NITEM], f64)> {
    let mut r: Vec<_> = (0..restarts)
        .into_par_iter()
        .map(|i| anneal(md, o, start, iters, seed_for(i), fixed, local))
        .collect();
    r.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    r
}

fn print_results(md: &Model, results: Vec<([usize; NITEM], f64)>, detail: usize, max: usize) {
    // how many restarts ended on each layout (convergence evidence), on stderr
    let mut hits: HashMap<String, (usize, f64)> = HashMap::new();
    for (pos, c) in &results {
        hits.entry(compact(pos)).or_insert((0, *c)).0 += 1;
    }
    let mut hv: Vec<_> = hits.into_iter().collect();
    hv.sort_by(|a, b| a.1 .1.partial_cmp(&b.1 .1).unwrap());
    for (l, (n, c)) in hv.iter().take(15) {
        eprintln!("hits {n:>3}  cost {c:.3}  {l}");
    }
    let mut seen = HashMap::new();
    for (pos, c) in results {
        let key = compact(&pos);
        if seen.insert(key.clone(), ()).is_some() {
            continue;
        }
        println!("cost {:.3}   {}", c, key);
        if seen.len() <= detail {
            print!("{}", show(&pos));
            report(&md.metrics(&pos, true));
        }
        if seen.len() >= max {
            break;
        }
    }
}

// Put item x at slot s by swapping with whatever is there.
fn place(pos: &mut [usize; NITEM], x: usize, s: usize) {
    let y = (0..NITEM).find(|&i| pos[i] == s).unwrap();
    pos.swap(x, y);
}

// ---------- exact per-layout corpus (for repeat-key modes) ----------
#[derive(Clone, Copy, PartialEq, Debug)]
enum RepMode {
    None,  // doubles are same-key presses
    Left,  // one repeat key on the left thumb
    Right, // one repeat key on the right (space) thumb
    Both,  // one on each thumb; the typist picks whichever avoids a same-thumb conflict
}

// Builds n-gram tables from words.tsv for one specific layout, the same way gen_corpus.py does,
// but replacing the second letter of each double with the repeat key chosen under `mode`.
// An adaptive key: after `prev`, pressing the key of `via` types `out` (ZMK antecedent-morph).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Adaptive {
    prev: usize,
    out: usize,
    via: usize,
}

fn words(dir: &str) -> &'static [(Vec<char>, f64)] {
    static WORDS: std::sync::OnceLock<Vec<(Vec<char>, f64)>> = std::sync::OnceLock::new();
    WORDS.get_or_init(|| {
        fs::read_to_string(format!("{dir}/words.tsv"))
            .expect("words.tsv")
            .lines()
            .map(|l| {
                let (w, f) = l.split_once('\t').unwrap();
                (w.chars().collect(), f.parse().unwrap())
            })
            .collect()
    })
}

fn corpus_from_words(dir: &str, pos: &[usize; NITEM], k: &[Key; NKEY], mode: RepMode, tri_cap: usize) -> Corpus {
    corpus_with_adaptives(dir, pos, k, mode, tri_cap, &[])
}

// Like corpus_from_words, with adaptive keys applied. Typing `prev` then the letter of `via` itself
// collides with a rule; those bigrams are reported as `ad_conflict` (the typist has to pause).
fn corpus_with_adaptives(dir: &str, pos: &[usize; NITEM], k: &[Key; NKEY], mode: RepMode, tri_cap: usize, rules: &[Adaptive]) -> Corpus {
    const P_COMMA: f64 = 0.055;
    const P_PERIOD: f64 = 0.050;
    const P_SEMI: f64 = 0.002;
    const JOINS: [(char, f64); 2] = [('-', 0.004), ('/', 0.0008)];
    let sfb = |a: usize, b: usize| (a != b && bigram_flags(k, a, b) & F_SFB != 0) as u8;
    let mut mono = vec![0.0; NITEM];
    let mut bi: HashMap<[usize; 2], f64> = HashMap::new();
    let mut skip: HashMap<[usize; 2], f64> = HashMap::new();
    let mut tri: HashMap<[usize; 3], f64> = HashMap::new();
    let (mut first, mut last) = (vec![0.0; NITEM], vec![0.0; NITEM]);
    let (mut words_tot, mut rep_space, mut conflict) = (0.0, 0.0, 0.0);
    // key that ends a word (last letter or trailing punctuation) / starts one, for pairs across a space
    let (mut end_key, mut start_key) = (vec![0.0; NITEM], vec![0.0; NITEM]);
    for (ch, f) in words(dir) {
        let f = *f;
        words_tot += f;
        if let Some(x) = item_id(ch[0]) { first[x] += f; }
        if let Some(x) = item_id(*ch.last().unwrap()) { last[x] += f; }
        for (tail, p) in [(Some(','), P_COMMA), (Some('.'), P_PERIOD), (Some(';'), P_SEMI), (None, 1.0 - P_COMMA - P_PERIOD - P_SEMI)] {
            let g = f * p;
            let mut seq: Vec<Option<usize>> = Vec::with_capacity(ch.len() + 1);
            let mut prev_rep = false;
            for i in 0..ch.len() {
                let c = ch[i];
                let double = mode != RepMode::None && i > 0 && c == ch[i - 1] && c.is_alphabetic() && !prev_rep;
                match (double, item_id(c)) {
                    (true, Some(x)) => {
                        // the key after the repeat press: next letter, trailing punctuation, or space (right thumb)
                        // None = space, which is its own key on the right thumb
                        let next = if i + 1 < ch.len() { item_id(ch[i + 1]).map(|y| Some(pos[y])) } else { Some(tail.map(|t| pos[item_id(t).unwrap()])) };
                        let clash = |r: usize| {
                            sfb(pos[x], pos[r])
                                + match next {
                                    Some(Some(n)) => sfb(pos[r], n),
                                    Some(None) => (k[pos[r]].finger == k[RTHUMB].finger) as u8,
                                    None => 0,
                                }
                        };
                        let r = match mode {
                            RepMode::Left => REP,
                            RepMode::Right => REP_R,
                            _ => match clash(REP).cmp(&clash(REP_R)) {
                                std::cmp::Ordering::Less => REP,
                                std::cmp::Ordering::Greater => REP_R,
                                _ => if k[pos[x]].hand == 0 { REP_R } else { REP },
                            },
                        };
                        if pos[r] == RTHUMB && i + 1 == ch.len() && tail.is_none() {
                            rep_space += g;
                        }
                        seq.push(Some(r));
                        prev_rep = true;
                    }
                    (_, id) => {
                        seq.push(id);
                        prev_rep = false;
                    }
                }
            }
            if let Some(t) = tail {
                seq.push(item_id(t));
            }
            if !rules.is_empty() {
                // the text as typed (before substitutions) decides which rule fires
                let text: Vec<Option<usize>> = ch.iter().map(|&c| item_id(c)).chain(tail.map(|t| item_id(t))).collect();
                for i in 1..seq.len() {
                    let (Some(p), Some(c)) = (text[i - 1], text[i]) else { continue };
                    for r in rules {
                        if r.prev == p && r.out == c && seq[i] == Some(c) {
                            seq[i] = Some(r.via);
                        } else if r.prev == p && r.via == c && r.out != c {
                            conflict += g;
                        }
                    }
                }
            }
            for x in seq.iter().flatten() {
                mono[*x] += g;
            }
            if let Some(Some(e)) = seq.last() { end_key[*e] += g; }
            if let Some(Some(b)) = seq.first() { start_key[*b] += g; }
            for w2 in seq.windows(2) {
                if let [Some(a), Some(b)] = w2 { *bi.entry([*a, *b]).or_default() += g; }
            }
            for w3 in seq.windows(3) {
                if let (Some(a), Some(c)) = (w3[0], w3[2]) { *skip.entry([a, c]).or_default() += g; }
                if let [Some(a), Some(b), Some(c)] = w3 { *tri.entry([*a, *b, *c]).or_default() += g; }
            }
        }
    }
    // hyphen / slash joins between words, from first/last-letter distributions
    for (sym, p) in JOINS {
        let s = item_id(sym).unwrap();
        mono[s] += words_tot * p;
        for a in 0..NITEM {
            if last[a] > 0.0 { *bi.entry([a, s]).or_default() += p * last[a]; }
            if first[a] > 0.0 { *bi.entry([s, a]).or_default() += p * first[a]; }
            for b in 0..NITEM {
                if last[a] > 0.0 && first[b] > 0.0 {
                    let g = p * last[a] * first[b] / words_tot;
                    *tri.entry([a, s, b]).or_default() += g;
                    *skip.entry([a, b]).or_default() += g;
                }
            }
        }
    }
    // a b across a space is a skipgram (the space press sits between them); words assumed independent
    if std::env::var("NOSPACE").is_err() {
        for a in 0..NITEM {
            for b in 0..NITEM {
                if end_key[a] > 0.0 && start_key[b] > 0.0 {
                    *skip.entry([a, b]).or_default() += end_key[a] * start_key[b] / words_tot;
                }
            }
        }
    }
    let pct = |v: f64, t: f64| v / t * 100.0;
    let mono_t: f64 = mono.iter().sum();
    let bi_t: f64 = bi.values().sum();
    let skip_t: f64 = skip.values().sum();
    let mut rep = vec![0.0; NITEM];
    for (g, f) in &bi {
        if g[0] == g[1] { rep[g[0]] += pct(*f, bi_t); }
    }
    let mut tri: Vec<_> = tri.into_iter().map(|(g, f)| (g[0], g[1], g[2], f)).collect();
    tri.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    tri.truncate(tri_cap);
    let tri_t: f64 = tri.iter().map(|t| t.3).sum();
    Corpus {
        rep_space: pct(rep_space, bi_t),
        rep_space_exact: true,
        ad_conflict: pct(conflict, bi_t),
        rep,
        mono: mono.iter().enumerate().filter(|(_, f)| **f > 0.0).map(|(i, f)| (i, pct(*f, mono_t))).collect(),
        bi: bi.iter().filter(|(g, _)| g[0] != g[1]).map(|(g, f)| (g[0], g[1], pct(*f, bi_t))).collect(),
        skip: skip.iter().filter(|(g, _)| g[0] != g[1]).map(|(g, f)| (g[0], g[1], pct(*f, skip_t))).collect(),
        tri: tri.into_iter().map(|(a, b, c, f)| (a, b, c, pct(f, tri_t))).collect(),
    }
}

// Metrics where `m` is worse than `b` by more than `tol` (percentage points; effort and ratios scaled).
fn regressions(m: &Metrics, b: &Metrics, tol: f64) -> Vec<String> {
    let pk = |x: &Metrics| x.fingers[0] + x.fingers[7];
    let lower = [
        ("SFB", m.sfb, b.sfb), ("SFS", m.sfs, b.sfs), ("LSB", m.lsb, b.lsb), ("FSB", m.fsb, b.fsb),
        ("HSB", m.hsb, b.hsb), ("pinky scissors", m.psc, b.psc), ("ring-pinky", m.ring_pinky, b.ring_pinky),
        ("redirects", m.redirect + m.weak_red, b.redirect + b.weak_red), ("weak redirects", m.weak_red, b.weak_red),
        ("effort x100", m.effort * 100.0, b.effort * 100.0), ("pinky load", pk(m), pk(b)),
        ("pinky off-home", m.pinky_off, b.pinky_off), ("same-finger dist", m.sf_dist, b.sf_dist),
        ("thumb doubles", m.thumb_rep, b.thumb_rep),
    ];
    let higher = [
        ("alternation", m.alt, b.alt), ("rolls in", m.roll_in, b.roll_in),
        ("in/out x10", 10.0 * m.roll_in / m.roll_out, 10.0 * b.roll_in / b.roll_out), ("home row", m.rows[1], b.rows[1]),
    ];
    let mut out = vec![];
    for (n, v, base) in lower {
        if v > base + tol { out.push(format!("{n} {v:.2} vs {base:.2}")); }
    }
    for (n, v, base) in higher {
        if v < base - tol { out.push(format!("{n} {v:.2} vs {base:.2}")); }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::env::var("CORPUS").unwrap_or(".".into());
    let md = Model::new(corpus_from_words(&dir, &parse("vwgmj;.'/=snthk,aeicfpdlx-uoyb r"), &keys(), RepMode::None, 4000));
    let local = std::env::var("LOCAL").is_ok();
    match args.get(1).map(|s| s.as_str()) {
        Some("analyze") => {
            for l in &args[2..] {
                let pos = parse(l);
                let m = md.metrics(&pos, true);
                print!("{}", show(&pos));
                report(&m);
                println!(
                    "  cost(eng) {:.3}  cost(vim .5) {:.3}\n",
                    cost(&m, &Obj::new(0.0, None, 0.0)),
                    cost(&m, &Obj::new(0.5, None, 0.0))
                );
            }
        }
        Some("top") => {
            let pos = parse(&args[2]);
            let ch: Vec<char> = ITEMS.chars().collect();
            for (name, flag) in [("SFB", F_SFB), ("LSB", F_LSB), ("FSB", F_FSB), ("HSB", F_HSB), ("PinkyScissor", F_PSC), ("RingPinky", F_RP)] {
                let mut v: Vec<_> = md.c.bi.iter().filter(|(a, b, _)| md.bflag[pos[*a]][pos[*b]] & flag != 0).collect();
                v.sort_by(|x, y| y.2.partial_cmp(&x.2).unwrap());
                let s: Vec<String> = v.iter().take(12).map(|(a, b, f)| format!("{}{} {:.3}", ch[*a], ch[*b], f)).collect();
                println!("{name}: {}", s.join(", "));
            }
        }
        Some("eval") => {
            // eval <none|left|right|both> <layout>...  exact repeat-key modelling, compared with PM (no repeat key)
            let mode = match args[2].as_str() {
                "left" => RepMode::Left,
                "right" => RepMode::Right,
                "both" => RepMode::Both,
                _ => RepMode::None,
            };
            let k = keys();
            let pm = parse("vwgmj;.'/=snthk,aeicfpdlx-uoyb r");
            let base = Model::new(corpus_from_words(&dir, &pm, &k, RepMode::None, 4000)).metrics(&pm, false);
            for l in &args[3..] {
                let pos = parse(l);
                let em = Model::new(corpus_from_words(&dir, &pos, &k, mode, 4000));
                let m = em.metrics(&pos, true);
                print!("{}", show(&pos));
                report(&m);
                let r = regressions(&m, &base, 0.02);
                println!("  vs PM ({} worse beyond 0.02): {}\n", r.len(), if r.is_empty() { "none - dominates PM".into() } else { r.join(", ") });
            }
        }
        Some("adapt") => {
            // adapt <layout> <max rules>: greedy adaptive-key search, scored exactly (weighted cost, no PM rule)
            let pos = parse(&args[2]);
            let max: usize = args[3].parse().unwrap();
            let k = keys();
            let o = Obj::new(0.0, None, 0.0);
            let ch: Vec<char> = ITEMS.chars().collect();
            let score = |rules: &[Adaptive]| {
                let em = Model::new(corpus_with_adaptives(&dir, &pos, &k, RepMode::None, 4000, rules));
                let m = em.metrics(&pos, false);
                (cost(&m, &o), m)
            };
            let mut rules: Vec<Adaptive> = vec![];
            let (mut cur, m0) = score(&rules);
            print!("{}", show(&pos));
            println!("no adaptives: cost {cur:.3}");
            report(&m0);
            for step in 0..max {
                // the worst bigrams under the current rules
                let em = Model::new(corpus_with_adaptives(&dir, &pos, &k, RepMode::None, 4000, &rules));
                let mut bad: Vec<(f64, usize, usize)> = em.c.bi.iter().filter_map(|&(a, b, f)| {
                    let fl = em.bflag[pos[a]][pos[b]];
                    let w = [(F_SFB, 12.0), (F_FSB, 6.0), (F_HSB, 3.0), (F_LSB, 5.0), (F_PSC, 6.0), (F_RP, 3.0)]
                        .iter().filter(|(x, _)| fl & x != 0).map(|(_, w)| w).sum::<f64>();
                    (w > 0.0 && a < NSLOT && b < NSLOT).then_some((f * w, a, b))
                }).collect();
                bad.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
                bad.truncate(20);
                let cands: Vec<Adaptive> = bad.iter().flat_map(|&(_, a, b)| {
                    let k = &k;
                    let pos = &pos;
                    (0..NSLOT).filter(move |&z| z != a && z != b && k[pos[z]].finger != k[pos[a]].finger)
                        .map(move |z| Adaptive { prev: a, out: b, via: z })
                }).filter(|r| !rules.iter().any(|x| x.prev == r.prev && (x.out == r.out || x.via == r.via))).collect();
                let best = cands.par_iter().map(|r| {
                    let mut rs = rules.clone();
                    rs.push(*r);
                    (score(&rs).0, *r)
                }).min_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                match best {
                    Some((c, r)) if c < cur - 0.05 => {
                        println!("rule {}: after {} press {} -> types {}   cost {:.3} -> {:.3}",
                            step + 1, ch[r.prev], ch[r.via], ch[r.out], cur, c);
                        cur = c;
                        rules.push(r);
                    }
                    _ => { println!("no further improving rule"); break; }
                }
            }
            let (c, m) = score(&rules);
            println!("with {} adaptives: cost {c:.3}", rules.len());
            report(&m);
        }
        Some("check") => {
            // incremental vs full scoring on random swaps
            let pos0 = parse(&args[2]);
            let mut pos = pos0;
            let mut m = md.metrics(&pos, true);
            let mut rng = Rng(12345);
            let o = Obj::new(1.5, Some(md.metrics(&pos0, true)), 0.05);
            let mut worst = 0.0f64;
            for _ in 0..100_000 {
                let (a, b) = (rng.n(NSLOT), rng.n(NSLOT));
                if !legal(&pos, a, b) {
                    continue;
                }
                md.swap(&mut m, &mut pos, a, b, true);
                worst = worst.max((cost(&m, &o) - cost(&md.metrics(&pos, true), &o)).abs());
            }
            println!("max |incremental - full| cost over 100k swaps: {worst:.2e}");
        }
        Some("opt") => {
            // opt <vim_w> <iters> <restarts> <base-layout|-> <slack> [fixed chars] [seed-layout]
            let vim_w: f64 = args[2].parse().unwrap();
            let iters: usize = args[3].parse().unwrap();
            let restarts: usize = args[4].parse().unwrap();
            let base = if args[5] == "-" { None } else { Some(md.metrics(&parse(&args[5]), true)) };
            let slack: f64 = args[6].parse().unwrap();
            let fixed: Vec<usize> = args.get(7).map(|s| s.chars().map(|c| item_id(c).unwrap()).collect()).unwrap_or_default();
            // start layout: explicit seed, else the baseline, else Promethium
            let start = match (args.get(8), args[5].as_str()) {
                (Some(seed), _) => parse(seed),
                (None, "-") => parse("vwgmj;.'/=snthk,aeicpfdlx-uoyb r"),
                (None, b) => parse(b),
            };
            let o = Obj::new(vim_w, base, slack);
            // TOPN=n lists up to n distinct results without detail (for post-hoc filtering)
            let (detail, max) = match std::env::var("TOPN").ok().and_then(|v| v.parse().ok()) {
                Some(n) => (0, n),
                None => (3, 6),
            };
            print_results(&md, multi_anneal(&md, &o, start, iters, restarts, &fixed, local), detail, max);
        }
        Some("sweep") => {
            // sweep <vim_w> <iters> <restarts> <base-layout> <slack> [seed-layout]
            // Pin every j/k + h/l pair placement, optimize the rest around it (in parallel across placements).
            let vim_w: f64 = args[2].parse().unwrap();
            let iters: usize = args[3].parse().unwrap();
            let restarts: usize = args[4].parse().unwrap();
            let base_pos = parse(&args[5]);
            let slack: f64 = args[6].parse().unwrap();
            let seed = args.get(7).map(|s| parse(s)).unwrap_or(base_pos);
            let o = Obj::new(vim_w, Some(md.metrics(&base_pos, true)), slack);
            let cols = [1usize, 2, 3, 5, 6, 7]; // pair start columns that keep both keys off the pinkies
            // j/k side by side on two different adjacent fingers (ring+middle or middle+index) so they can be spammed
            let jk: Vec<(usize, usize)> = (0..3).flat_map(|r| [1usize, 2, 6, 7].map(|c| (r * 10 + c, r * 10 + c + 1))).collect();
            let mut hl: Vec<(usize, usize)> = (0..3).flat_map(|r| cols.map(|c| (r * 10 + c, r * 10 + c + 1))).collect();
            // stacked h/l in either order (Engram puts l above h on the index)
            for r in 1..3 {
                for c in 1..9 {
                    hl.push((r * 10 + c, (r - 1) * 10 + c));
                    hl.push(((r - 1) * 10 + c, r * 10 + c));
                }
            }
            let (h, j, k, l) = (item_id('h').unwrap(), item_id('j').unwrap(), item_id('k').unwrap(), item_id('l').unwrap());
            let fixed = [h, j, k, l];
            let mut starts = vec![];
            for &(sj, sk) in &jk {
                for &(sh, sl) in &hl {
                    if [sj, sk, sh, sl].iter().collect::<std::collections::HashSet<_>>().len() < 4 {
                        continue;
                    }
                    let mut p = seed;
                    for (x, s) in [(h, sh), (j, sj), (k, sk), (l, sl)] {
                        place(&mut p, x, s);
                    }
                    if !(0..NITEM).all(|i| p[i] != THUMB || is_letter(i)) {
                        continue;
                    }
                    starts.push(p);
                }
            }
            eprintln!("{} placements x {} restarts x {} iters", starts.len(), restarts, iters);
            let mut results: Vec<_> = starts
                .par_iter()
                .flat_map_iter(|&p| (0..restarts).map(move |i| (p, i)))
                .map(|(p, i)| anneal(&md, &o, p, iters, seed_for(i * 7919 + p[h] * 31 + p[j]), &fixed, local))
                .collect();
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            print_results(&md, results, 5, 25);
        }
        _ => eprintln!("usage: kb analyze <layout>... | top <layout> | check <layout> | opt <vim_w> <iters> <restarts> <base|-> <slack> [fixed] [seed] | sweep <vim_w> <iters> <restarts> <base> <slack> [seed]"),
    }
}
