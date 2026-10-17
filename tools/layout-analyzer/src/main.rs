// Layout analyzer + simulated-annealing optimizer for a 3x10 + 1 thumb-letter Corne layout.
// Metrics mirror cyanophage's: SFB, SFS, LSB, scissors, alternation, rolls, redirects.
use std::collections::HashMap;
use std::fs;
use std::sync::Arc;
use std::thread;

const NSLOT: usize = 31; // 30 grid keys + left thumb
const THUMB: usize = 30;
const ITEMS: &str = "abcdefghijklmnoprstuvwxy,.';-/=";

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

fn keys() -> [Key; NSLOT] {
    // Effort for a column-staggered Corne, left hand pinky..inner; mirrored for right.
    let eff = [
        [3.0, 2.4, 2.0, 2.2, 3.0],
        [1.6, 1.3, 1.1, 1.0, 2.4],
        [3.2, 2.6, 2.3, 1.6, 3.0],
    ];
    let mut k = [Key { hand: 0, finger: 0, rank: 0, row: 0, col: 0, effort: 0.0 }; NSLOT];
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
    k
}

const F_SFB: u8 = 1;
const F_LSB: u8 = 2;
const F_FSB: u8 = 4;
const F_HSB: u8 = 8;
const F_PSC: u8 = 16; // any scissor that involves a pinky

fn bigram_flags(k: &[Key; NSLOT], a: usize, b: usize) -> u8 {
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
    if x.finger == 8 || y.finger == 8 {
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

fn tri_class(k: &[Key; NSLOT], a: usize, b: usize, c: usize) -> Tri {
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
        if x.rank != 3 && y.rank != 3 && z.rank != 3 && x.finger != 8 && y.finger != 8 && z.finger != 8 {
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
    rep: Vec<f64>, // same-letter double frequency (% of bigrams) per item
    mono: Vec<(usize, f64)>,
    bi: Vec<(usize, usize, f64)>,
    skip: Vec<(usize, usize, f64)>,
    tri: Vec<(usize, usize, usize, f64)>,
}

fn item_id(c: char) -> Option<usize> {
    ITEMS.chars().position(|x| x == c)
}

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
    let mut rep = vec![0.0; NSLOT];
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
    Corpus { rep, mono, bi, skip, tri }
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
#[derive(Default, Clone, Debug)]
struct Metrics {
    sfb: f64,
    sfs: f64,
    lsb: f64,
    fsb: f64,
    hsb: f64,
    psc: f64,
    alt: f64,
    roll_in: f64,
    roll_out: f64,
    one_in: f64,
    one_out: f64,
    redirect: f64,
    weak_red: f64,
    effort: f64,
    fingers: [f64; 9],
    rows: [f64; 4], // top, home, bottom, thumb
    pinky_off: f64,
    thumb_rep: f64, // double taps of the thumb letter (% of bigrams)
    sf_dist: f64, // same-finger travel (key units) per 100 bigrams, SFB + 0.5*SFS
    vim_effort: f64,
    vim_sfb: f64,
    vim_scis: f64,
    hjkl: f64,
}

struct Model {
    k: [Key; NSLOT],
    bflag: [[u8; NSLOT]; NSLOT],
    dist: [[f64; NSLOT]; NSLOT],
    tcls: Vec<Tri>,
    c: Corpus,
    vim: Vec<VimSeq>,
}

impl Model {
    fn new(c: Corpus) -> Self {
        let k = keys();
        let mut bflag = [[0u8; NSLOT]; NSLOT];
        let mut dist = [[0.0f64; NSLOT]; NSLOT];
        let mut tcls = vec![Tri::Other; NSLOT * NSLOT * NSLOT];
        for a in 0..NSLOT {
            for b in 0..NSLOT {
                bflag[a][b] = bigram_flags(&k, a, b);
                dist[a][b] = (((k[a].row - k[b].row) as f64).powi(2) + ((k[a].col - k[b].col) as f64).powi(2)).sqrt();
                for d in 0..NSLOT {
                    tcls[(a * NSLOT + b) * NSLOT + d] = tri_class(&k, a, b, d);
                }
            }
        }
        Model { k, bflag, dist, tcls, c, vim: vim_seqs() }
    }

    fn metrics(&self, pos: &[usize; NSLOT], with_vim: bool) -> Metrics {
        let mut m = Metrics::default();
        for &(c, f) in &self.c.mono {
            let s = pos[c];
            m.effort += f * self.k[s].effort / 100.0;
            m.fingers[self.k[s].finger as usize] += f;
            m.rows[self.k[s].row as usize] += f;
            if self.k[s].rank == 0 && self.k[s].row != 1 {
                m.pinky_off += f;
            }
        }
        for &(a, b, f) in &self.c.bi {
            let fl = self.bflag[pos[a]][pos[b]];
            if fl == 0 {
                continue;
            }
            if fl & F_SFB != 0 {
                m.sfb += f;
                m.sf_dist += f * self.dist[pos[a]][pos[b]];
            }
            if fl & F_LSB != 0 { m.lsb += f; }
            if fl & F_FSB != 0 { m.fsb += f; }
            if fl & F_HSB != 0 { m.hsb += f; }
            if fl & F_PSC != 0 { m.psc += f; }
        }
        for &(a, b, f) in &self.c.skip {
            if self.bflag[pos[a]][pos[b]] & F_SFB != 0 {
                m.sfs += f;
                m.sf_dist += 0.5 * f * self.dist[pos[a]][pos[b]];
            }
        }
        for (i, &r) in self.c.rep.iter().enumerate() {
            if pos[i] == THUMB {
                m.thumb_rep = r;
            }
        }
        let mut acc = [0.0f64; 8];
        for &(a, b, c, f) in &self.c.tri {
            acc[self.tcls[(pos[a] * NSLOT + pos[b]) * NSLOT + pos[c]] as usize] += f;
        }
        m.alt = acc[Tri::Alt as usize];
        m.roll_in = acc[Tri::RollIn as usize];
        m.roll_out = acc[Tri::RollOut as usize];
        m.one_in = acc[Tri::OneIn as usize];
        m.one_out = acc[Tri::OneOut as usize];
        m.redirect = acc[Tri::Redirect as usize];
        m.weak_red = acc[Tri::WeakRedirect as usize];
        if with_vim {
            let (mut keys, mut pairs) = (0.0, 0.0);
            for v in &self.vim {
                for (i, &c) in v.keys.iter().enumerate() {
                    m.vim_effort += v.w * self.k[pos[c]].effort;
                    keys += v.w;
                    if i > 0 {
                        let fl = self.bflag[pos[v.keys[i - 1]]][pos[c]];
                        pairs += v.w;
                        if fl & F_SFB != 0 { m.vim_sfb += v.w; }
                        if fl & (F_FSB | F_HSB | F_LSB) != 0 { m.vim_scis += v.w; }
                    }
                }
            }
            m.vim_effort /= keys;
            m.vim_sfb *= 100.0 / pairs;
            m.vim_scis *= 100.0 / pairs;
            m.hjkl = self.hjkl_penalty(pos);
        }
        m
    }

    // Pairs, Dvorak-style: j/k adjacent (j left of k on one row, or k stacked above j), and
    // h/l adjacent with h left of l on one row (no stacking). Neither pair on the thumb; pairs may live anywhere.
    fn hjkl_penalty(&self, pos: &[usize; NSLOT]) -> f64 {
        let id = |c| self.k[pos[item_id(c).unwrap()]];
        let (h, j, k, l) = (id('h'), id('j'), id('k'), id('l'));
        if [h, j, k, l].iter().any(|q| q.finger == 8) {
            return 10.0;
        }
        let pair = |a: Key, b: Key, vertical_ok: bool| {
            let horiz = ((b.row - a.row).abs() + (b.col - a.col - 1).abs()) as f64
                + if a.hand != b.hand { 2.0 } else { 0.0 };
            let vert = ((b.col - a.col).abs() + (a.row - b.row - 1).abs()) as f64;
            if vertical_ok { horiz.min(vert) } else { horiz }
        };
        pair(j, k, true) + pair(h, l, false)
    }
}

// ---------- objective ----------
#[derive(Clone)]
struct Obj {
    vim_w: f64,
    base: Option<Metrics>, // hinge: never worse than this on the headline metrics
    slack: f64,
}

fn cost(m: &Metrics, o: &Obj) -> f64 {
    let pinky = m.fingers[0] + m.fingers[7];
    let mut c = 12.0 * m.sfb + 3.0 * m.sfs + 5.0 * m.lsb + 6.0 * m.fsb + 3.0 * m.hsb + 6.0 * m.psc
        + 2.0 * m.redirect + 5.0 * m.weak_red
        - 0.4 * m.alt - 0.15 * m.roll_in - 0.05 * m.roll_out
        + 4.0 * m.effort
        + 6.0 * m.thumb_rep
        + 1.5 * pinky + 2.0 * m.pinky_off;
    for f in m.fingers.iter().take(8) {
        c += 1.0 * (f - 16.0).max(0.0);
    }
    if let Some(b) = &o.base {
        let s = o.slack;
        let hw: f64 = std::env::var("HINGE").ok().and_then(|v| v.parse().ok()).unwrap_or(100.0);
        let h = |x: f64| hw * x.max(0.0);
        c += h(m.sfb - b.sfb - s) + h(m.sfs - b.sfs - s) + h(m.lsb - b.lsb - s)
            + h((m.fsb + m.hsb) - (b.fsb + b.hsb) - s) + h(m.fsb - b.fsb - s) + h(m.psc - b.psc - s) + h(b.alt - m.alt - 4.0 * s)
            + h(m.redirect + m.weak_red - b.redirect - b.weak_red - 4.0 * s)
            + h(10.0 * (m.effort - b.effort) - s)
            + h(b.rows[1] - m.rows[1] - 4.0 * s)
            + h(m.pinky_off - b.pinky_off - s)
            + h(m.sf_dist - b.sf_dist - s)
            + h(m.roll_out / m.roll_in.max(1e-9) - b.roll_out / b.roll_in.max(1e-9) - s)
            + h(pinky - (b.fingers[0] + b.fingers[7]) - 4.0 * s);
    }
    if o.vim_w > 0.0 {
        c += o.vim_w * (2.0 * m.vim_effort + 0.3 * m.vim_sfb + 0.1 * m.vim_scis) + 30.0 * m.hjkl;
    }
    c
}

// ---------- layout I/O ----------
fn parse(s: &str) -> [usize; NSLOT] {
    let s: Vec<char> = s.chars().filter(|c| !c.is_whitespace()).collect();
    assert_eq!(s.len(), NSLOT, "need 30 grid chars + thumb");
    let mut pos = [usize::MAX; NSLOT];
    for (slot, &c) in s.iter().enumerate() {
        pos[item_id(c).unwrap_or_else(|| panic!("bad char {c}"))] = slot;
    }
    assert!(pos.iter().all(|&p| p != usize::MAX), "missing chars");
    pos
}

fn show(pos: &[usize; NSLOT]) -> String {
    let mut g = [' '; NSLOT];
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

fn compact(pos: &[usize; NSLOT]) -> String {
    let mut g = [' '; NSLOT];
    for (i, c) in ITEMS.chars().enumerate() {
        g[pos[i]] = c;
    }
    let s: String = g.iter().collect();
    format!("{} {} {} {}", &s[0..10], &s[10..20], &s[20..30], &s[30..])
}

fn report(m: &Metrics) {
    println!(
        "  SFB {:.2}%  SFS {:.2}%  LSB {:.2}%  FSB {:.2}%  HSB {:.2}%  (scissors {:.2}%, pinky scissors {:.2}%)",
        m.sfb, m.sfs, m.lsb, m.fsb, m.hsb, m.fsb + m.hsb, m.psc
    );
    println!(
        "  Alt {:.1}%  RollIn {:.1}%  RollOut {:.1}%  1hIn {:.1}%  1hOut {:.1}%  Redir {:.2}%  WeakRedir {:.2}%",
        m.alt, m.roll_in, m.roll_out, m.one_in, m.one_out, m.redirect, m.weak_red
    );
    let f = m.fingers;
    println!(
        "  Effort {:.3}  Fingers LP {:.1} LR {:.1} LM {:.1} LI {:.1} | LT {:.1} | RI {:.1} RM {:.1} RR {:.1} RP {:.1}  (L {:.1} / R {:.1})",
        m.effort, f[0], f[1], f[2], f[3], f[8], f[4], f[5], f[6], f[7],
        f[0] + f[1] + f[2] + f[3] + f[8], f[4] + f[5] + f[6] + f[7]
    );
    println!(
        "  Rows top {:.1}% home {:.1}% bottom {:.1}% thumb {:.1}%  PinkyOffHome {:.2}%  SameFingerDist {:.3}  In/Out {:.2}  ThumbDoubles {:.2}%",
        m.rows[0], m.rows[1], m.rows[2], m.rows[3], m.pinky_off, m.sf_dist, m.roll_in / m.roll_out, m.thumb_rep
    );
    println!(
        "  Vim: effort/key {:.3}  SFB {:.2}%  scissor+LSB {:.2}%  hjkl-geometry {:.0}",
        m.vim_effort, m.vim_sfb, m.vim_scis, m.hjkl
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

fn anneal(md: &Model, o: &Obj, start: [usize; NSLOT], iters: usize, seed: u64, fixed: &[usize]) -> ([usize; NSLOT], f64) {
    let mut rng = Rng(seed | 1);
    let vim = o.vim_w > 0.0;
    let mut pos = start;
    let free: Vec<usize> = (0..NSLOT).filter(|i| !fixed.contains(i)).collect();
    // shuffle free items among their slots
    for _ in 0..200 {
        let (a, b) = (free[rng.n(free.len())], free[rng.n(free.len())]);
        let (sa, sb) = (pos[a], pos[b]);
        if (sa == THUMB && !is_letter(b)) || (sb == THUMB && !is_letter(a)) {
            continue;
        }
        pos.swap(a, b);
    }
    let mut cur = cost(&md.metrics(&pos, vim), o);
    let (mut best, mut best_c) = (pos, cur);
    let (t0, t1) = (8.0f64, 0.002f64);
    for it in 0..iters {
        let t = t0 * (t1 / t0).powf(it as f64 / iters as f64);
        let (a, b) = (free[rng.n(free.len())], free[rng.n(free.len())]);
        if a == b {
            continue;
        }
        let (sa, sb) = (pos[a], pos[b]);
        if (sa == THUMB && !is_letter(b)) || (sb == THUMB && !is_letter(a)) {
            continue;
        }
        pos.swap(a, b);
        let c = cost(&md.metrics(&pos, vim), o);
        if c <= cur || rng.f() < ((cur - c) / t).exp() {
            cur = c;
            if c < best_c {
                best_c = c;
                best = pos;
            }
        } else {
            pos.swap(a, b);
        }
    }
    // steepest-descent polish over all legal swaps
    let mut pos = best;
    loop {
        let mut improved = false;
        for i in 0..free.len() {
            for j in i + 1..free.len() {
                let (a, b) = (free[i], free[j]);
                let (sa, sb) = (pos[a], pos[b]);
                if (sa == THUMB && !is_letter(b)) || (sb == THUMB && !is_letter(a)) {
                    continue;
                }
                pos.swap(a, b);
                let c = cost(&md.metrics(&pos, vim), o);
                if c < best_c - 1e-9 {
                    best_c = c;
                    improved = true;
                } else {
                    pos.swap(a, b);
                }
            }
        }
        if !improved {
            break;
        }
    }
    (pos, best_c)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::env::var("CORPUS").unwrap_or(".".into());
    let md = Arc::new(Model::new(load(&dir, 4000)));
    match args.get(1).map(|s| s.as_str()) {
        Some("analyze") => {
            for l in &args[2..] {
                let pos = parse(l);
                let m = md.metrics(&pos, true);
                print!("{}", show(&pos));
                report(&m);
                println!(
                    "  cost(eng) {:.3}  cost(vim .5) {:.3}\n",
                    cost(&m, &Obj { vim_w: 0.0, base: None, slack: 0.0 }),
                    cost(&m, &Obj { vim_w: 0.5, base: None, slack: 0.0 })
                );
            }
        }
        Some("top") => {
            let pos = parse(&args[2]);
            let ch: Vec<char> = ITEMS.chars().collect();
            for (name, flag) in [("SFB", F_SFB), ("LSB", F_LSB), ("FSB", F_FSB), ("HSB", F_HSB), ("PinkyScissor", F_PSC)] {
                let mut v: Vec<_> = md.c.bi.iter().filter(|(a, b, _)| md.bflag[pos[*a]][pos[*b]] & flag != 0).collect();
                v.sort_by(|x, y| y.2.partial_cmp(&x.2).unwrap());
                let s: Vec<String> = v.iter().take(12).map(|(a, b, f)| format!("{}{} {:.3}", ch[*a], ch[*b], f)).collect();
                println!("{name}: {}", s.join(", "));
            }
        }
        Some("opt") => {
            // opt <vim_w> <iters> <restarts> <base-layout|-> <slack> [fixed chars]
            let vim_w: f64 = args[2].parse().unwrap();
            let iters: usize = args[3].parse().unwrap();
            let restarts: usize = args[4].parse().unwrap();
            let base = if args[5] == "-" { None } else { Some(md.metrics(&parse(&args[5]), true)) };
            let slack: f64 = args[6].parse().unwrap();
            let start = if args[5] == "-" { parse("vwgmj;.'/=snthk,aeicpfdlx-uoyb r") } else { parse(&args[5]) };
            let fixed: Vec<usize> = args.get(7).map(|s| s.chars().map(|c| item_id(c).unwrap()).collect()).unwrap_or_default();
            let o = Obj { vim_w, base, slack };
            let threads = 4;
            let mut results = vec![];
            for round in 0..restarts.div_ceil(threads) {
                let hs: Vec<_> = (0..threads)
                    .map(|t| {
                        let (md, o, fixed) = (md.clone(), o.clone(), fixed.clone());
                        let seed = 0x9E3779B97F4A7C15u64.wrapping_mul((round * threads + t + 1) as u64 + std::process::id() as u64);
                        thread::spawn(move || anneal(&md, &o, start, iters, seed, &fixed))
                    })
                    .collect();
                for h in hs {
                    results.push(h.join().unwrap());
                }
            }
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let mut seen = HashMap::new();
            for (pos, c) in results {
                let key = compact(&pos);
                if seen.insert(key.clone(), ()).is_some() {
                    continue;
                }
                println!("cost {:.3}   {}", c, key);
                if seen.len() <= 3 {
                    print!("{}", show(&pos));
                    report(&md.metrics(&pos, true));
                }
                if seen.len() >= 6 {
                    break;
                }
            }
        }
        _ => eprintln!("usage: kb analyze <layout>... | kb opt <vim_w> <iters> <restarts> <base|-> <slack> [fixed]"),
    }
}
