// ==========================================================
// Quantum Emulator (single-file MVP++)
// MPS + SVD + truncation + canonicalization + OND-RNG
// Rust 1.75+
//
// Dependencies (Cargo.toml):
// num-complex = "0.4"
// nalgebra = "0.32"
// sha3 = "0.10"
// ==========================================================

use num_complex::Complex64;
use nalgebra::{DMatrix, SVD};
use sha3::{Shake256, digest::{Update, ExtendableOutput, XofReader}};
use std::time::Instant;

type C64 = Complex64;

/* =========================
   OND-RNG
   ========================= */

struct ONDRng {
    state: [u8; 32],
    step: u64,
}

impl ONDRng {
    fn new(seed: &[u8]) -> Self {
        let mut state = [0u8; 32];
        shake(&[seed, b"OND_INIT"], &mut state);
        Self { state, step: 0 }
    }

    fn next_f64(&mut self, ctx: &[u8]) -> f64 {
        self.step += 1;
        shake(&[&self.state, &self.step.to_be_bytes()], &mut self.state);
        let mut out = [0u8; 8];
        shake(&[&self.state, ctx], &mut out);
        if self.state[0] < 16 {
            shake(&[&self.state, b"SKIP"], &mut self.state);
        }
        (u64::from_be_bytes(out) as f64) / (u64::MAX as f64)
    }
}

fn shake(parts: &[&[u8]], out: &mut [u8]) {
    let mut h = Shake256::default();
    for p in parts {
        h.update(p);
    }
    let mut r = h.finalize_xof();
    r.read(out);
}

/* =========================
   Tensor3
   ========================= */

#[derive(Clone)]
struct Tensor3 {
    data: Vec<C64>,
    dl: usize,
    dp: usize,
    dr: usize,
}

impl Tensor3 {
    fn zeros(dl: usize, dp: usize, dr: usize) -> Self {
        Self {
            data: vec![C64::new(0.0, 0.0); dl * dp * dr],
            dl, dp, dr,
        }
    }

    fn idx(&self, l: usize, p: usize, r: usize) -> usize {
        (l * self.dp + p) * self.dr + r
    }

    fn get(&self, l: usize, p: usize, r: usize) -> C64 {
        self.data[self.idx(l, p, r)]
    }

    fn set(&mut self, l: usize, p: usize, r: usize, v: C64) {
        let i = self.idx(l, p, r);
        self.data[i] = v;
    }
}

/* =========================
   MPS
   ========================= */

struct Truncation {
    max_bond: usize,
    cutoff: f64,
}

struct MPS {
    sites: Vec<Tensor3>,
}

impl MPS {
    fn new_zero(n: usize) -> Self {
        let mut sites = Vec::with_capacity(n);
        for _ in 0..n {
            let mut t = Tensor3::zeros(1, 2, 1);
            t.set(0, 0, 0, C64::new(1.0, 0.0));
            sites.push(t);
        }
        Self { sites }
    }

    /* -------- 1-qubit gate -------- */

    fn apply_1q(&mut self, k: usize, u: [[C64; 2]; 2]) {
        let s = &self.sites[k];
        let mut out = Tensor3::zeros(s.dl, s.dp, s.dr);

        for l in 0..s.dl {
            for r in 0..s.dr {
                for p in 0..2 {
                    let mut acc = C64::new(0.0, 0.0);
                    for pp in 0..2 {
                        acc += u[p][pp] * s.get(l, pp, r);
                    }
                    out.set(l, p, r, acc);
                }
            }
        }
        self.sites[k] = out;
    }

    /* -------- 2-qubit gate via SVD -------- */

    fn apply_2q_svd(
        &mut self,
        k: usize,
        u: [[C64; 4]; 4],
        trunc: &Truncation,
    ) {
        let a = &self.sites[k];
        let b = &self.sites[k + 1];

        let dl = a.dl;
        let dr = b.dr;
        let chi = a.dr;

        // Build Θ matrix
        let mut theta = DMatrix::<C64>::zeros(dl * 2, 2 * dr);

        for l in 0..dl {
            for m in 0..chi {
                for r in 0..dr {
                    for p1 in 0..2 {
                        for p2 in 0..2 {
                            let mut v = C64::new(0.0, 0.0);
                            for q1 in 0..2 {
                                for q2 in 0..2 {
                                    let i = p1 * 2 + p2;
                                    let j = q1 * 2 + q2;
                                    v += u[i][j]
                                        * a.get(l, q1, m)
                                        * b.get(m, q2, r);
                                }
                            }
                            theta[(l * 2 + p1, p2 * dr + r)] += v;
                        }
                    }
                }
            }
        }

        let svd = SVD::new(theta, true, true);

        let mut kept = 0;
        for s in svd.singular_values.iter() {
            if *s > trunc.cutoff && kept < trunc.max_bond {
                kept += 1;
            }
        }

        let u_mat = svd.u.unwrap().columns(0, kept);
        let v_mat = svd.v_t.unwrap().rows(0, kept);
        let s_vals = &svd.singular_values[0..kept];

        let mut new_a = Tensor3::zeros(dl, 2, kept);
        for l in 0..dl {
            for p in 0..2 {
                for m in 0..kept {
                    new_a.set(
                        l,
                        p,
                        m,
                        u_mat[(l * 2 + p, m)] * s_vals[m],
                    );
                }
            }
        }

        let mut new_b = Tensor3::zeros(kept, 2, dr);
        for m in 0..kept {
            for p in 0..2 {
                for r in 0..dr {
                    new_b.set(
                        m,
                        p,
                        r,
                        v_mat[(m, p * dr + r)],
                    );
                }
            }
        }

        self.sites[k] = new_a;
        self.sites[k + 1] = new_b;
    }

    /* -------- Canonicalization -------- */

    fn left_canonicalize(&mut self) {
        for i in 0..self.sites.len() - 1 {
            let t = &self.sites[i];
            let m = DMatrix::from_iterator(
                t.dl * 2,
                t.dr,
                t.data.iter().cloned(),
            );

            let svd = SVD::new(m, true, true);
            let u = svd.u.unwrap();
            let s = svd.singular_values;
            let vt = svd.v_t.unwrap();

            let chi = s.len();

            let mut new_t = Tensor3::zeros(t.dl, 2, chi);
            for l in 0..t.dl {
                for p in 0..2 {
                    for k in 0..chi {
                        new_t.set(l, p, k, u[(l * 2 + p, k)]);
                    }
                }
            }

            let mut next = &mut self.sites[i + 1];
            let mut new_next = Tensor3::zeros(chi, 2, next.dr);

            for k in 0..chi {
                for r in 0..next.dr {
                    for p in 0..2 {
                        new_next.set(
                            k,
                            p,
                            r,
                            s[k] * vt[(k, r)] * next.get(0, p, r),
                        );
                    }
                }
            }

            self.sites[i] = new_t;
            self.sites[i + 1] = new_next;
        }
    }

    /* -------- Measurement -------- */

    fn measure_z(&mut self, k: usize, rng: &mut ONDRng) -> u8 {
        let s = &self.sites[k];
        let mut p0 = 0.0;
        let mut p1 = 0.0;

        for l in 0..s.dl {
            for r in 0..s.dr {
                p0 += s.get(l, 0, r).norm_sqr();
                p1 += s.get(l, 1, r).norm_sqr();
            }
        }

        let x = rng.next_f64(b"MEASURE");
        let outcome = if x < p0 / (p0 + p1) { 0 } else { 1 };

        let mut t = Tensor3::zeros(s.dl, 2, s.dr);
        let norm = if outcome == 0 { p0.sqrt() } else { p1.sqrt() };

        for l in 0..s.dl {
            for r in 0..s.dr {
                t.set(
                    l,
                    outcome as usize,
                    r,
                    s.get(l, outcome as usize, r) / norm,
                );
            }
        }

        self.sites[k] = t;
        outcome
    }
}

/* =========================
   Gates
   ========================= */

fn hadamard() -> [[C64; 2]; 2] {
    let s = 1.0 / 2.0_f64.sqrt();
    [
        [C64::new(s, 0.0), C64::new(s, 0.0)],
        [C64::new(s, 0.0), C64::new(-s, 0.0)],
    ]
}

/* =========================
   Noise (trajectory)
   ========================= */

fn depolarizing(rng: &mut ONDRng, p: f64) -> u8 {
    let x = rng.next_f64(b"DEPOL");
    if x < p / 3.0 { 1 }
    else if x < 2.0 * p / 3.0 { 2 }
    else if x < p { 3 }
    else { 0 }
}

/* =========================
   Benchmark
   ========================= */

fn benchmark(n: usize, depth: usize) {
    let trunc = Truncation { max_bond: 64, cutoff: 1e-8 };
    let mut psi = MPS::new_zero(n);
    let start = Instant::now();

    for t in 0..depth {
        psi.apply_1q(t % n, hadamard());
        if t + 1 < n {
            psi.apply_2q_svd(t % (n - 1), [[C64::new(1.0,0.0);4];4], &trunc);
        }
    }

    println!(
        "Benchmark: n={}, depth={} → {:.3} s",
        n, depth, start.elapsed().as_secs_f64()
    );
}

/* =========================
   Main
   ========================= */

fn main() {
    let mut rng = ONDRng::new(b"seed");
    let mut psi = MPS::new_zero(10);

    psi.apply_1q(0, hadamard());
    let m = psi.measure_z(0, &mut rng);
    println!("Measurement result: {}", m);

    benchmark(50, 100);
}
