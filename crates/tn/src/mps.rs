use crate::truncation::Truncation;
use faer::Mat;
use num_complex::Complex64;

pub type C64 = Complex64;

#[derive(Clone)]
pub struct Tensor3 {
    pub data: Vec<C64>,
    pub dl: usize,
    pub dp: usize,
    pub dr: usize,
}

impl Tensor3 {
    pub fn zeros(dl: usize, dp: usize, dr: usize) -> Self {
        Self {
            data: vec![C64::new(0.0, 0.0); dl * dp * dr],
            dl,
            dp,
            dr,
        }
    }

    #[inline]
    fn idx(&self, l: usize, p: usize, r: usize) -> usize {
        (l * self.dp + p) * self.dr + r
    }

    pub fn get(&self, l: usize, p: usize, r: usize) -> C64 {
        self.data[self.idx(l, p, r)]
    }

    pub fn set(&mut self, l: usize, p: usize, r: usize, v: C64) {
        let i = self.idx(l, p, r);
        self.data[i] = v;
    }
}

#[derive(Clone)]
pub struct MPS {
    pub sites: Vec<Tensor3>,
}

impl MPS {
    pub fn new_zero(n: usize) -> Self {
        let mut sites = Vec::with_capacity(n);
        for _ in 0..n {
            let mut t = Tensor3::zeros(1, 2, 1);
            t.set(0, 0, 0, C64::new(1.0, 0.0));
            sites.push(t);
        }
        Self { sites }
    }

    pub fn apply_1q(&mut self, k: usize, u: [[C64; 2]; 2]) {
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

    pub fn apply_2q_svd(&mut self, k: usize, u: [[C64; 4]; 4], trunc: Truncation) {
        let a = &self.sites[k];
        let b = &self.sites[k + 1];

        let dl = a.dl;
        let dr = b.dr;
        let chi = a.dr;

        let mut theta = Mat::<C64>::zeros(dl * 2, 2 * dr);

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
                                    v += u[i][j] * a.get(l, q1, m) * b.get(m, q2, r);
                                }
                            }
                            let row = l * 2 + p1;
                            let col = p2 * dr + r;
                            let cur = theta.read(row, col);
                            theta.write(row, col, cur + v);
                        }
                    }
                }
            }
        }

        let svd = theta.thin_svd();
        let s = svd.s_diagonal();

        let mut kept = 0;
        for i in 0..s.nrows() {
            let sv = s.read(i).re;
            if sv > trunc.cutoff && kept < trunc.max_bond {
                kept += 1;
            }
        }
        if kept == 0 {
            kept = 1;
        }

        let u_full = svd.u();
        let v_full = svd.v();
        let u_mat = u_full.submatrix(0, 0, u_full.nrows(), kept);
        let v_mat = v_full.submatrix(0, 0, v_full.nrows(), kept);
        let mut s_vals = Vec::with_capacity(kept);
        for i in 0..kept {
            s_vals.push(s.read(i).re);
        }

        let mut new_a = Tensor3::zeros(dl, 2, kept);
        for l in 0..dl {
            for p in 0..2 {
                for m in 0..kept {
                    let u_val = u_mat.read(l * 2 + p, m);
                    new_a.set(l, p, m, u_val * s_vals[m]);
                }
            }
        }

        let mut new_b = Tensor3::zeros(kept, 2, dr);
        for m in 0..kept {
            for p in 0..2 {
                for r in 0..dr {
                    let v_val = v_mat.read(p * dr + r, m).conj();
                    new_b.set(m, p, r, v_val);
                }
            }
        }

        self.sites[k] = new_a;
        self.sites[k + 1] = new_b;
    }
}
