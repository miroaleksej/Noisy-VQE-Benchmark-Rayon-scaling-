use crate::env::{left_env, right_env};
use rng::ONDRng;
use tn::mps::{C64, MPS, Tensor3};

pub fn measure_z(psi: &mut MPS, k: usize, rng: &mut ONDRng) -> u8 {
    let s = &psi.sites[k];
    let left = left_env(&psi.sites, k);
    let right = right_env(&psi.sites, k);

    let mut probs = vec![0.0f64; s.dp];
    for p in 0..s.dp {
        let mut acc = C64::new(0.0, 0.0);
        for l in 0..s.dl {
            for lp in 0..s.dl {
                let lval = left[l * s.dl + lp];
                for r in 0..s.dr {
                    for rp in 0..s.dr {
                        let rval = right[r * s.dr + rp];
                        acc += lval
                            * s.get(l, p, r)
                            * s.get(lp, p, rp).conj()
                            * rval;
                    }
                }
            }
        }
        let val = acc.re;
        probs[p] = if val < 0.0 { 0.0 } else { val };
    }

    let total: f64 = probs.iter().sum();
    if total == 0.0 {
        return 0;
    }

    let mut x = rng.next_f64(b"MEASURE_Z") * total;
    let mut outcome = 0usize;
    for (idx, p) in probs.iter().enumerate() {
        if x < *p {
            outcome = idx;
            break;
        }
        x -= *p;
    }

    let norm = probs[outcome].sqrt();
    if norm == 0.0 {
        return outcome as u8;
    }

    let mut t = Tensor3::zeros(s.dl, s.dp, s.dr);
    for l in 0..s.dl {
        for r in 0..s.dr {
            t.set(
                l,
                outcome,
                r,
                s.get(l, outcome, r) / norm,
            );
        }
    }

    psi.sites[k] = t;
    outcome as u8
}
