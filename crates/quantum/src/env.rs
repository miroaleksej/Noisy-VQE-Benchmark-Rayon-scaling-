use tn::mps::{C64, Tensor3};

pub(crate) fn left_env(sites: &[Tensor3], k: usize) -> Vec<C64> {
    let mut env = vec![C64::new(1.0, 0.0)];
    for i in 0..k {
        let a = &sites[i];
        let mut next = vec![C64::new(0.0, 0.0); a.dr * a.dr];
        for l in 0..a.dl {
            for lp in 0..a.dl {
                let lval = env[l * a.dl + lp];
                for p in 0..a.dp {
                    for r in 0..a.dr {
                        let aval = a.get(l, p, r);
                        for rp in 0..a.dr {
                            let idx = r * a.dr + rp;
                            next[idx] += lval * aval * a.get(lp, p, rp).conj();
                        }
                    }
                }
            }
        }
        env = next;
    }
    env
}

pub(crate) fn right_env(sites: &[Tensor3], k: usize) -> Vec<C64> {
    let mut env = vec![C64::new(1.0, 0.0)];
    for i in (k + 1..sites.len()).rev() {
        let a = &sites[i];
        let mut next = vec![C64::new(0.0, 0.0); a.dl * a.dl];
        for r in 0..a.dr {
            for rp in 0..a.dr {
                let rval = env[r * a.dr + rp];
                for p in 0..a.dp {
                    for l in 0..a.dl {
                        let aval = a.get(l, p, r);
                        for lp in 0..a.dl {
                            let idx = l * a.dl + lp;
                            next[idx] += aval * a.get(lp, p, rp).conj() * rval;
                        }
                    }
                }
            }
        }
        env = next;
    }
    env
}
