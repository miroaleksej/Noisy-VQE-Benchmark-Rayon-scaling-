use crate::env::{left_env, right_env};
use crate::gates::{pauli_x, pauli_y};
use tn::mps::{C64, MPS};

fn site_weight(psi: &MPS, k: usize, p: usize) -> f64 {
    let s = &psi.sites[k];
    let left = left_env(&psi.sites, k);
    let right = right_env(&psi.sites, k);

    let mut acc = C64::new(0.0, 0.0);
    for l in 0..s.dl {
        for lp in 0..s.dl {
            let lval = left[l * s.dl + lp];
            for r in 0..s.dr {
                for rp in 0..s.dr {
                    let rval = right[r * s.dr + rp];
                    acc += lval * s.get(l, p, r) * s.get(lp, p, rp).conj() * rval;
                }
            }
        }
    }

    let val = acc.re;
    if val < 0.0 { 0.0 } else { val }
}

fn site_element(psi: &MPS, k: usize, p: usize, pp: usize) -> C64 {
    let s = &psi.sites[k];
    let left = left_env(&psi.sites, k);
    let right = right_env(&psi.sites, k);

    let mut acc = C64::new(0.0, 0.0);
    for l in 0..s.dl {
        for lp in 0..s.dl {
            let lval = left[l * s.dl + lp];
            for r in 0..s.dr {
                for rp in 0..s.dr {
                    let rval = right[r * s.dr + rp];
                    acc += lval * s.get(l, p, r) * s.get(lp, pp, rp).conj() * rval;
                }
            }
        }
    }
    acc
}

fn expect_single_site(psi: &MPS, k: usize, op: [[C64; 2]; 2]) -> f64 {
    let s = &psi.sites[k];
    assert!(s.dp == 2, "expect_single_site supports qubits only");

    let w0 = site_weight(psi, k, 0);
    let w1 = site_weight(psi, k, 1);
    let denom = w0 + w1;
    if denom == 0.0 {
        return 0.0;
    }

    let mut numer = C64::new(0.0, 0.0);
    for p in 0..2 {
        for pp in 0..2 {
            numer += op[p][pp] * site_element(psi, k, p, pp);
        }
    }

    numer.re / denom
}

/// Expectation value ⟨Z_k⟩ for a qubit at site k.
pub fn expect_z(psi: &MPS, k: usize) -> f64 {
    let s = &psi.sites[k];
    assert!(s.dp == 2, "expect_z supports qubits only");

    let w0 = site_weight(psi, k, 0);
    let w1 = site_weight(psi, k, 1);
    let denom = w0 + w1;

    if denom == 0.0 {
        return 0.0;
    }

    (w0 - w1) / denom
}

/// Expectation value ⟨X_k⟩ for a qubit at site k.
pub fn expect_x(psi: &MPS, k: usize) -> f64 {
    expect_single_site(psi, k, pauli_x())
}

/// Expectation value ⟨Y_k⟩ for a qubit at site k.
pub fn expect_y(psi: &MPS, k: usize) -> f64 {
    expect_single_site(psi, k, pauli_y())
}

/// Expectation value ⟨Z_i Z_j⟩ for nearest neighbors (i, i+1).
pub fn expect_zz(psi: &MPS, i: usize, j: usize) -> f64 {
    assert!(j == i + 1, "expect_zz supports nearest neighbors only");

    let a = &psi.sites[i];
    let b = &psi.sites[j];
    assert!(a.dp == 2 && b.dp == 2, "expect_zz supports qubits only");

    let left = left_env(&psi.sites, i);
    let right = right_env(&psi.sites, j);

    let mut weights = [[0.0f64; 2]; 2];

    for pi in 0..2 {
        for pj in 0..2 {
            let mut acc = C64::new(0.0, 0.0);
            for l in 0..a.dl {
                for lp in 0..a.dl {
                    let lval = left[l * a.dl + lp];
                    for r in 0..b.dr {
                        for rp in 0..b.dr {
                            let rval = right[r * b.dr + rp];
                            for m in 0..a.dr {
                                for mp in 0..a.dr {
                                    acc += lval
                                        * a.get(l, pi, m)
                                        * b.get(m, pj, r)
                                        * a.get(lp, pi, mp).conj()
                                        * b.get(mp, pj, rp).conj()
                                        * rval;
                                }
                            }
                        }
                    }
                }
            }
            let val = acc.re;
            weights[pi][pj] = if val < 0.0 { 0.0 } else { val };
        }
    }

    let denom = weights[0][0] + weights[0][1] + weights[1][0] + weights[1][1];
    if denom == 0.0 {
        return 0.0;
    }

    let numer = weights[0][0] - weights[0][1] - weights[1][0] + weights[1][1];
    numer / denom
}

fn expect_two_site(psi: &MPS, i: usize, j: usize, op: [[C64; 4]; 4]) -> f64 {
    assert!(j == i + 1, "expect_two_site supports nearest neighbors only");

    let a = &psi.sites[i];
    let b = &psi.sites[j];
    assert!(a.dp == 2 && b.dp == 2, "expect_two_site supports qubits only");

    let left = left_env(&psi.sites, i);
    let right = right_env(&psi.sites, j);

    let mut denom = 0.0f64;
    let mut numer = C64::new(0.0, 0.0);

    for pi in 0..2 {
        for pj in 0..2 {
            for qi in 0..2 {
                for qj in 0..2 {
                    let op_val = op[pi * 2 + pj][qi * 2 + qj];
                    let mut acc = C64::new(0.0, 0.0);
                    for l in 0..a.dl {
                        for lp in 0..a.dl {
                            let lval = left[l * a.dl + lp];
                            for r in 0..b.dr {
                                for rp in 0..b.dr {
                                    let rval = right[r * b.dr + rp];
                                    for m in 0..a.dr {
                                        for mp in 0..a.dr {
                                            acc += lval
                                                * a.get(l, pi, m)
                                                * b.get(m, pj, r)
                                                * a.get(lp, qi, mp).conj()
                                                * b.get(mp, qj, rp).conj()
                                                * rval;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    numer += op_val * acc;
                    if pi == qi && pj == qj {
                        let val = acc.re;
                        denom += if val < 0.0 { 0.0 } else { val };
                    }
                }
            }
        }
    }

    if denom == 0.0 {
        return 0.0;
    }

    numer.re / denom
}

fn kron(a: [[C64; 2]; 2], b: [[C64; 2]; 2]) -> [[C64; 4]; 4] {
    let mut out = [[C64::new(0.0, 0.0); 4]; 4];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    out[i * 2 + k][j * 2 + l] = a[i][j] * b[k][l];
                }
            }
        }
    }
    out
}

/// Expectation value ⟨X_i X_j⟩ for nearest neighbors.
pub fn expect_xx(psi: &MPS, i: usize, j: usize) -> f64 {
    expect_two_site(psi, i, j, kron(pauli_x(), pauli_x()))
}

/// Expectation value ⟨Y_i Y_j⟩ for nearest neighbors.
pub fn expect_yy(psi: &MPS, i: usize, j: usize) -> f64 {
    expect_two_site(psi, i, j, kron(pauli_y(), pauli_y()))
}
