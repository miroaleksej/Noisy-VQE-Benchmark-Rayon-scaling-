use crate::hamiltonian::{Hamiltonian, Heisenberg};
use crate::observables::{expect_xx, expect_yy, expect_z, expect_zz};
use tn::mps::MPS;

/// Expectation value ⟨ψ|H|ψ⟩ for a diagonal Z/ZZ Hamiltonian.
pub fn energy(psi: &MPS, h: &Hamiltonian) -> f64 {
    let mut e = 0.0;

    for (i, &hi) in h.z_fields.iter().enumerate() {
        e += hi * expect_z(psi, i);
    }

    for (i, &j) in h.zz_couplings.iter().enumerate() {
        e += j * expect_zz(psi, i, i + 1);
    }

    e
}

/// Expectation value ⟨ψ|H|ψ⟩ for nearest-neighbor Heisenberg (XX + YY + ZZ).
pub fn energy_heisenberg(psi: &MPS, h: &Heisenberg) -> f64 {
    let mut e = 0.0;

    for i in 0..h.jx.len() {
        e += h.jx[i] * expect_xx(psi, i, i + 1);
    }
    for i in 0..h.jy.len() {
        e += h.jy[i] * expect_yy(psi, i, i + 1);
    }
    for i in 0..h.jz.len() {
        e += h.jz[i] * expect_zz(psi, i, i + 1);
    }

    e
}
