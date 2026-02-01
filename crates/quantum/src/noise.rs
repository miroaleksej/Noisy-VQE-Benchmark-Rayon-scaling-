use crate::gates::{pauli_x, pauli_y, pauli_z};
use rng::ONDRng;
use tn::mps::MPS;

/// Single-qubit depolarizing channel implemented via random Pauli kicks.
pub fn depolarizing_1q(psi: &mut MPS, k: usize, p: f64, rng: &mut ONDRng) {
    if p <= 0.0 {
        return;
    }

    let x = rng.next_f64(b"DEPOL_1Q");
    if x >= p {
        return;
    }

    let r = x / p;
    if r < 1.0 / 3.0 {
        psi.apply_1q(k, pauli_x());
    } else if r < 2.0 / 3.0 {
        psi.apply_1q(k, pauli_y());
    } else {
        psi.apply_1q(k, pauli_z());
    }
}
