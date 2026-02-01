use crate::hamiltonian::Hamiltonian;
use crate::shot_estimator::{estimate_z_shots, estimate_zz_shots};
use rng::ONDRng;
use tn::mps::MPS;

/// Estimate ⟨ψ|H|ψ⟩ via shots for a diagonal Z/ZZ Hamiltonian.
pub fn estimate_energy_shots(
    psi: &MPS,
    h: &Hamiltonian,
    rng: &mut ONDRng,
    shots: usize,
) -> f64 {
    let mut e = 0.0;

    for (i, &hi) in h.z_fields.iter().enumerate() {
        e += hi * estimate_z_shots(psi, i, rng, shots);
    }

    for (i, &j) in h.zz_couplings.iter().enumerate() {
        e += j * estimate_zz_shots(psi, i, i + 1, rng, shots);
    }

    e
}
