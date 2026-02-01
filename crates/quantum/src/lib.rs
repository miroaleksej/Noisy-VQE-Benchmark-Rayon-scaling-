pub mod gates;
pub mod measurement;
pub mod observables;
pub mod hamiltonian;
pub mod energy;
pub mod shot_estimator;
pub mod energy_shots;
pub mod noise;
mod env;

use tn::{mps::MPS, truncation::Truncation};

pub fn apply_cnot(psi: &mut MPS, k: usize, trunc: Truncation) {
    psi.apply_2q_svd(k, gates::cnot(), trunc);
}

pub fn apply_cz(psi: &mut MPS, k: usize, trunc: Truncation) {
    psi.apply_2q_svd(k, gates::cz(), trunc);
}
