use quantum::{apply_cnot, energy::energy, gates::hadamard, hamiltonian::Hamiltonian};
use tn::{mps::MPS, truncation::Truncation};

#[test]
fn bell_energy_ising() {
    let trunc = Truncation {
        max_bond: 8,
        cutoff: 1e-12,
    };
    let mut psi = MPS::new_zero(2);

    psi.apply_1q(0, hadamard());
    apply_cnot(&mut psi, 0, trunc);

    let h = Hamiltonian {
        z_fields: vec![0.0, 0.0],
        zz_couplings: vec![1.0],
    };

    let e = energy(&psi, &h);
    assert!((e - 1.0).abs() < 1e-12);
}
