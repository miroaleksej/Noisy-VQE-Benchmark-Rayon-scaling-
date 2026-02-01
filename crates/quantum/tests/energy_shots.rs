use quantum::{
    apply_cnot,
    energy::energy,
    energy_shots::estimate_energy_shots,
    gates::hadamard,
    hamiltonian::Hamiltonian,
};
use rng::ONDRng;
use tn::{mps::MPS, truncation::Truncation};

#[test]
fn shot_energy_converges() {
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

    let exact = energy(&psi, &h);
    let mut rng = ONDRng::new(b"shots");

    let est = estimate_energy_shots(&psi, &h, &mut rng, 5000);

    assert!((est - exact).abs() < 0.05);
}
