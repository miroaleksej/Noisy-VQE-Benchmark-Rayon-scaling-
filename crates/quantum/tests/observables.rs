use quantum::{apply_cnot, gates::hadamard, observables::{expect_z, expect_zz}};
use tn::{mps::MPS, truncation::Truncation};

#[test]
fn bell_observables() {
    let trunc = Truncation {
        max_bond: 8,
        cutoff: 1e-12,
    };
    let mut psi = MPS::new_zero(2);

    psi.apply_1q(0, hadamard());
    apply_cnot(&mut psi, 0, trunc);

    assert!(expect_z(&psi, 0).abs() < 1e-12);
    assert!(expect_z(&psi, 1).abs() < 1e-12);
    assert!((expect_zz(&psi, 0, 1) - 1.0).abs() < 1e-12);
}
