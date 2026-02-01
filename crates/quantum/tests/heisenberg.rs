use quantum::{
    apply_cnot,
    energy::energy_heisenberg,
    gates::hadamard,
    hamiltonian::Heisenberg,
    observables::{expect_xx, expect_yy, expect_zz},
};
use tn::{mps::MPS, truncation::Truncation};

#[test]
fn bell_heisenberg_observables() {
    let trunc = Truncation {
        max_bond: 8,
        cutoff: 1e-12,
    };
    let mut psi = MPS::new_zero(2);

    psi.apply_1q(0, hadamard());
    apply_cnot(&mut psi, 0, trunc);

    let xx = expect_xx(&psi, 0, 1);
    let yy = expect_yy(&psi, 0, 1);
    let zz = expect_zz(&psi, 0, 1);

    assert!((xx - 1.0).abs() < 1e-12, "XX = {}", xx);
    assert!((yy + 1.0).abs() < 1e-12, "YY = {}", yy);
    assert!((zz - 1.0).abs() < 1e-12, "ZZ = {}", zz);

    let h = Heisenberg {
        jx: vec![1.0],
        jy: vec![2.0],
        jz: vec![3.0],
    };
    let e = energy_heisenberg(&psi, &h);
    assert!((e - 2.0).abs() < 1e-12, "E = {}", e);
}

#[test]
fn heisenberg_bell_sanity_energy() {
    let trunc = Truncation {
        max_bond: 8,
        cutoff: 1e-12,
    };
    let mut psi = MPS::new_zero(2);

    psi.apply_1q(0, hadamard());
    apply_cnot(&mut psi, 0, trunc);

    let jx = 1.0;
    let jy = 2.0;
    let jz = 3.0;
    let h = Heisenberg {
        jx: vec![jx],
        jy: vec![jy],
        jz: vec![jz],
    };
    let e = energy_heisenberg(&psi, &h);
    let expected = jx - jy + jz;
    assert!((e - expected).abs() < 1e-12, "E = {}", e);
}
