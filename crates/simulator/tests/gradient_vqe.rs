use quantum::{energy::energy_heisenberg, gates::rx, hamiltonian::Heisenberg};
use simulator::gradient_vqe::vqe_gradient;
use tn::mps::MPS;

#[test]
fn gradient_vqe_converges() {
    let h = Heisenberg::uniform(2, 1.0);

    let energy_fn = |theta: f64| {
        let mut psi = MPS::new_zero(2);
        psi.apply_1q(0, rx(theta));
        energy_heisenberg(&psi, &h)
    };

    let (_theta, e) = vqe_gradient(0.3, energy_fn, 0.2, 60);

    assert!(e < -0.9, "E = {}", e);
}
