use quantum::gates::hadamard;
use std::time::Instant;
use tn::mps::{C64, MPS};
use tn::truncation::Truncation;

pub mod grad;
pub mod gradient_vqe;
mod output;
pub mod vqe;
pub use vqe::{noisy_vqe_sweep, vqe_sweep, vqe_sweep_shots, vqe_sweep_steps};

pub fn benchmark(n: usize, depth: usize) {
    let trunc = Truncation {
        max_bond: 64,
        cutoff: 1e-8,
    };
    let mut psi = MPS::new_zero(n);

    let ident = [[C64::new(1.0, 0.0); 4]; 4];

    let start = Instant::now();
    for t in 0..depth {
        psi.apply_1q(t % n, hadamard());
        if t + 1 < n {
            psi.apply_2q_svd(t % (n - 1), ident, trunc);
        }
    }

    println!(
        "Benchmark: n={}, depth={} → {:.3} s",
        n,
        depth,
        start.elapsed().as_secs_f64()
    );
}
