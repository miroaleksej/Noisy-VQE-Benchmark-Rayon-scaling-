use clap::{Parser, ValueEnum};

use quantum::{
    apply_cnot,
    energy::energy,
    gates::hadamard,
    hamiltonian::Hamiltonian,
    measurement::measure_z,
    observables::{expect_z, expect_zz},
};
use rng::ONDRng;
use simulator::{benchmark, noisy_vqe_sweep, vqe_sweep, vqe_sweep_shots, vqe_sweep_steps};
use tn::{mps::MPS, truncation::Truncation};

/// Quantum MPS Emulator (OND-RNG)
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// VQE mode: analytic | shots | noisy
    #[arg(long, value_enum)]
    mode: Option<Mode>,

    /// Number of shots for shot-based VQE
    #[arg(long, default_value_t = 50)]
    shots: usize,

    /// Number of trajectories for noisy VQE
    #[arg(long, default_value_t = 5)]
    trajectories: usize,

    /// Depolarizing noise probability
    #[arg(long, default_value_t = 0.01)]
    p: f64,

    /// Number of theta steps in VQE sweep
    #[arg(long, default_value_t = 200)]
    theta_steps: usize,

    /// RNG seed (full reproducibility)
    #[arg(long, default_value = "default-seed")]
    seed: String,

    /// Number of Rayon worker threads (0 = Rayon default)
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Run MPS benchmark
    #[arg(long)]
    benchmark: bool,
}

#[derive(ValueEnum, Clone, Debug)]
enum Mode {
    Analytic,
    Shots,
    Noisy,
}

fn main() {
    let args = Args::parse();

    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .expect("Failed to build Rayon thread pool");
    }

    // --------------------------------------------------
    // Demo state: Bell pair (UNCHANGED default behavior)
    // --------------------------------------------------
    let trunc = Truncation {
        max_bond: 64,
        cutoff: 1e-8,
    };

    let mut rng = ONDRng::new(args.seed.as_bytes());
    let mut psi = MPS::new_zero(2);

    psi.apply_1q(0, hadamard());
    apply_cnot(&mut psi, 0, trunc);

    // Observables (same as before)
    println!("Z0 = {:.3}", expect_z(&psi, 0));
    println!("Z1 = {:.3}", expect_z(&psi, 1));
    println!("Z0Z1 = {:.3}", expect_zz(&psi, 0, 1));

    let h = Hamiltonian::ising(2, 0.0, 1.0);
    println!("Energy = {:.3}", energy(&psi, &h));

    let m0 = measure_z(&mut psi, 0, &mut rng);
    let m1 = measure_z(&mut psi, 1, &mut rng);
    println!("Bell measurement: {}, {}", m0, m1);

    // --------------------------------------------------
    // VQE dispatch (NEW, but default = legacy demo)
    // --------------------------------------------------
    match args.mode {
        None => {
            benchmark(40, 80);
            vqe_sweep();
            vqe_sweep_shots(60, 50, &args.seed);
            noisy_vqe_sweep(40, 5, 50, 0.01, &args.seed);
        }
        Some(Mode::Analytic) => {
            vqe_sweep_steps(args.theta_steps);
            if args.benchmark {
                benchmark(40, 80);
            }
        }
        Some(Mode::Shots) => {
            vqe_sweep_shots(args.theta_steps, args.shots, &args.seed);
            if args.benchmark {
                benchmark(40, 80);
            }
        }
        Some(Mode::Noisy) => {
            noisy_vqe_sweep(
                args.theta_steps,
                args.trajectories,
                args.shots,
                args.p,
                &args.seed,
            );
            if args.benchmark {
                benchmark(40, 80);
            }
        }
    }
}
