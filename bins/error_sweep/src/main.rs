use clap::Parser;
use quantum::{
    apply_cnot,
    energy::{energy, energy_heisenberg},
    gates::{hadamard, rx},
    hamiltonian::{Hamiltonian, Heisenberg},
};
use rng::ONDRng;
use tn::{mps::MPS, truncation::Truncation};

use std::fs::File;
use std::io::{BufWriter, Write};

enum HMode {
    Ising(Hamiltonian),
    Heisenberg(Heisenberg),
}

impl HMode {
    fn energy(&self, psi: &MPS) -> f64 {
        match self {
            HMode::Ising(h) => energy(psi, h),
            HMode::Heisenberg(h) => energy_heisenberg(psi, h),
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about = "MPS energy error sweep vs bond dimension")]
struct Args {
    /// Number of qubits
    #[arg(long, default_value_t = 40)]
    n: usize,

    /// Circuit depth (brickwork layers)
    #[arg(long, default_value_t = 50)]
    depth: usize,

    /// Comma-separated list of test bond dimensions
    #[arg(long, default_value = "8,16,32")]
    chi_test: String,

    /// Reference bond dimension
    #[arg(long, default_value_t = 128)]
    chi_ref: usize,

    /// Optional check bond dimension for reference convergence (0 = disabled)
    #[arg(long, default_value_t = 0)]
    chi_ref_check: usize,

    /// Hamiltonian: ising | heisenberg
    #[arg(long, default_value = "heisenberg")]
    h: String,

    /// Heisenberg coupling Jx (only used when --h heisenberg)
    #[arg(long, default_value_t = 1.0)]
    heisenberg_jx: f64,

    /// Heisenberg coupling Jy (only used when --h heisenberg)
    #[arg(long, default_value_t = 1.0)]
    heisenberg_jy: f64,

    /// Heisenberg coupling Jz (only used when --h heisenberg)
    #[arg(long, default_value_t = 1.0)]
    heisenberg_jz: f64,

    /// Run Bell-state sanity check for Heisenberg energy and exit
    #[arg(long)]
    sanity: bool,

    /// SVD cutoff
    #[arg(long, default_value_t = 1e-8)]
    cutoff: f64,

    /// RNG seed
    #[arg(long, default_value = "err-40")]
    seed: String,

    /// Output CSV path
    #[arg(long, default_value = "error_sweep.csv")]
    out: String,
}

fn main() {
    let args = Args::parse();

    if args.sanity {
        run_sanity(&args);
        return;
    }

    let chi_test = parse_list(&args.chi_test);
    if chi_test.is_empty() {
        eprintln!("chi_test must contain at least one integer value");
        std::process::exit(1);
    }

    if args.chi_ref_check > 0 && args.chi_ref_check <= args.chi_ref {
        eprintln!(
            "ERROR: --chi-ref-check ({}) must be > --chi-ref ({})",
            args.chi_ref_check, args.chi_ref
        );
        std::process::exit(1);
    }

    let h_mode = match args.h.as_str() {
        "ising" => HMode::Ising(Hamiltonian::ising(args.n, 0.0, 1.0)),
        "heisenberg" => {
            let bonds = args.n.saturating_sub(1);
            HMode::Heisenberg(Heisenberg {
                jx: vec![args.heisenberg_jx; bonds],
                jy: vec![args.heisenberg_jy; bonds],
                jz: vec![args.heisenberg_jz; bonds],
            })
        }
        other => {
            eprintln!("ERROR: --h must be 'ising' or 'heisenberg', got '{}'", other);
            std::process::exit(1);
        }
    };

    let e_ref = run_energy(
        args.n,
        args.depth,
        Truncation {
            max_bond: args.chi_ref,
            cutoff: args.cutoff,
        },
        &args.seed,
        &h_mode,
    );

    if args.chi_ref_check > 0 {
        let e_check = run_energy(
            args.n,
            args.depth,
            Truncation {
                max_bond: args.chi_ref_check,
                cutoff: args.cutoff,
            },
            &args.seed,
            &h_mode,
        );
        let diff = (e_ref - e_check).abs();
        const REF_TOL: f64 = 1e-6;
        if diff > REF_TOL {
            eprintln!(
                "WARNING: reference not converged: |E({}) - E({})| = {:.3e}",
                args.chi_ref, args.chi_ref_check, diff
            );
        }
    }

    let file = File::create(&args.out).expect("failed to create CSV file");
    let mut w = BufWriter::new(file);
    writeln!(w, "chi,energy,error_energy").expect("failed to write header");

    for &chi in &chi_test {
        let e = run_energy(
            args.n,
            args.depth,
            Truncation {
                max_bond: chi,
                cutoff: args.cutoff,
            },
            &args.seed,
            &h_mode,
        );
        let err = (e - e_ref).abs();
        writeln!(w, "{},{},{}", chi, e, err).expect("failed to write row");
        println!("chi={}  E={}  |dE|={:.3e}", chi, e, err);
    }
}

fn run_energy(
    n: usize,
    depth: usize,
    trunc: Truncation,
    seed: &str,
    h: &HMode,
) -> f64 {
    let mut rng = ONDRng::new(seed.as_bytes());
    let mut psi = MPS::new_zero(n);

    for _ in 0..depth {
        apply_brickwork_layer(&mut psi, trunc, &mut rng);
    }

    h.energy(&psi)
}

fn apply_brickwork_layer(psi: &mut MPS, trunc: Truncation, rng: &mut ONDRng) {
    let n = psi.sites.len();
    apply_pairs(psi, trunc, rng, n, 0);
    apply_pairs(psi, trunc, rng, n, 1);
}

fn apply_pairs(psi: &mut MPS, trunc: Truncation, rng: &mut ONDRng, n: usize, start: usize) {
    let mut i = start;
    while i + 1 < n {
        apply_random_2q(psi, i, trunc, rng);
        i += 2;
    }
}

fn apply_random_2q(psi: &mut MPS, k: usize, trunc: Truncation, rng: &mut ONDRng) {
    let a0 = rand_angle(rng, b"RZ0");
    let b0 = rand_angle(rng, b"RX0");
    let c0 = rand_angle(rng, b"RZ1");
    let a1 = rand_angle(rng, b"RZ2");
    let b1 = rand_angle(rng, b"RX1");
    let c1 = rand_angle(rng, b"RZ3");

    psi.apply_1q(k, rz(a0));
    psi.apply_1q(k, rx(b0));
    psi.apply_1q(k, rz(c0));
    psi.apply_1q(k + 1, rz(a1));
    psi.apply_1q(k + 1, rx(b1));
    psi.apply_1q(k + 1, rz(c1));

    apply_cnot(psi, k, trunc);
}

fn rand_angle(rng: &mut ONDRng, ctx: &[u8]) -> f64 {
    rng.next_f64(ctx) * 2.0 * std::f64::consts::PI
}

fn rz(theta: f64) -> [[quantum::gates::C64; 2]; 2] {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    let z = quantum::gates::C64::new(0.0, 0.0);
    [
        [quantum::gates::C64::new(c, -s), z],
        [z, quantum::gates::C64::new(c, s)],
    ]
}

fn parse_list(input: &str) -> Vec<usize> {
    input
        .split(',')
        .filter_map(|s| {
            let t = s.trim();
            if t.is_empty() {
                None
            } else {
                t.parse::<usize>().ok()
            }
        })
        .collect()
}

fn run_sanity(args: &Args) {
    let trunc = Truncation {
        max_bond: 8,
        cutoff: 1e-12,
    };

    let mut psi = MPS::new_zero(2);
    psi.apply_1q(0, hadamard());
    apply_cnot(&mut psi, 0, trunc);

    let h = Heisenberg {
        jx: vec![args.heisenberg_jx],
        jy: vec![args.heisenberg_jy],
        jz: vec![args.heisenberg_jz],
    };
    let e = energy_heisenberg(&psi, &h);
    let expected = args.heisenberg_jx - args.heisenberg_jy + args.heisenberg_jz;
    let err = (e - expected).abs();

    assert!(
        err < 1e-12,
        "Sanity check failed: E={} expected={} err={}",
        e,
        expected,
        err
    );
    println!("Sanity OK: E = {}", e);
}
