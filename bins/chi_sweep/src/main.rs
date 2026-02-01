use clap::Parser;
use quantum::{apply_cnot, gates::rx};
use rng::ONDRng;
use tn::{mps::MPS, truncation::Truncation};

use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about = "MPS chi growth sweep (brickwork 1D)")]
struct Args {
    /// Number of qubits
    #[arg(long, default_value_t = 64)]
    n: usize,

    /// Maximum circuit depth (number of brickwork layers)
    #[arg(long, default_value_t = 200)]
    depth_max: usize,

    /// Depth step between measurements
    #[arg(long, default_value_t = 5)]
    depth_step: usize,

    /// Comma-separated list of max bond dimensions
    #[arg(long, default_value = "16,32,64")]
    max_bond: String,

    /// SVD cutoff
    #[arg(long, default_value_t = 1e-8)]
    cutoff: f64,

    /// Base RNG seed (shared across max_bond sweeps)
    #[arg(long, default_value = "chi-sweep")]
    seed: String,

    /// Output CSV path
    #[arg(long, default_value = "chi_sweep.csv")]
    out: String,
}

fn main() {
    let args = Args::parse();

    if args.depth_step == 0 {
        eprintln!("depth_step must be > 0");
        std::process::exit(1);
    }

    let max_bonds = parse_max_bonds(&args.max_bond);
    if max_bonds.is_empty() {
        eprintln!("max_bond must contain at least one integer value");
        std::process::exit(1);
    }

    let mut rows: Vec<(usize, usize, usize, f64)> = Vec::new();

    for &max_bond in &max_bonds {
        let trunc = Truncation {
            max_bond,
            cutoff: args.cutoff,
        };
        let mut rng = ONDRng::new(args.seed.as_bytes());
        let mut psi = MPS::new_zero(args.n);

        let mut depth = 0usize;
        while depth < args.depth_max {
            let layers = (args.depth_max - depth).min(args.depth_step);
            let start = Instant::now();
            for _ in 0..layers {
                apply_brickwork_layer(&mut psi, trunc, &mut rng);
                depth += 1;
            }
            let elapsed = start.elapsed().as_secs_f64();
            let layer_ms = (elapsed / layers as f64) * 1000.0;
            let chi = chi_max(&psi);

            rows.push((max_bond, depth, chi, layer_ms));
            println!(
                "max_bond={} depth={} chi_max={} layer_ms={:.3}",
                max_bond, depth, chi, layer_ms
            );
        }
    }

    write_csv(&args.out, &rows);
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

fn chi_max(psi: &MPS) -> usize {
    psi.sites
        .iter()
        .map(|s| s.dl.max(s.dr))
        .max()
        .unwrap_or(1)
}

fn write_csv(path: &str, rows: &[(usize, usize, usize, f64)]) {
    let file = File::create(path).expect("failed to create CSV file");
    let mut w = BufWriter::new(file);
    writeln!(w, "max_bond,depth,chi_max,layer_ms").expect("failed to write header");
    for (max_bond, depth, chi, layer_ms) in rows {
        writeln!(w, "{},{},{},{}", max_bond, depth, chi, layer_ms)
            .expect("failed to write row");
    }
}

fn parse_max_bonds(input: &str) -> Vec<usize> {
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
