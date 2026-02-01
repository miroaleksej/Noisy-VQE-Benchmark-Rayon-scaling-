use clap::Parser;
use quantum::{apply_cnot, gates::rx};
use rng::ONDRng;
use tn::{mps::C64, mps::MPS, truncation::Truncation};

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(author, version, about = "MPS fidelity sweep vs bond dimension (n <= 30)")]
struct Args {
    /// Number of qubits (recommended <= 30)
    #[arg(long, default_value_t = 24)]
    n: usize,

    /// Circuit depth (brickwork layers)
    #[arg(long, default_value_t = 30)]
    depth: usize,

    /// Sweep depth from 1..=depth and output a depth x chi surface
    #[arg(long)]
    depth_sweep: bool,

    /// Depth step for --depth-sweep
    #[arg(long, default_value_t = 1)]
    depth_step: usize,

    /// Start depth for --depth-sweep (inclusive)
    #[arg(long, default_value_t = 1)]
    depth_start: usize,

    /// End depth for --depth-sweep (inclusive). 0 = use --depth
    #[arg(long, default_value_t = 0)]
    depth_end: usize,

    /// Comma-separated list of test bond dimensions
    #[arg(long, default_value = "4,8,16,32")]
    chi_test: String,

    /// Reference bond dimension
    #[arg(long, default_value_t = 64)]
    chi_ref: usize,

    /// SVD cutoff
    #[arg(long, default_value_t = 1e-8)]
    cutoff: f64,

    /// RNG seed
    #[arg(long, default_value = "fid-24")]
    seed: String,

    /// Output CSV path
    #[arg(long, default_value = "fidelity_sweep.csv")]
    out: String,
}

fn main() {
    let args = Args::parse();

    if args.depth_step == 0 {
        eprintln!("depth_step must be > 0");
        std::process::exit(1);
    }

    let depth_end = if args.depth_end == 0 {
        args.depth
    } else {
        args.depth_end
    };
    if args.depth_sweep {
        if args.depth_start == 0 {
            eprintln!("depth_start must be >= 1");
            std::process::exit(1);
        }
        if depth_end < args.depth_start {
            eprintln!(
                "depth_end must be >= depth_start ({} < {})",
                depth_end, args.depth_start
            );
            std::process::exit(1);
        }
    }

    if args.n > 30 {
        eprintln!("WARNING: fidelity sweep is intended for n <= 30 (got n={})", args.n);
    }

    let chi_test = parse_list(&args.chi_test);
    if chi_test.is_empty() {
        eprintln!("chi_test must contain at least one integer value");
        std::process::exit(1);
    }

    let max_test = *chi_test.iter().max().unwrap_or(&0);
    if args.chi_ref <= max_test {
        eprintln!(
            "WARNING: chi_ref ({}) should be > max chi_test ({})",
            args.chi_ref, max_test
        );
    }

    if args.depth_sweep {
        let depth_out = depth_output_path(&args.out);
        println!("depth-sweep output: {}", depth_out.display());
        let file = File::create(&depth_out).expect("failed to create CSV file");
        let mut w = BufWriter::new(file);
        writeln!(w, "depth,chi,fidelity,one_minus_fidelity").expect("failed to write header");

        let mut rng = ONDRng::new(args.seed.as_bytes());
        let trunc_ref = Truncation {
            max_bond: args.chi_ref,
            cutoff: args.cutoff,
        };
        let truncs: Vec<Truncation> = chi_test
            .iter()
            .map(|&chi| Truncation {
                max_bond: chi,
                cutoff: args.cutoff,
            })
            .collect();

        let mut psi_ref = MPS::new_zero(args.n);
        let mut psi_tests: Vec<MPS> = chi_test.iter().map(|_| MPS::new_zero(args.n)).collect();

        let mut depth = 0usize;
        while depth < depth_end {
            let layer = build_layer_params(args.n, &mut rng);

            apply_layer_params(&mut psi_ref, trunc_ref, &layer);
            for (psi, trunc) in psi_tests.iter_mut().zip(truncs.iter()) {
                apply_layer_params(psi, *trunc, &layer);
            }

            depth += 1;
            if depth < args.depth_start {
                continue;
            }
            if (depth - args.depth_start) % args.depth_step == 0 || depth == depth_end {
                let ref_norm = overlap(&psi_ref, &psi_ref).re;
                for (idx, &chi) in chi_test.iter().enumerate() {
                    let psi = &psi_tests[idx];
                    let ov = overlap(psi, &psi_ref);
                    let norm = overlap(psi, psi).re;
                    let fidelity = ov.norm_sqr() / (norm * ref_norm);
                    let one_minus = 1.0 - fidelity;

                    self_check(chi, args.chi_ref, one_minus);

                    writeln!(w, "{},{},{},{}", depth, chi, fidelity, one_minus)
                        .expect("failed to write row");
                }
                println!("depth={}  wrote {} rows", depth, chi_test.len());
            }
        }
    } else {
        let file = File::create(&args.out).expect("failed to create CSV file");
        let mut w = BufWriter::new(file);
        let psi_ref = build_state(
            args.n,
            args.depth,
            Truncation {
                max_bond: args.chi_ref,
                cutoff: args.cutoff,
            },
            &args.seed,
        );

        let ref_norm = overlap(&psi_ref, &psi_ref).re;

        writeln!(w, "chi,fidelity,one_minus_fidelity").expect("failed to write header");
        for &chi in &chi_test {
            let psi = build_state(
                args.n,
                args.depth,
                Truncation {
                    max_bond: chi,
                    cutoff: args.cutoff,
                },
                &args.seed,
            );

            let ov = overlap(&psi, &psi_ref);
            let norm = overlap(&psi, &psi).re;
            let fidelity = ov.norm_sqr() / (norm * ref_norm);
            let one_minus = 1.0 - fidelity;

            self_check(chi, args.chi_ref, one_minus);

            writeln!(w, "{},{},{}", chi, fidelity, one_minus).expect("failed to write row");
            println!("chi={}  1-fidelity={:.3e}", chi, one_minus);
        }
    }
}

fn build_state(n: usize, depth: usize, trunc: Truncation, seed: &str) -> MPS {
    let mut rng = ONDRng::new(seed.as_bytes());
    let mut psi = MPS::new_zero(n);

    for _ in 0..depth {
        apply_brickwork_layer(&mut psi, trunc, &mut rng);
    }

    psi
}

fn apply_brickwork_layer(psi: &mut MPS, trunc: Truncation, rng: &mut ONDRng) {
    let layer = build_layer_params(psi.sites.len(), rng);
    apply_layer_params(psi, trunc, &layer);
}

#[derive(Clone, Copy)]
struct GateParams {
    k: usize,
    a0: f64,
    b0: f64,
    c0: f64,
    a1: f64,
    b1: f64,
    c1: f64,
}

fn build_layer_params(n: usize, rng: &mut ONDRng) -> Vec<GateParams> {
    let mut layer = Vec::with_capacity(n);
    for start in [0usize, 1usize] {
        let mut i = start;
        while i + 1 < n {
            layer.push(GateParams {
                k: i,
                a0: rand_angle(rng, b"RZ0"),
                b0: rand_angle(rng, b"RX0"),
                c0: rand_angle(rng, b"RZ1"),
                a1: rand_angle(rng, b"RZ2"),
                b1: rand_angle(rng, b"RX1"),
                c1: rand_angle(rng, b"RZ3"),
            });
            i += 2;
        }
    }
    layer
}

fn apply_layer_params(psi: &mut MPS, trunc: Truncation, layer: &[GateParams]) {
    for gate in layer {
        apply_gate_params(psi, trunc, *gate);
    }
}

fn apply_gate_params(psi: &mut MPS, trunc: Truncation, gate: GateParams) {
    psi.apply_1q(gate.k, rz(gate.a0));
    psi.apply_1q(gate.k, rx(gate.b0));
    psi.apply_1q(gate.k, rz(gate.c0));
    psi.apply_1q(gate.k + 1, rz(gate.a1));
    psi.apply_1q(gate.k + 1, rx(gate.b1));
    psi.apply_1q(gate.k + 1, rz(gate.c1));

    apply_cnot(psi, gate.k, trunc);
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

fn overlap(a: &MPS, b: &MPS) -> C64 {
    assert_eq!(a.sites.len(), b.sites.len(), "MPS length mismatch");
    let mut env = vec![C64::new(0.0, 0.0); a.sites[0].dl * b.sites[0].dl];
    env[0] = C64::new(1.0, 0.0);

    for (sa, sb) in a.sites.iter().zip(b.sites.iter()) {
        let mut next = vec![C64::new(0.0, 0.0); sa.dr * sb.dr];
        for la in 0..sa.dl {
            for lb in 0..sb.dl {
                let env_val = env[la * sb.dl + lb];
                if env_val == C64::new(0.0, 0.0) {
                    continue;
                }
                for ra in 0..sa.dr {
                    for rb in 0..sb.dr {
                        let mut acc = C64::new(0.0, 0.0);
                        for p in 0..sa.dp {
                            acc += sa.get(la, p, ra).conj() * sb.get(lb, p, rb);
                        }
                        next[ra * sb.dr + rb] += env_val * acc;
                    }
                }
            }
        }
        env = next;
    }

    env.into_iter().fold(C64::new(0.0, 0.0), |a, b| a + b)
}

fn self_check(chi: usize, chi_ref: usize, one_minus: f64) {
    if chi == chi_ref {
        const SELF_TOL: f64 = 1e-8;
        if one_minus > SELF_TOL {
            eprintln!(
                "ERROR: self-fidelity check failed for chi_ref={} (1-fidelity={:.3e})",
                chi_ref, one_minus
            );
            std::process::exit(1);
        }
    }
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

fn depth_output_path(out: &str) -> PathBuf {
    let path = Path::new(out);
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("fidelity_sweep");
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("csv");
    let file_name = format!("{stem}_depth.{ext}");
    match path.parent() {
        Some(parent) => parent.join(file_name),
        None => PathBuf::from(file_name),
    }
}
