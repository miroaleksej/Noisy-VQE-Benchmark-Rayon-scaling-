use crate::output::write_csv;
use quantum::{
    energy::energy,
    energy_shots::estimate_energy_shots,
    gates::rx,
    hamiltonian::Hamiltonian,
    noise::depolarizing_1q,
};
use rayon::prelude::*;
use rng::ONDRng;
use tn::mps::MPS;

pub fn vqe_sweep() {
    vqe_sweep_steps(200);
}

pub fn vqe_sweep_steps(steps: usize) {
    let h = Hamiltonian {
        z_fields: vec![0.0, 0.0],
        zz_couplings: vec![1.0],
    };

    let mut best_theta = 0.0;
    let mut best_energy = f64::INFINITY;
    let mut rows = Vec::with_capacity(steps + 1);

    for i in 0..=steps {
        let theta = 2.0 * std::f64::consts::PI * (i as f64) / (steps as f64);

        let mut psi = MPS::new_zero(2);
        psi.apply_1q(0, rx(theta));

        let e = energy(&psi, &h);
        rows.push((theta, e));
        if e < best_energy {
            best_energy = e;
            best_theta = theta;
        }
    }

    if let Err(err) = write_csv("vqe_analytic.csv", &rows) {
        eprintln!("Failed to write CSV to vqe_analytic.csv: {}", err);
    }

    println!(
        "VQE result: min E = {:.6} at theta = {:.3} rad",
        best_energy, best_theta
    );
}

pub fn vqe_sweep_shots(steps: usize, shots: usize, seed: &str) {
    let h = Hamiltonian {
        z_fields: vec![0.0, 0.0],
        zz_couplings: vec![1.0],
    };

    let mut best_theta = 0.0;
    let mut best_energy = f64::INFINITY;
    let mut rows = Vec::with_capacity(steps + 1);

    for i in 0..=steps {
        let theta = 2.0 * std::f64::consts::PI * (i as f64) / (steps as f64);

        let mut psi = MPS::new_zero(2);
        psi.apply_1q(0, rx(theta));

        let seed_str = format!("{}-vqe-shots-{}", seed, i);
        let mut rng = ONDRng::new(seed_str.as_bytes());
        let e = estimate_energy_shots(&psi, &h, &mut rng, shots);
        rows.push((theta, e));

        if e < best_energy {
            best_energy = e;
            best_theta = theta;
        }
    }

    if let Err(err) = write_csv("vqe_shots.csv", &rows) {
        eprintln!("Failed to write CSV to vqe_shots.csv: {}", err);
    }

    println!(
        "VQE shots: min E = {:.6} at theta = {:.3} rad (shots = {})",
        best_energy, best_theta, shots
    );
}

fn noisy_vqe_energy(
    theta: f64,
    h: &Hamiltonian,
    trajectories: usize,
    shots: usize,
    p: f64,
    seed: &str,
    step: usize,
) -> f64 {
    let energies: Vec<f64> = (0..trajectories)
        .into_par_iter()
        .map(|t| {
            let seed_str = format!("{}-theta-{}-traj-{}", seed, step, t);
            let mut rng = ONDRng::new(seed_str.as_bytes());
            let mut psi = MPS::new_zero(2);
            psi.apply_1q(0, rx(theta));
            depolarizing_1q(&mut psi, 0, p, &mut rng);

            estimate_energy_shots(&psi, h, &mut rng, shots)
        })
        .collect();

    let mut total = 0.0;
    for e in energies {
        total += e;
    }

    total / trajectories as f64
}

pub fn noisy_vqe_sweep(
    steps: usize,
    trajectories: usize,
    shots: usize,
    p: f64,
    seed: &str,
) {
    let h = Hamiltonian {
        z_fields: vec![0.0, 0.0],
        zz_couplings: vec![1.0],
    };

    let mut best_theta = 0.0;
    let mut best_energy = f64::INFINITY;
    let mut rows = Vec::with_capacity(steps + 1);

    for i in 0..=steps {
        let theta = 2.0 * std::f64::consts::PI * (i as f64) / (steps as f64);
        let e = noisy_vqe_energy(theta, &h, trajectories, shots, p, seed, i);
        rows.push((theta, e));

        if e < best_energy {
            best_energy = e;
            best_theta = theta;
        }
    }

    if let Err(err) = write_csv("vqe_noisy.csv", &rows) {
        eprintln!("Failed to write CSV to vqe_noisy.csv: {}", err);
    }

    println!(
        "VQE noisy: min E = {:.6} at theta = {:.3} rad (traj = {}, shots = {}, p = {:.3})",
        best_energy, best_theta, trajectories, shots, p
    );
}

#[cfg(test)]
mod tests {
    use super::noisy_vqe_energy;
    use quantum::hamiltonian::Hamiltonian;

    #[test]
    fn noisy_energy_deterministic_with_seed() {
        let h = Hamiltonian {
            z_fields: vec![0.0, 0.0],
            zz_couplings: vec![1.0],
        };

        let e1 = noisy_vqe_energy(0.7, &h, 8, 20, 0.01, "seed", 3);
        let e2 = noisy_vqe_energy(0.7, &h, 8, 20, 0.01, "seed", 3);

        assert!((e1 - e2).abs() < 1e-12, "e1 = {}, e2 = {}", e1, e2);
    }
}
