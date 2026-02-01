use crate::measurement::measure_z;
use rng::ONDRng;
use tn::mps::MPS;

/// Estimate ⟨Z_k⟩ via projective measurements (shots).
pub fn estimate_z_shots(psi: &MPS, k: usize, rng: &mut ONDRng, shots: usize) -> f64 {
    if shots == 0 {
        return 0.0;
    }

    let mut sum = 0.0;
    for _ in 0..shots {
        let mut psi_copy = psi.clone();
        let m = measure_z(&mut psi_copy, k, rng);
        sum += if m == 0 { 1.0 } else { -1.0 };
    }

    sum / shots as f64
}

/// Estimate ⟨Z_i Z_j⟩ via projective measurements (shots).
pub fn estimate_zz_shots(
    psi: &MPS,
    i: usize,
    j: usize,
    rng: &mut ONDRng,
    shots: usize,
) -> f64 {
    if shots == 0 {
        return 0.0;
    }

    let mut sum = 0.0;
    for _ in 0..shots {
        let mut psi_copy = psi.clone();
        let mi = measure_z(&mut psi_copy, i, rng);
        let mj = measure_z(&mut psi_copy, j, rng);

        let zi = if mi == 0 { 1.0 } else { -1.0 };
        let zj = if mj == 0 { 1.0 } else { -1.0 };

        sum += zi * zj;
    }

    sum / shots as f64
}
