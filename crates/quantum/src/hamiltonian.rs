#[derive(Clone)]
pub struct Hamiltonian {
    /// local fields h_i * Z_i
    pub z_fields: Vec<f64>,
    /// nearest-neighbor couplings J_i * Z_i Z_{i+1}
    pub zz_couplings: Vec<f64>,
}

impl Hamiltonian {
    pub fn ising(n: usize, h: f64, j: f64) -> Self {
        Self {
            z_fields: vec![h; n],
            zz_couplings: vec![j; n.saturating_sub(1)],
        }
    }
}

#[derive(Clone)]
pub struct Heisenberg {
    pub jx: Vec<f64>,
    pub jy: Vec<f64>,
    pub jz: Vec<f64>,
}

impl Heisenberg {
    pub fn uniform(n: usize, j: f64) -> Self {
        Self {
            jx: vec![j; n.saturating_sub(1)],
            jy: vec![j; n.saturating_sub(1)],
            jz: vec![j; n.saturating_sub(1)],
        }
    }
}
