use num_complex::Complex64;

pub type C64 = Complex64;

pub fn hadamard() -> [[C64; 2]; 2] {
    let s = 1.0 / 2.0_f64.sqrt();
    [
        [C64::new(s, 0.0), C64::new(s, 0.0)],
        [C64::new(s, 0.0), C64::new(-s, 0.0)],
    ]
}

pub fn pauli_x() -> [[C64; 2]; 2] {
    let z = C64::new(0.0, 0.0);
    let o = C64::new(1.0, 0.0);
    [[z, o], [o, z]]
}

pub fn pauli_y() -> [[C64; 2]; 2] {
    let z = C64::new(0.0, 0.0);
    let i = C64::new(0.0, 1.0);
    let ni = C64::new(0.0, -1.0);
    [[z, ni], [i, z]]
}

pub fn pauli_z() -> [[C64; 2]; 2] {
    let z = C64::new(0.0, 0.0);
    let o = C64::new(1.0, 0.0);
    let m = C64::new(-1.0, 0.0);
    [[o, z], [z, m]]
}

pub fn rx(theta: f64) -> [[C64; 2]; 2] {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [
        [C64::new(c, 0.0), C64::new(0.0, -s)],
        [C64::new(0.0, -s), C64::new(c, 0.0)],
    ]
}

/// |00>→|00>, |01>→|01>, |10>→|11>, |11>→|10>
pub fn cnot() -> [[C64; 4]; 4] {
    let z = C64::new(0.0, 0.0);
    let o = C64::new(1.0, 0.0);
    [
        [o, z, z, z],
        [z, o, z, z],
        [z, z, z, o],
        [z, z, o, z],
    ]
}

/// diag(1, 1, 1, -1)
pub fn cz() -> [[C64; 4]; 4] {
    let z = C64::new(0.0, 0.0);
    let o = C64::new(1.0, 0.0);
    let m = C64::new(-1.0, 0.0);
    [
        [o, z, z, z],
        [z, o, z, z],
        [z, z, o, z],
        [z, z, z, m],
    ]
}
