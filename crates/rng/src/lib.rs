use sha3::{digest::{ExtendableOutput, Update, XofReader}, Shake256};

pub struct ONDRng {
    state: [u8; 32],
    step: u64,
}

impl ONDRng {
    pub fn new(seed: &[u8]) -> Self {
        let mut state = [0u8; 32];
        shake(&[seed, b"OND_INIT"], &mut state);
        Self { state, step: 0 }
    }

    pub fn next_f64(&mut self, ctx: &[u8]) -> f64 {
        self.step += 1;

        let state = self.state;
        let step_bytes = self.step.to_be_bytes();
        let mut next_state = self.state;
        shake(&[&state, &step_bytes, b"QSIM"], &mut next_state);
        self.state = next_state;

        let mut out = [0u8; 8];
        shake(&[&self.state, ctx], &mut out);

        if self.state[0] < 16 {
            let state = self.state;
            let mut next_state = self.state;
            shake(&[&state, b"SKIP"], &mut next_state);
            self.state = next_state;
        }

        (u64::from_be_bytes(out) as f64) / (u64::MAX as f64)
    }
}

fn shake(parts: &[&[u8]], out: &mut [u8]) {
    let mut h = Shake256::default();
    for p in parts {
        h.update(p);
    }
    let mut r = h.finalize_xof();
    r.read(out);
}
