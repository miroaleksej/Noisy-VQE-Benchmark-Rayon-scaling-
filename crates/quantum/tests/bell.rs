use quantum::{apply_cnot, gates::hadamard, measurement::measure_z};
use rng::ONDRng;
use tn::{mps::MPS, truncation::Truncation};

#[test]
fn bell_state_z_correlation() {
    let trunc = Truncation {
        max_bond: 8,
        cutoff: 1e-12,
    };

    let mut counts = [[0usize; 2]; 2];

    for shot in 0..100 {
        let mut rng = ONDRng::new(format!("seed-{}", shot).as_bytes());
        let mut psi = MPS::new_zero(2);

        psi.apply_1q(0, hadamard());
        apply_cnot(&mut psi, 0, trunc);

        let m0 = measure_z(&mut psi, 0, &mut rng);
        let m1 = measure_z(&mut psi, 1, &mut rng);

        counts[m0 as usize][m1 as usize] += 1;
    }

    assert_eq!(counts[0][1], 0, "Found |01> in Bell state");
    assert_eq!(counts[1][0], 0, "Found |10> in Bell state");

    assert!(
        counts[0][0] > 0,
        "Never observed |00>, counts = {:?}",
        counts
    );
    assert!(
        counts[1][1] > 0,
        "Never observed |11>, counts = {:?}",
        counts
    );
}
