pub fn parameter_shift<F>(theta: f64, energy_fn: &F) -> f64
where
    F: Fn(f64) -> f64,
{
    let shift = std::f64::consts::FRAC_PI_2;
    0.5 * (energy_fn(theta + shift) - energy_fn(theta - shift))
}
