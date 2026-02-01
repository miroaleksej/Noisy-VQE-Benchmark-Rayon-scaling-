use crate::grad::parameter_shift;

pub fn vqe_gradient<F>(mut theta: f64, energy_fn: F, lr: f64, steps: usize) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    for _ in 0..steps {
        let grad = parameter_shift(theta, &energy_fn);
        theta -= lr * grad;
    }

    let e = energy_fn(theta);
    (theta, e)
}
