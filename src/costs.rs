use smolmatrix::*;

pub fn mse<const S: usize>(x: Vector<S>, e: &Vector<S>) -> f32 {
    squared_error(x, e)
        .inner
        .into_iter()
        .map(|a| a[0])
        .sum::<f32>()
}

pub fn mse_derivative<const S: usize>(x: Vector<S>, e: &Vector<S>) -> f32 {
    squared_error_derivative(x, e)
        .inner
        .into_iter()
        .map(|a| a[0])
        .sum::<f32>()
}

pub fn squared_error<const S: usize>(x: Vector<S>, e: &Vector<S>) -> Vector<S> {
    (x.clone() - e) * &(x - e)
}

pub fn squared_error_derivative<const S: usize>(x: Vector<S>, e: &Vector<S>) -> Vector<S> {
    (x - e) * 2.0
}
