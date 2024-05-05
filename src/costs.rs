use smolmatrix::*;

pub fn mse<const S: usize>(x: Vector<S>, e: &Vector<S>) -> Vector<S> {
    (x.clone() - e) * &(x - e)
}

pub fn mse_derivative<const S: usize>(x: Vector<S>, e: &Vector<S>) -> Vector<S> {
    (x - e) * 2.0
}
