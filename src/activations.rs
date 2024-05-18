use smolmatrix::*;

pub fn linear<const S: usize>(v: Vector<S>) -> Vector<S> {
    v
}

pub fn linear_derivative<const S: usize>(_v: Vector<S>) -> Vector<S> {
    Vector { inner: [[1.0]; S] }
}

pub fn relu<const S: usize>(mut v: Vector<S>) -> Vector<S> {
    for y in 0..S {
        v[(0, y)] = v[(0, y)].max(0.0);
    }

    v
}

pub fn relu_derivative<const S: usize>(mut v: Vector<S>) -> Vector<S> {
    for y in 0..S {
        v[(0, y)] = v[(0, y)].signum().max(0.0);
    }

    v
}

pub fn tanh<const S: usize>(mut v: Vector<S>) -> Vector<S> {
    for y in 0..S {
        v[(0, y)] = v[(0, y)].tanh();
    }

    v
}

pub fn tanh_derivative<const S: usize>(mut v: Vector<S>) -> Vector<S> {
    for y in 0..S {
        let tanh = v[(0, y)].tanh();
        v[(0, y)] = 1.0 - tanh * tanh;
    }

    v
}

pub fn softmax<const S: usize>(mut v: Vector<S>) -> Vector<S> {
    let mut sum = 0.0;

    for y in 0..S {
        let yv = v[(0, y)].exp();
        sum += yv;
        v[(0, y)] = yv;
    }

    for y in 0..S {
        v[(0, y)] /= sum;
    }

    v
}

pub fn softmax_derivative<const S: usize>(v: Vector<S>, t: &Vector<S>) -> Vector<S> {
    softmax(v) - t
}

pub fn softmax_cost<const S: usize>() -> Vector<S> { // HACK: wacky simple solution
    Vector { inner: [[1.0]; S] }
}
