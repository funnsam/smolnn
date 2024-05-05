use smolmatrix::*;

pub fn linear<const S: usize>(v: Vector<S>) -> Vector<S> {
    v
}

pub fn linear_derivative<const S: usize>(_v: Vector<S>) -> Vector<S> {
    Vector { inner: [[1.0]; S], }
}

pub fn relu<const S: usize>(mut v: Vector<S>) -> Vector<S> {
    for y in 0..S {
        v[(0, y)] = v[(0, y)].max(0.0);
    }

    v
}
