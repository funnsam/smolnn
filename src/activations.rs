use smolmatrix::*;

pub fn linear<const S: usize>(v: Vector<S>) -> Vector<S> {
    v
}

pub fn linear_derivative<const S: usize>(_v: Vector<S>) -> Vector<S> {
    Vector { inner: [[1.0]; S] }
}

pub fn relu<const S: usize>(v: Vector<S>) -> Vector<S> {
    v.map_each(|v| *v = v.max(0.0))
}

pub fn relu_derivative<const S: usize>(v: Vector<S>) -> Vector<S> {
    v.map_each(|v| *v = v.signum().max(0.0))
}

pub fn tanh<const S: usize>(v: Vector<S>) -> Vector<S> {
    v.map_each(|v| *v = v.tanh())
}

pub fn tanh_derivative<const S: usize>(v: Vector<S>) -> Vector<S> {
    v.map_each(|v| *v = 1.0 - v.tanh().powi(2))
}

pub fn stable_softmax<const S: usize>(mut v: Vector<S>) -> Vector<S> {
    let max = v.inner.iter().flatten().max_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)).unwrap();
    v -= *max;
    softmax(v)
}

pub fn stable_softmax_derivative<const S: usize>(v: Vector<S>, t: &Vector<S>) -> Vector<S> {
    stable_softmax(v) - t
}

pub fn softmax<const S: usize>(mut v: Vector<S>) -> Vector<S> {
    v.map_each_in_place(|i| *i = i.exp());
    v /= v.inner.iter().flatten().sum::<f32>();
    v
}

pub fn softmax_derivative<const S: usize>(v: Vector<S>, t: &Vector<S>) -> Vector<S> {
    softmax(v) - t
}

pub fn softmax_cost<const S: usize>(v: Vector<S>, t: &mut Vector<S>) -> Vector<S> {
    t.map_each_in_place(|i| *i = (*i + 1e-15).ln());
    -(v * &*t)
}
