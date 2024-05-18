use smolmatrix::*;

pub trait Optimizer<const I: usize, const O: usize> {
    fn update_weights(&mut self, p: Matrix<I, O>, g: Matrix<I, O>) -> Matrix<I, O>;
    fn update_biases(&mut self, p: Vector<O>, g: Vector<O>) -> Vector<O>;
}

struct Sgd(f32);
impl<const I: usize, const O: usize> Optimizer<I, O> for Sgd {
    fn update_weights(&mut self, p: Matrix<I, O>, g: Matrix<I, O>) -> Matrix<I, O> {
        p - &(g * self.0)
    }

    fn update_biases(&mut self, p: Vector<O>, g: Vector<O>) -> Vector<O> {
        p - &(g * self.0)
    }
}

pub fn sgd<const I: usize, const O: usize>(learning_rate: f32) -> impl Optimizer<I, O> {
    Sgd(learning_rate)
}

struct SgdMomentum<const I: usize, const O: usize> {
    alpha: f32,
    beta: f32,
    v_weight: Matrix<I, O>,
    v_bias: Vector<O>,
}

impl<const I: usize, const O: usize> Optimizer<I, O> for SgdMomentum<I, O> {
    fn update_weights(&mut self, p: Matrix<I, O>, g: Matrix<I, O>) -> Matrix<I, O> {
        self.v_weight = self.v_weight.clone() * self.beta + &(g * self.alpha);
        p - &self.v_weight
    }

    fn update_biases(&mut self, p: Vector<O>, g: Vector<O>) -> Vector<O> {
        self.v_bias = self.v_bias.clone() * self.beta + &(g * self.alpha);
        p - &self.v_bias
    }
}

pub fn sgd_momentum<const I: usize, const O: usize>(
    learning_rate: f32,
    beta: f32,
) -> impl Optimizer<I, O> {
    SgdMomentum {
        alpha: learning_rate * (1.0 - beta),
        beta,
        v_weight: Matrix::new_zeroed(),
        v_bias: Matrix::new_zeroed(),
    }
}
