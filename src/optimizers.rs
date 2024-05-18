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

struct Adam<const I: usize, const O: usize> {
    alpha: f32,
    beta1: f32,
    beta2: f32,
    m_weight: Matrix<I, O>,
    m_bias: Vector<O>,
    v_weight: Matrix<I, O>,
    v_bias: Vector<O>,
    t: i32,
}

impl<const I: usize, const O: usize> Optimizer<I, O> for Adam<I, O> {
    fn update_weights(&mut self, p: Matrix<I, O>, g: Matrix<I, O>) -> Matrix<I, O> {
        // LIGHT:
        // m^t = β_1 m^(t-1) + (1-β_1) g^t
        // v^t = β_2 v^(t-1) + (1-β_2) (g^t o g^t)
        // ~
        // ^m^t = m^t / 1-β_1^t
        // ^v^t = v^t / 1-β_2^t
        // ~
        //        α^t-1 √(1-β_2^t)
        // α^t = ------------------
        //           1 - β_1^t
        // ~
        //                   α^t ^m^t
        // θ^t = θ^(t-1) - -----------
        //                  √^v^t + ε

        self.m_weight = self.m_weight.clone() * self.beta1 + &(g.clone() * (1.0 - self.beta1));
        self.v_weight = self.v_weight.clone() * self.beta2 + &(g.map_each(|i| *i *= *i) * (1.0 - self.beta2));
        let m_hat = self.m_weight.clone() / (1.0 - self.beta1.powi(self.t));
        let v_hat = self.v_weight.clone() / (1.0 - self.beta2.powi(self.t));
        self.alpha = (self.alpha * (1.0 - self.beta2.powi(self.t)).sqrt()) / (1.0 - self.beta1.powi(self.t));
        self.t += 1;
        p - &((m_hat * self.alpha) / &(v_hat.map_each(|i| *i = i.sqrt()) + f32::EPSILON))
    }

    fn update_biases(&mut self, p: Vector<O>, g: Vector<O>) -> Vector<O> {
        self.m_bias = self.m_bias.clone() * self.beta1 + &(g.clone() * (1.0 - self.beta1));
        self.v_bias = self.v_bias.clone() * self.beta2 + &(g.map_each(|i| *i *= *i) * (1.0 - self.beta2));
        let m_hat = self.m_bias.clone() / (1.0 - self.beta1.powi(self.t));
        let v_hat = self.v_bias.clone() / (1.0 - self.beta2.powi(self.t));
        self.alpha = (self.alpha * (1.0 - self.beta2.powi(self.t)).sqrt()) / (1.0 - self.beta2.powi(self.t));
        self.t += 1;
        p - &((m_hat * self.alpha) / &(v_hat.map_each(|i| *i = i.sqrt()) + f32::EPSILON))
    }
}

pub fn adam<const I: usize, const O: usize>(
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
) -> impl Optimizer<I, O> {
    Adam {
        alpha: learning_rate,
        beta1,
        beta2,
        m_weight: Matrix::new_zeroed(),
        m_bias: Matrix::new_zeroed(),
        v_weight: Matrix::new_zeroed(),
        v_bias: Matrix::new_zeroed(),
        t: 1,
    }
}
