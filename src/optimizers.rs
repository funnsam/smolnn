use smolmatrix::*;

pub trait Optimizer<const I: usize, const O: usize> {
    fn update_weights(&mut self, p: &mut Matrix<I, O>, g: &mut Matrix<I, O>);
    fn update_biases(&mut self, p: &mut Vector<O>, g: &mut Vector<O>);
}

#[cfg(feature = "alloc")]
impl<const I: usize, const O: usize, Opt: Optimizer<I, O>> Optimizer<I, O> for alloc::boxed::Box<Opt> {
    fn update_weights(&mut self, p: &mut Matrix<I, O>, g: &mut Matrix<I, O>) {
        Opt::update_weights(self, p, g)
    }

    fn update_biases(&mut self, p: &mut Vector<O>, g: &mut Vector<O>) {
        Opt::update_biases(self, p, g)
    }
}

pub struct Sgd(f32);
impl<const I: usize, const O: usize> Optimizer<I, O> for Sgd {
    fn update_weights(&mut self, p: &mut Matrix<I, O>, g: &mut Matrix<I, O>) {
        *g *= self.0;
        *p -= &*g;
    }

    fn update_biases(&mut self, p: &mut Vector<O>, g: &mut Vector<O>) {
        *g *= self.0;
        *p -= &*g;
    }
}

pub fn sgd(learning_rate: f32) -> Sgd {
    Sgd(learning_rate)
}

pub struct SgdMomentum<const I: usize, const O: usize> {
    alpha: f32,
    beta: f32,
    v_weight: Matrix<I, O>,
    v_bias: Vector<O>,
}

impl<const I: usize, const O: usize> Optimizer<I, O> for SgdMomentum<I, O> {
    fn update_weights(&mut self, p: &mut Matrix<I, O>, g: &mut Matrix<I, O>) {
        *g *= self.alpha;
        self.v_weight *= self.beta;
        self.v_weight += &*g;
        *p -= &self.v_weight;
    }

    fn update_biases(&mut self, p: &mut Vector<O>, g: &mut Vector<O>) {
        *g *= self.alpha;
        self.v_bias *= self.beta;
        self.v_bias += &*g;
        *p -= &self.v_bias;
    }
}

pub fn sgd_momentum<const I: usize, const O: usize>(
    learning_rate: f32,
    beta: f32,
) -> SgdMomentum<I, O> {
    SgdMomentum {
        alpha: learning_rate * (1.0 - beta),
        beta,
        v_weight: Matrix::new_zeroed(),
        v_bias: Matrix::new_zeroed(),
    }
}

#[derive(Debug)]
pub struct Adam<const I: usize, const O: usize> {
    alpha: f32,
    beta1: f32,
    beta2: f32,
    m_weight: Matrix<I, O>,
    m_bias: Vector<O>,
    v_weight: Matrix<I, O>,
    v_bias: Vector<O>,
    t: i32,
    epsilon: f32,
}

impl<const I: usize, const O: usize> Optimizer<I, O> for Adam<I, O> {
    fn update_weights(&mut self, p: &mut Matrix<I, O>, g: &mut Matrix<I, O>) {
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

        self.m_weight *= self.beta1;
        let g2 = g.clone() * (1.0 - self.beta1);
        self.m_weight += &g2;

        self.v_weight *= self.beta2;
        g.map_each_in_place(|i| *i *= *i);
        *g *= 1.0 - self.beta2;
        self.v_weight += &*g;

        let m_hat = self.m_weight.clone() / (1.0 - self.beta1.powi(self.t));
        let mut v_hat = self.v_weight.clone() / (1.0 - self.beta2.powi(self.t));

        // alpha already updated
        self.t += 1;

        v_hat.map_each_in_place(|i| *i = i.sqrt());
            *p -= &((m_hat * self.alpha) / &(v_hat + self.epsilon));
    }

    fn update_biases(&mut self, p: &mut Vector<O>, g: &mut Vector<O>) {
        self.m_bias *= self.beta1;
        let g2 = g.clone() * (1.0 - self.beta1);
        self.m_bias += &g2;

        self.v_bias *= self.beta2;
        g.map_each_in_place(|i| *i *= *i);
        *g *= 1.0 - self.beta2;
        self.v_bias += &*g;

        let m_hat = self.m_bias.clone() / (1.0 - self.beta1.powi(self.t));
        let mut v_hat = self.v_bias.clone() / (1.0 - self.beta2.powi(self.t));

        // self.alpha = (self.alpha * (1.0 - self.beta2.powi(self.t)).sqrt()) / (1.0 - self.beta1.powi(self.t));
        // t should update later

        v_hat.map_each_in_place(|i| *i = i.sqrt());
        *p -= &((m_hat * self.alpha) / &(v_hat + self.epsilon));
    }
}

pub fn adam<const I: usize, const O: usize>(
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
) -> Adam<I, O> {
    Adam {
        alpha: learning_rate,
        beta1,
        beta2,
        m_weight: Matrix::new_zeroed(),
        m_bias: Matrix::new_zeroed(),
        v_weight: Matrix::new_zeroed(),
        v_bias: Matrix::new_zeroed(),
        t: 1,
        epsilon: 1e-7,
    }
}
