use smolmatrix::*;
use smolnn::*;

const IN: usize = 784;
const L0: usize = 256;
const L1: usize = 64;
const OUT: usize = 10;

const LEARNING_RATE: f32 = 0.001;

pub struct Model {
    l0: Box<Layer<IN, L0>>,
    l1: Box<Layer<L0, L1>>,
    l2: Box<Layer<L1, OUT>>,

    l0_opt: Box<optimizers::Adam<IN, L0>>,
    l1_opt: Box<optimizers::Adam<L0, L1>>,
    l2_opt: Box<optimizers::Adam<L1, OUT>>,
}

impl Model {
    pub fn new() -> Self {
        Self {
            l0: Box::new(Layer::new_randomized()),
            l1: Box::new(Layer::new_randomized()),
            l2: Box::new(Layer::new_randomized()),

            l0_opt: Box::new(optimizers::adam(LEARNING_RATE, 0.9, 0.999)),
            l1_opt: Box::new(optimizers::adam(LEARNING_RATE, 0.9, 0.999)),
            l2_opt: Box::new(optimizers::adam(LEARNING_RATE, 0.9, 0.999)),
        }
    }

    pub fn evaluate(&self, i: &Vector<IN>) -> Vector<OUT> {
        let l0 = activations::tanh(self.l0.evaluate(i));
        let l1 = activations::tanh(self.l1.evaluate(&l0));
        let l2 = activations::stable_softmax(self.l2.evaluate(&l1));

        l2
    }

    pub fn feed(&self, epoch: &mut Epoch, i: &Vector<IN>, t: u8) {
        let mut t = one_at(t);

        let z0 = self.l0.evaluate(i);
        let a0 = activations::tanh(z0.clone());
        let z1 = self.l1.evaluate(&a0);
        let a1 = activations::tanh(z1.clone());
        let z2 = self.l2.evaluate(&a1);
        let _a2 = activations::stable_softmax(z2.clone());

        let actv_der_0 = activations::tanh_derivative(z0.clone());
        let actv_der_1 = activations::tanh_derivative(z1.clone());
        let actv_der_2 = activations::stable_softmax_derivative(z2.clone(), &t);

        let cost_der = activations::softmax_cost(z2.clone(), &mut t);
        epoch.c += cost_der.inner.iter().flatten().copied().sum::<f32>();
        drop(t);

        let cost_der = self.l2.back_prop(&mut epoch.l2, &a1, actv_der_2.clone(), &cost_der, &actv_der_1);
        let cost_der = self.l1.back_prop(&mut epoch.l1, &a0, actv_der_1.clone(), &cost_der, &actv_der_0);
        self.l0.back_prop(&mut epoch.l0, i, actv_der_0.clone(), &cost_der, &Vector::new_zeroed());
    }

    pub fn apply(&mut self, mut epoch: Epoch) -> f32 {
        self.l0.apply_in_place(&mut epoch.l0, 1.0 / crate::BATCH_SIZE as f32, &mut self.l0_opt);
        self.l1.apply_in_place(&mut epoch.l1, 1.0 / crate::BATCH_SIZE as f32, &mut self.l1_opt);
        self.l2.apply_in_place(&mut epoch.l2, 1.0 / crate::BATCH_SIZE as f32, &mut self.l2_opt);

        epoch.c / crate::BATCH_SIZE as f32
    }
}

pub struct Epoch {
    l0: Box<BackPropAcc<IN, L0>>,
    l1: Box<BackPropAcc<L0, L1>>,
    l2: Box<BackPropAcc<L1, OUT>>,
    c: f32,
}

impl Epoch {
    pub fn new() -> Self {
        Self {
            l0: Box::new(BackPropAcc::new()),
            l1: Box::new(BackPropAcc::new()),
            l2: Box::new(BackPropAcc::new()),
            c: 0.0,
        }
    }
}

fn one_at(id: u8) -> Vector<OUT> {
    let mut v = Vector::new_zeroed();
    v[(0, id as usize)] = 1.0;
    v
}
