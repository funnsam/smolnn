use smolmatrix::*;
use smolnn::*;

const IN: usize = 784;
const L0: usize = 256;
const L1: usize = 256;
const OUT: usize = 10;

const LEARNING_RATE: f32 = 3e-4;

pub struct Model {
    l0: Layer<IN, L0>,
    l1: Layer<L0, L1>,
    l2: Layer<L1, OUT>,

    l0_opt: optimizers::Adam<IN, L0>,
    l1_opt: optimizers::Adam<L0, L1>,
    l2_opt: optimizers::Adam<L1, OUT>,
}

impl Model {
    pub fn new() -> Self {
        Self {
            l0: Layer::new_randomized(),
            l1: Layer::new_randomized(),
            l2: Layer::new_randomized(),

            l0_opt: optimizers::adam(LEARNING_RATE, 0.9, 0.999),
            l1_opt: optimizers::adam(LEARNING_RATE, 0.9, 0.999),
            l2_opt: optimizers::adam(LEARNING_RATE, 0.9, 0.999),
        }
    }

    pub fn evaluate(&self, i: &Vector<IN>) -> Vector<OUT> {
        let l0 = activations::tanh(self.l0.evaluate(&i));
        let l1 = activations::tanh(self.l1.evaluate(&l0));
        let l2 = activations::tanh(self.l2.evaluate(&l1));

        l2
    }

    pub fn feed(&self, epoch: &mut Epoch, i: &Vector<IN>, t: &Vector<OUT>) {
        let l0 = activations::tanh(self.l0.evaluate(&i));
        let l1 = activations::tanh(self.l1.evaluate(&l0));
        let l2 = activations::tanh(self.l2.evaluate(&l1));
        epoch.c += costs::mse(l2.clone(), t);

        let cost_der = activations::softmax_cost();

        let actv_der_0 = activations::tanh_derivative(l0.clone());
        let actv_der_1 = activations::tanh_derivative(l1.clone());
        let actv_der_2 = activations::softmax_derivative(l2.clone(), t);

        let cost_der = self.l2.back_prop(&mut epoch.l2, &l1, actv_der_2.clone(), &cost_der, &actv_der_1);
        let cost_der = self.l1.back_prop(&mut epoch.l1, &l0, actv_der_1.clone(), &cost_der, &actv_der_0);
        self.l0.back_prop(&mut epoch.l0, i, actv_der_0.clone(), &cost_der, &Vector::new_zeroed());
    }

    pub fn apply(&mut self, epoch: Epoch) -> f32 {
        self.l0.apply(epoch.l0, 1.0 / crate::BATCH_SIZE as f32, &mut self.l0_opt);
        self.l1.apply(epoch.l1, 1.0 / crate::BATCH_SIZE as f32, &mut self.l1_opt);
        self.l2.apply(epoch.l2, 1.0 / crate::BATCH_SIZE as f32, &mut self.l2_opt);

        epoch.c / crate::BATCH_SIZE as f32
    }
}

pub struct Epoch {
    l0: BackPropAcc<IN, L0>,
    l1: BackPropAcc<L0, L1>,
    l2: BackPropAcc<L1, OUT>,
    c: f32,
}

impl Epoch {
    pub fn new() -> Self {
        Self {
            l0: BackPropAcc::new(),
            l1: BackPropAcc::new(),
            l2: BackPropAcc::new(),
            c: 0.0,
        }
    }
}
