#![no_std]

use smolmatrix::*;

pub mod activations;
pub mod costs;

pub struct Layer<const IN: usize, const OUT: usize> {
    pub weights: Matrix<IN, OUT>,
    pub biases: Vector<OUT>,
}

impl<const IN: usize, const OUT: usize> Layer<IN, OUT> {
    pub fn new_zeroed() -> Self {
        Self { weights: Matrix::new_zeroed(), biases: Matrix::new_zeroed() }
    }

    #[cfg(feature = "alea")]
    pub fn new_randomized() -> Self {
        let mut s = Self::new_zeroed();

        fn r<const W: usize, const H: usize>(m: &mut Matrix<W, H>) {
            for i in m.inner.iter_mut() {
                for j in i.iter_mut() {
                    *j = alea::f32();
                }
            }
        }

        r(&mut s.weights);
        r(&mut s.biases);

        s
    }

    pub fn evaluate(&self, i: &Vector<IN>) -> Vector<OUT> {
        &self.weights * i + &self.biases
    }

    pub fn back_prop(&mut self, i: &Vector<IN>, act_der: Vector<OUT>, cost_der: &Vector<OUT>, prev_act_der: &Vector<IN>, rate: f32) -> Vector<IN> {
    // pub fn back_prop(&mut self, i: &Vector<IN>, act_der: Vector<OUT>, cost_der: f32, rate: f32) {
        let dc_db = act_der.clone() * cost_der;
        let dc_dw = (&dc_db) * &i.transpose();

        self.biases = self.biases.clone() - &(dc_db * rate);
        self.weights = self.weights.clone() - &(dc_dw * rate);

        // LIGHT:
        //   p
        // ( Σ  c'σ' w_i^1) σ' i_0
        //  i=1
        //
        (&(act_der * cost_der).transpose() * &self.weights).transpose() * prev_act_der * i
    }
}
