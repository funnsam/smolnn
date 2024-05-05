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

    pub fn evaluate(&self, i: &Vector<IN>) -> Vector<OUT> {
        &self.weights * i + &self.biases
    }

    pub fn back_prop(&mut self, i: &Vector<IN>, act_der: Vector<OUT>, cost_der: &Vector<OUT>, rate: f32) {
        let dc_db = act_der * cost_der;
        let dc_dw = i * &dc_db.transpose();

        self.biases = self.biases.clone() - &(dc_db * rate);
        self.weights = self.weights.clone() - &(dc_dw.transpose() * rate);
    }
}
