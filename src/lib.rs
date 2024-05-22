#![no_std]

#![warn(
    clippy::complexity,
    clippy::correctness,
    clippy::perf,
    clippy::nursery,
    clippy::suspicious,
    clippy::style
)]
#![allow(
    clippy::semicolon_inside_block,
    clippy::just_underscores_and_digits,
    unknown_lints,
    cast_ref_to_mut,
    invalid_reference_casting,
    mutable_transmutes
)]

#[cfg(feature = "alloc")]
extern crate alloc;

use optimizers::Optimizer;
use smolmatrix::*;

pub mod activations;
pub mod costs;
pub mod optimizers;

#[derive(Clone, Debug)]
pub struct Layer<const IN: usize, const OUT: usize> {
    pub weights: Matrix<IN, OUT>,
    pub biases: Vector<OUT>,
}

impl<const IN: usize, const OUT: usize> Layer<IN, OUT> {
    pub const fn new_zeroed() -> Self {
        Self {
            weights: Matrix::new_zeroed(),
            biases: Matrix::new_zeroed(),
        }
    }

    #[cfg(feature = "alea")]
    pub fn new_randomized() -> Self {
        Self::new_each_as(|| alea::f32() - 0.5)
    }

    pub fn new_each_as<F: Fn() -> f32>(f: F) -> Self {
        let mut s = Self::new_zeroed();

        fn r<F: Fn() -> f32, const W: usize, const H: usize>(m: &mut Matrix<W, H>, f: &F) {
            for i in m.inner.iter_mut() {
                for j in i.iter_mut() {
                    *j = f();
                }
            }
        }

        r(&mut s.weights, &f);
        r(&mut s.biases, &f);

        s
    }

    pub fn evaluate(&self, i: &Vector<IN>) -> Vector<OUT> {
        &self.weights * i + &self.biases
    }

    pub fn back_prop(
        &self,
        back_prop: &mut BackPropAcc<IN, OUT>,
        i: &Vector<IN>,
        act_der: Vector<OUT>,
        cost_der: &Vector<OUT>,
        prev_act_der: &Vector<IN>,
    ) -> Vector<IN> {
        let dc_db = act_der.clone() * cost_der;
        let dc_dw = (&dc_db) * &i.transpose();

        back_prop.0.biases = back_prop.0.biases.clone() + &dc_db;
        back_prop.0.weights = back_prop.0.weights.clone() + &dc_dw;

        // LIGHT:
        // ┌  p             ┐
        // │  Σ  c'σ' w_i^1 │ σ' i_0
        // └ i=1            ┘

        (&(act_der * cost_der).transpose() * &self.weights).transpose() * prev_act_der * i
    }

    pub fn apply<Opt: Optimizer<IN, OUT>>(
        &mut self,
        bp: BackPropAcc<IN, OUT>,
        sf: f32,
        opt: &mut Opt,
    ) {
        opt.update_biases(&mut self.biases, &mut (bp.0.biases * sf));
        opt.update_weights(&mut self.weights, &mut (bp.0.weights * sf));
    }

    pub fn apply_in_place<Opt: Optimizer<IN, OUT>>(
        &mut self,
        bp: &mut BackPropAcc<IN, OUT>,
        sf: f32,
        opt: &mut Opt,
    ) {
        bp.0.biases *= sf;
        opt.update_biases(&mut self.biases, &mut bp.0.biases);
        bp.0.weights *= sf;
        opt.update_weights(&mut self.weights, &mut bp.0.weights);
    }
}

#[derive(Debug)]
pub struct BackPropAcc<const I: usize, const O: usize>(Layer<I, O>);
impl<const I: usize, const O: usize> Default for BackPropAcc<I, O> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const I: usize, const O: usize> BackPropAcc<I, O> {
    pub const fn new() -> Self {
        Self(Layer::new_zeroed())
    }
}
