use smolmatrix::*;
use smolnn::{optimizers::Optimizer, *};

const SAMPLES_SQRT: isize = 10;
const SAMPLES: isize = SAMPLES_SQRT * SAMPLES_SQRT;
const TARGET_COST: f32 = 0.002;

fn main() {
    let rate = 0.4;

    let l0 = Layer::new_randomized();

    let mut expected = Vec::new();
    for x in (-SAMPLES_SQRT / 2)..SAMPLES_SQRT {
        let x = x as f32 / SAMPLES_SQRT as f32;

        for y in (-SAMPLES_SQRT / 2)..SAMPLES_SQRT {
            let y = y as f32 / SAMPLES_SQRT as f32;
            expected.push((f(x, y), vector!(2 [x, y])));
        }
    }

    fn train_w_opt(l0: &Layer<2, 1>, expected: &[(Matrix<1, 1>, Matrix<1, 2>)], mut l0_opt: impl Optimizer<2, 1>) {
        let mut l0 = l0.clone();

        for i in 1.. {
            let mut r = Vec::new();
            for (_, x) in expected.iter() {
                let r0 = l0.evaluate(x);
                r.push(r0);
            }

            let mut c = 0.0;
            let mut bp0 = BackPropAcc::new();
            for (r0, (e, x)) in r.into_iter().zip(expected.iter()) {
                c += costs::mse(r0.clone(), e);
                let cost_der = costs::squared_error_derivative(r0.clone(), e);

                let actv_der_0 = activations::linear_derivative(r0.clone());

                l0.back_prop(&mut bp0, x, actv_der_0, &cost_der, &Vector::new_zeroed());
            }

            l0.apply(bp0, 1.0 / SAMPLES as f32, &mut l0_opt);

            println!("{i:>5} {}", c / SAMPLES as f32);

            if !c.is_finite() || (c / SAMPLES as f32) < TARGET_COST {
                break;
            }
        }

        println!("{}", l0.weights);
        println!("{}", l0.biases);
    }

    train_w_opt(&l0, &expected, optimizers::adam(rate, 0.9, 0.999));
    train_w_opt(&l0, &expected, optimizers::sgd_momentum(rate, 0.9));
    train_w_opt(&l0, &expected, optimizers::sgd(rate));
}

fn f(x: f32, y: f32) -> Vector<1> {
    vector!(1 [
        x * 6.9 + y * 4.2,
    ])
}
