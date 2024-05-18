use smolmatrix::*;
use smolnn::*;

const SAMPLES_SQRT: isize = 10;
const SAMPLES: isize = SAMPLES_SQRT * SAMPLES_SQRT;

fn main() {
    let learn_rate = 1.0;
    let rate = learn_rate / SAMPLES as f32 / SAMPLES as f32;

    let mut l0: Layer<2, 16> = Layer::new_randomized();
    let mut lf: Layer<16, 2> = Layer::new_randomized();

    let mut expected = Vec::new();
    for x in (-SAMPLES_SQRT / 2)..SAMPLES_SQRT {
        let x = x as f32 / SAMPLES_SQRT as f32;

        for y in (-SAMPLES_SQRT / 2)..SAMPLES_SQRT {
            let y = y as f32 / SAMPLES_SQRT as f32;
            expected.push((f(x, y), vector!(2 [x, y])));
        }
    }

    // let mut l0_opt = optimizers::sgd(rate);
    // let mut lf_opt = optimizers::sgd(rate);
    // let mut l0_opt = optimizers::sgd_momentum(rate, 0.9);
    // let mut lf_opt = optimizers::sgd_momentum(rate, 0.9);

    for i in 1.. {
        let mut r = Vec::new();
        for (_, x) in expected.iter() {
            let r0 = activations::relu(l0.evaluate(x));
            let rf = activations::linear(lf.evaluate(&r0));
            r.push((r0, rf));
        }

        let mut c = 0.0;
        for ((r0, rf), (e, x)) in r.into_iter().zip(expected.iter()) {
            c += costs::mse(rf.clone(), e);
            let cost_der = costs::squared_error_derivative(rf.clone(), e);

            let actv_der_0 = activations::relu_derivative(r0.clone());
            let actv_der_f = activations::linear_derivative(rf);

            // let cost_der = lf.back_prop(&r0, actv_der_f.clone(), &cost_der, &actv_der_0, &mut lf_opt);
            // l0.back_prop(x, actv_der_0, &cost_der, &actv_der_f, &mut l0_opt);
        }

        println!("{i:>5} {}", c / SAMPLES as f32);

        if !c.is_finite() || (c / SAMPLES as f32) < 0.2 {
            break;
        }
    }

    println!("{}", l0.weights);
    println!("{}", l0.biases);
    println!("{}", lf.weights);
    println!("{}", lf.biases);
}

fn f(x: f32, y: f32) -> Vector<2> {
    vector!(2 [
        (x * 6.0).min(0.0),
        (y * 9.0).min(0.0),
    ])
}
