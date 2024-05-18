use smolmatrix::*;
use smolnn::*;

const SAMPLES_SQRT: isize = 10;
const SAMPLES: isize = SAMPLES_SQRT * SAMPLES_SQRT;

fn main() {
    let learn_rate = 3e-4;

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

    let mut l0_opt = optimizers::adam(learn_rate, 0.9, 0.999);
    let mut lf_opt = optimizers::adam(learn_rate, 0.9, 0.999);

    for i in 1.. {
        let mut r = Vec::new();
        for (_, x) in expected.iter() {
            let r0 = activations::relu(l0.evaluate(x));
            let rf = activations::softmax(lf.evaluate(&r0));
            r.push((r0, rf));
        }

        let mut c = 0.0;
        let mut bp0 = BackPropAcc::new();
        let mut bpf = BackPropAcc::new();
        for ((r0, rf), (e, x)) in r.into_iter().zip(expected.iter()) {
            c += costs::mse(rf.clone(), e);
            // NOTE: normally you use this for the cost:
            // ```
            // let cost_der = costs::squared_error_derivative(rf.clone(), e);
            // ```
            // , however due to impl details, to calculate the cost derivative of a softmax layer
            // do this instead
            let cost_der = activations::softmax_cost();

            let actv_der_0 = activations::relu_derivative(r0.clone());
            let actv_der_f = activations::softmax_derivative(rf, e);

            let cost_der = lf.back_prop(&mut bpf, &r0, actv_der_f.clone(), &cost_der, &actv_der_0);
            l0.back_prop(&mut bp0, x, actv_der_0, &cost_der, &actv_der_f);
        }

        l0.apply(bp0, 1.0 / SAMPLES as f32, &mut l0_opt);
        lf.apply(bpf, 1.0 / SAMPLES as f32, &mut lf_opt);

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
    if x < y {
        vector!(2 [1.0, 0.0])
    } else {
        vector!(2 [0.0, 1.0])
    }
}
