use smolnn::*;
use smolmatrix::*;

const SAMPLES_SQRT: isize = 10;
const SAMPLES: isize = SAMPLES_SQRT * SAMPLES_SQRT;

fn main() {
    let learn_rate = 1.0;
    let rate = learn_rate / SAMPLES as f32 / SAMPLES as f32;

    let mut lf: Layer<2, 1> = Layer::new_zeroed();

    let mut expected = Vec::new();
    for x in (-SAMPLES_SQRT / 2)..SAMPLES_SQRT {
        let x = x as f32 / 10.0;

        for y in (-SAMPLES_SQRT / 2)..SAMPLES_SQRT {
            let y = y as f32 / 10.0;
            expected.push((f(x, y), vector!(2 [x, y])));
        }
    }

    loop {
        let mut r = Vec::new();
        for (_, x) in expected.iter() {
            r.push(lf.evaluate(x));
        }

        let mut c = 0.0;
        for (r, (s, x)) in r.into_iter().zip(expected.iter()) {
            let e = vector!(1 [*s]);
            c += costs::mse(r.clone(), &e)[(0, 0)];
            lf.back_prop(&x, vector!(1 [1.0]), &costs::mse_derivative(r, &e), rate);
        }

        println!("{}", c / SAMPLES as f32);

        if (c / SAMPLES as f32) < 0.01 {
            break;
        }

        std::thread::sleep_ms(100);
    }

    println!("{}", lf.weights);
    println!("{}\n\n", lf.biases);
}

fn f(x: f32, y: f32) -> f32 {
    x * 6.0 + y * 9.0 + 4.2
}
