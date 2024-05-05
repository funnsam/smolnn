use smolnn::*;
use smolmatrix::*;

const SAMPLES: isize = 10;

fn main() {
    let learn_rate = 0.8;
    let rate = learn_rate / SAMPLES as f32 / SAMPLES as f32;

    let mut lf: Layer<1, 1> = Layer::new_zeroed();

    let mut expected = Vec::new();
    for i in (-SAMPLES / 2)..SAMPLES {
        let x = i as f32 / 10.0;
        expected.push((f(x), x));
    }

    loop {
        let mut r = Vec::new();
        for (_, x) in expected.iter() {
            r.push(lf.evaluate(&vector!(1 [*x])));
        }

        let mut c = 0.0;
        for (r, (s, x)) in r.into_iter().zip(expected.iter()) {
            let i = vector!(1 [*x]);
            let e = vector!(1 [*s]);
            c += costs::mse(r.clone(), &e)[(0, 0)];
            lf.back_prop(&i, vector!(1 [1.0]), &costs::mse_derivative(r, &e), rate);
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

fn f(x: f32) -> f32 {
    x * 6.0 + 9.0
}
