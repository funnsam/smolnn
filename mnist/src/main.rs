use smolmatrix::*;

use core::sync::atomic::*;

mod model;

// training params
const BATCH_SIZE: usize = 25;

// visualization params
static BAR_LENGTH: AtomicUsize = AtomicUsize::new(0);

mod reader;

fn main() {
    BAR_LENGTH.store(term_size::dimensions().unwrap().0 - 11, Ordering::Relaxed);

    let (images, labels) = reader::read_data("t10k", None).unwrap();
    let mut model = model::Model::new();

    for i in 1..=25 {
        let mut epoch = model::Epoch::new();

        for _ in 0..BATCH_SIZE {
            let id = alea::u64_in_range(0, images.len() as u64) as usize;
            model.feed(&mut epoch, &images[id], labels[id]);
        }

        println!("{i:>5} {}", model.apply(epoch));
    }

    for (i, l) in images.iter().zip(labels.iter()) {
        visualize(i);
        println!("Expected: {}", l);

        let f = model.evaluate(i);
        let p = f
            .inner
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1[0]
                    .partial_cmp(&b.1[0])
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .unwrap();

        println!("Predicted: {} ({:.1}%)", p.0, p.1[0] * 100.0);
        bar(&f);
        std::thread::sleep_ms(1000);
    }
}

// fn visualize_first_ten(t: &str) {
//     let (images, labels) = reader::read_data(t, Some(10)).unwrap();

//     for i in 0..images.len() {
//         println!("Correct: {}", labels[i]);
//         visualize(&images[i]);
//         std::thread::sleep_ms(1000);
//     }

//     println!("First 10 of {t} end");
// }

fn visualize(fb: &Vector<784>) {
    for yhalf in 0..28 / 2 {
        for x in 0..28 {
            plot(fb[(0, x + yhalf * 56)], fb[(0, x + yhalf * 56 + 28)]);
        }

        println!("\x1b[0m");
    }
}

fn plot(a: f32, b: f32) {
    let a = (a * 255.0) as u8;
    let b = (b * 255.0) as u8;
    print!("\x1b[38;2;{a};{a};{a}m\x1b[48;2;{b};{b};{b}m▀");
}

fn bar(v: &Vector<10>) {
    for (i, [v]) in v.inner.iter().enumerate() {
        let len = ((v + 1.0).log2() * BAR_LENGTH.load(Ordering::Relaxed) as f32).floor() as usize;
        println!(
            "{i} {:━<len$}{:<2$} ({3:.1}%)",
            "",
            "",
            BAR_LENGTH.load(Ordering::Relaxed) - len,
            v * 100.0
        );
    }
}
