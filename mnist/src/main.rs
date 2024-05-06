use smolmatrix::*;

mod reader;

fn main() {
    let images = std::fs::read("db/t10k-images-idx3-ubyte").unwrap();
    let images = reader::read_images(&mut images.as_slice()).unwrap();
    let labels = std::fs::read("db/t10k-labels-idx1-ubyte").unwrap();
    let labels = reader::read_labels(&mut labels.as_slice()).unwrap();

    for i in 0.. {
        visualize(&images[i]);
        println!("{}", labels[i]);
        std::thread::sleep_ms(1000);
    }
}

fn visualize(fb: &Vector<784>) {
    for yhalf in 0..28/2 {
        for x in 0..28 {
            plot(fb[(0, x + yhalf * 56)], fb[(0, x + yhalf * 56 + 28)]);
        }

        println!("\x1b[0m");
    }
}

fn plot(a: f32, b: f32) {
    let a = (a * 255.0) as u8;
    let b = (b * 255.0) as u8;
    print!("\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m▀", a, a, a, b, b, b);
}
