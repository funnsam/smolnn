use std::io;
use smolmatrix::*;

pub fn read_images<R: io::Read>(b: &mut R) -> io::Result<Vec<Vector<784>>> {
    b.read_exact(&mut [0; 4])?;

    let mut size = [0; 4];
    b.read_exact(&mut size)?;

    let mut images = Vec::with_capacity(u32::from_be_bytes(size) as usize);

    b.read_exact(&mut [0; 8])?;

    let mut buf = [0; 784];
    while let Ok(()) = b.read_exact(&mut buf) {
        let mut v = Vector::new_zeroed();

        for (yi, y) in buf.chunks(28).enumerate() {
            for (xi, i) in y.iter().enumerate() {
                v[(0, xi + yi * 28)] = *i as f32 / 255.0;
            }
        }

        images.push(v);
    }

    Ok(images)
}

pub fn read_labels<R: io::Read>(b: &mut R) -> io::Result<Vec<u8>> {
    b.read_exact(&mut [0; 4])?;

    let mut size = [0; 4];
    b.read_exact(&mut size)?;

    let mut labels = Vec::with_capacity(u32::from_be_bytes(size) as usize);
    let mut buf = [0; 1];

    while let Ok(()) = b.read_exact(&mut buf) {
        labels.push(buf[0]);
    }

    Ok(labels)
}
