use std::fs::File;
use std::io::Read;
use std::path::Path;

use itertools::Itertools;

use crate::lsh::Lsh;

mod lsh;

fn z_score(vec: &Vec<f32>) -> Vec<f32> {
    let mean = vec.iter().sum::<f32>() / vec.len() as f32;
    let std = f32::sqrt(
        vec.iter().map(|f| f32::powi(f - mean, 2)).sum::<f32>() / (vec.len() as f32 - 1.0),
    );
    vec.iter().map(|f| (f - mean) / std).collect()
}

fn convert_doc_to_f32(text: &String, max_len: usize) -> Vec<Vec<f32>> {
    let mut out: Vec<Vec<f32>> =
        z_score(&text.as_str().as_bytes().iter().map(|&b| b as f32).collect())
            .chunks(max_len)
            .map(|chunk| chunk.to_vec())
            .collect();

    // Add padding.
    if out.last().unwrap().len() < max_len {
        out.last_mut().unwrap().resize(max_len, 0.0);
    }
    out
}

fn main() {
    let input_path = Path::new("./data/romeo_juliet.txt");
    let mut input_file = match File::open(&input_path) {
        Ok(file) => file,
        Err(e) => panic!(
            "Error opening file at {}: {e}",
            input_path.to_str().unwrap()
        ),
    };
    let mut input = String::new();
    input_file.read_to_string(&mut input).unwrap();

    const MAX_LENGTH: usize = 10;
    const NUM_BITS: u8 = 8;

    let data = convert_doc_to_f32(
        &input.lines().into_iter().map(|s| s.to_string()).collect(),
        MAX_LENGTH,
    );

    let mut collection = Lsh::new(MAX_LENGTH, NUM_BITS);

    for embedding in data {
        collection.insert(embedding);
    }




}
