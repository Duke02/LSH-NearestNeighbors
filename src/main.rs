use std::fs::File;
use std::io::Read;
use std::path::Path;
use crate::lshash::LSHash;
use crate::lshashtable::LSHashTable;

mod lshash;
mod lshashtable;
mod utils;

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

fn test_neighbors<H: LSHash>(collection: &LSHashTable<H>, i: usize, embedding: &Vec<f32>) {
    let neighbors = collection.nearest_neighbors(embedding);
    match neighbors {
        None => println!("No neighbors found for query #{i}!"),
        Some(n) => println!("{} neighbors found for query #{i}!", n.len()),
    }
}

fn test_k_neighbors<H: LSHash>(collection: &LSHashTable<H>, i: usize, embedding: &Vec<f32>) {
    let k = 5;
    let neighbors = collection.top_k_neighbors(embedding, k);
    match neighbors {
        None => println!("No neighbors found for query #{i} with k={k}!"),
        Some(n) => println!("{} neighbors found for query #{i} with k={k}!", n.len()),
    }
}

fn test_closest_neighbor<H: LSHash>(collection: &LSHashTable<H>, i: usize, embedding: &Vec<f32>) {
    let closest_neighbor = collection.closest_neighbor(embedding);
    match closest_neighbor {
        None => println!("No closest neighbor found for query #{i}!"),
        Some(c) => println!(
            "Closest neighbor to query #{i} had a sum of {}!",
            c.iter().sum::<f32>()
        ),
    }
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
    const NUM_BITS: u8 = 16;

    println!("Gathering data...");
    let euclidean_data = convert_doc_to_f32(
        &input.lines().into_iter().map(|s| s.to_string()).collect(),
        MAX_LENGTH,
    );
    let cosine_data = euclidean_data.clone();

    println!("Creating hashers and collections...");
    let euclidean_hasher = lshash::EuclideanHash::new_with_optimal_bin_width(NUM_BITS, &euclidean_data[0..f32::sqrt(euclidean_data.len() as f32) as usize].to_vec());
    let mut euclidean_collection = LSHashTable::new(euclidean_hasher, NUM_BITS);

    let cosine_hasher = lshash::CosineHash::new(NUM_BITS, MAX_LENGTH);
    let mut cosine_collection = LSHashTable::new(cosine_hasher, NUM_BITS);

    println!("Inserting...");
    euclidean_collection.bulk_insert(euclidean_data);
    cosine_collection.bulk_insert(cosine_data);

    let query_string = "Romeo, Romeo, where you at, Romeo? \
Did you just flip me off, my guy? \
Yeah and? What you gonna do about it, scrub? \
But daddy I love him!";

    let query_embeddings = convert_doc_to_f32(&query_string.to_string(), MAX_LENGTH);

    println!("Euclidean Collection has a total size of {}", euclidean_collection.len());
    println!("Cosine Collection has a total size of {}", cosine_collection.len());

    for (i, embedding) in query_embeddings.iter().enumerate() {
        println!("Euclidean");
        test_neighbors(&euclidean_collection, i, &embedding);
        println!("Cosine");
        test_neighbors(&cosine_collection, i, &embedding);

        println!("Euclidean");
        test_k_neighbors(&euclidean_collection, i, &embedding);
        println!("Cosine");
        test_k_neighbors(&cosine_collection, i, &embedding);

        println!("Euclidean");
        test_closest_neighbor(&euclidean_collection, i, &embedding);
        println!("Cosine");
        test_closest_neighbor(&cosine_collection, i, &embedding);
    }
}
