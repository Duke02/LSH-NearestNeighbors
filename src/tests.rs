#![feature(test)]

extern crate test;

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;
    use rand::distributions::Standard;
    use rand::Rng;
    use crate::lshash::{CosineHash, EuclideanHash, LSHash};
    use crate::lshashtable::LSHashTable;
    use crate::utils::to_unit_vector;

    #[bench]
    fn bench_euclidean_hash(b: &mut Bencher) {
        let hasher = EuclideanHash::new(8, 4, 0.5);
        b.iter(|| hasher.hash(&vec![0.1, 0.2, 1.5, -0.9]));
    }

    #[bench]
    fn bench_cosine_hash(b: &mut Bencher) {
        let hasher = CosineHash::new(8, 4);
        b.iter(|| hasher.hash(&vec![0.1, 0.2, 1.5, -0.9]));
    }

    #[bench]
    fn bench_euclidean_distance(b: &mut Bencher) {
        let hasher = EuclideanHash::new(8, 4, 0.5);
        b.iter(|| hasher.distance(&vec![0.1, 0.2, 1.5, -0.9], &vec![0.1, 0.2, 1.5, -0.9]));
    }

    #[bench]
    fn bench_cosine_distance(b: &mut Bencher) {
        let hasher = CosineHash::new(8, 4);
        b.iter(|| hasher.distance(&vec![0.1, 0.2, 1.5, -0.9], &vec![0.1, 0.2, 1.5, -0.9]));
    }

    fn generate_data(dimensions: usize, count: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::thread_rng();
        (0..count).map(|_| rng.clone().sample_iter(&Standard).take(dimensions).collect()).collect()
    }

    #[bench]
    fn bench_cosine_bulk_insert(b: &mut Bencher) {
        let hasher = CosineHash::new(8, 4);
        let mut hashtable = LSHashTable::new(hasher, 8);
        let data = generate_data(4, 1000);
        b.iter(|| hashtable.bulk_insert(data.clone()));
    }

    #[bench]
    fn bench_euclidean_bulk_insert(b: &mut Bencher) {
        let hasher = EuclideanHash::new(8, 4, 0.5);
        let mut hashtable = LSHashTable::new(hasher, 8);
        let data = generate_data(4, 1000);
        b.iter(|| hashtable.bulk_insert(data.clone()));
    }

    #[bench]
    fn bench_cosine_insert(b: &mut Bencher) {
        let hasher = CosineHash::new(8, 4);
        let mut hashtable = LSHashTable::new(hasher, 8);
        let data = vec![0.5, 0.4, -0.25, 0.0];
        b.iter(|| hashtable.insert(data.clone()));
    }

    #[bench]
    fn bench_euclidean_insert(b: &mut Bencher) {
        let hasher = EuclideanHash::new(8, 4, 0.5);
        let mut hashtable = LSHashTable::new(hasher, 8);
        let data = vec![0.5, 0.4, -0.25, 0.0];
        b.iter(|| hashtable.insert(data.clone()));
    }

    #[bench]
    fn bench_cosine_top_k_neighbors(b: &mut Bencher) {
        let hasher = CosineHash::new(8, 4);
        let mut hashtable = LSHashTable::new(hasher, 8);
        let data = generate_data(4, 1000);
        hashtable.bulk_insert(data);
        let query = vec![0.0, 3.2, -0.2, -5.0];
        b.iter(|| hashtable.top_k_neighbors(&query, 5));
    }

    #[bench]
    fn bench_euclidean_top_k_neighbors(b: &mut Bencher) {
        let hasher = EuclideanHash::new(8, 4, 0.5);
        let mut hashtable = LSHashTable::new(hasher, 8);
        let data = generate_data(4, 1000);
        hashtable.bulk_insert(data);
        let query = vec![0.0, 3.2, -0.2, -5.0];
        b.iter(|| hashtable.top_k_neighbors(&query, 5));
    }
    
    #[bench]
    fn bench_cosine_nearest_neighbors(b: &mut Bencher) {
        let hasher = CosineHash::new(8, 4);
        let mut hashtable = LSHashTable::new(hasher, 8);
        let data = generate_data(4, 1000);
        hashtable.bulk_insert(data);
        let query = vec![0.0, 3.2, -0.2, -5.0];
        b.iter(|| hashtable.nearest_neighbors(&query));
    }

    #[bench]
    fn bench_euclidean_nearest_neighbors(b: &mut Bencher) {
        let hasher = EuclideanHash::new(8, 4, 0.5);
        let mut hashtable = LSHashTable::new(hasher, 8);
        let data = generate_data(4, 1000);
        hashtable.bulk_insert(data);
        let query = vec![0.0, 3.2, -0.2, -5.0];
        b.iter(|| hashtable.nearest_neighbors(&query));
    }

    #[bench]
    fn bench_cosine_closest_neighbor(b: &mut Bencher) {
        let hasher = CosineHash::new(8, 4);
        let mut hashtable = LSHashTable::new(hasher, 8);
        let data = generate_data(4, 1000);
        hashtable.bulk_insert(data);
        let query = vec![0.0, 3.2, -0.2, -5.0];
        b.iter(|| hashtable.closest_neighbor(&query));
    }

    #[bench]
    fn bench_cosine_closest_neighbor_slight_optim(b: &mut Bencher) {
        let hasher = CosineHash::new(8, 4);
        let mut hashtable = LSHashTable::new(hasher, 8);
        let data = generate_data(4, 1000);
        hashtable.bulk_insert(data);
        let query = to_unit_vector(&vec![0.0, 3.2, -0.2, -5.0]);
        b.iter(|| hashtable.closest_neighbor(&query));
    }

    #[bench]
    fn bench_euclidean_closest_neighbor(b: &mut Bencher) {
        let hasher = EuclideanHash::new(8, 4, 0.5);
        let mut hashtable = LSHashTable::new(hasher, 8);
        let data = generate_data(4, 1000);
        hashtable.bulk_insert(data);
        let query = vec![0.0, 3.2, -0.2, -5.0];
        b.iter(|| hashtable.closest_neighbor(&query));
    }
}