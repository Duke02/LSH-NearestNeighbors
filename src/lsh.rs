use rand::{thread_rng, Rng};
use std::hash::Hash;

pub struct Lsh {
    buckets: Vec<Vec<Vec<f32>>>,
    num_hashes: usize,
    projection_matrix: Vec<Vec<f32>>,
    num_buckets: usize,
}

fn dot(one: &Vec<f32>, two: &Vec<f32>) -> f32 {
    one.iter().zip(two.iter()).map(|(a, b)| a * b).sum()
}

fn distance(one: &Vec<f32>, two: &Vec<f32>) -> f32 {
    // Euclidean distance
    one.iter()
        .zip(two.iter())
        .map(|(a, b)| a - b)
        .map(|d| f32::powi(d, 2))
        .sum()
}

impl Lsh {
    pub fn new(num_buckets: usize, num_dimensions: usize, num_bits: usize) -> Self {
        let mut rng = thread_rng();
        Self {
            num_buckets,
            buckets: vec![vec![]; num_buckets],
            num_hashes: 0,
            projection_matrix: (0..num_bits)
                .map(|_| (0..num_dimensions).map(|_| rng.gen()).collect())
                .collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.num_hashes == 0
    }

    fn hash(&self, object: &Vec<f32>) -> usize {
        self.projection_matrix
            .iter()
            .map(|proj| dot(proj, object))
            .map(|r| r > 0.0)
            .map(|b| if b { 1 } else { 0 })
            .enumerate()
            .map(|(i, b)| b << i)
            .sum()
    }

    fn get_index(&self, object: &Vec<f32>) -> usize {
        let hash = self.hash(object);
        hash % self.num_buckets
    }

    pub fn insert(&mut self, object: Vec<f32>) {
        let index = self.get_index(&object);
        self.num_hashes += 1;
        self.buckets[index].push(object);
    }

    pub fn nearest_neighbors(&self, object: &Vec<f32>) -> Vec<Vec<f32>> {
        if self.is_empty() {
            return vec![];
        }

        let index = self.get_index(object);
        match self.buckets[index].len() {
            0 => {
                // Ping-Pong from before the index to after
                // until we find the closest set of neighbors.
                let mut radius: usize = 1;
                while ((index - radius) > 0) || ((index + radius) < self.num_buckets) {
                    if (index - radius) > 0 && self.buckets[index - radius].len() > 0 {
                        return self.buckets[index - radius].clone();
                    } else if (index + radius) < self.num_buckets
                        && self.buckets[index + radius].len() > 0
                    {
                        return self.buckets[index + radius].clone();
                    } else {
                        radius += 1;
                    }
                }
                // Since we do the "Are we empty" check above,
                // it should never reach this but just in case.
                vec![]
            }
            _ => self.buckets[index].clone(),
        }
    }

    pub fn closest_neighbor(&self, object: &Vec<f32>) -> Vec<f32> {
        if self.is_empty() {
            return vec![];
        }
        let neighbors = self.nearest_neighbors(object);
        match neighbors.iter().min_by_key(distance) {
            None => vec![],
            Some(neighbor) => neighbor.to_vec(),
        }
    }
}