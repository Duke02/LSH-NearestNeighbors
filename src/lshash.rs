use itertools::Itertools;
use rand::{thread_rng, Rng};

use crate::utils::{dot, magnitude, to_unit_vector};

pub trait LSHash {
    fn hash(&self, data: &Vec<f32>) -> usize;
    fn distance(&self, one: &Vec<f32>, two: &Vec<f32>) -> f32;
}

fn create_hyperplane<R: Rng>(num_dimensions: usize, num_bits: u8, rng: &mut R) -> Vec<Vec<f32>> {
    (0..num_bits)
        .map(|_| (0..num_dimensions).map(|_| rng.gen()).collect())
        .collect()
}

pub struct EuclideanHash {
    projection_matrix: Vec<Vec<f32>>,
    offsets: Vec<f32>,
    bin_width: f32,
}

impl EuclideanHash {
    pub fn new(num_bits: u8, num_dimensions: usize, bin_width: f32) -> Self {
        let mut rng = thread_rng();
        Self {
            projection_matrix: create_hyperplane(num_dimensions, num_bits, &mut rng),
            offsets: (0..num_bits)
                .map(|_| rng.gen_range(0.0..bin_width))
                .collect(),
            bin_width,
        }
    }

    pub fn new_with_optimal_bin_width(num_bits: u8, sample_data: &Vec<Vec<f32>>) -> Self {
        let start = EuclideanHash::new(num_bits, sample_data[0].len(), 0.5);
        let distances = sample_data
            .iter()
            .permutations(2)
            .map(|p| (p[0], p[1]))
            .filter(|(a, b)| a != b)
            .map(|(a, b)| start.distance(a, b))
            .collect_vec();
        // Rule of Thumb is Half of average distance.
        let bin_width = distances.iter().sum::<f32>() / (2.0 * distances.len() as f32);

        EuclideanHash::new(num_bits, sample_data[0].len(), bin_width)
    }
}

impl LSHash for EuclideanHash {
    fn hash(&self, data: &Vec<f32>) -> usize {
        self.projection_matrix
            .iter()
            .zip(self.offsets.iter())
            .map(|(proj, offset)| ((dot(proj, data) + offset) / self.bin_width).floor() as usize)
            .map(|r| r > 0)
            .map(|b| if b { 1 } else { 0 })
            .enumerate()
            .map(|(i, b)| b << i)
            .sum()
    }

    fn distance(&self, one: &Vec<f32>, two: &Vec<f32>) -> f32 {
        // Euclidean distance
        one.iter()
            .zip(two.iter())
            .map(|(a, b)| a - b)
            .map(|d| f32::powi(d, 2))
            .sum()
    }
}

pub struct CosineHash {
    projection_matrix: Vec<Vec<f32>>,
}

impl CosineHash {
    pub fn new(num_bits: u8, num_dimensions: usize) -> Self {
        let mut rng = thread_rng();
        Self {
            projection_matrix: create_hyperplane(num_dimensions, num_bits, &mut rng)
                .iter()
                .map(to_unit_vector)
                .collect(),
        }
    }
}

impl LSHash for CosineHash {
    fn hash(&self, data: &Vec<f32>) -> usize {
        let normed_data = to_unit_vector(data);
        self.projection_matrix
            .iter()
            .map(|proj| dot(&normed_data, proj))
            .map(|r| r > 0.0)
            .map(|b| if b { 1 } else { 0 })
            .enumerate()
            .map(|(i, b)| b << i)
            .sum()
    }

    fn distance(&self, one: &Vec<f32>, two: &Vec<f32>) -> f32 {
        let mag_one = magnitude(one);
        let mag_two = magnitude(two);
        1.0 - (dot(one, two) / (mag_one * mag_two))
    }
}
