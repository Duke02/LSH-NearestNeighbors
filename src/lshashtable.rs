use crate::lshash::LSHash;
use itertools::{Itertools, MinMaxResult};

pub struct LSHashTable<H: LSHash> {
    buckets: Vec<Vec<Vec<f32>>>,
    num_hashes: usize,
    num_buckets: usize,
    hasher: H,
}

impl<H: LSHash> LSHashTable<H> {
    pub fn new(hasher: H, num_dimensions: usize, num_bits: u8) -> Self {
        let num_buckets: usize = 2 << num_bits as usize;
        Self {
            num_buckets,
            buckets: vec![vec![]; num_buckets],
            num_hashes: 0,
            hasher,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.num_hashes == 0
    }

    pub fn len(&self) -> usize {
        self.num_hashes
    }

    fn get_index(&self, object: &Vec<f32>) -> usize {
        let hash = self.hasher.hash(object);
        hash % self.num_buckets
    }

    pub fn insert(&mut self, object: Vec<f32>) {
        let index = self.get_index(&object);
        self.num_hashes += 1;
        self.buckets[index].push(object);
    }

    pub fn contains(&self, object: &Vec<f32>) -> bool {
        if self.is_empty() {
            return false;
        }
        let index = self.get_index(object);
        let bucket = &self.buckets[index];
        if bucket.is_empty() {
            return false;
        }
        bucket.contains(&object)
    }

    pub fn remove(&mut self, object: &Vec<f32>) {
        if self.is_empty() {
            panic!("Tried to remove an object from an empty LSH.");
        }
        if !self.contains(object) {
            panic!("Tried to remove an object that doesn't exist in this LSH.");
        }

        let index = self.get_index(object);
        self.buckets[index].remove(index);
    }

    pub fn nearest_neighbors(&self, object: &Vec<f32>) -> Option<Vec<Vec<f32>>> {
        if self.is_empty() {
            return None;
        }

        let index = self.get_index(object);
        match self.buckets[index].len() {
            0 => {
                // Ping-Pong from before the index to after
                // until we find the closest set of neighbors.
                let mut radius: usize = 1;
                while ((index - radius) > 0) || ((index + radius) < self.num_buckets) {
                    if (index - radius) > 0 && self.buckets[index - radius].len() > 0 {
                        return Some(self.buckets[index - radius].clone());
                    } else if (index + radius) < self.num_buckets
                        && self.buckets[index + radius].len() > 0
                    {
                        return Some(self.buckets[index + radius].clone());
                    } else {
                        radius += 1;
                    }
                }
                // Since we do the "Are we empty" check above,
                // it should never reach this but just in case.
                None
            }
            _ => Some(self.buckets[index].clone()),
        }
    }

    pub fn top_k_neighbors(&self, object: &Vec<f32>, k: usize) -> Option<Vec<Vec<f32>>> {
        if self.is_empty() {
            return None;
        }

        let neighbors = match self.nearest_neighbors(object) {
            None => {
                return None;
            }
            Some(n) => n,
        };

        if neighbors.len() <= k {
            return Some(neighbors);
        }

        let k_nearest_neighbors = neighbors
            .iter()
            .k_smallest_by(k, |n1, n2| {
                f32::total_cmp(
                    &self.hasher.distance(n1, object),
                    &self.hasher.distance(n2, object),
                )
            })
            .map(|n| n.clone())
            .collect_vec();

        Some(k_nearest_neighbors)
    }

    pub fn closest_neighbor(&self, object: &Vec<f32>) -> Option<Vec<f32>> {
        if self.is_empty() {
            return None;
        }
        let neighbors = self.nearest_neighbors(object);
        match neighbors {
            Some(neighbors) => {
                let distances = neighbors.iter().map(|n| self.hasher.distance(object, &n));
                let closest_index = match distances.position_minmax() {
                    MinMaxResult::NoElements => {
                        return None;
                    }
                    MinMaxResult::OneElement(i) => i,
                    MinMaxResult::MinMax(mini, _) => mini,
                };
                Some(neighbors[closest_index].clone())
            }
            None => None,
        }
    }
}
