pub fn dot(one: &Vec<f32>, two: &Vec<f32>) -> f32 {
    one.iter().zip(two.iter()).map(|(a, b)| a * b).sum()
}

