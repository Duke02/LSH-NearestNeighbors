pub fn dot(one: &Vec<f32>, two: &Vec<f32>) -> f32 {
    one.iter().zip(two.iter()).map(|(a, b)| a * b).sum()
}

pub fn magnitude(one: &Vec<f32>) -> f32 {
    f32::sqrt(dot(one, one))
}

pub fn to_unit_vector(vec: &Vec<f32>) -> Vec<f32> {
    let mag = magnitude(vec);
    vec.iter().map(|&a| a / mag).collect()
}