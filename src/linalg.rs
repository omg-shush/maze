pub fn transpose(mat: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    [0, 1, 2, 3].map(|i| mat.map(|inner| inner[i]))
}

pub fn mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    // Dot rows of a with columns of b
    // Array of columns
    let mut prod = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            // Dot column i of b with row j of a
            let col = b[i];
            let row = a.map(|col| col[j]);
            for k in 0..4 {
                prod[i][j] += col[k] * row[k];
            }
        }
    }
    prod
}

pub fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    let mut sum = a.clone();
    for i in 0..3 {
        sum[i] += b[i];
    }
    sum
}

pub fn sub(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let mut diff = a.clone();
    for i in 0..4 {
        diff[i] -= b[i];
    }
    diff
}

pub fn translate(translation: [f32; 3]) -> [[f32; 4]; 4] {
    transpose([
        [1.0, 0.0, 0.0, translation[0]],
        [0.0, 1.0, 0.0, translation[1]],
        [0.0, 0.0, 1.0, translation[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])
}

pub fn rotate(rotation: [f32; 3]) -> [[f32; 4]; 4] {
    let t = rotation[0];
    let rot_x = transpose([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, t.cos(), -1.0 * t.sin(), 0.0],
        [0.0, t.sin(), t.cos(), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]);
    let t = rotation[1];
    let rot_y = transpose([
        [t.cos(), 0.0, t.sin(), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-1.0 * t.sin(), 0.0, t.cos(), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]);
    let t = rotation[2];
    let rot_z = transpose([
        [t.cos(), -1.0 * t.sin(), 0.0, 0.0],
        [t.sin(), t.cos(), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]);
    mul(rot_z, mul(rot_y, rot_x))
}

pub fn model(rotation: [f32; 3], scale: [f32; 3], translation: [f32; 3]) -> [[f32; 4]; 4] {
    let trans_scale = transpose([
        [scale[0], 0.0, 0.0, translation[0]],
        [0.0, scale[1], 0.0, translation[1]],
        [0.0, 0.0, scale[2], translation[2]],
        [0.0, 0.0, 0.0, 1.0]
    ]);
    mul(trans_scale, rotate(rotation))
}

pub fn view(rotation: [f32; 3], scale: [f32; 3], translation: [f32; 3]) -> [[f32; 4]; 4] {
    let trans_scale = transpose([
        [1.0, 0.0, 0.0, translation[0]].map(|x| x * scale[0]),
        [0.0, 1.0, 0.0, translation[1]].map(|x| x * scale[1]),
        [0.0, 0.0, 1.0, translation[2]].map(|x| x * scale[2]),
        [0.0, 0.0, 0.0, 1.0]
    ]);
    mul(rotate(rotation), trans_scale)
}

pub fn projection(near: f32, far: f32, focal: f32, aspect: f32) -> [[f32; 4]; 4] {
    transpose([
        [focal / aspect, 0.0,   0.0,                         0.0],
        [0.0,            focal, 0.0,                         0.0],
        [0.0,            0.0,   (near + far) / (near - far), (2.0 * near * far) / (near - far)],
        [0.0,            0.0,   -1.0,                        0.0]
    ])
}

pub fn _identity() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]
}
