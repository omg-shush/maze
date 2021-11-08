use std::time::Instant;

pub struct Camera {
    position: [f32; 3],
    scale: [f32; 3],
    dest_position: [f32; 3],
    dest_speed: f32,
    last_update: Instant,
    rotation: [f32; 3]
}

impl Camera {
    pub fn new() -> Camera {
        Camera {
            position: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            dest_position: [0.0, 0.0, 0.0],
            dest_speed: 0.0,
            last_update: Instant::now(),
            rotation: [0.0, 0.0, 0.0]
        }
    }

    pub fn update(&mut self) {
        let now = Instant::now();
        for i in 0..3 {
            self.position[i] += (self.dest_position[i] - self.position[i]) * self.dest_speed * (now - self.last_update).as_secs_f32();
        }
    }

    pub fn position(&mut self, position: [f32; 3]) {
        self.position = position;
        self.dest_position = position;
    }

    pub fn adjust(&mut self, delta: [f32; 3], seconds: f32, speed: f32) {
        self.dest_position[0] += delta[0];
        self.dest_position[1] += delta[1];
        self.dest_position[2] += delta[2];
        self.last_update = Instant::now();
        self.dest_speed = speed / seconds as f32;
    }

    pub fn turn(&mut self, delta: [f32; 3]) {
        for i in 0..3 {
            self.rotation[i] += delta[i];
        }
    }

    pub fn view(&self) -> [[f32; 4]; 4] {
        let t = self.rotation[0];
        let rot_x = transpose([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, t.cos(), -1.0 * t.sin(), 0.0],
            [0.0, t.sin(), t.cos(), 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]);
        let t = self.rotation[1];
        let rot_y = transpose([
            [t.cos(), 0.0, t.sin(), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-1.0 * t.sin(), 0.0, t.cos(), 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]);
        let t = self.rotation[2];
        let rot_z = transpose([
            [t.cos(), -1.0 * t.sin(), 0.0, 0.0],
            [t.sin(), t.cos(), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]);
        let trans_scale = transpose([
            [1.0, 0.0, 0.0, -self.position[0]].map(|x| x * self.scale[0]),
            [0.0, 1.0, 0.0, -self.position[1]].map(|x| x * self.scale[1]),
            [0.0, 0.0, 1.0, -self.position[2]].map(|x| x * self.scale[2]),
            [0.0, 0.0, 0.0, 1.0]
        ]);
        mul(rot_z, mul(rot_y, mul(rot_x, trans_scale)))
    }

    pub fn projection(&self) -> [[f32; 4]; 4] {
        let (n, f) = (0.1, 100.0);
        let (focal_length, aspect_ratio) = (1.0 / (90.0f32 / 2.0).to_radians().tan(), 1.0);
        transpose([
            [focal_length / aspect_ratio, 0.0,                0.0,         0.0],
            [0.0,                         focal_length,       0.0,         0.0],
            [0.0,                         0.0,                (n + f) / (n - f), (2.0 * n * f) / (n - f)],
            [0.0,                         0.0,               -1.0,         0.0]
        ])
    }
}

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
