use std::time::{Instant, Duration};

pub struct Camera {
    position: [f32; 3],
    scale: [f32; 3],
    dest_position: [f32; 3],
    dest_speed: f32,
    last_update: Instant
}

impl Camera {
    pub fn new() -> Camera {
        Camera {
            position: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            dest_position: [0.0, 0.0, 0.0],
            dest_speed: 0.0,
            last_update: Instant::now()
        }
    }

    pub fn update(&mut self) {
        let now = Instant::now();
        // Need to animate camera
        for i in 0..3 {
            self.position[i] += (self.dest_position[i] - self.position[i]) * self.dest_speed * (now - self.last_update).as_secs_f32();
        }
    }

    pub fn position(&mut self, position: [f32; 3]) {
        self.position = position
    }

    pub fn get_position(&self) -> [f32; 3] {
        self.dest_position
    }

    pub fn adjust(&mut self, delta: [f32; 3], seconds: f32, speed: f32) {
        self.dest_position[0] += delta[0];
        self.dest_position[1] += delta[1];
        self.dest_position[2] += delta[2];
        self.last_update = Instant::now();
        self.dest_speed = speed / seconds as f32;
    }

    pub fn scale(&mut self, scale: [f32; 3]) {
        self.scale = scale;
    }

    pub fn view(&self) -> [[f32; 4]; 4] {
        transpose([
            [1.0, 0.0, 0.0, -self.position[0]].map(|x| x * self.scale[0]),
            [0.0, 1.0, 0.0, -self.position[1]].map(|x| x * self.scale[1]),
            [0.0, 0.0, 1.0, -self.position[2]].map(|x| x * self.scale[2]),
            [0.0, 0.0, 0.0, 1.0]
        ])
    }
}

fn transpose(mat: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    [0, 1, 2, 3].map(|i| mat.map(|inner| inner[i]))
}
