use crate::linalg;

pub struct Camera {
    position: [f32; 3],
    scale: [f32; 3],
    rotation: [f32; 3],
    aspect_ratio: f32,
    fov: u32
}

impl Camera {
    pub fn new(resolution: [u32; 2], fov: u32) -> Camera {
        Camera {
            position: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            rotation: [0.0, 0.0, 0.0],
            aspect_ratio: {
                let [x, y] = resolution;
                x as f32 / y as f32
            },
            fov
        }
    }

    pub fn position(&mut self, position: [f32; 3]) {
        self.position = position;
    }

    pub fn turn(&mut self, delta: [f32; 3]) {
        for i in 0..3 {
            self.rotation[i] += delta[i];
        }
    }

    pub fn view(&self) -> [[f32; 4]; 4] {
        linalg::view(self.rotation, self.scale, self.position.map(|x| -x))
    }

    pub fn projection(&self) -> [[f32; 4]; 4] {
        linalg::projection(0.1, 100.0, 1.0 / (self.fov as f32 / 2.0).to_radians().tan(), self.aspect_ratio)
    }
}
