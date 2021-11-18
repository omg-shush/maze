use crate::linalg;

pub struct Camera {
    position: [f32; 3],
    scale: [f32; 3],
    rotation: [f32; 3]
}

impl Camera {
    pub fn new() -> Camera {
        Camera {
            position: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            rotation: [0.0, 0.0, 0.0]
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
        linalg::projection(0.1, 100.0, 1.0 / (90.0f32 / 2.0).to_radians().tan(), 1.0)
    }
}
