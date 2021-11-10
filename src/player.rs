use std::time::Instant;

use super::camera::Camera;
use super::world;

pub struct Player {
    dest_position: [i32; 3],
    position: [f32; 3],
    dest_speed: f32,
    last_update: Instant,
    pub complete: bool,
    pub camera: Camera
}

impl Player {
    pub fn new() -> Player {
        let mut p = Player {
            dest_position: [0, 0, 0],
            position: [0.0, 0.0, 0.0],
            dest_speed: 0.0,
            last_update: Instant::now(),
            complete: false,
            camera: Camera::new()
        };
        p.camera.turn([15.0, 0.0, 0.0].map(|f: f32| f.to_radians()));
        p.camera.position([0.0, 0.4, 4.0]);
        p
    }

    pub fn move_position(&mut self, delta: [i32; 3], seconds: f32) {
        for i in 0..3 {
            self.dest_position[i] += delta[i];
        }
        self.last_update = Instant::now();
        let dist = delta.map(|i| i * i).iter().fold(0.0, |acc, x| acc + *x as f32).sqrt();
        self.dest_speed = dist / seconds;
    }

    pub fn get_position(&self) -> [f32; 3] {
        self.position
    }

    pub fn cell(&self) -> [i32; 3] {
        self.dest_position
    }

    pub fn update(&mut self) {
        let now = Instant::now();
        let delta: [f32; 3] = [0, 1, 2].map(|i| (self.dest_position[i] as f32 - self.position[i]) * self.dest_speed * (now - self.last_update).as_secs_f32());
        for i in 0..3 {
            self.position[i] += delta[i];
        }
        self.camera.adjust(delta);
        if self.position[0].round() as usize >= world::WIDTH {
            self.complete = true;
        }
    }
}
