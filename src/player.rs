use std::time::{Duration, Instant};
use std::rc::Rc;
use std::cell::RefCell;

use super::world::World;
use super::camera::Camera;
use super::world;

pub struct Player {
    dest_position: [i32; 3],
    position: [f32; 3],
    dest_speed: f32,
    last_update: Instant,
    world: Rc<RefCell<World>>,
    pub complete: bool,
    pub solve: Option<(usize, Instant)>,
    pub camera: Camera
}

impl Player {
    pub fn new(world: Rc<RefCell<World>>) -> Player {
        let mut p = Player {
            dest_position: [0, 0, 0],
            position: [0.0, 0.0, 0.0],
            dest_speed: 0.0,
            last_update: Instant::now(),
            world: world,
            complete: false,
            solve: None,
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

        // Interpolate position
        let delta: [f32; 3] = [0, 1, 2].map(|i| (self.dest_position[i] as f32 - self.position[i]) * self.dest_speed * (now - self.last_update).as_secs_f32());
        for i in 0..3 {
            self.position[i] += delta[i];
        }

        // Auto-solve
        if let Some((i, time)) = self.solve {
            if now > time {
                let n = self.world.borrow_mut().solution[i];
                let p = self.cell();
                self.move_position([n[0] - p[0], n[1] - p[1], n[2] - p[2]], 0.5);
                if i + 1 < self.world.borrow_mut().solution.len() {
                    let next_move = now + Duration::from_secs_f32(0.5);
                    self.solve = Some((i + 1, next_move));
                } else {
                    self.solve = None;
                }
            }
        }

        // Tracking camera
        self.camera.adjust(delta);

        // Check for victory
        if self.position[0].round() as usize >= world::WIDTH {
            self.complete = true;
        }
    }
}
