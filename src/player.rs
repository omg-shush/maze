use std::time::{Duration, Instant};
use std::rc::Rc;
use std::cell::RefCell;
use std::sync::Arc;

use vulkano::buffer::{BufferUsage, CpuBufferPool, ImmutableBuffer, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::SingleLayoutDescSetPool;
use vulkano::device::{Device, Queue};
use vulkano::pipeline::PipelineBindPoint;
use vulkano::sync::GpuFuture;

use crate::parameters::RAINBOW;
use crate::world::World;
use crate::camera::Camera;
use crate::linalg;
use crate::pipeline::{InstanceModel, Pipeline};
use crate::pipeline::cs::ty::Vertex;
use crate::pipeline::fs::ty::PlayerPositionData;
use crate::pipeline::vs::ty::ViewProjectionData;

const CAMERA_OFFSET: [f32; 3] = [0.0, 1.6, 4.0];

pub struct Player {
    dest_position: [i32; 4],
    position: [f32; 4],
    dest_speed: f32,
    last_update: Instant,
    reach_dest: Instant,
    world: Rc<RefCell<World>>,
    pub complete: bool,
    pub solve: Option<(usize, Instant)>,
    pub camera: Camera,
    vertex_buffer: Arc<ImmutableBuffer<[Vertex]>>,
    instance_buffer_pool: CpuBufferPool<[InstanceModel; 1]>,
    player_position_buffer_pool: CpuBufferPool<PlayerPositionData>
}

impl Player {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, world: Rc<RefCell<World>>) -> (Box<Player>, Box<dyn GpuFuture>) {
        let (vertex_buffer, future) = ImmutableBuffer::from_iter(
            player_buffer().into_iter(),
            BufferUsage::vertex_buffer(),
            queue).unwrap();
        let mut p = Player {
            dest_position: [0, 0, 0, 0],
            position: [0.0, 0.0, 0.0, 0.0],
            dest_speed: 0.0,
            last_update: Instant::now(),
            reach_dest: Instant::now(),
            world: world,
            complete: false,
            solve: None,
            camera: Camera::new(),
            vertex_buffer,
            instance_buffer_pool: CpuBufferPool::new(device.clone(), BufferUsage::vertex_buffer()),
            player_position_buffer_pool: CpuBufferPool::new(device.clone(), BufferUsage::uniform_buffer())
        };
        p.camera.turn([30.0, 0.0, 0.0].map(|f: f32| f.to_radians()));
        p.camera.position(CAMERA_OFFSET);
        println!("Initialized player");
        (Box::new(p), future.boxed())
    }

    pub fn render(&self, desc_set_pool: &mut SingleLayoutDescSetPool, builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>, pipeline: &Pipeline) {
        let instance_buffer = self.instance_buffer_pool.next([
            InstanceModel { m: linalg::model([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], self.position[0..3].try_into().unwrap()) }
        ]).unwrap();
        let player_position_buffer = self.player_position_buffer_pool.next(
            PlayerPositionData { pos: linalg::add(self.position[0..3].try_into().unwrap(), [0.0, 0.0, 0.8]) }).unwrap();
        let descripter_set = {
            let mut builder = desc_set_pool.next();
            builder.add_buffer(Arc::new(player_position_buffer)).unwrap();
            builder.build().unwrap()
        };
        let view_projection = linalg::mul(self.camera.projection(), self.camera.view());
        builder
            .bind_vertex_buffers(0, (self.vertex_buffer.clone(), instance_buffer.clone()))
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.graphics_pipeline.layout().clone(),
                0,
                descripter_set)
            .push_constants(pipeline.graphics_pipeline.layout().clone(), 0, ViewProjectionData {
                vp: view_projection,
                pushColor: RAINBOW[self.cell()[3] as usize % RAINBOW.len()]})
            .draw(
                self.vertex_buffer.len() as u32,
                instance_buffer.len() as u32,
                0,
                0).unwrap();
    }

    pub fn move_position(&mut self, delta: [i32; 4], seconds: f32) {
        for i in 0..delta.len() {
            self.dest_position[i] += delta[i];
        }
        self.last_update = Instant::now();
        self.reach_dest = self.last_update + Duration::from_secs_f32(seconds);
        if seconds <= 0.1 {
            self.position = self.dest_position.map(|i| i as f32);
        } else {
            let dist = delta.map(|i| i * i).iter().fold(0.0, |acc, x| acc + *x as f32).sqrt();
            self.dest_speed = dist / seconds;
        }
    }

    pub fn get_position(&self) -> [f32; 4] {
        self.position
    }

    pub fn cell(&self) -> [i32; 4] {
        self.dest_position
    }

    pub fn update(&mut self) {
        let now = Instant::now();

        // Interpolate position
        if now > self.reach_dest {
            self.position = self.dest_position.map(|i| i as f32);
        } else {
            let delta = [0, 1, 2, 3].map(|i| (self.dest_position[i] as f32 - self.position[i]) * self.dest_speed * (now - self.last_update).as_secs_f32());
            for i in 0..delta.len() {
                self.position[i] += delta[i];
            }
        }

        // Auto-solve
        const MOVE_TIME: f32 = 0.5;
        if let Some((i, time)) = self.solve {
            if now > time {
                let n = self.world.borrow_mut().solution[i];
                let p = self.cell();
                self.move_position([n[0] - p[0], n[1] - p[1], n[2] - p[2], n[3] - p[3]], MOVE_TIME);
                if i + 1 < self.world.borrow_mut().solution.len() {
                    let next_move = now + Duration::from_secs_f32(MOVE_TIME);
                    self.solve = Some((i + 1, next_move));
                } else {
                    self.solve = None;
                }
            }
        }

        // Tracking camera
        self.camera.position(linalg::add(self.position[0..3].try_into().unwrap(), CAMERA_OFFSET));

        // Check for victory
        if self.position[0].round() as usize >= self.world.borrow().width {
            self.complete = true;
        }
    }
}

fn player_buffer() -> Vec<Vertex> {
    const PLAYER_COLOR: [f32; 3] = [ 0.2, 0.2, 0.8 ];
    const HALF_SIZE: f32 = 0.2;
    let (x, y) = (0.0, 0.0);
    [
        Vertex { position: [ x + HALF_SIZE, y + HALF_SIZE, 0.5 ], color: PLAYER_COLOR, normal: [0.0, 0.0, 1.0], .. Default::default() },
        Vertex { position: [ x + HALF_SIZE, y - HALF_SIZE, 0.5 ], color: PLAYER_COLOR, normal: [0.0, 0.0, 1.0], .. Default::default() },
        Vertex { position: [ x - HALF_SIZE, y - HALF_SIZE, 0.5 ], color: PLAYER_COLOR, normal: [0.0, 0.0, 1.0], .. Default::default() },
        Vertex { position: [ x - HALF_SIZE, y - HALF_SIZE, 0.5 ], color: PLAYER_COLOR, normal: [0.0, 0.0, 1.0], .. Default::default() },
        Vertex { position: [ x - HALF_SIZE, y + HALF_SIZE, 0.5 ], color: PLAYER_COLOR, normal: [0.0, 0.0, 1.0], .. Default::default() },
        Vertex { position: [ x + HALF_SIZE, y + HALF_SIZE, 0.5 ], color: PLAYER_COLOR, normal: [0.0, 0.0, 1.0], .. Default::default() }
    ].to_vec()
}
