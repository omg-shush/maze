use rand::{Rng, thread_rng};
use std::rc::Rc;
use std::cell::RefCell;
use std::time::{Duration, Instant};
use std::sync::Arc;

use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::device::Queue;
use vulkano::buffer::{BufferUsage, CpuBufferPool, ImmutableBuffer, TypedBufferAccess};
use vulkano::sync::GpuFuture;
use vulkano::descriptor_set::SingleLayoutDescSetPool;
use vulkano::pipeline::PipelineBindPoint;

use crate::pipeline::InstanceModel;
use crate::player::Player;
use crate::world::World;
use crate::parameters::Params;
use crate::pipeline::cs::ty::Vertex;
use crate::pipeline::fs::ty::PlayerPositionData;
use crate::pipeline::vs::ty::ViewProjectionData;
use crate::pipeline::Pipeline;
use crate::linalg;

pub struct Ghost {
    position: [f32; 4],
    color: [f32; 3],
    reach_dest: Instant,
    dest_position: [usize; 4],
    init_position: [usize; 4],
    move_time: f32,
    instant_start: Instant,
    world: Rc<RefCell<World>>,
    vertex_buffer: Arc<ImmutableBuffer<[Vertex]>>,
    instance_buffer_pool: CpuBufferPool<[InstanceModel; 1]>,
    player_position_buffer_pool: CpuBufferPool<PlayerPositionData>
}

impl Ghost {
    pub fn new(params: &Params, queue: Arc<Queue>, world: Rc<RefCell<World>>, color: [f32; 3]) -> (Ghost, Box<dyn GpuFuture>) {
        let mut rng = thread_rng();
        let dest_position = params.dimensions.map(|d| rng.gen_range(0..d));
        let position = dest_position.map(|i| i as f32);

        let (vertex_buffer, future) = ImmutableBuffer::from_iter(
            ghost_buffer(color),
            BufferUsage::vertex_buffer(),
            queue.clone()).unwrap();
        
        (Ghost {
            position,
            color,
            reach_dest: Instant::now(),
            dest_position,
            init_position: dest_position,
            move_time: params.ghost_move_time,
            instant_start: Instant::now(),
            world,
            vertex_buffer,
            instance_buffer_pool: CpuBufferPool::new(queue.device().clone(), BufferUsage::vertex_buffer()),
            player_position_buffer_pool: CpuBufferPool::new(queue.device().clone(), BufferUsage::uniform_buffer())
        }, future.boxed())
    }

    pub fn update(&mut self, player: &Box<Player>) {
        let now = Instant::now();
        if now > self.reach_dest {
            self.position = self.dest_position.map(|i| i as f32);
            self.init_position = self.dest_position;
            // Did we reach the player?
            if self.dest_position == player.cell().map(|i| i as usize) {
                // Player defeat
                // TODO
                return;
            }
            // Otherwise, use BFS to track player
            let ghost_pos = (self.dest_position[0] as usize, self.dest_position[1] as usize, self.dest_position[2] as usize, self.dest_position[3] as usize);
            let player_pos = (player.cell()[0] as usize, player.cell()[1] as usize, player.cell()[2] as usize, player.cell()[3] as usize);
            // Next target position
            let (x, y, z, w) = self.world.borrow_mut().bfs(ghost_pos, player_pos)[1]; // IDK why mutable borrow is necessary
            self.dest_position = [x, y, z, w];
            self.reach_dest = now + Duration::from_secs_f32(self.move_time);
        } else {
            // Animate movement
            let progress = 1.0 - (self.reach_dest - now).as_secs_f32() / self.move_time; // ranges from 0.0 at start to 1.0 at dest
            self.position = [0, 1, 2, 3].map(|i| self.init_position[i] as f32 + (self.dest_position[i] as f32 - self.init_position[i] as f32) * progress);
        }
    }

    pub fn render(&self, player: &Box<Player>, desc_set_pool: &mut SingleLayoutDescSetPool, builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>, pipeline: &Pipeline) {
        let x = self.position[0] + (self.position[3] - player.get_position()[3]) * ((self.world.borrow_mut().width + 1) as f32);
        let z = self.position[2] + ((Instant::now() - self.instant_start).as_secs_f32() * 3.0).sin() / 4.0;
        let instance_buffer = self.instance_buffer_pool.next([InstanceModel {
            m: linalg::translate([x, self.position[1], z])
        }]).unwrap();
        let player_position_buffer = self.player_position_buffer_pool.next(
            PlayerPositionData { pos: linalg::add(self.position[0..3].try_into().unwrap(), [0.0, 0.0, 1.0]) }).unwrap(); // Always light self
        let descriptor_set = {
            let mut builder = desc_set_pool.next();
            builder.add_buffer(Arc::new(player_position_buffer)).unwrap();
            builder.build().unwrap()
        };
        let view_projection = linalg::mul(player.camera.projection(), player.camera.view());
        builder
            .bind_vertex_buffers(0, (self.vertex_buffer.clone(), instance_buffer.clone()))
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.graphics_pipeline.layout().clone(),
                0,
                descriptor_set)
            .push_constants(pipeline.graphics_pipeline.layout().clone(), 0, ViewProjectionData {
                vp: view_projection,
                pushColor: self.color})
            .draw(
                self.vertex_buffer.len() as u32,
                instance_buffer.len() as u32,
                0,
                0).unwrap();
    }
}

fn ghost_buffer(color: [f32; 3]) -> Vec<Vertex> {
    const HALF_SIZE: f32 = 0.2;
    let (x, y) = (0.0, 0.0);
    [
        Vertex { position: [ x + HALF_SIZE, y + HALF_SIZE, 0.6 ], color: color, normal: [0.0, 0.0, 1.0], .. Default::default() },
        Vertex { position: [ x + HALF_SIZE, y - HALF_SIZE, 0.6 ], color: color, normal: [0.0, 0.0, 1.0], .. Default::default() },
        Vertex { position: [ x - HALF_SIZE, y - HALF_SIZE, 0.6 ], color: color, normal: [0.0, 0.0, 1.0], .. Default::default() },
        Vertex { position: [ x - HALF_SIZE, y - HALF_SIZE, 0.6 ], color: color, normal: [0.0, 0.0, 1.0], .. Default::default() },
        Vertex { position: [ x - HALF_SIZE, y + HALF_SIZE, 0.6 ], color: color, normal: [0.0, 0.0, 1.0], .. Default::default() },
        Vertex { position: [ x + HALF_SIZE, y + HALF_SIZE, 0.6 ], color: color, normal: [0.0, 0.0, 1.0], .. Default::default() }
    ].to_vec()
}
