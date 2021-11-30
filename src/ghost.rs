use rand::{Rng, thread_rng};
use std::time::{Duration, Instant};
use std::sync::Arc;

use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::device::Queue;
use vulkano::buffer::{BufferUsage, CpuBufferPool, ImmutableBuffer, TypedBufferAccess};
use vulkano::sync::GpuFuture;
use vulkano::descriptor_set::SingleLayoutDescSetPool;
use vulkano::pipeline::PipelineBindPoint;

use crate::pipeline::InstanceModel;
use crate::player::{GameState, Player};
use crate::world::World;
use crate::config::Config;
use crate::pipeline::cs::ty::Vertex;
use crate::pipeline::vs::ty::{ViewProjectionData, PlayerPositionData};
use crate::pipeline::Pipeline;
use crate::linalg;

pub struct Ghost {
    grace: bool, // Grace period where ghost doesn't move till first food eaten
    position: [f32; 4],
    color: [f32; 3],
    reach_dest: Instant,
    dest_position: [usize; 4],
    init_position: [usize; 4],
    move_time: f32,
    current_move_time: f32, // Incorporates speed penalties for 3rd or 4th dimensional movement
    instant_start: Instant,
    vertex_buffer: Arc<ImmutableBuffer<[Vertex]>>,
    instance_buffer_pool: CpuBufferPool<[InstanceModel; 1]>,
    player_position_buffer_pool: CpuBufferPool<PlayerPositionData>
}

impl Ghost {
    pub fn new(config: &Config, queue: Arc<Queue>, color: [f32; 3]) -> (Ghost, Box<dyn GpuFuture>) {
        let mut rng = thread_rng();
        let dest_position = config.dimensions.map(|d| rng.gen_range(d/2..d));
        let position = dest_position.map(|i| i as f32);

        let (vertex_buffer, future) = ImmutableBuffer::from_iter(
            ghost_buffer(color),
            BufferUsage::vertex_buffer(),
            queue.clone()).unwrap();
        
        (Ghost {
            grace: true,
            position,
            color,
            reach_dest: Instant::now(),
            dest_position,
            init_position: dest_position,
            move_time: config.ghost_move_time,
            current_move_time: config.ghost_move_time,
            instant_start: Instant::now(),
            vertex_buffer,
            instance_buffer_pool: CpuBufferPool::new(queue.device().clone(), BufferUsage::vertex_buffer()),
            player_position_buffer_pool: CpuBufferPool::new(queue.device().clone(), BufferUsage::uniform_buffer())
        }, future.boxed())
    }

    pub fn update(&mut self, player: &mut Player, world: &World) {
        if self.grace {
            if player.score > 0 {
                self.grace = false;
            } else {
                return;
            }
        }

        let now = Instant::now();
        
        // Did we reach the player?
        let player_dist = linalg::sub(self.position, player.get_position()).map(|i| i * i).iter().fold(0.0, |acc, i| acc + i);
        if player_dist < 0.2 {
            player.game_state = GameState::Lost; // Player defeat
                return;
        }

        if now > self.reach_dest {
            self.position = self.dest_position.map(|i| i as f32);
            self.init_position = self.dest_position;
            // Otherwise, use BFS to track player
            let ghost_pos = (self.dest_position[0] as usize, self.dest_position[1] as usize, self.dest_position[2] as usize, self.dest_position[3] as usize);
            let player_pos = (player.cell()[0] as usize, player.cell()[1] as usize, player.cell()[2] as usize, player.cell()[3] as usize);
            // Next target position
            let (x, y, z, w) = *world.bfs(ghost_pos, player_pos).get(1).unwrap_or(&ghost_pos);
            self.dest_position = [x, y, z, w];
            self.current_move_time = self.move_time *
                if self.dest_position[2] != self.init_position[2] {
                    2.0 // Vertical penalty
                } else if self.dest_position[3] != self.dest_position[3] {
                    5.0 // Fourth penalty
                } else {
                    1.0
                };
            self.reach_dest = now + Duration::from_secs_f32(self.current_move_time);
        } else {
            // Animate movement
            let progress = 1.0 - (self.reach_dest - now).as_secs_f32() / self.current_move_time; // ranges from 0.0 at start to 1.0 at dest
            self.position = [0, 1, 2, 3].map(|i| self.init_position[i] as f32 + (self.dest_position[i] as f32 - self.init_position[i] as f32) * progress);
        }
    }

    pub fn render(&self, player: &Player, world: &World, desc_set_pool: &mut SingleLayoutDescSetPool, builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>, pipeline: &Pipeline) {
        let position = self.world_position(player, world);
        let instance_buffer = self.instance_buffer_pool.next([InstanceModel {
            m: linalg::translate(position) }]).unwrap();
        let player_position_buffer = self.player_position_buffer_pool.next(PlayerPositionData {
                player_pos: player.get_position()[0..3].try_into().unwrap(),
                ghost_pos: linalg::add(position, [0.0, 0.0, 1.0]),
                ..Default::default() }).unwrap();
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

    pub fn position(&self) -> [f32; 4] {
        self.position
    }

    pub fn world_position(&self, player: &Player, world: &World) -> [f32; 3] {
        let x = self.position[0] + (self.position[3] - player.get_position()[3]) * ((world.width + 1) as f32);
        let z = self.position[2] + ((Instant::now() - self.instant_start).as_secs_f32() * 3.0).sin() / 4.0;
        [x, self.position[1], z]
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
