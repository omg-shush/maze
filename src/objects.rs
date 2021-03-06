use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::device::Queue;

use crate::model::Model;
use crate::pipeline::vs::ty::ViewProjectionData;
use crate::pipeline::{InstanceModel, Pipeline};
use crate::player::Player;
use crate::world::{Cell, Coordinate, World};
use crate::parameters::RAINBOW;
use crate::config::Config;
use crate::linalg;

struct Food {
    model: InstanceModel
}

pub struct Objects {
    time_start: Instant,
    food: HashMap<Coordinate, Food>,
    food_buffer: Arc<CpuAccessibleBuffer<[InstanceModel]>>,
    buffer_len: u32,
    pub dirty_buffer: bool
}

impl Objects {
    pub fn new(queue: Arc<Queue>, world: &mut World, config: &Config) -> Objects {
        let food = generate_food(world, config);
        let food_buffer = CpuAccessibleBuffer::from_iter(
            queue.device().clone(),
            BufferUsage::vertex_buffer_transfer_destination(),
            false,
            food.values().map(|f| f.model)).unwrap();
        Objects {
            time_start: Instant::now(),
            food,
            buffer_len: food_buffer.len() as u32,
            food_buffer,
            dirty_buffer: true
        }
    }

    pub fn update(&mut self, player: &Player) {
        if self.dirty_buffer {
            if let Ok (mut access) = self.food_buffer.write() {
                self.dirty_buffer = false;
                let instances: Vec<InstanceModel> = self.food.iter().filter_map(|((x, y, z, w), food)| {
                    let (_x, _y, z, w) = (*x as i32, *y as i32, *z as i32, *w as i32);
                    if z <= player.cell()[2] && z > player.cell()[2] - 6 && w >= player.cell()[3] - 1 && w <= player.cell()[3] + 1 {
                        Some (food.model)
                    } else {
                        None
                    }
                }).collect();
                self.buffer_len = instances.len() as u32;
                for i in 0..instances.len() {
                    access[i] = instances[i];
                }
            }
        }
    }

    pub fn render(&self, player: &Player, world: &World, models: &HashMap<String, Model>, builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>, pipeline: &Pipeline) {
        let food_color = RAINBOW[2];
        // let instances: Vec<InstanceModel> = self.food.values().map(|food| { food.model }).collect();
        // builder.update_buffer(self.food_buffer.clone(), &instances[..]).unwrap();

        // Render food objects
        // TODO use own shader pipeline for customizability
        let vp = linalg::mul(player.camera.projection(), player.camera.view());
        let x_offset = (-player.get_position()[3]) * ((world.width + 1) as f32);
        let z_offset = ((Instant::now() - self.time_start).as_secs_f32() * 2.0).sin() / 5.0;
        let vp = linalg::mul(vp, linalg::translate([x_offset, 0.0, z_offset]));
        builder
            .bind_pipeline_graphics(pipeline.graphics_pipeline.clone())
            .push_constants(
                pipeline.graphics_pipeline.layout().clone(),
            0,
            ViewProjectionData { pushColor: food_color, vp })
            .bind_vertex_buffers(0, (models["ceiling"].vertices.clone(), self.food_buffer.clone()))
            .draw(
                models["ceiling"].vertices.len() as u32,
                self.buffer_len,
                0,
                0).unwrap();
    }

    pub fn remove_food(&mut self, pos: Coordinate) {
        self.food.remove(&pos);
        self.dirty_buffer = true;
    }
}

fn generate_food(world: &mut World, config: &Config) -> HashMap<Coordinate, Food> {
    (0..config.food_count).map(|_| {
        let (x, y, z, w) = world.random_empty_cell();
        world.cells[w][z][y][x] = Cell::Food;
        let world_transform = world.world_transform(w, 0.0);
        let model = linalg::model(
            [90f32.to_radians(), 0.0, 45f32.to_radians()],
            [0.5, 0.5, 1.0],
            [x as f32, y as f32, z as f32 + 0.6]);
        ((x, y, z, w), Food { model: InstanceModel {
            m: linalg::mul(world_transform, model) } })
    }).collect()
}
