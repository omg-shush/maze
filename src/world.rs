use rand::seq::SliceRandom;
use rand::thread_rng;
use vulkano::pipeline::PipelineBindPoint;
use std::collections::hash_map::HashMap;
use std::collections::hash_set::HashSet;
use std::collections::vec_deque::VecDeque;
use std::rc::Rc;
use std::cell::RefCell;
use std::sync::Arc;

use vulkano::buffer::{BufferUsage, CpuBufferPool, ImmutableBuffer, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::SingleLayoutDescSetPool;
use vulkano::device::Queue;
use vulkano::sync::{now, GpuFuture};

use crate::linalg;
use crate::pipeline::Pipeline;
use crate::disjoint_set;
use crate::pipeline::InstanceModel;
use crate::pipeline::fs::ty::PlayerPositionData;
use crate::player::Player;
use crate::model::Model;
use crate::pipeline::vs::ty::ViewProjectionData;
use crate::parameters::{Params, RAINBOW};

type Coordinate = (usize, usize, usize, usize);

#[derive(Debug, Clone, Copy)]
pub enum Cell {
    Empty
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wall {
    NoWall,
    SolidWall
}

struct LevelInstances {
    walls: Vec<InstanceModel>,
    floors: Vec<InstanceModel>,
    ceilings: Vec<InstanceModel>,
    corners: Vec<InstanceModel>,
    left_portals: Vec<InstanceModel>,
    right_portals: Vec<InstanceModel>
}

impl LevelInstances {
    fn into_iter(self) -> std::array::IntoIter<Vec<InstanceModel>, 6> {
        [self.walls, self.floors, self.ceilings, self.corners, self.left_portals, self.right_portals].into_iter()
    }
}

struct LevelBuffers {
    walls: Arc<ImmutableBuffer<[InstanceModel]>>,
    floors: Arc<ImmutableBuffer<[InstanceModel]>>,
    ceilings: Arc<ImmutableBuffer<[InstanceModel]>>,
    corners: Arc<ImmutableBuffer<[InstanceModel]>>,
    left_portals: Arc<ImmutableBuffer<[InstanceModel]>>,
    right_portals: Arc<ImmutableBuffer<[InstanceModel]>>
}

impl From<Vec<Arc<ImmutableBuffer<[InstanceModel]>>>> for LevelBuffers {
    fn from(list: Vec<Arc<ImmutableBuffer<[InstanceModel]>>>) -> Self {
        LevelBuffers {
            walls: list[0].clone(),
            floors: list[1].clone(),
            ceilings: list[2].clone(),
            corners: list[3].clone(),
            left_portals: list[4].clone(),
            right_portals: list[5].clone()
        }
    }
}

pub struct World {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub fourth: usize,

    // Dimensions: fourth x depth x height x width
    pub cells: Box<[Box<[Box<[Box<[Cell]>]>]>]>,
    // Vertical walls, fourth x depth x height x (width + 1)
    pub xwalls: Box<[Box<[Box<[Box<[Wall]>]>]>]>,
    // Horizontal walls, fourth x depth x (height + 1) x width
    pub ywalls: Box<[Box<[Box<[Box<[Wall]>]>]>]>,
    // Floors/Ceilings, fourth x (depth + 1) x height x width
    pub zwalls: Box<[Box<[Box<[Box<[Wall]>]>]>]>,
    // I don't even know any more, (fourth + 1) x depth x height x width
    pub wwalls: Box<[Box<[Box<[Box<[Wall]>]>]>]>,

    pub start: Coordinate,
    pub finish: Coordinate,
    pub solution: Vec<([i32; 4])>,

    player_position_buffer_pool: CpuBufferPool<[PlayerPositionData; 1]>,
    vertex_buffers: Vec<Vec<LevelBuffers>> // lists of model matrices, indexed by: fourth -> level
}

impl World {
    pub fn new(params: &Params, queue: Arc<Queue>) -> (Rc<RefCell<World>>, Box<dyn GpuFuture>) {
        // Start by creating a 2D grid, with walls around each cell
        let [width, height, depth, fourth] = params.dimensions;
        let mut world = World {
            cells: vec![vec![vec![vec![Cell::Empty; width].into_boxed_slice(); height].into_boxed_slice(); depth].into_boxed_slice(); fourth].into_boxed_slice(),
            xwalls: vec![vec![vec![vec![Wall::SolidWall; width + 1].into_boxed_slice(); height].into_boxed_slice(); depth].into_boxed_slice(); fourth].into_boxed_slice(),
            ywalls: vec![vec![vec![vec![Wall::SolidWall; width].into_boxed_slice(); height + 1].into_boxed_slice(); depth].into_boxed_slice(); fourth].into_boxed_slice(),
            zwalls: vec![vec![vec![vec![Wall::SolidWall; width].into_boxed_slice(); height].into_boxed_slice(); depth + 1].into_boxed_slice(); fourth].into_boxed_slice(),
            wwalls: vec![vec![vec![vec![Wall::SolidWall; width].into_boxed_slice(); height].into_boxed_slice(); depth].into_boxed_slice(); fourth + 1].into_boxed_slice(),
            start: (0, 0, 0, 0),
            finish: (width - 1, height - 1, depth - 1, fourth - 1),
            solution: Vec::new(),
            player_position_buffer_pool: CpuBufferPool::new(queue.device().clone(), BufferUsage::uniform_buffer()),
            vertex_buffers: Vec::new(),
            width,
            height,
            depth,
            fourth
        };
        world.generate_maze();
        
        let world_data: Vec<Vec<LevelInstances>> = (0..fourth).map(|fourth| (0..depth).map(|level| world.vertex_buffer(fourth, level)).collect()).collect();
        let world_buffer: Vec<Vec<_>> =
            world_data.into_iter().map(|fourths| {
                fourths.into_iter().map(|instance_buffers| {
                    instance_buffers.into_iter().map(|ibuf| {
                        ImmutableBuffer::from_iter(
                            ibuf,
                            BufferUsage::vertex_buffer(),
                            queue.clone()
                        ).expect("Failed to construct buffer")
                    })
                }).collect()
            }).collect();
        let future = now(queue.device().clone()).boxed();
        let future = world_buffer.into_iter().fold(future, |future, fourth| {
            let mut fourth_buffers = Vec::new();
            let future = fourth.into_iter().fold(future, |future, level| {
                let mut level_buffers = Vec::new();
                let future = level.into_iter().fold(future, |future, (buf, upload)| {
                    level_buffers.push(buf);
                    future.join(upload).boxed()
                });
                fourth_buffers.push(LevelBuffers::from(level_buffers));
                future.then_signal_fence_and_flush().unwrap().boxed()
            });
            world.vertex_buffers.push(fourth_buffers);
            future.then_signal_fence_and_flush().unwrap().boxed()
        });
        println!("Initialized world");
        (Rc::new(RefCell::new(world)), future)
    }

    pub fn render(&self, models: &HashMap<String, Box<Model>>, player: &Box<Player>, desc_set_pool: &mut SingleLayoutDescSetPool, builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>, pipeline: &Pipeline) {
        let player_position_buffer = self.player_position_buffer_pool.next([
            PlayerPositionData { pos: linalg::add(player.get_position()[0..3].try_into().unwrap(), [0.0, 0.0, 0.4]) }
        ]).unwrap();
        let descriptor_set = {
            let mut builder = desc_set_pool.next();
            builder.add_buffer(Arc::new(player_position_buffer)).unwrap();
            builder.build().unwrap()
        };
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.graphics_pipeline.layout().clone(),
                0,
                descriptor_set
            );
        let view_projection = linalg::mul(player.camera.projection(), player.camera.view());

        let fourth = player.cell()[3];
        let between = player.get_position()[3];
        fn world_transform(world: &World, fourth: usize, between: f32) -> [[f32; 4]; 4] {
            let spacing = (world.width + 1) as f32;
            linalg::translate([(fourth as f32 - between) * spacing, 0.0, 0.0])
        }

        for w in fourth - 1..fourth + 2 {
            if w >= 0 && w < self.fourth as i32 {
                let w = w as usize;
                let wvp = linalg::mul(view_projection, world_transform(self, w, between));
                self.render_fourth(w, wvp, player, models, builder, pipeline);
            }
        }
    }

    fn render_fourth(&self, fourth: usize, view_projection: [[f32; 4]; 4], player: &Box<Player>, models: &HashMap<String, Box<Model>>, builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>, pipeline: &Pipeline) {
        let fourth_color = RAINBOW[fourth % RAINBOW.len()];
        let left_color = RAINBOW[(fourth as i32 - 1).rem_euclid(RAINBOW.len() as i32) as usize];
        let right_color = RAINBOW[(fourth + 1) % RAINBOW.len()];
        let corner_color = fourth_color.map(|f| (f * 1.2).clamp(0.0, 1.0));
        let floor_color = fourth_color.map(|f| f * 0.1);
        let ascend_color = [1.0, 1.0, 1.0];
        let (min_level, max_level) = ((player.cell()[2] - 6).clamp(0, self.depth as i32) as usize, player.cell()[2] as usize);
        for level in min_level..max_level + 1 {
            let level_buffers = &self.vertex_buffers[fourth][level];
            builder
                .push_constants(
                    pipeline.graphics_pipeline.layout().clone(),
                    0,
                    ViewProjectionData { vp: view_projection, pushColor: fourth_color })
                .bind_vertex_buffers(0, (models["wall"].vertices.clone(), level_buffers.walls.clone()))
                .draw(
                    models["wall"].vertices.len() as u32,
                    level_buffers.walls.len() as u32,
                    0,
                    0)
                .unwrap()
                .push_constants(
                    pipeline.graphics_pipeline.layout().clone(),
                    0,
                    ViewProjectionData { vp: view_projection, pushColor: floor_color })
                .bind_vertex_buffers(0, (models["floor"].vertices.clone(), level_buffers.floors.clone()))
                .draw(
                    models["floor"].vertices.len() as u32,
                    level_buffers.floors.len() as u32,
                    0,
                    0)
                .unwrap()
                .push_constants(
                    pipeline.graphics_pipeline.layout().clone(),
                    0,
                    ViewProjectionData { vp: view_projection, pushColor: corner_color })
                .bind_vertex_buffers(0, (models["corner"].vertices.clone(), level_buffers.corners.clone()))
                .draw(
                    models["corner"].vertices.len() as u32,
                    level_buffers.corners.len() as u32,
                    0,
                    0)
                .unwrap()
                .push_constants(
                    pipeline.graphics_pipeline.layout().clone(),
                    0,
                    ViewProjectionData { vp: view_projection, pushColor: ascend_color })
                .bind_vertex_buffers(0, (models["ceiling"].vertices.clone(), level_buffers.ceilings.clone()))
                .draw(
                    models["ceiling"].vertices.len() as u32,
                    level_buffers.ceilings.len() as u32,
                    0,
                    0)
                .unwrap()
                .push_constants(
                    pipeline.graphics_pipeline.layout().clone(),
                    0,
                    ViewProjectionData { vp: view_projection, pushColor: left_color })
                .bind_vertex_buffers(0, (models["ceiling"].vertices.clone(), level_buffers.left_portals.clone()))
                .draw(
                    models["ceiling"].vertices.len() as u32,
                    level_buffers.left_portals.len() as u32,
                    0,
                    0)
                .unwrap()
                .push_constants(
                    pipeline.graphics_pipeline.layout().clone(),
                    0,
                    ViewProjectionData { vp: view_projection, pushColor: right_color })
                .bind_vertex_buffers(0, (models["ceiling"].vertices.clone(), level_buffers.right_portals.clone()))
                .draw(
                    models["ceiling"].vertices.len() as u32,
                    level_buffers.right_portals.len() as u32,
                    0,
                    0)
                .unwrap();
        }
    }

    pub fn generate_maze(&mut self) {
        // Use randomized kruskal's algorithm

        // Random list of edges
        #[derive(Debug)]
        enum MazeEdge {
            XWall (Coordinate),
            YWall (Coordinate),
            ZWall (Coordinate),
            WWall (Coordinate)
        }
        let mut edges: Vec<MazeEdge> = Vec::new();
        for w in 0..self.fourth {
            for z in 0..self.depth {
                for y in 0..self.height {
                    for x in 0..self.width {
                        if x != 0 {
                            edges.push(MazeEdge::XWall((x, y, z, w)))
                        }
                        if y != 0 {
                            edges.push(MazeEdge::YWall((x, y, z, w)))
                        }
                        if z != 0 {
                            edges.push(MazeEdge::ZWall((x, y, z, w)))
                        }
                        if w != 0 {
                            edges.push(MazeEdge::WWall((x, y, z, w)))
                        }
                    }
                }
            }
        }
        edges.shuffle(&mut thread_rng());

        // Initialize disjoint set of cells
        let mut cells = disjoint_set::DisjointSet::new();
        for w in 0..self.fourth {
            for z in 0..self.depth {
                for y in 0..self.height {
                    for x in 0..self.width {
                        // Use tuples to hash correctly hopefully
                        cells.add(&(x, y, z, w));
                    }
                }
            }
        }

        // Take a random edge and check if the neighbor cells are connected
        // If not, remove the edge to merge them
        // Also generate map from each cell to accessible neighbors
        let mut neighbors: HashMap<Coordinate, Vec<Coordinate>> = HashMap::new();
        for edge in edges.iter() {
            let (cell_a, cell_b) =
                match edge {
                    MazeEdge::XWall ((x, y, z, w)) => ((*x - 1, *y, *z, *w), (*x, *y, *z, *w)),
                    MazeEdge::YWall ((x, y, z, w)) => ((*x, *y - 1, *z, *w), (*x, *y, *z, *w)),
                    MazeEdge::ZWall ((x, y, z, w)) => ((*x, *y, *z - 1, *w), (*x, *y, *z, *w)),
                    MazeEdge::WWall ((x, y, z, w)) => ((*x, *y, *z, *w - 1), (*x, *y, *z, *w))
                };
            let set_a = cells.find(&cell_a);
            let set_b = cells.find(&cell_b);
            if set_a != set_b {
                // Remove edge between these cells in world
                match edge {
                    MazeEdge::XWall ((x, y, z, w)) => self.xwalls[*w][*z][*y][*x] = Wall::NoWall,
                    MazeEdge::YWall ((x, y, z, w)) => self.ywalls[*w][*z][*y][*x] = Wall::NoWall,
                    MazeEdge::ZWall ((x, y, z, w)) => self.zwalls[*w][*z][*y][*x] = Wall::NoWall,
                    MazeEdge::WWall ((x, y, z, w)) => self.wwalls[*w][*z][*y][*x] = Wall::NoWall
                }
                // Mark them as neighbors for BFS later
                if !neighbors.contains_key(&cell_a) {
                    neighbors.insert(cell_a, Vec::new());
                }
                if !neighbors.contains_key(&cell_b) {
                    neighbors.insert(cell_b, Vec::new());
                }
                neighbors.get_mut(&cell_a).unwrap().push(cell_b);
                neighbors.get_mut(&cell_b).unwrap().push(cell_a);
                // And merge the sets they belong to
                cells.union(&set_a, &set_b);
            }
        }
        // Results in minimum spanning tree connecting all cells of maze

        // Generate exit at bottom right corner of top layer in last w
        self.xwalls[self.fourth - 1][self.depth - 1][self.height - 1][self.width] = Wall::NoWall;

        // Use breadth-first search to find solution
        let mut queue: VecDeque<Coordinate> = VecDeque::new();
        queue.push_back((0, 0, 0, 0));
        let mut visited: HashSet<Coordinate> = HashSet::new();
        visited.insert((0, 0, 0, 0));
        let mut backtrack: HashMap<Coordinate, Coordinate> = HashMap::new();
        while !queue.is_empty() {
            // Take next cell from queue
            let cell = queue.pop_front().unwrap();

            // Add unvisited neighbors to the queue
            for n in neighbors.get(&cell).unwrap_or(&Vec::new()) {
                if !visited.contains(n) {
                    visited.insert(*n);
                    queue.push_back(*n);
                    backtrack.insert(*n, cell);
                }
            }
        }
        // Use backtracking information to recover path
        let mut previous = self.finish;
        self.solution.push({
            let (x, y, z, w) = self.finish;
            [x, y, z, w].map(|u| u as i32)
        });
        while previous != self.start {
            previous = *backtrack.get(&previous).expect("Backtracking after BFS failed, impossible");
            let (x, y, z, w) = previous;
            self.solution.push([x, y, z, w].map(|u| u as i32));
        }
        self.solution.reverse(); // Get finish at the end of the vec
    }

    fn vertex_buffer(&self, w: usize, z: usize) -> LevelInstances {
        // Generate vertex data for maze
        let mut walls: Vec<InstanceModel> = Vec::new();

        // Mark cells with open ceilings
        let ceilings: Vec<InstanceModel> = self.cells[w][z].iter().enumerate().map(|(y, row)| {
            row.iter().enumerate().filter_map(move |(x, _cell)| {
                match self.zwalls[w][z + 1][y][x] {
                    Wall::SolidWall => None,
                    Wall::NoWall => {
                        let (x, y, z) = (x as f32, y as f32, z as f32 + 0.8);
                        Some (InstanceModel { m: linalg::model([90f32.to_radians(), 0.0, 0.0], [1.0, 1.0, 1.0], [x, y, z]) })
                    }
                }
            })
        }).flatten().collect();

        // Mark fourth-dimensional portals i guess
        let left_portals: Vec<InstanceModel> = self.cells[w][z].iter().enumerate().map(|(y, row)| {
            row.iter().enumerate().filter_map(move |(x, _cell)| {
                // Check "left" fourth dimension adjacent cell
                match self.wwalls[w][z][y][x] {
                    Wall::SolidWall => None,
                    Wall::NoWall => {
                        let (x, y, z) = (x as f32 - 0.3, y as f32, z as f32 + 0.4);
                        Some (InstanceModel { m: linalg::model([90f32.to_radians(), 90f32.to_radians(), 0.0], [0.5, 1.0, 1.0], [x, y, z]) })
                    }
                }
            })
        })
        .flatten()
        .collect();

        let right_portals: Vec<InstanceModel> = self.cells[w][z].iter().enumerate().map(|(y, row)| {
            row.iter().enumerate().filter_map(move |(x, _cell)| {
                // Check "right" fourth dimension adjacent cell
                match self.wwalls[w + 1][z][y][x] {
                    Wall::SolidWall => None,
                    Wall::NoWall => {
                        let (x, y, z) = (x as f32 + 0.3, y as f32, z as f32 + 0.4);
                        Some (InstanceModel { m: linalg::model([90f32.to_radians(), 270f32.to_radians(), 0.0], [0.5, 1.0, 1.0], [x, y, z]) })
                    }
                }
            })
        })
        .flatten()
        .collect();

        // Map xwalls to rectangles
        walls.append(&mut self.xwalls[w][z].iter().enumerate().map(|(y, row)| {
            row.iter().enumerate().filter_map(move |(x, wall)| {
                // Draw a wall between cells (x - 1, y, z) and (x, y, z)
                let (x, y, z) = (x as f32 - 0.5, y as f32, z as f32);
                match wall {
                    Wall::SolidWall => Some (
                            InstanceModel { m: linalg::model([90f32.to_radians(), 0.0, 90f32.to_radians()], [1.0, 1.0, 1.0], [x, y, z]) }
                        ),
                    Wall::NoWall => None
                }
            })
        })
        .flatten()
        .collect::<Vec<_>>());

        // Map ywalls to rectangles
        walls.append(&mut self.ywalls[w][z].iter().enumerate().map(|(y, row)| {
            row.iter().enumerate().filter_map(move |(x, wall)| {
                // Draw a wall between cells (x, y - 1, z) and (x, y, z)
                let (x, y, z) = (x as f32, y as f32 - 0.5, z as f32);
                match wall {
                    Wall::SolidWall => Some (
                            InstanceModel { m: linalg::model([90f32.to_radians(), 0.0, 0.0], [1.0, 1.0, 1.0], [x, y, z]) }
                        ),
                    Wall::NoWall => None
                }
            })
        }).flatten().collect::<Vec<_>>());

        // Map zwalls to rectangles
        let floors: Vec<InstanceModel> = self.zwalls[w][z].iter().enumerate().map(|(y, row)| {
            row.iter().enumerate().filter_map(move |(x, wall)| {
                // Draw a floor between cells (x, y, z - 1) and (x, y, z)
                let (x, y, z) = (x as f32, y as f32, z as f32 - 0.05);
                match wall {
                    Wall::SolidWall => Some (
                            InstanceModel { m: linalg::model([90f32.to_radians(), 0.0, 0.0], [1.0, 1.0, 1.0], [x, y, z]) }
                        ),
                    Wall::NoWall => None
                }
            })
        }).flatten().collect();

        // Generate wall corners
        let mut corners: Vec<InstanceModel> = Vec::new();
        for x in 0..self.width + 1 {
            for y in 0..self.height + 1 {
                // Draw a wall corner between cells (x - 1, y - 1, z) and (x, y, z)
                let (x, y, z) = (x as f32 - 0.5, y as f32 - 0.5, z as f32);
                corners.push(InstanceModel { m: linalg::model([90f32.to_radians(), 0.0, 0.0], [1.0, 1.0, 1.0], [x, y, z]) });
            }
        }

        LevelInstances { walls, floors, corners, ceilings, left_portals, right_portals }
    }

    pub fn check_move(&self, current: [i32; 4], delta: [i32; 4]) -> bool {
        let (x, y, z, w) = (current[0] as usize, current[1] as usize, current[2] as usize, current[3] as usize);
        match delta {
            // Move left
            [-1, 0, 0, 0] => match self.xwalls[w][z][y][x] {
                Wall::SolidWall => false,
                Wall::NoWall => true
            },
            // Move right
            [1, 0, 0, 0] => match self.xwalls[w][z][y][x + 1] {
                Wall::SolidWall => false,
                Wall::NoWall => true
            },
            // Move up
            [0, -1, 0, 0] => match self.ywalls[w][z][y][x] {
                Wall::SolidWall => false,
                Wall::NoWall => true
            },
            // Move down
            [0, 1, 0, 0] => match self.ywalls[w][z][y + 1][x] {
                Wall::SolidWall => false,
                Wall::NoWall => true
            },
            // Ascend
            [0, 0, 1, 0] => match self.zwalls[w][z + 1][y][x] {
                Wall::SolidWall => false,
                Wall::NoWall => true
            }
            // Descend
            [0, 0, -1, 0] => match self.zwalls[w][z][y][x] {
                Wall::SolidWall => false,
                Wall::NoWall => true
            }
            // Increment fourth
            [0, 0, 0, 1] => match self.wwalls[w + 1][z][y][x] {
                Wall::SolidWall => false,
                Wall::NoWall => true
            }
            // Decrement fourth
            [0, 0, 0, -1] => match self.wwalls[w][z][y][x] {
                Wall::SolidWall => false,
                Wall::NoWall => true
            }
            _ => false // Invalid move
        }
    }
}
