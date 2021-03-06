use rand::seq::SliceRandom;
use rand::{Rng, thread_rng};
use rand::rngs::ThreadRng;
use std::collections::hash_map::HashMap;
use std::collections::hash_set::HashSet;
use std::collections::vec_deque::VecDeque;
use std::sync::Arc;

use vulkano::pipeline::PipelineBindPoint;
use vulkano::buffer::{BufferUsage, CpuBufferPool, ImmutableBuffer, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::SingleLayoutDescSetPool;
use vulkano::device::Queue;
use vulkano::sync::{now, GpuFuture};

use crate::ghost::Ghost;
use crate::linalg;
use crate::pipeline::Pipeline;
use crate::disjoint_set;
use crate::pipeline::InstanceModel;
use crate::player::Player;
use crate::model::Model;
use crate::pipeline::vs::ty::{ViewProjectionData, PlayerPositionData};
use crate::parameters::RAINBOW;
use crate::config::Config;

pub type Coordinate = (usize, usize, usize, usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    Empty,
    Food
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
    pub cells: Vec<Vec<Vec<Vec<Cell>>>>,
    // Vertical walls, fourth x depth x height x (width + 1)
    pub xwalls: Vec<Vec<Vec<Vec<Wall>>>>,
    // Horizontal walls, fourth x depth x (height + 1) x width
    pub ywalls: Vec<Vec<Vec<Vec<Wall>>>>,
    // Floors/Ceilings, fourth x (depth + 1) x height x width
    pub zwalls: Vec<Vec<Vec<Vec<Wall>>>>,
    // I don't even know any more, (fourth + 1) x depth x height x width
    pub wwalls: Vec<Vec<Vec<Vec<Wall>>>>,

    player_position_buffer_pool: CpuBufferPool<[PlayerPositionData; 1]>,
    vertex_buffers: Vec<Vec<LevelBuffers>>, // lists of model matrices, indexed by: fourth -> level
    neighbors: HashMap<Coordinate, Vec<Coordinate>>
}

impl World {
    pub fn new(config: &Config, queue: Arc<Queue>) -> (World, Box<dyn GpuFuture>) {
        // Start by creating a 2D grid, with walls around each cell
        let [width, height, depth, fourth] = config.dimensions;
        let mut world = World {
            cells: vec![vec![vec![vec![Cell::Empty; width]; height]; depth]; fourth],
            xwalls: vec![vec![vec![vec![Wall::SolidWall; width + 1]; height]; depth]; fourth],
            ywalls: vec![vec![vec![vec![Wall::SolidWall; width]; height + 1]; depth]; fourth],
            zwalls: vec![vec![vec![vec![Wall::SolidWall; width]; height]; depth + 1]; fourth],
            wwalls: vec![vec![vec![vec![Wall::SolidWall; width]; height]; depth]; fourth + 1],
            player_position_buffer_pool: CpuBufferPool::new(queue.device().clone(), BufferUsage::uniform_buffer()),
            vertex_buffers: Vec::new(),
            neighbors: HashMap::new(),
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
        (world, future)
    }

    pub fn render(&self, models: &HashMap<String, Model>, player: &Player, ghost: &Ghost, desc_set_pool: &mut SingleLayoutDescSetPool, builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>, pipeline: &Pipeline) {
        let view_projection = linalg::mul(player.camera.projection(), player.camera.view());

        let fourth = player.cell()[3];
        let between = player.get_position()[3];

        for w in fourth - 2..=fourth + 2 {
            if w >= 0 && w < self.fourth as i32 {
                let w = w as usize;

                let player_position_buffer = self.player_position_buffer_pool.next([
                    PlayerPositionData {
                        player_pos: {
                            let diff = w as f32 - player.get_position()[3];
                            let mut arr: [f32; 3] = player.get_position()[0..3].try_into().unwrap();
                            arr[0] -= diff * (1 + self.width) as f32;
                            arr
                        },
                        ghost_pos: {
                            let diff = w as f32 - ghost.position()[3];
                            let mut arr: [f32; 3] = ghost.position()[0..3].try_into().unwrap();
                            arr[0] -= diff * (1 + self.width) as f32;
                            arr
                        },
                        ..Default::default()
                    }
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

                let wvp = linalg::mul(view_projection, self.world_transform(w, between));
                self.render_fourth(w, wvp, player, models, builder, pipeline);
            }
        }
    }

    pub fn world_transform(&self, fourth: usize, between: f32) -> [[f32; 4]; 4] {
        let spacing = (self.width + 1) as f32;
        linalg::translate([(fourth as f32 - between) * spacing, 0.0, 0.0])
    }

    fn render_fourth(&self, fourth: usize, view_projection: [[f32; 4]; 4], player: &Player, models: &HashMap<String, Model>, builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>, pipeline: &Pipeline) {
        let fourth_color = RAINBOW[fourth % RAINBOW.len()];
        let left_color = RAINBOW[(fourth as i32 - 1).rem_euclid(RAINBOW.len() as i32) as usize];
        let right_color = RAINBOW[(fourth + 1) % RAINBOW.len()];
        let corner_color = fourth_color.map(|f| (f * 1.2).clamp(0.0, 1.0));
        let floor_color = fourth_color.map(|f| f * 0.1);
        let ascend_color = [1.0, 1.0, 1.0];
        let (min_level, max_level) = ((player.cell()[2] - 6).clamp(0, self.depth as i32) as usize, player.cell()[2] as usize);
        for level in min_level..=max_level {
            let level_buffers = &self.vertex_buffers[fourth][level];
            let draws = [
                (fourth_color, &models["wall"], level_buffers.walls.clone()),
                (floor_color, &models["floor"], level_buffers.floors.clone()),
                (corner_color, &models["corner"], level_buffers.corners.clone()),
                (ascend_color, &models["ceiling"], level_buffers.ceilings.clone()),
                (left_color, &models["ceiling"], level_buffers.left_portals.clone()),
                (right_color, &models["ceiling"], level_buffers.right_portals.clone()),
            ];
            for (color, model, instances) in draws {
                builder
                    .push_constants(
                        pipeline.graphics_pipeline.layout().clone(),
                        0,
                        ViewProjectionData { vp: view_projection, pushColor: color })
                    .bind_vertex_buffers(0, (model.vertices.clone(), instances.clone()))
                    .draw(
                        model.vertices.len() as u32,
                        instances.len() as u32,
                        0,
                        0)
                    .unwrap();
            }
        }
    }

    fn generate_maze(&mut self) {
        // Use randomized kruskal's algorithm
        let mut rng = thread_rng();

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
        edges.shuffle(&mut rng);

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
            let within_level = match edge {
                MazeEdge::XWall (_) | MazeEdge::YWall (_) => true,
                _ => false
            };
            if set_a != set_b || (within_level && rng.gen_bool(0.3)) {
                // Remove edge between these cells in world
                match edge {
                    MazeEdge::XWall ((x, y, z, w)) => self.xwalls[*w][*z][*y][*x] = Wall::NoWall,
                    MazeEdge::YWall ((x, y, z, w)) => self.ywalls[*w][*z][*y][*x] = Wall::NoWall,
                    MazeEdge::ZWall ((x, y, z, w)) => self.zwalls[*w][*z][*y][*x] = Wall::NoWall,
                    MazeEdge::WWall ((x, y, z, w)) => self.wwalls[*w][*z][*y][*x] = Wall::NoWall
                }
                // Mark them as neighbors for BFS later
                if !self.neighbors.contains_key(&cell_a) {
                    self.neighbors.insert(cell_a, Vec::new());
                }
                if !self.neighbors.contains_key(&cell_b) {
                    self.neighbors.insert(cell_b, Vec::new());
                }
                self.neighbors.get_mut(&cell_a).unwrap().push(cell_b);
                self.neighbors.get_mut(&cell_b).unwrap().push(cell_a);
                // And merge the sets they belong to
                cells.union(&set_a, &set_b);
            }
        }
        // Results in minimum spanning tree connecting all cells of maze
    }

    pub fn random_empty_cell(&self) -> Coordinate {
        fn gen(world: &World, rng: &mut ThreadRng) -> Coordinate {
            (rng.gen_range(0..world.width), rng.gen_range(0..world.height), rng.gen_range(0..world.depth), rng.gen_range(0..world.fourth))
        }
        let mut rng = thread_rng();
        let (mut x, mut y, mut z, mut w) = gen(self, &mut rng);
        while self.cells[w][z][y][x] != Cell::Empty {
            let (nx, ny, nz, nw) = gen(self, &mut rng);
            x = nx;
            y = ny;
            z = nz;
            w = nw;
        }
        (x, y, z, w)
    }

    pub fn bfs(&self, start: Coordinate, finish: Coordinate) -> Vec<Coordinate> {
        // Use breadth-first search to find solution
        let mut queue: VecDeque<Coordinate> = VecDeque::new();
        queue.push_back(start);
        let mut visited: HashSet<Coordinate> = HashSet::new();
        visited.insert(start);
        let mut backtrack: HashMap<Coordinate, Coordinate> = HashMap::new();
        while !queue.is_empty() {
            // Take next cell from queue
            let cell = queue.pop_front().unwrap();

            // Add unvisited neighbors to the queue
            for n in self.neighbors.get(&cell).unwrap_or(&Vec::new()) {
                if !visited.contains(n) {
                    visited.insert(*n);
                    queue.push_back(*n);
                    backtrack.insert(*n, cell);
                    if *n == finish {
                        break;
                    }
                }
            }
        }
        // Use backtracking information to recover path
        let mut solution: Vec<Coordinate> = Vec::new();
        let mut previous = finish;
        solution.push(finish);
        while previous != start {
            previous = *backtrack.get(&previous).expect("Backtracking after BFS failed, impossible");
            solution.push(previous);
        }
        solution.reverse(); // Get finish at the end of the vec
        solution
    }

    // Given fixed w and z coordinates, generate a list of instances of each type of object within the level
    fn vertex_buffer(&self, w: usize, z: usize) -> LevelInstances {
        // Mark fourth-dimensional portals i guess
        let left_portals: Vec<InstanceModel> = self.cells[w][z].iter().enumerate().flat_map(|(y, row)| {
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
        }).collect();
        let right_portals: Vec<InstanceModel> = self.cells[w][z].iter().enumerate().flat_map(|(y, row)| {
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
        }).collect();

        // Map horizontal walls
        let top_to_down = self.xwalls[w][z].iter().enumerate().flat_map(|(y, row)| {
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
        });
        let left_to_right = self.ywalls[w][z].iter().enumerate().flat_map(|(y, row)| {
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
        });
        let walls: Vec<InstanceModel> = top_to_down.chain(left_to_right).collect();

        // Map floors to rectangles
        let floors: Vec<InstanceModel> = self.zwalls[w][z].iter().enumerate().flat_map(|(y, row)| {
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
        }).collect();

        // Mark cells with open ceilings
        let ceilings: Vec<InstanceModel> = self.cells[w][z].iter().enumerate().flat_map(|(y, row)| {
            row.iter().enumerate().filter_map(move |(x, _cell)| {
                match self.zwalls[w][z + 1][y][x] {
                    Wall::SolidWall => None,
                    Wall::NoWall => {
                        let (x, y, z) = (x as f32, y as f32, z as f32 + 0.8);
                        Some (InstanceModel { m: linalg::model([90f32.to_radians(), 0.0, 0.0], [1.0, 1.0, 1.0], [x, y, z]) })
                    }
                }
            })
        }).collect();

        // Generate wall corners
        let mut corners: Vec<InstanceModel> = Vec::new();
        for x in 0..self.width + 1 {
            for y in 0..self.height + 1 {
                // Only add corner if at least 1 horizontal wall is touching
                if (y < self.height && self.xwalls[w][z][y][x] != Wall::NoWall)
                || (x < self.width && self.ywalls[w][z][y][x] != Wall::NoWall)
                || self.xwalls[w][z][y - 1][x] != Wall::NoWall
                || self.ywalls[w][z][y][x - 1] != Wall::NoWall {
                    // Draw a wall corner between cells (x - 1, y - 1, z) and (x, y, z)
                    let (x, y, z) = (x as f32 - 0.5, y as f32 - 0.5, z as f32);
                    corners.push(InstanceModel { m: linalg::model([90f32.to_radians(), 0.0, 0.0], [1.0, 1.0, 1.0], [x, y, z]) });
                }
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
