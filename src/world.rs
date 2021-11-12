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

pub const WIDTH: usize = 10;
pub const HEIGHT: usize = 10;
pub const DEPTH: usize = 10;

#[derive(Debug, Clone, Copy)]
pub enum Cell {
    Empty
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wall {
    NoWall,
    SolidWall
}

pub struct World {
    // Dimensions: DEPTH x HEIGHT x WIDTH
    pub cells: Box<[Box<[Box<[Cell]>]>]>,
    // Vertical walls, DEPTH x HEIGHT x (WIDTH + 1)
    pub xwalls: Box<[Box<[Box<[Wall]>]>]>,
    // Horizontal walls, DEPTH x (HEIGHT + 1) x WIDTH
    pub ywalls: Box<[Box<[Box<[Wall]>]>]>,
    // Floors/Ceilings, (DEPTH + 1) x HEIGHT x WIDTH
    pub zwalls: Box<[Box<[Box<[Wall]>]>]>,

    pub start: [i32; 3],
    pub finish: [i32; 3],
    pub solution: Vec<([i32; 3])>,

    player_position_buffer_pool: CpuBufferPool<[PlayerPositionData; 1]>,
    vertex_buffers: Vec<Vec<Arc<ImmutableBuffer<[InstanceModel]>>>>
}

impl World {
    pub fn new(queue: Arc<Queue>) -> (Rc<RefCell<World>>, Box<dyn GpuFuture>) {
        // Start by creating a 2D grid, with walls around each cell
        let mut world = World {
            cells: vec![vec![vec![Cell::Empty; WIDTH].into_boxed_slice(); HEIGHT].into_boxed_slice(); DEPTH].into_boxed_slice(),
            xwalls: vec![vec![vec![Wall::SolidWall; WIDTH + 1].into_boxed_slice(); HEIGHT].into_boxed_slice(); DEPTH].into_boxed_slice(),
            ywalls: vec![vec![vec![Wall::SolidWall; WIDTH].into_boxed_slice(); HEIGHT + 1].into_boxed_slice(); DEPTH].into_boxed_slice(),
            zwalls: vec![vec![vec![Wall::SolidWall; WIDTH].into_boxed_slice(); HEIGHT].into_boxed_slice(); DEPTH + 1].into_boxed_slice(),
            start: [0, 0, 0],
            finish: [WIDTH as i32 - 1, HEIGHT as i32 - 1, DEPTH as i32 - 1],
            solution: Vec::new(),
            player_position_buffer_pool: CpuBufferPool::new(queue.device().clone(), BufferUsage::uniform_buffer()),
            vertex_buffers: Vec::new()
        };
        world.generate_maze();
        
        let world_data: Vec<_> = (0..DEPTH).map(|level| world.vertex_buffer(level)).collect();
        let world_buffer: Vec<_> =
            world_data.into_iter().map(|instance_buffers| {
                instance_buffers.map(|ibuf| {
                    ImmutableBuffer::from_iter(
                        ibuf,
                        BufferUsage::vertex_buffer(),
                        queue.clone()
                    ).expect("Failed to construct buffer")
                })
            }).collect();
        let future = now(queue.device().clone()).boxed();
        let future = world_buffer.into_iter().fold(future, |future, level| {
            let mut level_buffers = Vec::new();
            let future = level.into_iter().fold(future, |future, (buf, upload)| {
                level_buffers.push(buf);
                future.join(upload).boxed()
            });
            world.vertex_buffers.push(level_buffers);
            future.then_signal_fence_and_flush().unwrap().boxed()
        });
        println!("Initialized world");
        (Rc::new(RefCell::new(world)), future)
    }

    pub fn render(&self, models: &HashMap<String, Box<Model>>, player: &Box<Player>, desc_set_pool: &mut SingleLayoutDescSetPool, builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>, pipeline: &Pipeline) {
        let player_position_buffer = self.player_position_buffer_pool.next([
            PlayerPositionData { pos: linalg::add(player.get_position(), [0.0, 0.0, 0.4]) }
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
        let (min_level, max_level) = ((player.cell()[2] - 6).clamp(0, DEPTH as i32) as usize, player.cell()[2] as usize);
        for level in min_level..max_level + 1 {
            let bufs: Vec<_> = self.vertex_buffers[level].iter().map(|arc| arc.clone()).collect();
            let (walls, floors, corners, ceilings) = (bufs[0].clone(), bufs[1].clone(), bufs[2].clone(), bufs[3].clone());
            builder
                .push_constants(
                    pipeline.graphics_pipeline.layout().clone(),
                    0,
                    ViewProjectionData { vp: view_projection, pushColor: [0.0, 0.4, 0.8] })
                .bind_vertex_buffers(0, (models["wall"].vertices.clone(), walls.clone()))
                .draw(
                    models["wall"].vertices.len() as u32,
                    walls.len() as u32,
                    0,
                    0)
                .unwrap()
                .push_constants(
                    pipeline.graphics_pipeline.layout().clone(),
                    0,
                    ViewProjectionData { vp: view_projection, pushColor: [0.1, 0.6, 0.9] })
                .bind_vertex_buffers(0, (models["floor"].vertices.clone(), floors.clone()))
                .draw(
                    models["floor"].vertices.len() as u32,
                    floors.len() as u32,
                    0,
                    0)
                .unwrap()
                .push_constants(
                    pipeline.graphics_pipeline.layout().clone(),
                    0,
                    ViewProjectionData { vp: view_projection, pushColor: [0.0, 0.1, 0.3] })
                .bind_vertex_buffers(0, (models["corner"].vertices.clone(), corners.clone()))
                .draw(
                    models["corner"].vertices.len() as u32,
                    corners.len() as u32,
                    0,
                    0)
                .unwrap()
                .push_constants(
                    pipeline.graphics_pipeline.layout().clone(),
                    0,
                    ViewProjectionData { vp: view_projection, pushColor: [0.2, 0.8, 0.2] })
                .bind_vertex_buffers(0, (models["ceiling"].vertices.clone(), ceilings.clone()))
                .draw(
                    models["ceiling"].vertices.len() as u32,
                    ceilings.len() as u32,
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
            XWall ([usize; 3]),
            YWall ([usize; 3]),
            ZWall ([usize; 3])
        }
        let mut edges: Vec<MazeEdge> = Vec::new();
        for z in 0..DEPTH {
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    if x != 0 {
                        edges.push(MazeEdge::XWall([x, y, z]))
                    }
                    if y != 0 {
                        edges.push(MazeEdge::YWall([x, y, z]))
                    }
                    if z != 0 {
                        edges.push(MazeEdge::ZWall([x, y, z]))
                    }
                }
            }
        }
        edges.shuffle(&mut thread_rng());

        // Initialize disjoint set of cells
        let mut cells = disjoint_set::DisjointSet::new();
        for z in 0..DEPTH {
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    // Use tuples to hash correctly hopefully
                    cells.add(&(x, y, z));
                }
            }
        }

        // Take a random edge and check if the neighbor cells are connected
        // If not, remove the edge to merge them
        // Also generate map from each cell to accessible neighbors
        let mut neighbors: HashMap<(usize, usize, usize), Vec<(usize, usize, usize)>> = HashMap::new();
        for edge in edges.iter() {
            let (cell_a, cell_b) =
                match edge {
                    MazeEdge::XWall ([x, y, z]) => ((*x - 1, *y, *z), (*x, *y, *z)),
                    MazeEdge::YWall ([x, y, z]) => ((*x, *y - 1, *z), (*x, *y, *z)),
                    MazeEdge::ZWall ([x, y, z]) => ((*x, *y, *z - 1), (*x, *y, *z))
                };
            let set_a = cells.find(&cell_a);
            let set_b = cells.find(&cell_b);
            if set_a != set_b {
                // Remove edge between these cells in world
                match edge {
                    MazeEdge::XWall ([x, y, z]) => self.xwalls[*z][*y][*x] = Wall::NoWall,
                    MazeEdge::YWall ([x, y, z]) => self.ywalls[*z][*y][*x] = Wall::NoWall,
                    MazeEdge::ZWall ([x, y, z]) => self.zwalls[*z][*y][*x] = Wall::NoWall
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

        // Generate exit at bottom right corner of top layer
        self.xwalls[DEPTH - 1][HEIGHT - 1][WIDTH] = Wall::NoWall;

        // Use breadth-first search to find solution
        let mut queue: VecDeque<(usize, usize, usize)> = VecDeque::new();
        queue.push_back((0, 0, 0));
        let mut visited: HashSet<(usize, usize, usize)> = HashSet::new();
        visited.insert((0, 0, 0));
        let mut backtrack: HashMap<(usize, usize, usize), (usize, usize, usize)> = HashMap::new();
        while !queue.is_empty() {
            // Take next cell from queue
            let cell = queue.pop_front().unwrap();

            // Add unvisited neighbors to the queue
            for n in neighbors.get(&cell).unwrap() {
                if !visited.contains(n) {
                    visited.insert(*n);
                    queue.push_back(*n);
                    backtrack.insert(*n, cell);
                }
            }
        }
        // Use backtracking information to recover path
        let start = (self.start[0] as usize, self.start[1] as usize, self.start[2] as usize);
        let finish = (self.finish[0] as usize, self.finish[1] as usize, self.finish[2] as usize);
        let mut previous = finish;
        self.solution.push(self.finish);
        while previous != start {
            previous = *backtrack.get(&previous).expect("Backtracking after BFS failed, impossible");
            let (x, y, z) = previous;
            self.solution.push([x, y, z].map(|i| i as i32));
        }
        self.solution.reverse(); // Get finish at the end of the vec
    }

    pub fn vertex_buffer(&self, level: usize) -> [Vec<InstanceModel>; 4] {
        // Generate vertex data for maze
        // const FLOOR_COLOR: [f32; 3] = [ 0.9, 0.5, 0.5 ];
        // const RAINBOW: [[f32; 3]; 7] = [
        //     [ 0.8, 0.0, 0.0 ],
        //     [ 0.8, 0.4, 0.0 ],
        //     [ 0.4, 0.8, 0.0 ],
        //     [ 0.0, 0.8, 0.0 ],
        //     [ 0.0, 0.4, 0.8 ],
        //     [ 0.0, 0.0, 0.8 ],
        //     [ 0.4, 0.0, 0.8 ]
        // ];
        // const ASCEND_COLOR: [f32; 3]= [ 0.4, 1.0, 0.0 ];
        // let wall_color = RAINBOW[level % RAINBOW.len()];
        // let floor_color = wall_color.map(|f| f * 0.2);
        // let ascend_color = wall_color.map(|f| (f * 1.2).clamp(0.0, 1.0));
        let mut walls: Vec<InstanceModel> = Vec::new();
        let mut floors: Vec<InstanceModel> = Vec::new();
        let mut corners: Vec<InstanceModel> = Vec::new();
        let mut ceilings: Vec<InstanceModel> = Vec::new();

        // Mark cells with open ceilings
        ceilings.append(&mut self.cells[level].iter().enumerate().map(|(y, row)| {
            row.iter().enumerate().filter_map(move |(x, _cell)| {
                match self.zwalls[level + 1][y][x] {
                    Wall::SolidWall => None,
                    Wall::NoWall => {
                        let (x, y, z) = (x as f32, y as f32, level as f32 + 0.8);
                        Some (InstanceModel { m: linalg::model([90f32.to_radians(), 0.0, 0.0], [1.0, 1.0, 1.0], [x, y, z]) })
                    }
                }
            })
        })
        .flatten()
        .collect::<Vec<_>>());

        // Map xwalls to rectangles
        walls.append(&mut self.xwalls[level].iter().enumerate().map(|(y, row)| {
            row.iter().enumerate().filter_map(move |(x, wall)| {
                // Draw a wall between cells (x - 1, y, z) and (x, y, z)
                let (x, y, z) = (x as f32 - 0.5, y as f32, level as f32);
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
        walls.append(&mut self.ywalls[level].iter().enumerate().map(|(y, row)| {
            row.iter().enumerate().filter_map(move |(x, wall)| {
                // Draw a wall between cells (x, y - 1, z) and (x, y, z)
                let (x, y, z) = (x as f32, y as f32 - 0.5, level as f32);
                match wall {
                    Wall::SolidWall => Some (
                            InstanceModel { m: linalg::model([90f32.to_radians(), 0.0, 0.0], [1.0, 1.0, 1.0], [x, y, z]) }
                        ),
                    Wall::NoWall => None
                }
            })
        })
        .flatten()
        .collect::<Vec<_>>());

        // Map zwalls to rectangles
        floors.append(&mut self.zwalls[level].iter().enumerate().map(|(y, row)| {
            row.iter().enumerate().filter_map(move |(x, wall)| {
                // Draw a floor between cells (x, y, z - 1) and (x, y, z)
                let (x, y, z) = (x as f32, y as f32, level as f32 - 0.05);
                match wall {
                    Wall::SolidWall => Some (
                            InstanceModel { m: linalg::model([90f32.to_radians(), 0.0, 0.0], [1.0, 1.0, 1.0], [x, y, z]) }
                        ),
                    Wall::NoWall => None
                }
            })
        })
        .flatten()
        .collect::<Vec<_>>());

        // Generate wall corners
        for x in 0..WIDTH + 1 {
            for y in 0..HEIGHT + 1 {
                // Draw a wall corner between cells (x - 1, y - 1, z) and (x, y, z)
                let (x, y, z) = (x as f32 - 0.5, y as f32 - 0.5, level as f32);
                corners.push(InstanceModel { m: linalg::model([90f32.to_radians(), 0.0, 0.0], [1.0, 1.0, 1.0], [x, y, z]) });
            }
        }

        [walls, floors, corners, ceilings]
    }

    pub fn check_move(&self, current: [i32; 3], delta: [i32; 3]) -> bool {
        let (x, y, z) = (current[0] as usize, current[1] as usize, current[2] as usize);
        match delta {
            // Move up
            [0, -1, 0] => match self.ywalls[z][y][x] {
                Wall::SolidWall => false,
                Wall::NoWall => true
            },
            // Move down
            [0, 1, 0] => match self.ywalls[z][y + 1][x] {
                Wall::SolidWall => false,
                Wall::NoWall => true
            },
            // Move left
            [-1, 0, 0] => match self.xwalls[z][y][x] {
                Wall::SolidWall => false,
                Wall::NoWall => true
            },
            // Move right
            [1, 0, 0] => match self.xwalls[z][y][x + 1] {
                Wall::SolidWall => false,
                Wall::NoWall => true
            },
            // Ascend
            [0, 0, 1] => match self.zwalls[z + 1][y][x] {
                Wall::SolidWall => false,
                Wall::NoWall => true
            }
            // Descend
            [0, 0, -1] => match self.zwalls[z][y][x] {
                Wall::SolidWall => false,
                Wall::NoWall => true
            }
            _ => false // Invalid move
        }
    }
}
