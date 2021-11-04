use std::sync::Arc;

use rand::seq::SliceRandom;
use rand::thread_rng;

use vulkano::device::Device;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::impl_vertex;
use vulkano::buffer::BufferUsage;

use super::disjoint_set;

#[derive(Default, Debug, Clone)]
pub struct Vertex {
    pub position: [f32; 2],
    pub color: [f32; 3]
}
impl_vertex!(Vertex, position, color);

const WIDTH: i32 = 20;
const HEIGHT: i32 = 20;

#[derive(Debug, Clone, Copy)]
pub enum Cell {
    Empty
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wall {
    NoWall,
    SolidWall
}

#[derive(Debug, Clone)]
pub struct World {
    pub cells: [[Cell; WIDTH as usize]; HEIGHT as usize],
    // Vertical walls
    pub xwalls: [[Wall; WIDTH as usize + 1]; HEIGHT as usize],
    // Horizontal walls
    pub ywalls: [[Wall; WIDTH as usize]; HEIGHT as usize + 1],
    pub start: [i32; 2],
    pub finish: [i32; 2]
}

pub fn generate_maze(world: &mut World) {
    // Use randomized kruskal's algorithm

    // Random list of edges
    #[derive(Debug)]
    enum MazeEdge {
        XWall ([i32; 2]),
        YWall ([i32; 2])
    }
    let mut edges: Vec<MazeEdge> = Vec::new();
    for (y, row) in world.xwalls.iter().enumerate() {
        for (x, _) in row.iter().enumerate() {
            if x != 0 && x != WIDTH as usize {
                edges.push(MazeEdge::XWall([x as i32, y as i32]));
            }
        }
    }
    for (y, row) in world.ywalls.iter().enumerate() {
        for (x, _) in row.iter().enumerate() {
            if y != 0 && y != HEIGHT as usize {
                edges.push(MazeEdge::YWall([x as i32, y as i32]));
            }
        }
    }
    edges.shuffle(&mut thread_rng());

    // Initialize disjoint set of cells
    let mut cells = disjoint_set::DisjointSet::new();
    for (y, row) in world.cells.iter().enumerate() {
        for (x, _) in row.iter().enumerate() {
            // Use tuples to hash correclty hopefully
            cells.add(&(x as i32, y as i32));
        }
    }

    // Take a random edge and check if the neighbor cells are connected
    // If not, remove the edge to merge them
    for edge in edges.iter() {
        let (cell_a, cell_b) =
            match edge {
                MazeEdge::XWall ([x, y]) => ((*x - 1, *y), (*x, *y)),
                MazeEdge::YWall ([x, y]) => ((*x, *y - 1), (*x, *y))
            };
        let set_a = cells.find(&cell_a);
        let set_b = cells.find(&cell_b);
        if set_a != set_b {
            // Remove edge between these cells in world
            match edge {
                MazeEdge::XWall ([x, y]) => world.xwalls[*y as usize][*x as usize] = Wall::NoWall,
                MazeEdge::YWall ([x, y]) => world.ywalls[*y as usize][*x as usize] = Wall::NoWall
            }
            // ... and merge the sets they belong to
            cells.union(&set_a, &set_b);
        }
    }
    // Results in minimum spanning tree connecting all cells of maze
    println!("# cells in maze: {}", cells.len())
}

pub fn player_buffer(device: Arc<Device>) -> Arc<CpuAccessibleBuffer<[Vertex]>> {
    const PLAYER_COLOR: [f32; 3] = [ 0.8, 0.2, 0.2 ];
    const HALF_SIZE: f32 = 0.2;
    let (x, y) = (0.0, 0.0);
    let data = [
        Vertex { position: [ x + HALF_SIZE, y + HALF_SIZE ], color: PLAYER_COLOR },
        Vertex { position: [ x + HALF_SIZE, y - HALF_SIZE ], color: PLAYER_COLOR },
        Vertex { position: [ x - HALF_SIZE, y - HALF_SIZE ], color: PLAYER_COLOR },
        Vertex { position: [ x - HALF_SIZE, y - HALF_SIZE ], color: PLAYER_COLOR },
        Vertex { position: [ x - HALF_SIZE, y + HALF_SIZE ], color: PLAYER_COLOR },
        Vertex { position: [ x + HALF_SIZE, y + HALF_SIZE ], color: PLAYER_COLOR }
    ].to_vec();

    CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::vertex_buffer(),
        false,
        data
    ).unwrap()
}

pub fn vertex_buffer(device: Arc<Device>) -> Arc<CpuAccessibleBuffer<[Vertex]>> {
    // Start by creating a 2D grid, with walls around each cell
    let mut world = World {
        cells: [[Cell::Empty; WIDTH as usize]; HEIGHT as usize],
        xwalls: [[Wall::SolidWall; WIDTH as usize + 1]; HEIGHT as usize],
        ywalls: [[Wall::SolidWall; WIDTH as usize]; HEIGHT as usize + 1],
        start: [0, 0],
        finish: [WIDTH - 1, HEIGHT - 1]
    };

    // Construct a random maze by setting certain walls to NoWall
    generate_maze(&mut world);

    // Generate vertex data for maze
    const CELL_COLOR: [f32; 3] = [ 0.9, 0.5, 0.5 ];
    const WALL_COLOR: [f32; 3] = [ 0.0, 0.0, 0.0 ];
    let mut data: Vec<Vertex> = Vec::new();

    // Map cells to squares
    data.append(&mut world.cells.iter().enumerate().map(|(y, row)| {
        row.iter().enumerate().map(move |(x, _cell)| {
            // Draw a square around cell (x, y)
            const HALF_SIZE: f32 = 0.5;
            let (x, y) = (x as f32, y as f32);
            [
                Vertex { position: [ x + HALF_SIZE, y + HALF_SIZE ], color: CELL_COLOR },
                Vertex { position: [ x + HALF_SIZE, y - HALF_SIZE ], color: CELL_COLOR },
                Vertex { position: [ x - HALF_SIZE, y - HALF_SIZE ], color: CELL_COLOR },
                Vertex { position: [ x - HALF_SIZE, y - HALF_SIZE ], color: CELL_COLOR },
                Vertex { position: [ x - HALF_SIZE, y + HALF_SIZE ], color: CELL_COLOR },
                Vertex { position: [ x + HALF_SIZE, y + HALF_SIZE ], color: CELL_COLOR }
            ]
        })
    })
    .flatten()
    .flatten()
    .collect::<Vec<_>>());

    // Map xwalls to rectangles
    data.append(&mut world.xwalls.iter().enumerate().map(|(y, row)| {
        row.iter().enumerate().filter_map(move |(x, wall)| {
            // Draw a wall between cells (x - 1, y) and (x, y)
            const HALF_WIDTH: f32 = 0.1;
            const HALF_HEIGHT: f32 = 0.4;
            let (x, y) = (x as f32 - 0.5, y as f32);
            match wall {
                Wall::SolidWall => Some ([
                        Vertex { position: [ x + HALF_WIDTH, y + HALF_HEIGHT ], color: WALL_COLOR },
                        Vertex { position: [ x + HALF_WIDTH, y - HALF_HEIGHT ], color: WALL_COLOR },
                        Vertex { position: [ x - HALF_WIDTH, y - HALF_HEIGHT ], color: WALL_COLOR },
                        Vertex { position: [ x - HALF_WIDTH, y - HALF_HEIGHT ], color: WALL_COLOR },
                        Vertex { position: [ x - HALF_WIDTH, y + HALF_HEIGHT ], color: WALL_COLOR },
                        Vertex { position: [ x + HALF_WIDTH, y + HALF_HEIGHT ], color: WALL_COLOR }
                    ]),
                Wall::NoWall => None
            }
            
        })
    })
    .flatten()
    .flatten()
    .collect::<Vec<_>>());

    // Map ywalls to rectangles
    data.append(&mut world.ywalls.iter().enumerate().map(|(y, row)| {
        row.iter().enumerate().filter_map(move |(x, wall)| {
            // Draw a wall between cells (x, y - 1) and (x, y)
            const HALF_WIDTH: f32 = 0.4;
            const HALF_HEIGHT: f32 = 0.1;
            let (x, y) = (x as f32, y as f32 - 0.5);
            match wall {
                Wall::SolidWall => Some ([
                        Vertex { position: [ x + HALF_WIDTH, y + HALF_HEIGHT ], color: WALL_COLOR },
                        Vertex { position: [ x + HALF_WIDTH, y - HALF_HEIGHT ], color: WALL_COLOR },
                        Vertex { position: [ x - HALF_WIDTH, y - HALF_HEIGHT ], color: WALL_COLOR },
                        Vertex { position: [ x - HALF_WIDTH, y - HALF_HEIGHT ], color: WALL_COLOR },
                        Vertex { position: [ x - HALF_WIDTH, y + HALF_HEIGHT ], color: WALL_COLOR },
                        Vertex { position: [ x + HALF_WIDTH, y + HALF_HEIGHT ], color: WALL_COLOR }
                    ]),
                Wall::NoWall => None
            }
        })
    })
    .flatten()
    .flatten()
    .collect::<Vec<_>>());

    // Generate wall corners, if at least one adjacent x- or y- wall is solid
    for x in 0..WIDTH + 1 {
        for y in 0..HEIGHT + 1 {
            // let (xu, yu) = (x as usize, y as usize);
            // if x == 0 || x == WIDTH - 1 || y == 0 || y == HEIGHT - 1 ||
            //    world.xwalls[yu][xu] == Wall::SolidWall ||
            //    world.xwalls[yu + 1][xu] == Wall::SolidWall ||
            //    world.ywalls[yu][xu] == Wall::SolidWall ||
            //    world.ywalls[yu][xu + 1] == Wall::SolidWall {

            // Draw a wall corner between cells [x - 1, y - 1] and [x, y]
            const HALF_SIZE: f32 = 0.1;
            let (x, y) = (x as f32 - 0.5, y as f32 - 0.5);
            data.append(&mut [
                Vertex { position: [ x + HALF_SIZE, y + HALF_SIZE ], color: WALL_COLOR },
                Vertex { position: [ x + HALF_SIZE, y - HALF_SIZE ], color: WALL_COLOR },
                Vertex { position: [ x - HALF_SIZE, y - HALF_SIZE ], color: WALL_COLOR },
                Vertex { position: [ x - HALF_SIZE, y - HALF_SIZE ], color: WALL_COLOR },
                Vertex { position: [ x - HALF_SIZE, y + HALF_SIZE ], color: WALL_COLOR },
                Vertex { position: [ x + HALF_SIZE, y + HALF_SIZE ], color: WALL_COLOR }
            ].to_vec())
        }
    }

    CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::vertex_buffer(),
        false,
        data
    ).unwrap()
}
