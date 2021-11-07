use rand::seq::SliceRandom;
use rand::thread_rng;

use vulkano::impl_vertex;

use super::disjoint_set;
use super::pipeline::cs::ty::Vertex;

impl_vertex!(Vertex, position, color);

pub const WIDTH: i32 = 10;
pub const HEIGHT: i32 = 10;

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

impl World {
    pub fn new() -> World {
        // Start by creating a 2D grid, with walls around each cell
        World {
            cells: [[Cell::Empty; WIDTH as usize]; HEIGHT as usize],
            xwalls: [[Wall::SolidWall; WIDTH as usize + 1]; HEIGHT as usize],
            ywalls: [[Wall::SolidWall; WIDTH as usize]; HEIGHT as usize + 1],
            start: [0, 0],
            finish: [WIDTH - 1, HEIGHT - 1]
        }
    }

    pub fn generate_maze(&mut self) {
        // Use randomized kruskal's algorithm

        // Random list of edges
        #[derive(Debug)]
        enum MazeEdge {
            XWall ([i32; 2]),
            YWall ([i32; 2])
        }
        let mut edges: Vec<MazeEdge> = Vec::new();
        for (y, row) in self.xwalls.iter().enumerate() {
            for (x, _) in row.iter().enumerate() {
                if x != 0 && x != WIDTH as usize {
                    edges.push(MazeEdge::XWall([x as i32, y as i32]));
                }
            }
        }
        for (y, row) in self.ywalls.iter().enumerate() {
            for (x, _) in row.iter().enumerate() {
                if y != 0 && y != HEIGHT as usize {
                    edges.push(MazeEdge::YWall([x as i32, y as i32]));
                }
            }
        }
        edges.shuffle(&mut thread_rng());

        // Initialize disjoint set of cells
        let mut cells = disjoint_set::DisjointSet::new();
        for (y, row) in self.cells.iter().enumerate() {
            for (x, _) in row.iter().enumerate() {
                // Use tuples to hash correctly hopefully
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
                    MazeEdge::XWall ([x, y]) => self.xwalls[*y as usize][*x as usize] = Wall::NoWall,
                    MazeEdge::YWall ([x, y]) => self.ywalls[*y as usize][*x as usize] = Wall::NoWall
                }
                // ... and merge the sets they belong to
                cells.union(&set_a, &set_b);
            }
        }
        // Results in minimum spanning tree connecting all cells of maze

        // Generate exit at bottom right corner
        self.xwalls[HEIGHT as usize - 1][WIDTH as usize] = Wall::NoWall;
    }

    pub fn player_buffer(&self) -> Vec<Vertex> {
        const PLAYER_COLOR: [f32; 3] = [ 0.2, 0.2, 0.8 ];
        const HALF_SIZE: f32 = 0.2;
        let (x, y) = (0.0, 0.0);
        let data = [
            Vertex { position: [ x + HALF_SIZE, y + HALF_SIZE ], color: PLAYER_COLOR, .. Default::default() },
            Vertex { position: [ x + HALF_SIZE, y - HALF_SIZE ], color: PLAYER_COLOR, .. Default::default() },
            Vertex { position: [ x - HALF_SIZE, y - HALF_SIZE ], color: PLAYER_COLOR, .. Default::default() },
            Vertex { position: [ x - HALF_SIZE, y - HALF_SIZE ], color: PLAYER_COLOR, .. Default::default() },
            Vertex { position: [ x - HALF_SIZE, y + HALF_SIZE ], color: PLAYER_COLOR, .. Default::default() },
            Vertex { position: [ x + HALF_SIZE, y + HALF_SIZE ], color: PLAYER_COLOR, .. Default::default() }
        ].to_vec();

        data
    }

    pub fn vertex_buffer(&self) -> Vec<Vertex> {
        // Generate vertex data for maze
        const CELL_COLOR: [f32; 3] = [ 0.9, 0.5, 0.5 ];
        const WALL_COLOR: [f32; 3] = [ 0.0, 0.0, 0.0 ];
        let mut data: Vec<Vertex> = Vec::new();

        // Map cells to squares
        data.append(&mut self.cells.iter().enumerate().map(|(y, row)| {
            row.iter().enumerate().map(move |(x, _cell)| {
                // Draw a square around cell (x, y)
                const HALF_SIZE: f32 = 0.5;
                let (x, y) = (x as f32, y as f32);
                [
                    Vertex { position: [ x + HALF_SIZE, y + HALF_SIZE ], color: CELL_COLOR, .. Default::default() },
                    Vertex { position: [ x + HALF_SIZE, y - HALF_SIZE ], color: CELL_COLOR, .. Default::default() },
                    Vertex { position: [ x - HALF_SIZE, y - HALF_SIZE ], color: CELL_COLOR, .. Default::default() },
                    Vertex { position: [ x - HALF_SIZE, y - HALF_SIZE ], color: CELL_COLOR, .. Default::default() },
                    Vertex { position: [ x - HALF_SIZE, y + HALF_SIZE ], color: CELL_COLOR, .. Default::default() },
                    Vertex { position: [ x + HALF_SIZE, y + HALF_SIZE ], color: CELL_COLOR, .. Default::default() }
                ]
            })
        })
        .flatten()
        .flatten()
        .collect::<Vec<_>>());

        // Map xwalls to rectangles
        data.append(&mut self.xwalls.iter().enumerate().map(|(y, row)| {
            row.iter().enumerate().filter_map(move |(x, wall)| {
                // Draw a wall between cells (x - 1, y) and (x, y)
                const HALF_WIDTH: f32 = 0.1;
                const HALF_HEIGHT: f32 = 0.4;
                let (x, y) = (x as f32 - 0.5, y as f32);
                match wall {
                    Wall::SolidWall => Some ([
                            Vertex { position: [ x + HALF_WIDTH, y + HALF_HEIGHT ], color: WALL_COLOR, .. Default::default() },
                            Vertex { position: [ x + HALF_WIDTH, y - HALF_HEIGHT ], color: WALL_COLOR, .. Default::default() },
                            Vertex { position: [ x - HALF_WIDTH, y - HALF_HEIGHT ], color: WALL_COLOR, .. Default::default() },
                            Vertex { position: [ x - HALF_WIDTH, y - HALF_HEIGHT ], color: WALL_COLOR, .. Default::default() },
                            Vertex { position: [ x - HALF_WIDTH, y + HALF_HEIGHT ], color: WALL_COLOR, .. Default::default() },
                            Vertex { position: [ x + HALF_WIDTH, y + HALF_HEIGHT ], color: WALL_COLOR, .. Default::default() }
                        ]),
                    Wall::NoWall => None
                }
                
            })
        })
        .flatten()
        .flatten()
        .collect::<Vec<_>>());

        // Map ywalls to rectangles
        data.append(&mut self.ywalls.iter().enumerate().map(|(y, row)| {
            row.iter().enumerate().filter_map(move |(x, wall)| {
                // Draw a wall between cells (x, y - 1) and (x, y)
                const HALF_WIDTH: f32 = 0.4;
                const HALF_HEIGHT: f32 = 0.1;
                let (x, y) = (x as f32, y as f32 - 0.5);
                match wall {
                    Wall::SolidWall => Some ([
                            Vertex { position: [ x + HALF_WIDTH, y + HALF_HEIGHT ], color: WALL_COLOR, .. Default::default() },
                            Vertex { position: [ x + HALF_WIDTH, y - HALF_HEIGHT ], color: WALL_COLOR, .. Default::default() },
                            Vertex { position: [ x - HALF_WIDTH, y - HALF_HEIGHT ], color: WALL_COLOR, .. Default::default() },
                            Vertex { position: [ x - HALF_WIDTH, y - HALF_HEIGHT ], color: WALL_COLOR, .. Default::default() },
                            Vertex { position: [ x - HALF_WIDTH, y + HALF_HEIGHT ], color: WALL_COLOR, .. Default::default() },
                            Vertex { position: [ x + HALF_WIDTH, y + HALF_HEIGHT ], color: WALL_COLOR, .. Default::default() }
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
                    Vertex { position: [ x + HALF_SIZE, y + HALF_SIZE ], color: WALL_COLOR, .. Default::default() },
                    Vertex { position: [ x + HALF_SIZE, y - HALF_SIZE ], color: WALL_COLOR, .. Default::default() },
                    Vertex { position: [ x - HALF_SIZE, y - HALF_SIZE ], color: WALL_COLOR, .. Default::default() },
                    Vertex { position: [ x - HALF_SIZE, y - HALF_SIZE ], color: WALL_COLOR, .. Default::default() },
                    Vertex { position: [ x - HALF_SIZE, y + HALF_SIZE ], color: WALL_COLOR, .. Default::default() },
                    Vertex { position: [ x + HALF_SIZE, y + HALF_SIZE ], color: WALL_COLOR, .. Default::default() }
                ].to_vec())
            }
        }

        data
    }

    pub fn check_move(&self, current: [i32; 2], proposed: [i32; 2]) -> bool {
        // if current[0] != proposed[0] && current[1] == proposed[1] {
        //     return false
        // }
        // if (current[0] - proposed[0]).abs() > 1 || (current[1] - proposed[1]).abs() > 1 {
        //     return false
        // }
        // if proposed[0] < 0 || proposed[0] >= WIDTH || proposed[1] < 0 || proposed[1] >= HEIGHT {
        //     return false
        // }
        let (x, y) = (current[0] as usize, current[1] as usize);
        match (proposed[0] - current[0], proposed[1] - current[1]) {
            // Move up
            (0, -1) => match self.ywalls[y][x] {
                Wall::SolidWall => false,
                Wall::NoWall => true
            },
            // Move down
            (0, 1) => match self.ywalls[y + 1][x] {
                Wall::SolidWall => false,
                Wall::NoWall => true
            },
            // Move left
            (-1, 0) => match self.xwalls[y][x] {
                Wall::SolidWall => false,
                Wall::NoWall => true
            },
            // Move right
            (1, 0) => match self.xwalls[y][x + 1] {
                Wall::SolidWall => false,
                Wall::NoWall => true
            },
            _ => false
        }
    }
}
