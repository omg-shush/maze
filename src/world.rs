use rand::seq::SliceRandom;
use rand::thread_rng;

use super::disjoint_set;
use super::pipeline::cs::ty::{Rectangle, Vertex};

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

#[derive(Debug, Clone)]
pub struct World {
    pub cells: [[[Cell; WIDTH]; HEIGHT]; DEPTH],
    // Vertical walls
    pub xwalls: [[[Wall; WIDTH + 1]; HEIGHT]; DEPTH],
    // Horizontal walls
    pub ywalls: [[[Wall; WIDTH]; HEIGHT + 1]; DEPTH],
    // Floors/Ceilings
    pub zwalls: [[[Wall; WIDTH]; HEIGHT]; DEPTH + 1],

    pub start: [i32; 2],
    pub finish: [i32; 2]
}

impl World {
    pub fn new() -> World {
        // Start by creating a 2D grid, with walls around each cell
        World {
            cells: [[[Cell::Empty; WIDTH]; HEIGHT]; DEPTH],
            xwalls: [[[Wall::SolidWall; WIDTH + 1]; HEIGHT]; DEPTH],
            ywalls: [[[Wall::SolidWall; WIDTH]; HEIGHT + 1]; DEPTH],
            zwalls: [[[Wall::SolidWall; WIDTH]; HEIGHT]; DEPTH + 1],
            start: [0, 0],
            finish: [WIDTH as i32 - 1, HEIGHT as i32 - 1]
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
                // ... and merge the sets they belong to
                cells.union(&set_a, &set_b);
            }
        }
        // Results in minimum spanning tree connecting all cells of maze

        // Generate exit at bottom right corner of top layer
        self.xwalls[DEPTH - 1][HEIGHT - 1][WIDTH] = Wall::NoWall;
    }

    pub fn player_buffer(&self) -> Vec<Vertex> {
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

    pub fn vertex_buffer(&self, level: usize) -> Vec<Rectangle> {
        // Generate vertex data for maze
        const FLOOR_COLOR: [f32; 3] = [ 0.9, 0.5, 0.5 ];
        const WALL_COLOR: [f32; 3] = [ 0.0, 0.0, 0.8 ];
        const ASCEND_COLOR: [f32; 3]= [ 0.4, 1.0, 0.0 ];
        let mut data: Vec<Rectangle> = Vec::new();

        // Mark cells with open ceilings
        data.append(&mut self.cells[level].iter().enumerate().map(|(y, row)| {
            row.iter().enumerate().map(move |(x, _cell)| {
                match self.zwalls[level + 1][y][x] {
                    Wall::SolidWall => [].to_vec(),
                    Wall::NoWall => {
                        let (x, y, z) = (x as f32, y as f32, level as f32 + 0.8);
                        [
                            Rectangle { position: [x, y - 0.2, z], color: ASCEND_COLOR, width: 0.4, height: 0.05, depth: 0.05, .. Default::default() },
                            Rectangle { position: [x, y + 0.2, z], color: ASCEND_COLOR, width: 0.4, height: 0.05, depth: 0.05, .. Default::default() },
                            Rectangle { position: [x - 0.2, y, z], color: ASCEND_COLOR, width: 0.05, height: 0.4, depth: 0.05, .. Default::default() },
                            Rectangle { position: [x + 0.2, y, z], color: ASCEND_COLOR, width: 0.05, height: 0.4, depth: 0.05, .. Default::default() }
                        ].to_vec()
                    }
                }
            }).flatten()
        })
        .flatten()
        .collect::<Vec<_>>());

        // Map xwalls to rectangles
        data.append(&mut self.xwalls[level].iter().enumerate().map(|(y, row)| {
            row.iter().enumerate().filter_map(move |(x, wall)| {
                // Draw a wall between cells (x - 1, y, z) and (x, y, z)
                let (x, y, z) = (x as f32 - 0.5, y as f32, level as f32);
                match wall {
                    Wall::SolidWall => Some (
                            Rectangle { position: [x, y, z], color: WALL_COLOR, width: 0.2, height: 0.8, depth: 1.0, .. Default::default() }
                        ),
                    Wall::NoWall => None
                }
                
            })
        })
        .flatten()
        .collect::<Vec<_>>());

        // Map ywalls to rectangles
        data.append(&mut self.ywalls[level].iter().enumerate().map(|(y, row)| {
            row.iter().enumerate().filter_map(move |(x, wall)| {
                // Draw a wall between cells (x, y - 1, z) and (x, y, z)
                let (x, y, z) = (x as f32, y as f32 - 0.5, level as f32);
                match wall {
                    Wall::SolidWall => Some (
                            Rectangle { position: [x, y, z], color: WALL_COLOR, width: 0.8, height: 0.2, depth: 1.0, .. Default::default() }
                        ),
                    Wall::NoWall => None
                }
            })
        })
        .flatten()
        .collect::<Vec<_>>());

        // Map zwalls to rectangles
        data.append(&mut self.zwalls[level].iter().enumerate().map(|(y, row)| {
            row.iter().enumerate().filter_map(move |(x, wall)| {
                // Draw a floor/ceiling between cells (x, y, z - 1) and (x, y, z)
                let (x, y, z) = (x as f32, y as f32, level as f32 - 0.05);
                match wall {
                    Wall::SolidWall => Some (
                            Rectangle { position: [x, y, z], color: FLOOR_COLOR, width: 1.0, height: 1.0, depth: 0.1, .. Default::default() }
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
                // Draw a wall corner between cells ([)x - 1, y - 1, z) and (x, y, z)
                let (x, y, z) = (x as f32 - 0.5, y as f32 - 0.5, level as f32);
                data.push(Rectangle { position: [x, y, z], color: WALL_COLOR, width: 0.2, height: 0.2, depth: 1.0, .. Default::default() });
            }
        }

        data
    }

    pub fn check_move(&self, current: [i32; 3], delta: [i32; 3]) -> bool {
        // if current[0] != proposed[0] && current[1] == proposed[1] {
        //     return false
        // }
        // if (current[0] - proposed[0]).abs() > 1 || (current[1] - proposed[1]).abs() > 1 {
        //     return false
        // }
        // if proposed[0] < 0 || proposed[0] >= WIDTH || proposed[1] < 0 || proposed[1] >= HEIGHT {
        //     return false
        // }
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
