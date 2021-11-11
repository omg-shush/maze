use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::hash_map::HashMap;
use std::collections::hash_set::HashSet;
use std::collections::vec_deque::VecDeque;
use std::rc::Rc;
use std::cell::RefCell;

use crate::linalg;

use super::disjoint_set;
use super::pipeline::InstanceModel;
use super::pipeline::cs::ty::Vertex;

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
    pub solution: Vec<([i32; 3])>
}

impl World {
    pub fn new() -> Rc<RefCell<World>> {
        // Start by creating a 2D grid, with walls around each cell
        Rc::new(RefCell::new(World {
            cells: vec![vec![vec![Cell::Empty; WIDTH].into_boxed_slice(); HEIGHT].into_boxed_slice(); DEPTH].into_boxed_slice(),
            xwalls: vec![vec![vec![Wall::SolidWall; WIDTH + 1].into_boxed_slice(); HEIGHT].into_boxed_slice(); DEPTH].into_boxed_slice(),
            ywalls: vec![vec![vec![Wall::SolidWall; WIDTH].into_boxed_slice(); HEIGHT + 1].into_boxed_slice(); DEPTH].into_boxed_slice(),
            zwalls: vec![vec![vec![Wall::SolidWall; WIDTH].into_boxed_slice(); HEIGHT].into_boxed_slice(); DEPTH + 1].into_boxed_slice(),
            start: [0, 0, 0],
            finish: [WIDTH as i32 - 1, HEIGHT as i32 - 1, DEPTH as i32 - 1],
            solution: Vec::new()
        }))
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

    pub fn vertex_buffer(&self, level: usize) -> (Vec<InstanceModel>, Vec<InstanceModel>, Vec<InstanceModel>, Vec<InstanceModel>) {
        // Generate vertex data for maze
        // const FLOOR_COLOR: [f32; 3] = [ 0.9, 0.5, 0.5 ];
        const RAINBOW: [[f32; 3]; 7] = [
            [ 0.8, 0.0, 0.0 ],
            [ 0.8, 0.4, 0.0 ],
            [ 0.4, 0.8, 0.0 ],
            [ 0.0, 0.8, 0.0 ],
            [ 0.0, 0.4, 0.8 ],
            [ 0.0, 0.0, 0.8 ],
            [ 0.4, 0.0, 0.8 ]
        ];
        // const ASCEND_COLOR: [f32; 3]= [ 0.4, 1.0, 0.0 ];
        let wall_color = RAINBOW[level % RAINBOW.len()];
        let floor_color = wall_color.map(|f| f * 0.2);
        let ascend_color = wall_color.map(|f| (f * 1.2).clamp(0.0, 1.0));
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
                        // [
                        //     Rectangle { position: [x, y - 0.2, z], color: ascend_color, width: 0.4, height: 0.05, depth: 0.05, .. Default::default() },
                        //     Rectangle { position: [x, y + 0.2, z], color: ascend_color, width: 0.4, height: 0.05, depth: 0.05, .. Default::default() },
                        //     Rectangle { position: [x - 0.2, y, z], color: ascend_color, width: 0.05, height: 0.4, depth: 0.05, .. Default::default() },
                        //     Rectangle { position: [x + 0.2, y, z], color: ascend_color, width: 0.05, height: 0.4, depth: 0.05, .. Default::default() }
                        // ].to_vec()
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
                            // Rectangle { position: [x, y, z], color: wall_color, width: 0.2, height: 0.8, depth: 1.0, .. Default::default() }
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
                            // Rectangle { position: [x, y, z], color: wall_color, width: 0.8, height: 0.2, depth: 1.0, .. Default::default() }
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
                            // Rectangle { position: [x, y, z], color: floor_color, width: 1.0, height: 1.0, depth: 0.1, .. Default::default() }
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
                // data.push(Rectangle { position: [x, y, z], color: wall_color, width: 0.2, height: 0.2, depth: 1.0, .. Default::default() });
                corners.push(InstanceModel { m: linalg::model([90f32.to_radians(), 0.0, 0.0], [1.0, 1.0, 1.0], [x, y, z]) });
            }
        }

        (walls, floors, corners, ceilings)
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
