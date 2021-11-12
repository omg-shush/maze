use std::fs;
use std::io::{BufRead, BufReader};
use std::sync::Arc;

use vulkano::buffer::{BufferUsage, ImmutableBuffer};
use vulkano::device::Queue;
use vulkano::sync::GpuFuture;

use crate::pipeline::cs::ty::Vertex;

pub struct Model {
    pub file: String,
    pub vertices: Arc<ImmutableBuffer<[Vertex]>>
}

impl Model {
    pub fn new(queue: Arc<Queue>, filename: &str) -> (Box<Model>, Box<dyn GpuFuture>) {
        let mut vertices = Vec::new();
        let file = fs::File::open(filename).expect(&format!("Failed to load model `{}'", filename));
        let reader = BufReader::new(file);
        let mut v: Vec<[f32; 3]> = Vec::new();
        let mut vn: Vec<[f32; 3]> = Vec::new();
        for res in reader.lines() {
            if let Ok(lin) = res {
                match &lin[..2] {
                    "v " => {
                        let vertex = lin[2..]
                            .split_ascii_whitespace()
                            .map(|f| f.parse::<f32>().expect("Invalid float"))
                            .collect::<Vec<f32>>();
                        v.push([vertex[0], vertex[1], vertex[2]]);
                    },
                    "vn" => {
                        let normal = lin[3..]
                            .split_ascii_whitespace()
                            .map(|f| f.parse::<f32>().expect("Invalid float"))
                            .collect::<Vec<f32>>();
                        vn.push([normal[0], normal[1], normal[2]]);
                    }
                    "f " => {
                        let face = lin[2..]
                            .split_ascii_whitespace()
                            .map(|v| v.split('/').map(|f| f.parse::<usize>().unwrap_or_default())
                            .collect::<Vec<usize>>())
                            .collect::<Vec<Vec<usize>>>();
                        for i in 0..3 {
                            vertices.push(Vertex {
                                position: v[face[i][0] - 1], // Subtract 1 since .OBJ is 1-indexed
                                color: [ 0.0, 0.4, 0.8 ], // TODO uv's
                                normal: vn[face[i][2] - 1],
                                .. Vertex::default()
                            })
                        }
                    },
                    _ => ()
                }
            }
        }
        println!("Loaded model {}", filename);
        let (vertices, future) = ImmutableBuffer::from_iter(
            vertices,
            BufferUsage::vertex_buffer(),
            queue
        ).unwrap();
        (Box::new(Model {
            file: filename.split('.').next().unwrap().to_owned(),
            vertices
        }), future.boxed())
    }
}