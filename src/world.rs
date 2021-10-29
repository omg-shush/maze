use std::sync::Arc;

use vulkano::device::Device;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::impl_vertex;
use vulkano::buffer::BufferUsage;

#[derive(Default, Debug, Clone)]
    pub struct Vertex {
        pub position: [f32; 2],
    }
    impl_vertex!(Vertex, position);

pub fn vertex_buffer(device: Arc<Device>) -> Arc<CpuAccessibleBuffer<[Vertex]>> {
    CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        [
            Vertex { position: [-0.75, 0.5] },
            Vertex { position: [0.75, 0.5] },
            Vertex { position: [0.0, -0.5] }
        ].iter().cloned()
    ).unwrap()
}
