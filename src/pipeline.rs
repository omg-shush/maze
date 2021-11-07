use std::sync::Arc;

use winit::window::Window;

use vulkano::device::Device;
use vulkano::swapchain::Swapchain;
use vulkano::pipeline::{ComputePipeline, GraphicsPipeline};
use vulkano::render_pass::Subpass;
use vulkano::pipeline::vertex::Vertex;
use vulkano::render_pass::RenderPass;

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
        #version 450
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec3 color;
        layout(push_constant) uniform ViewProjectionData {
            mat4 vp;
        } vpd;
        layout(location = 0) out vec3 passColor;
        void main() {
            gl_Position = vpd.vp * vec4(position, 0.0, 1.0);
            passColor = color;
        }
        "
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
        #version 450
        layout(location = 0) in vec3 passColor;
        layout(location = 0) out vec4 f_color;
        void main() {
            f_color = vec4(passColor, 1.0);
        }
        "
    }
}

pub mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
        #version 450
        layout(local_size_x = 256) in;
        struct Vertex {
            vec2 position;
            vec3 color;
        };
        layout(push_constant) uniform SourceLength {
            int len;
        } sl;
        layout(set = 0, binding = 0) readonly buffer SourceBuffer {
            Vertex data[];
        } src;
        layout(set = 0, binding = 1) buffer DestBuffer {
            Vertex data[];
        } dst;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i < sl.len) {
                dst.data[i] = src.data[i];
            }
        }
        ",
        types_meta: {
            #[derive(Clone, Copy, PartialEq, Debug, Default)]
        }
    }
}

pub struct Pipeline {
    pub render_pass: Arc<RenderPass>,
    pub graphics_pipeline: Arc<GraphicsPipeline>,
    pub compute_pipeline: Arc<ComputePipeline>
}

pub fn compile_shaders<T: Vertex>(
        device: Arc<Device>,
        swapchain: &Swapchain<Window>) -> Pipeline {
    let vertex_shader = vs::Shader::load(device.clone()).expect("Failed to load vertex shader");
    let fragment_shader = fs::Shader::load(device.clone()).expect("Failed to load fragment shader");
    let compute_shader = cs::Shader::load(device.clone()).expect("Failed to load compute shader");

    let render_pass = Arc::new(
        vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color_value: {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                }
            },
            pass: {
                color: [color_value],
                depth_stencil: {}
            }
        ).unwrap()
    );

    let graphics_pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer::<T>()
            .vertex_shader(vertex_shader.main_entry_point(), ())
            .fragment_shader(fragment_shader.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap()
    );

    let compute_pipeline = Arc::new(
        ComputePipeline::new(device.clone(), &compute_shader.main_entry_point(), &(), None, |_| {}).unwrap()
    );

    Pipeline {render_pass, graphics_pipeline, compute_pipeline}
}
