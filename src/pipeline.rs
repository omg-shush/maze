use std::sync::Arc;

use winit::window::Window;

use vulkano::device::Device;
use vulkano::swapchain::Swapchain;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::Subpass;
use vulkano::pipeline::vertex::Vertex;
use vulkano::render_pass::RenderPass;

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
        #version 450
        layout(location = 0) in vec2 position;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
        }
        "
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
        #version 450
        layout(location = 0) out vec4 f_color;
        void main() {
            f_color = vec4(0.0, 1.0, 0.0, 1.0);
        }
        "
    }
}

pub struct Pipeline {
    pub render_pass: Arc<RenderPass>,
    pub graphics_pipeline: Arc<GraphicsPipeline>
}

pub fn compile_shaders<T: Vertex>(device: Arc<Device>, swapchain: &Swapchain<Window>) -> Pipeline {
    let vertex_shader = vs::Shader::load(device.clone()).unwrap();
    let fragment_shader = fs::Shader::load(device.clone()).unwrap();

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

    Pipeline {render_pass, graphics_pipeline}
}
