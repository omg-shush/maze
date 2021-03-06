use std::sync::Arc;

use winit::window::Window;
use vulkano::device::Device;
use vulkano::swapchain::Swapchain;
use vulkano::pipeline::{ComputePipeline, GraphicsPipeline};
use vulkano::render_pass::Subpass;
use vulkano::pipeline::vertex::{BuffersDefinition, Vertex};
use vulkano::render_pass::RenderPass;
use vulkano::impl_vertex;
use vulkano::format::Format;

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
        #version 450
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 color;
        layout(location = 2) in vec3 normal;
        layout(location = 3) in mat4 m;
        layout(push_constant) uniform ViewProjectionData {
            mat4 vp;
            vec3 pushColor;
        } vpd;
        layout(set = 0, binding = 0) uniform PlayerPositionData {
            vec3 player_pos;
            vec3 ghost_pos;
        } ppd;
        layout(location = 0) out vec3 passPosition;
        layout(location = 1) out vec3 passColor;
        layout(location = 2) out vec3 passNormal;
        layout(location = 3) out vec3 playerVec;
        layout(location = 4) out vec3 ghostVec;
        void main() {
            vec4 worldPosition = m * vec4(position, 1.0);
            gl_Position = vpd.vp * worldPosition;
            passPosition = worldPosition.xyz;
            passColor = vpd.pushColor;
            passNormal = normalize((m * vec4(normal, 0.0)).xyz);
            playerVec = ppd.player_pos - worldPosition.xyz;
            ghostVec = ppd.ghost_pos - worldPosition.xyz;
        }
        ",
        types_meta: {
            #[derive(Clone, Copy, PartialEq, Debug, Default)]
        }
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
        #version 450
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 color;
        layout(location = 2) in vec3 normal;
        layout(location = 3) in vec3 playerVec;
        layout(location = 4) in vec3 ghostVec;
        layout(location = 0) out vec4 f_color;

        float point_light(vec3 light_pos) {
            float distance2 = length(light_pos);
            distance2 *= distance2;
            return clamp((1.0 / distance2) * clamp(dot(normal, normalize(light_pos)), 0.0, 1.0), 0.0, 1.0);
        }

        void main() {
            vec3 directional_light = normalize(vec3(1, -2, 3));
            float ambient = 0.02;
            float directional = 0.33 * clamp(dot(normal, -directional_light), 0.0, 1.0);
            float point = 0.65 *  clamp(point_light(playerVec) + point_light(ghostVec), 0.0, 1.0);
            float brightness = ambient + directional + point;
            f_color = vec4(color * brightness, 1.0);
        }
        ",
        types_meta: {
            #[derive(Clone, Copy, PartialEq, Debug, Default)]
        }
    }
}

pub mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
        #version 450
        layout(local_size_x = 256) in;
        struct Rectangle {
            vec3 position;
            vec3 color;
            float width;
            float height;
            float depth;
        };
        struct Vertex {
            vec3 position;
            vec3 color;
            vec3 normal;
        };
        layout(push_constant) uniform SourceLength {
            uint len;
        } sl;
        layout(set = 0, binding = 0) readonly buffer SourceBuffer {
            Rectangle data[];
        } src;
        layout(set = 0, binding = 1) buffer DestBuffer {
            Vertex data[];
        } dst;
        void main() {
            // Called once per rectangular prism
            uint i = gl_GlobalInvocationID.x;
            Rectangle wall = src.data[i];
            uint per = 36;
            if (i < sl.len) {
                // Bottom
                dst.data[i * per +  0].position = wall.position + vec3(wall.width / 2.0, wall.height / 2.0, 0.0);
                dst.data[i * per +  1].position = wall.position + vec3(wall.width / 2.0, wall.height / -2.0, 0.0);
                dst.data[i * per +  2].position = wall.position + vec3(wall.width / -2.0, wall.height / -2.0, 0.0);
                dst.data[i * per +  3].position = wall.position + vec3(wall.width / -2.0, wall.height / -2.0, 0.0);
                dst.data[i * per +  4].position = wall.position + vec3(wall.width / -2.0, wall.height / 2.0, 0.0);
                dst.data[i * per +  5].position = wall.position + vec3(wall.width / 2.0, wall.height / 2.0, 0.0);
                
                // Top
                dst.data[i * per +  6].position = wall.position + vec3(wall.width / 2.0, wall.height / 2.0, wall.depth);
                dst.data[i * per +  7].position = wall.position + vec3(wall.width / 2.0, wall.height / -2.0, wall.depth);
                dst.data[i * per +  8].position = wall.position + vec3(wall.width / -2.0, wall.height / -2.0, wall.depth);
                dst.data[i * per +  9].position = wall.position + vec3(wall.width / -2.0, wall.height / -2.0, wall.depth);
                dst.data[i * per + 10].position = wall.position + vec3(wall.width / -2.0, wall.height / 2.0, wall.depth);
                dst.data[i * per + 11].position = wall.position + vec3(wall.width / 2.0, wall.height / 2.0, wall.depth);

                // Front
                dst.data[i * per + 12].position = wall.position + vec3(wall.width / -2.0, wall.height / -2.0, wall.depth);
                dst.data[i * per + 13].position = wall.position + vec3(wall.width / -2.0, wall.height / -2.0, 0.0);
                dst.data[i * per + 14].position = wall.position + vec3(wall.width / 2.0, wall.height / -2.0, 0.0);
                dst.data[i * per + 15].position = wall.position + vec3(wall.width / 2.0, wall.height / -2.0, 0.0);
                dst.data[i * per + 16].position = wall.position + vec3(wall.width / 2.0, wall.height / -2.0, wall.depth);
                dst.data[i * per + 17].position = wall.position + vec3(wall.width / -2.0, wall.height / -2.0, wall.depth);

                // Back
                dst.data[i * per + 18].position = wall.position + vec3(wall.width / -2.0, wall.height / 2.0, wall.depth);
                dst.data[i * per + 19].position = wall.position + vec3(wall.width / -2.0, wall.height / 2.0, 0.0);
                dst.data[i * per + 20].position = wall.position + vec3(wall.width / 2.0, wall.height / 2.0, 0.0);
                dst.data[i * per + 21].position = wall.position + vec3(wall.width / 2.0, wall.height / 2.0, 0.0);
                dst.data[i * per + 22].position = wall.position + vec3(wall.width / 2.0, wall.height / 2.0, wall.depth);
                dst.data[i * per + 23].position = wall.position + vec3(wall.width / -2.0, wall.height / 2.0, wall.depth);

                // Right
                dst.data[i * per + 24].position = wall.position + vec3(wall.width / 2.0, wall.height / 2.0, wall.depth);
                dst.data[i * per + 25].position = wall.position + vec3(wall.width / 2.0, wall.height / -2.0, wall.depth);
                dst.data[i * per + 26].position = wall.position + vec3(wall.width / 2.0, wall.height / -2.0, 0.0);
                dst.data[i * per + 27].position = wall.position + vec3(wall.width / 2.0, wall.height / -2.0, 0.0);
                dst.data[i * per + 28].position = wall.position + vec3(wall.width / 2.0, wall.height / 2.0, 0.0);
                dst.data[i * per + 29].position = wall.position + vec3(wall.width / 2.0, wall.height / 2.0, wall.depth);

                // Left
                dst.data[i * per + 30].position = wall.position + vec3(wall.width / -2.0, wall.height / 2.0, wall.depth);
                dst.data[i * per + 31].position = wall.position + vec3(wall.width / -2.0, wall.height / -2.0, wall.depth);
                dst.data[i * per + 32].position = wall.position + vec3(wall.width / -2.0, wall.height / -2.0, 0.0);
                dst.data[i * per + 33].position = wall.position + vec3(wall.width / -2.0, wall.height / -2.0, 0.0);
                dst.data[i * per + 34].position = wall.position + vec3(wall.width / -2.0, wall.height / 2.0, 0.0);
                dst.data[i * per + 35].position = wall.position + vec3(wall.width / -2.0, wall.height / 2.0, wall.depth);
            }
            for (int j = 0; j < 12; j++) {
                dst.data[i * per + j].color = wall.color;
                dst.data[i * per + j].normal = vec3(0.0, 0.0, 1.0);;
            }
            for (int j = 12; j < 18; j++) { // -y
                dst.data[i * per + j].color = wall.color;
                dst.data[i * per + j].normal = vec3(0.0, -1.0, 0.0);;
            }
            for (int j = 18; j < 24; j++) { // +y
                dst.data[i * per + j].color = wall.color;
                dst.data[i * per + j].normal = vec3(0.0, 1.0, 0.0);;
            }
            for (int j = 24; j < 30; j++) { // +x
                dst.data[i * per + j].color = wall.color;
                dst.data[i * per + j].normal = vec3(1.0, 0.0, 0.0);;
            }
            for (int j = 30; j < 36; j++) { // -x
                dst.data[i * per + j].color = wall.color;
                dst.data[i * per + j].normal = vec3(-1.0, 0.0, 0.0);;
            }
        }
        ",
        types_meta: {
            #[derive(Clone, Copy, PartialEq, Debug, Default)]
        }
    }
}

impl_vertex!(cs::ty::Rectangle, position, color, width, height);
impl_vertex!(cs::ty::Vertex, position, color, normal);
#[derive(Default, Clone, Copy)]
pub struct InstanceModel {
    pub m: [[f32; 4]; 4]
}
impl_vertex!(InstanceModel, m);

pub struct Pipeline {
    pub render_pass: Arc<RenderPass>,
    pub graphics_pipeline: Arc<GraphicsPipeline>,
    pub compute_pipeline: Arc<ComputePipeline>
}

pub fn compile_shaders<T: Vertex>(
        device: Arc<Device>,
        swapchain: &Swapchain<Window>,
        samples: u32) -> Pipeline {
    let vertex_shader = vs::Shader::load(device.clone()).expect("Failed to load vertex shader");
    let fragment_shader = fs::Shader::load(device.clone()).expect("Failed to load fragment shader");
    let compute_shader = cs::Shader::load(device.clone()).expect("Failed to load compute shader");

    let render_pass = Arc::new(
        vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                msaa_image: {
                    load: Clear,
                    store: DontCare,
                    format: swapchain.format(),
                    samples: samples,
                },
                color_image: {
                    load: DontCare,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                },
                depth_image: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D16_UNORM,
                    samples: samples,
                }
            },
            pass: {
                color: [msaa_image],
                depth_stencil: {depth_image},
                resolve: [color_image]
            }
        ).unwrap()
    );

    let graphics_pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input(
                BuffersDefinition::new()
                .vertex::<cs::ty::Vertex>()
                .instance::<InstanceModel>())
            .vertex_shader(vertex_shader.main_entry_point(), ())
            .fragment_shader(fragment_shader.main_entry_point(), ())
            .depth_stencil_simple_depth()
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
