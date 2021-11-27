use std::collections::HashMap;
use std::sync::Arc;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::pipeline::{GraphicsPipeline, PipelineBindPoint};
use vulkano::render_pass::{RenderPass, Subpass};
use vulkano::sampler::Sampler;
use vulkano::sync::{self, GpuFuture};
use vulkano::device::{Queue, Device};
use vulkano::impl_vertex;

use crate::player::Player;
use crate::texture::Texture;

pub struct UserInterface {
    textures: HashMap<String, Texture>,
    graphics_pipeline: Arc<GraphicsPipeline>,
    rect_buffer: Arc<CpuAccessibleBuffer<[UIVertex; 6]>>,
    elements: Vec<UIElement>
}

struct UIElement {
    texture_descriptor: Arc<PersistentDescriptorSet>,
    size_offset: SizeOffset
}

#[derive(Clone, Copy)]
struct SizeOffset {
    size: [f32; 2],
    offset: [f32; 2]
}

impl UserInterface {
    pub fn new(queue: Arc<Queue>, render_pass: Arc<RenderPass>) -> (UserInterface, Box<dyn GpuFuture>) {
        // Load textures
        let mut futures = Vec::new();
        let textures: HashMap<String, Texture> = [
            Texture::new(queue.clone(), "up.png")
        ].map(|(texture, future)| {
            futures.push(future);
            (texture.file.to_owned(), texture)
        }).into_iter().collect();
        let future = futures.into_iter().fold(sync::now(queue.device().clone()).boxed(), |acc, fut| {
            acc.join(fut).boxed()
        });

        // Initialize pipeline for displaying UI
        let graphics_pipeline = graphics_pipeline(queue.device().clone(), render_pass.clone());

        // Initialize texture samplers
        let sampler = Sampler::simple_repeat_linear_no_mipmap(queue.device().clone());
        let layout = graphics_pipeline.layout().descriptor_set_layouts()[0].clone();
        let desc_set = {
            let mut builder = PersistentDescriptorSet::start(layout.clone());
            builder.add_sampled_image(textures.get("up.png").unwrap().access(), sampler).unwrap();
            Arc::new(builder.build().unwrap())
        };

        // Build rect buffer
        let rect_buffer = CpuAccessibleBuffer::from_data(
            queue.device().clone(),
            BufferUsage::vertex_buffer(),
            false,
            [
                [-1.0, -1.0],
                [-1.0,  1.0],
                [ 1.0, -1.0],
                [ 1.0, -1.0],
                [-1.0,  1.0],
                [ 1.0,  1.0]
            ].map(|xy| UIVertex { position: xy, uv: xy.map(|f| f.clamp(0.0, 1.0)) })).unwrap();

        // Build UI elements
        let up = UIElement { texture_descriptor: desc_set, size_offset: SizeOffset { size: [0.1, 0.1], offset: [0.9, 0.9] } };
        let elements = vec![up];

        (UserInterface { textures, graphics_pipeline, rect_buffer, elements }, future)
    }

    pub fn render(&self, player: &Box<Player>, builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) {
        builder
            .bind_pipeline_graphics(self.graphics_pipeline.clone());
        let layout = self.graphics_pipeline.layout();
        // Render each UI element
        for element in self.elements.iter() {
            builder
                .bind_descriptor_sets(PipelineBindPoint::Graphics,
                    layout.clone(),
                    0,
                    element.texture_descriptor.clone())
                .push_constants(layout.clone(),
                0,
                element.size_offset)
                .bind_vertex_buffers(0, self.rect_buffer.clone())
                .draw(6, 1, 0, 0).unwrap();
        }
    }
}

fn graphics_pipeline(device: Arc<Device>, render_pass: Arc<RenderPass>) -> Arc<GraphicsPipeline> {
    let vertex_shader = vs::Shader::load(device.clone()).expect("Failed to compile UI vertex shader");
    let fragment_shader = fs::Shader::load(device.clone()).expect("Failed to compile UI fragment shader");

    Arc::new(
    GraphicsPipeline::start()
        .vertex_input_single_buffer::<UIVertex>()
        .vertex_shader(vertex_shader.main_entry_point(), ())
        .fragment_shader(fragment_shader.main_entry_point(), ())
        .depth_stencil_simple_depth()
        .triangle_list()
        .blend_alpha_blending()
        .viewports_dynamic_scissors_irrelevant(1)
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap()
    )
}

#[derive(Default, Clone, Copy)]
struct UIVertex {
    position: [f32; 2],
    uv: [f32; 2]
}
impl_vertex!(UIVertex, position, uv);

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
        #version 450
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec2 uv;
        layout(push_constant) uniform SizeOffset {
            vec2 size;
            vec2 offset;
        } ps;
        layout(location = 0) out vec2 passUv;
        void main() {
            gl_Position = vec4(position * ps.size + ps.offset, 0.0, 1.0);
            passUv = uv;
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
        layout(location = 0) in vec2 passUv;
        layout(set = 0, binding = 0) uniform sampler2D tex;
        layout(location = 0) out vec4 f_color;
        void main() {
            f_color = texture(tex, passUv);
        }
        "
    }
}
