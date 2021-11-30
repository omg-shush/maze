use std::collections::HashMap;
use std::iter::empty;
use std::sync::Arc;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor_set::layout::DescriptorSetLayout;
use vulkano::pipeline::{GraphicsPipeline, PipelineBindPoint};
use vulkano::render_pass::{RenderPass, Subpass};
use vulkano::sampler::Sampler;
use vulkano::device::{Queue, Device};
use vulkano::impl_vertex;

use crate::config::Config;
use crate::player::{GameState, Player};
use crate::texture::Texture;
use crate::world::World;

const DIGIT_WIDTH: f32 = 1.0 / 10.0;
const DIGIT_HEIGHT: f32 = 100.0 / 512.0;
const CONTROL_WIDTH: f32 = 0.093;
const CONTROL_HEIGHT: f32 = 100.0 / 512.0;

pub struct UserInterface {
    graphics_pipeline: Arc<GraphicsPipeline>,
    rect_buffer: Arc<CpuAccessibleBuffer<[UIVertex; 6]>>,
    scale_x: f32,
    scale_y: f32,
    controls: Vec<([i32; 4], UIElement, UIElement)>,
    digits: Vec<UIElement>,
    slash: UIElement,
    win: UIElement,
    lose: UIElement
}

#[derive(Clone)]
struct UIElement {
    texture_descriptor: Arc<PersistentDescriptorSet>,
    shader_constant: ShaderConstant
}

type ShaderConstant = vs::ty::ShaderConstant;

fn tex_desc_set(layout: Arc<DescriptorSetLayout>, sampler: Arc<Sampler>, texture: &Texture) -> Arc<PersistentDescriptorSet> {
    let mut builder = PersistentDescriptorSet::start(layout);
    builder.add_sampled_image(texture.access(), sampler.clone()).unwrap();
    Arc::new(builder.build().unwrap())
}

impl UserInterface {
    pub fn new(queue: Arc<Queue>, render_pass: Arc<RenderPass>, textures: &HashMap<String, Texture>, resolution: [u32; 2], config: &Config) -> UserInterface {
        // Initialize pipeline for displaying UI
        let graphics_pipeline = graphics_pipeline(queue.device().clone(), render_pass.clone());

        // Initialize texture samplers
        let sampler = Sampler::simple_repeat_linear_no_mipmap(queue.device().clone());
        let layout = graphics_pipeline.layout().descriptor_set_layouts()[0].clone();

        // Build rect buffer
        let rect_buffer = CpuAccessibleBuffer::from_data(
            queue.device().clone(),
            BufferUsage::vertex_buffer(),
            false,
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0]
            ].map(|xy| UIVertex { position: xy, uv: xy.map(|f| f.clamp(0.0, 1.0)) })).unwrap();

        // Use UI scaling
        let [digit_ui_width, digit_ui_height] =
            [DIGIT_WIDTH, DIGIT_HEIGHT].map(|f| f * config.ui_scale);

        // Build UI elements
        let controls_desc = tex_desc_set(layout.clone(), sampler.clone(), &textures["controls"]);
        let controls_dim_desc = tex_desc_set(layout.clone(), sampler.clone(), &textures["controls_dim"]);
        let control_ui_width = 0.1 * config.ui_scale;
        let control_ui_height = 0.16 * config.ui_scale;
        let [mut control_w, mut control_a, mut control_s, mut control_d,
            mut control_q, mut control_e, mut control_space, mut control_lctrl] =
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0].map(|i| {
                UIElement { texture_descriptor: controls_desc.clone(), shader_constant: ShaderConstant {
                    texture_region: [i * CONTROL_WIDTH, 0.0, (i + 1.0) * CONTROL_WIDTH, CONTROL_HEIGHT],
                    size: [control_ui_width, control_ui_height], offset: [0.0, 0.0] } } });
        let (control_x_pos, control_y_pos) = (-0.84, -0.92);
        control_w.shader_constant.offset = [control_x_pos, control_y_pos];
        control_a.shader_constant.offset = [control_x_pos - 0.66 * control_ui_width, control_y_pos + control_ui_height];
        control_s.shader_constant.offset = [control_x_pos + 0.33 * control_ui_width, control_y_pos + control_ui_height];
        control_d.shader_constant.offset = [control_x_pos + 1.33 * control_ui_width, control_y_pos + control_ui_height];
        control_q.shader_constant.offset = [control_x_pos - control_ui_width, control_y_pos];
        control_e.shader_constant.offset = [control_x_pos + control_ui_width, control_y_pos];
        control_space.shader_constant.offset = [control_x_pos + control_ui_width * 2.5, control_y_pos];
        control_lctrl.shader_constant.offset = [control_x_pos + control_ui_width * 2.5, control_y_pos + control_ui_height];
        let controls = [
            ([0, -1, 0, 0], control_w),
            ([-1, 0, 0, 0], control_a),
            ([0, 1, 0, 0], control_s),
            ([1, 0, 0, 0], control_d),
            ([0, 0, 0, -1], control_q),
            ([0, 0, 0, 1], control_e),
            ([0, 0, 1, 0], control_space),
            ([0, 0, -1, 0], control_lctrl)].map(|(delta, control)| {
                let mut dim = control.clone();
                dim.texture_descriptor = controls_dim_desc.clone();
                (delta, control, dim)
            }).to_vec();

        let digits_desc_set = tex_desc_set(layout.clone(), sampler.clone(), &textures["digits"]);
        let digits: Vec<UIElement> = (0..=9).map(|i| {
            UIElement { texture_descriptor: digits_desc_set.clone(), shader_constant: ShaderConstant {
                texture_region: [DIGIT_WIDTH * i as f32, 0.0, DIGIT_WIDTH * (i + 1) as f32, DIGIT_HEIGHT],
                size: [digit_ui_width, digit_ui_height],
                offset: [0.0, 0.0] // Will be set later, when needed
            } } }).collect();
        let slash = UIElement {
            texture_descriptor: digits_desc_set,
            shader_constant: ShaderConstant {
                texture_region: [0.0, DIGIT_HEIGHT, DIGIT_WIDTH, 2.0 * DIGIT_HEIGHT],
                size: [digit_ui_width, digit_ui_height],
                offset: [1.0 - 3.0 * digit_ui_width, 1.0 - digit_ui_height] } };

        let win = UIElement { texture_descriptor: tex_desc_set(layout.clone(), sampler.clone(), &textures["win"]),
            shader_constant: ShaderConstant {
                texture_region: [0.0, 0.0, 1.0, 1.0],
                size: [2.0, 2.0],
                offset: [-1.0, -1.0]
            } };
        let lose = UIElement { texture_descriptor: tex_desc_set(layout.clone(), sampler.clone(), &textures["lose"]),
            shader_constant: ShaderConstant {
                texture_region: [0.0, 0.0, 1.0, 1.0],
                size: [2.0, 2.0],
                offset: [-1.0, -1.0]
            } };

        // Compensate for aspect ratio
        let [x, y] = resolution;
        let ratio = x as f32 / y as f32;
        let (scale_x, scale_y) = if ratio >= 1.0 { (ratio, 1.0) } else { (1.0, 1.0 / ratio) };

        UserInterface { graphics_pipeline, rect_buffer, scale_x, scale_y, controls, digits, slash, win, lose }
    }

    pub fn render(&self, player: &Player, world: &World, config: &Config, builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) {
        // Display valid controls
        let controls = self.controls.iter().filter_map(|(delta, control, dim)| {
            if world.check_move(player.cell(), *delta) {
                Some (control)
            } else {
                Some (dim)
            }
        });

        let [digit_ui_width, digit_ui_height] = [DIGIT_WIDTH, DIGIT_HEIGHT].map(|f| f * config.ui_scale);

        // Display player's score
        let mut score_ones = self.digits[player.score as usize % 10].clone();
        score_ones.shader_constant.offset = [1.0 - 4.0 * digit_ui_width, 1.0 - digit_ui_height];
        let mut score_tens = self.digits[player.score as usize / 10 % 10].clone();
        score_tens.shader_constant.offset = [1.0 - 5.0 * digit_ui_width, 1.0 - digit_ui_height];
        let mut max_ones = self.digits[config.food_count % 10].clone();
        max_ones.shader_constant.offset = [1.0 - 1.0 * digit_ui_width, 1.0 - digit_ui_height];
        let mut max_tens = self.digits[config.food_count / 10 % 10].clone();
        max_tens.shader_constant.offset = [1.0 - 2.0 * digit_ui_width, 1.0 - digit_ui_height];
        let score = [score_tens, score_ones, self.slash.clone(), max_tens, max_ones];

        // Display win/lose screens
        let screens = vec![self.lose.clone(), self.win.clone()];
        let game_state_elements = match player.game_state {
            GameState::Playing => &screens[0..0],
            GameState::Lost => &screens[0..1],
            GameState::Won => &screens[1..2]
        }.iter();

        let mut elements = Box::new(empty()) as Box<dyn Iterator<Item = &UIElement>>;
        if config.display_controls {
            elements = Box::new(elements.chain(controls));
        }
        elements = Box::new(elements.chain(score.iter()));

        // TODO do this ahead of time!
        // Anchor to edges and compensate for aspect ratio
        let mut elements = Box::new(elements.map(|e| {
            let mut e = e.clone();
            e.shader_constant.size[0] /= self.scale_x;
            e.shader_constant.size[1] /= self.scale_y;
            e.shader_constant.offset[0] /= self.scale_x;
            e.shader_constant.offset[1] /= self.scale_y;
            e.shader_constant.offset[0] += e.shader_constant.offset[0].signum() * (self.scale_x - 1.0) / 2.0;
            e.shader_constant.offset[1] += e.shader_constant.offset[0].signum() * (self.scale_y - 1.0) / 2.0;
            e
        })) as Box<dyn Iterator<Item = UIElement>>;

        // Centered elements only compensate for aspect ratio
        let game_state_elements = game_state_elements.map(|e| {
            let mut e = e.clone();
            e.shader_constant.size[0] /= self.scale_x;
            e.shader_constant.size[1] /= self.scale_y;
            e.shader_constant.offset[0] /= self.scale_x;
            e.shader_constant.offset[1] /= self.scale_y;
            e
        });
        elements = Box::new(elements.chain(game_state_elements));

        builder
            .bind_pipeline_graphics(self.graphics_pipeline.clone());
        let layout = self.graphics_pipeline.layout();
        // Render each UI element
        for element in elements {
            builder
                .bind_descriptor_sets(PipelineBindPoint::Graphics,
                    layout.clone(),
                    0,
                    element.texture_descriptor.clone())
                .push_constants(layout.clone(),
                0,
                element.shader_constant)
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
        .depth_stencil_disabled() // Ignore depth testing for overlaying UI images
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
        layout(push_constant) uniform ShaderConstant {
            vec4 texture_region;
            vec2 size;
            vec2 offset;
        } sc;
        layout(location = 0) out vec2 passUv;
        void main() {
            vec2 tex_start = sc.texture_region.xy;
            vec2 tex_finish = sc.texture_region.zw;
            gl_Position = vec4(position * sc.size + sc.offset, 0.0, 1.0);
            passUv = vec2(uv.x * (tex_finish.x - tex_start.x) + tex_start.x, uv.y * (tex_finish.y - tex_start.y) + tex_start.y);
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
