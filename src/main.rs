use std::borrow::Cow;
use std::vec;
use std::sync::Arc;

use vulkano::descriptor_set::{SingleLayoutDescSetPool};
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, KeyboardInput, VirtualKeyCode, WindowEvent, ElementState};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{WindowBuilder};
use winit::dpi::{PhysicalSize, PhysicalPosition};

use vulkano::device::{Device, Features, DeviceExtensions};
use vulkano::device::physical::{PhysicalDevice};
use vulkano::instance::{Instance, ApplicationInfo};
use vulkano::Version;
use vulkano::image::ImageUsage;
use vulkano::image::view::ImageView;
use vulkano::swapchain;
use vulkano::swapchain::{Swapchain, AcquireError, SwapchainCreationError};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents};
use vulkano::buffer::cpu_access::CpuAccessibleBuffer;
use vulkano::buffer::{BufferUsage, TypedBufferAccess};
use vulkano::pipeline::viewport::Viewport;
use vulkano::render_pass::{Framebuffer, FramebufferAbstract};
use vulkano::sync;
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::pipeline::PipelineBindPoint;

mod world;
mod pipeline;
mod disjoint_set;

fn main() {
    // Create vulkan instance
    let app_infos = ApplicationInfo {
        application_name: Some(Cow::from("maze")),
        application_version: Some(Version::V1_2),
        engine_name: None,
        engine_version: None };
    let instance_exts = vulkano_win::required_extensions();
    let instance = Instance::new(Some(&app_infos), Version::V1_2, &instance_exts, None).unwrap();

    // for layer in instance::layers_list().unwrap() {
    //     println!("Layer: {}", layer.name())
    // }

    // for device in PhysicalDevice::enumerate(&instance) {
    //     println!("Device: {}", device.properties().device_name)
    // }
    let card = PhysicalDevice::enumerate(&instance).next().unwrap();
    println!("Using card {}", card.properties().device_name);

    // Create logical device
    let features = Features {
        robust_buffer_access: true,
        .. Features::none()
    };
    let extensions = DeviceExtensions {
        khr_swapchain: true,
        .. DeviceExtensions::none()
    };
    let draw_queue = card.queue_families().find(|&q| q.supports_graphics()).unwrap();
    let transfer_queue = card.queue_families().find(|&q| q.explicitly_supports_transfers()).unwrap();
    let queues = [(draw_queue, 1.0), (transfer_queue, 0.0)];

    let (device, mut qs) = Device::new(card, &features, &extensions, queues.iter().cloned()).unwrap();
    println!("Created logical vulkan device {:?}", device);

    let draw_queue = qs.next().unwrap();
    let transfer_queue = qs.next().unwrap();

    // Create window
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .with_inner_size(PhysicalSize { width: 800, height: 720 })
        .with_position(PhysicalPosition { x : 300, y: 200 })
        .with_resizable(false)
        .with_title("maze or something i guess")
        .build_vk_surface(&event_loop, instance.clone()).unwrap();

    // Create swapchain
    let surface_caps = surface.capabilities(card).unwrap();
    let resolution = surface_caps.max_image_extent;
    let buffers = 2;
    let transform = surface_caps.current_transform;
    let (format, color_space) = surface_caps.supported_formats[0];
    let usage = ImageUsage {
        color_attachment: true,
        .. ImageUsage::none()
    };
    let (mut swapchain, images) = Swapchain::start(device.clone(), surface.clone())
                                     .num_images(buffers)
                                     .format(format)
                                     .dimensions(resolution)
                                     .usage(usage)
                                     .transform(transform)
                                     .build().unwrap();
    println!("Created swapchain {:?}", swapchain);

    // Generate vertex data
    let vertex_buffer = world::vertex_buffer(device.clone());
    let player_buffer = world::player_buffer(device.clone());

    // Compile shader pipeline
    let pipeline = pipeline::compile_shaders::<world::Vertex>(device.clone(), &swapchain);

    // Initialize framebuffers
    let dimensions = images[0].dimensions();
    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0
    };
    let mut framebuffers = images
        .iter()
        .map(|image| {
            let view = ImageView::new(image.clone()).unwrap();
            Arc::new(
                Framebuffer::start(pipeline.render_pass.clone())
                    .add(view)
                    .unwrap()
                    .build()
                    .unwrap()
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        }).collect::<Vec<_>>();

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());
    let mut recreate_swapchain = false;

    let mut player_position_data = pipeline::vs::ty::PlayerPositionData {
        player_position: [0.0, 0.0]
    };
    // Up, down, left, right
    let mut keys = [ElementState::Released; 4];

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested, ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(_), ..
        } => {
            recreate_swapchain = true;
        }
        Event::WindowEvent {
            event: WindowEvent::KeyboardInput {
                input: KeyboardInput {
                    virtual_keycode: Some (keycode),
                    state, ..
                }, ..
            }, ..
        } => {
            match keycode {
                VirtualKeyCode::W => keys[0] = state,
                VirtualKeyCode::S => keys[1] = state,
                VirtualKeyCode::A => keys[2] = state,
                VirtualKeyCode::D => keys[3] = state,
                _ => {}
            }
        }
        Event::RedrawEventsCleared => {
            previous_frame_end.as_mut().unwrap().cleanup_finished();

            if recreate_swapchain {
                let dimensions: [u32; 2] = surface.window().inner_size().into();
                viewport = Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                    depth_range: 0.0..1.0
                };
                let (new_swapchain, new_images) =
                    match swapchain.recreate().dimensions(dimensions).build() {
                        Ok (r) => r,
                        Err(SwapchainCreationError::UnsupportedDimensions) => return,
                        _ => panic!("Failed to recreate swapchain!")
                    };
                swapchain = new_swapchain;
                framebuffers = new_images
                    .iter()
                    .map(|image| {
                        let view = ImageView::new(image.clone()).unwrap();
                        Arc::new(
                            Framebuffer::start(pipeline.render_pass.clone())
                                .add(view)
                                .unwrap()
                                .build()
                                .unwrap()
                        ) as Arc<dyn FramebufferAbstract + Send + Sync>
                    }).collect::<Vec<_>>();
                recreate_swapchain = false;
            }

            let (image_num, suboptimal, acquire_future) =
                match swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next framebuffer! {}", e)
                };
            if suboptimal {
                recreate_swapchain = true;
            }

            let clear_values = vec![[0.5, 0.5, 0.85, 1.0].into()];
            let mut builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                draw_queue.family(),
                CommandBufferUsage::OneTimeSubmit
            ).unwrap();

            // Update push constants
            if keys[0] == ElementState::Pressed { // Up
                player_position_data.player_position[1] -= 0.01;
            }
            if keys[1] == ElementState::Pressed { // Down
                player_position_data.player_position[1] += 0.01;
            }
            if keys[2] == ElementState::Pressed { // Left
                player_position_data.player_position[0] -= 0.01;
            }
            if keys[3] == ElementState::Pressed { // Right
                player_position_data.player_position[0] += 0.01;
            }

            // Update uniforms
            let vp_buffer = CpuAccessibleBuffer::from_data(
                device.clone(),
                BufferUsage::uniform_buffer_transfer_destination(),
                false,
                pipeline::vs::ty::ViewProjectionData { vp: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]] }
            ).unwrap();
            let mut pool = SingleLayoutDescSetPool::new(
                pipeline.graphics_pipeline.layout().descriptor_set_layouts()[0].clone()
            );
            let descriptor_set = {
                let mut builder = pool.next();
                builder.add_buffer(vp_buffer.clone()).unwrap();
                builder.build().unwrap()
            };

            builder
                .begin_render_pass(
                    framebuffers[image_num].clone(),
                    SubpassContents::Inline,
                    clear_values
                ).unwrap()
                .set_viewport(0, [viewport.clone()])
                .bind_pipeline_graphics(pipeline.graphics_pipeline.clone())
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    pipeline.graphics_pipeline.layout().clone(),
                    0,
                    descriptor_set
                )
                .push_constants(pipeline.graphics_pipeline.layout().clone(), 0, player_position_data)
                .draw(vertex_buffer.len() as u32, 1, 0, 0).unwrap()
                .bind_vertex_buffers(0, player_buffer.clone())
                .push_constants(pipeline.graphics_pipeline.layout().clone(), 0, pipeline::vs::ty::PlayerPositionData { player_position: [0.0, 0.0] })
                .draw(player_buffer.len() as u32, 1, 0, 0).unwrap()
                .end_render_pass().unwrap();
            let command_buffer = builder.build().unwrap();

            let future = previous_frame_end
                .take().unwrap()
                .join(acquire_future)
                .then_execute(draw_queue.clone(), command_buffer).unwrap()
                .then_swapchain_present(draw_queue.clone(), swapchain.clone(), image_num)
                .then_signal_fence_and_flush();

            match future {
                Ok(future) => {
                    previous_frame_end = Some(future.boxed());
                }
                Err(FlushError::OutOfDate) => {
                    recreate_swapchain = true;
                    previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
                Err(e) => {
                    println!("Failed to flush future: {:?}", e);
                    previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
            }
        }
        _ => ()
    });
}
