use std::borrow::Cow;
use std::vec;
use std::sync::Arc;

use vulkano::descriptor_set::{SingleLayoutDescSetPool};
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, KeyboardInput, VirtualKeyCode, WindowEvent, ElementState};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{WindowBuilder};
use winit::dpi::{PhysicalPosition, LogicalSize};

use vulkano::device::{Device, Features, DeviceExtensions};
use vulkano::device::physical::{PhysicalDevice};
use vulkano::instance::{Instance, ApplicationInfo};
use vulkano::Version;
use vulkano::image::ImageUsage;
use vulkano::image::view::ImageView;
use vulkano::image::attachment::AttachmentImage;
use vulkano::swapchain;
use vulkano::swapchain::{Swapchain, AcquireError, SwapchainCreationError};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents};
use vulkano::buffer::cpu_access::CpuAccessibleBuffer;
use vulkano::buffer::{BufferUsage, DeviceLocalBuffer, TypedBufferAccess};
use vulkano::pipeline::viewport::Viewport;
use vulkano::render_pass::{Framebuffer, FramebufferAbstract};
use vulkano::sync;
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::pipeline::PipelineBindPoint;
use vulkano::format::{ClearValue, Format};

use pipeline::cs::ty::Vertex;
use parameters::Params;
use player::Player;
use pipeline::vs::ty::ViewProjectionData;

mod world;
mod pipeline;
mod disjoint_set;
mod camera;
mod parameters;
mod player;
mod linalg;

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
    let _transfer_queue = qs.next().unwrap();

    // Create window
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .with_inner_size(LogicalSize { width: 800, height: 800 })
        .with_position(PhysicalPosition { x : 300, y: 200 })
        .with_resizable(false)
        .with_title("maze or something i guess")
        .build_vk_surface(&event_loop, instance.clone()).unwrap();

    // Configure parameters
    let params = Params::new(device.clone());
    println!("{:?}", params);

    // Create swapchain
    let surface_caps = surface.capabilities(card).unwrap();
    let resolution = surface_caps.max_image_extent;
    let buffers = 2;
    let transform = surface_caps.current_transform;
    let (format, _color_space) = surface_caps.supported_formats[0];
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

    // Generate world data
    let mut world = world::World::new();
    world.generate_maze();
    let world_data = world.vertex_buffer();
    let world_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::storage_buffer(),
        false,
        world_data.clone()
    ).expect("Failed to construct buffer");
    let player_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::vertex_buffer(),
        false,
        world.player_buffer()
    ).expect("Failed to construct buffer");

    // Compile shader pipeline
    let pipeline = pipeline::compile_shaders::<Vertex>(device.clone(), &swapchain, &params);

    // Use compute shader to elaborate vertex data
    let vertex_buffer: Arc<DeviceLocalBuffer<[Vertex]>> = DeviceLocalBuffer::array(
        device.clone(),
        36 * world_buffer.len(), // 6 vertices per rectangle
        BufferUsage {
            storage_buffer: true,
            vertex_buffer: true,
            .. BufferUsage::none()
        },
        [draw_queue.family()]
    ).unwrap();
    let compute_descriptor_set = {
        let mut pool = SingleLayoutDescSetPool::new(
            pipeline.compute_pipeline.layout().descriptor_set_layouts()[0].clone()
        );
        let mut builder = pool.next();
        builder.add_buffer(world_buffer.clone()).unwrap();
        builder.add_buffer(vertex_buffer.clone()).unwrap();
        builder.build().unwrap()
    };
    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        draw_queue.family(),
        CommandBufferUsage::OneTimeSubmit
    ).unwrap();
    builder
        .bind_pipeline_compute(pipeline.compute_pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.compute_pipeline.layout().clone(),
            0,
            compute_descriptor_set
        )
        .push_constants(
            pipeline.compute_pipeline.layout().clone(),
            0,
            pipeline::cs::ty::SourceLength { len: world_data.len() as i32 }
        )
        .dispatch([(world_data.len() / 256 + 1) as u32, 1, 1])
        .unwrap();
    let compute_command_buffer = builder.build().unwrap();
    sync::now(device.clone())
        .then_execute(draw_queue.clone(), compute_command_buffer).unwrap()
        .then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    // Initialize framebuffers
    let dimensions = images[0].dimensions();
    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0
    };
    let dview = ImageView::new(AttachmentImage::transient_multisampled(device.clone(), dimensions, params.sample_count, Format::D16_UNORM).unwrap()).unwrap();
    let mut framebuffers = images
        .iter()
        .map(|image| {
            let view = ImageView::new(image.clone()).unwrap();
            let mview = ImageView::new(AttachmentImage::transient_multisampled(device.clone(), dimensions, params.sample_count, format).unwrap()).unwrap();
            Arc::new(
                Framebuffer::start(pipeline.render_pass.clone())
                    .add(mview).unwrap()
                    .add(view).unwrap()
                    .add(dview.clone()).unwrap()
                    .build().unwrap()
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        }).collect::<Vec<_>>();

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());
    let mut recreate_swapchain = false;
    let mut desc_set_pool = SingleLayoutDescSetPool::new(
        pipeline.graphics_pipeline.layout().descriptor_set_layouts()[0].clone()
    );

    let mut player = Player::new();

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
            if player.complete {
                return;
            }
            match keycode {
                VirtualKeyCode::W | VirtualKeyCode::Up => {
                    if state == ElementState::Pressed && keys[0] == ElementState::Released {
                        if world.check_move([player.cell()[0], player.cell()[1]], [player.cell()[0], player.cell()[1] - 1]) {
                            player.move_position([0, -1, 0], 0.5);
                        }
                    }
                    keys[0] = state;
                },
                VirtualKeyCode::S | VirtualKeyCode::Down => {
                    if state == ElementState::Pressed && keys[0] == ElementState::Released {
                        if world.check_move([player.cell()[0], player.cell()[1]], [player.cell()[0], player.cell()[1] + 1]) {
                            player.move_position([0, 1, 0], 0.5);
                        }
                    }
                    keys[1] = state
                },
                VirtualKeyCode::A | VirtualKeyCode::Left => {
                    if state == ElementState::Pressed && keys[0] == ElementState::Released {
                        if world.check_move([player.cell()[0], player.cell()[1]], [player.cell()[0] - 1, player.cell()[1]]) {
                            player.move_position([-1, 0, 0], 0.5);
                        }
                    }
                    keys[2] = state
                },
                VirtualKeyCode::D | VirtualKeyCode::Right => {
                    if state == ElementState::Pressed && keys[0] == ElementState::Released {
                        if world.check_move([player.cell()[0], player.cell()[1]], [player.cell()[0] + 1, player.cell()[1]]) {
                            player.move_position([1, 0, 0], 0.5);
                        }
                    }
                    keys[3] = state
                },
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
                let dview = ImageView::new(AttachmentImage::transient_multisampled(device.clone(), dimensions, params.sample_count, Format::D16_UNORM).unwrap()).unwrap();
                framebuffers = new_images
                    .iter()
                    .map(|image| {
                        let view = ImageView::new(image.clone()).unwrap();
                        let mview = ImageView::new(AttachmentImage::transient_multisampled(device.clone(), dimensions, params.sample_count, format).unwrap()).unwrap();
                        Arc::new(
                            Framebuffer::start(pipeline.render_pass.clone())
                                .add(mview).unwrap()
                                .add(view).unwrap()
                                .add(dview.clone()).unwrap()
                                .build().unwrap()
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

            let clear_values = vec![[0.0, 0.0, 0.0, 1.0].into(), ClearValue::None, ClearValue::Depth(1.0)];
            let destination_values = vec![[0.4, 0.85, 0.4, 1.0].into(), ClearValue::None, ClearValue::Depth(1.0)];
            let mut builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                draw_queue.family(),
                CommandBufferUsage::OneTimeSubmit
            ).unwrap();

            // Update uniforms
            let ppd_buffer = CpuAccessibleBuffer::from_data(
                device.clone(),
                BufferUsage::uniform_buffer_transfer_destination(),
                false,
                pipeline::fs::ty::PlayerPositionData { pos: {
                    let mut p = player.get_position().map(|i| i as f32);
                    p[2] += 0.8;
                    p
                } }
            ).unwrap();
            let world_descriptor_set = {
                let mut builder = desc_set_pool.next();
                builder.add_buffer(ppd_buffer.clone()).unwrap();
                builder.build().unwrap()
            };
            let ppd_buffer = CpuAccessibleBuffer::from_data(
                device.clone(),
                BufferUsage::uniform_buffer_transfer_destination(),
                false,
                pipeline::fs::ty::PlayerPositionData { pos: [0.0, 0.0, 0.8] }
            ).unwrap();
            let player_descriptor_set = {
                let mut builder = desc_set_pool.next();
                builder.add_buffer(ppd_buffer.clone()).unwrap();
                builder.build().unwrap()
            };

            // Update game state
            player.update();
            let proj = player.camera.projection();
            let view_projection = ViewProjectionData { vp: linalg::mul(proj, player.camera.view()) };
            let player_view = {
                let mut array = [0.0; 3];
                let p = player.get_position();
                let c = player.camera.get_position();
                for i in 0..3 {
                    array[i] = p[i] - c[i];
                }
                linalg::view(player.camera.get_rotation(), player.camera.get_scale(), array)
            };
            let view_projection_no_translation = ViewProjectionData { vp: linalg::mul(proj, player_view) };

            if player.complete {
                // Destination reached
                builder
                    .begin_render_pass(
                        framebuffers[image_num].clone(),
                        SubpassContents::Inline,
                        destination_values
                    ).unwrap()
                    .set_viewport(0, [viewport.clone()])
                    .bind_pipeline_graphics(pipeline.graphics_pipeline.clone())
                    .end_render_pass().unwrap();
            } else {
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
                        world_descriptor_set
                    )
                    .push_constants(pipeline.graphics_pipeline.layout().clone(), 0, view_projection)
                    .draw(vertex_buffer.len() as u32, 1, 0, 0).unwrap()
                    .bind_vertex_buffers(0, player_buffer.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipeline.graphics_pipeline.layout().clone(),
                        0,
                        player_descriptor_set
                    )
                    .push_constants(pipeline.graphics_pipeline.layout().clone(), 0, view_projection_no_translation)
                    .draw(player_buffer.len() as u32, 1, 0, 0).unwrap()
                    .end_render_pass().unwrap();
            }
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
