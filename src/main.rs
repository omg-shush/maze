use std::borrow::Cow;
use std::collections::HashMap;
use std::vec;
use std::sync::Arc;
use std::time::{Duration, Instant};

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
use pipeline::InstanceModel;
use model::Model;

mod world;
mod pipeline;
mod disjoint_set;
mod camera;
mod parameters;
mod player;
mod linalg;
mod model;

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

    // Compile shader pipeline
    let pipeline = pipeline::compile_shaders::<Vertex>(device.clone(), &swapchain, &params);

    // Load models
    let models: HashMap<String, Box<Model>> = [
        Model::new(device.clone(), "wall.obj"),
        Model::new(device.clone(), "floor.obj"),
        Model::new(device.clone(), "corner.obj"),
        Model::new(device.clone(), "ceiling.obj")
    ].map(|m| (m.file.to_owned(), m)).into_iter().collect();

    // Generate world data
    let world = world::World::new();
    world.borrow_mut().generate_maze();
    let world_data: Vec<(Vec<InstanceModel>, Vec<InstanceModel>, Vec<InstanceModel>, Vec<InstanceModel>)> = (0..world::DEPTH).map(|level| world.borrow_mut().vertex_buffer(level)).collect();
    let world_buffer: Vec<[Arc<CpuAccessibleBuffer<[InstanceModel]>>; 4]> =
        world_data.into_iter().map(|(walls, floors, corners, ceilings)| { [
            CpuAccessibleBuffer::from_iter(
                device.clone(),
                BufferUsage::vertex_buffer(),
                false,
                walls
            ).expect("Failed to construct buffer"),
            CpuAccessibleBuffer::from_iter(
                device.clone(),
                BufferUsage::vertex_buffer(),
                false,
                floors
            ).expect("Failed to construct buffer"),
            CpuAccessibleBuffer::from_iter(
                device.clone(),
                BufferUsage::vertex_buffer(),
                false,
                corners
            ).expect("Failed to construct buffer"),
            CpuAccessibleBuffer::from_iter(
                device.clone(),
                BufferUsage::vertex_buffer(),
                false,
                ceilings
            ).expect("Failed to construct buffer")
        ] }).collect();
    let player_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::vertex_buffer(),
        false,
        world.borrow_mut().player_buffer()
    ).expect("Failed to construct buffer");

    // Use compute shader to elaborate vertex data
    // let vertex_buffer: Vec<Arc<DeviceLocalBuffer<[Vertex]>>> = world_buffer.iter().map(|level_buffer| {
    //     DeviceLocalBuffer::array(
    //         device.clone(),
    //         36 * level_buffer.len() as u64, // 6 vertices per rectangle, 6 rectangles per box
    //         BufferUsage {
    //             storage_buffer: true,
    //             vertex_buffer: true,
    //             .. BufferUsage::none()
    //         },
    //         [draw_queue.family()]
    //     ).unwrap()
    // }).collect();

    // let mut builder = AutoCommandBufferBuilder::primary(
    //     device.clone(),
    //     draw_queue.family(),
    //     CommandBufferUsage::OneTimeSubmit
    // ).unwrap();
    // builder.bind_pipeline_compute(pipeline.compute_pipeline.clone());
    // let mut compute_desc_set_pool = SingleLayoutDescSetPool::new(
    //     pipeline.compute_pipeline.layout().descriptor_set_layouts()[0].clone()
    // );
    // for level in 0..world::DEPTH {
    //     let input_len = world_buffer[level].len() as u32;
    //     let compute_descriptor_set = {
    //         let mut builder = compute_desc_set_pool.next();
    //         builder.add_buffer(world_buffer[level].clone()).unwrap();
    //         builder.add_buffer(vertex_buffer[level].clone()).unwrap();
    //         builder.build().unwrap()
    //     };
    //     builder
    //         .bind_descriptor_sets(
    //             PipelineBindPoint::Compute,
    //             pipeline.compute_pipeline.layout().clone(),
    //             0,
    //             compute_descriptor_set
    //         )
    //         .push_constants(
    //             pipeline.compute_pipeline.layout().clone(),
    //             0,
    //             pipeline::cs::ty::SourceLength { len: input_len }
    //         )
    //         .dispatch([input_len / 256 + 1, 1, 1])
    //         .unwrap();
    // }
    // let compute_command_buffer = builder.build().unwrap();
    // sync::now(device.clone())
    //     .then_execute(draw_queue.clone(), compute_command_buffer).unwrap()
    //     .then_signal_fence_and_flush().unwrap()
    //     .wait(None).unwrap();

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

    let mut player = Player::new(world.clone());

    // Up, down, left, right, ascend, descend
    let mut keys = [ElementState::Released; 6];

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
            if player.complete || player.solve.is_some() {
                return;
            }
            let world = world.borrow_mut();
            let seconds = 0.5;
            match keycode {
                VirtualKeyCode::W | VirtualKeyCode::Up => {
                    if state == ElementState::Pressed && keys[0] == ElementState::Released {
                        if world.check_move(player.cell(), [0, -1, 0]) {
                            player.move_position([0, -1, 0], seconds);
                        }
                    }
                    keys[0] = state;
                },
                VirtualKeyCode::S | VirtualKeyCode::Down => {
                    if state == ElementState::Pressed && keys[1] == ElementState::Released {
                        if world.check_move(player.cell(), [0, 1, 0]) {
                            player.move_position([0, 1, 0], seconds);
                        }
                    }
                    keys[1] = state
                },
                VirtualKeyCode::A | VirtualKeyCode::Left => {
                    if state == ElementState::Pressed && keys[2] == ElementState::Released {
                        if world.check_move(player.cell(), [-1, 0, 0]) {
                            player.move_position([-1, 0, 0], seconds);
                        }
                    }
                    keys[2] = state
                },
                VirtualKeyCode::D | VirtualKeyCode::Right => {
                    if state == ElementState::Pressed && keys[3] == ElementState::Released {
                        if world.check_move(player.cell(), [1, 0, 0]) {
                            player.move_position([1, 0, 0], seconds);
                        }
                    }
                    keys[3] = state
                },
                VirtualKeyCode::Space => {
                    if state == ElementState::Pressed && keys[4] == ElementState::Released {
                        if world.check_move(player.cell(), [0, 0, 1]) {
                            player.move_position([0, 0, 1], seconds);
                        }
                    }
                    keys[4] = state
                },
                VirtualKeyCode::LControl => {
                    if state == ElementState::Pressed && keys[5] == ElementState::Released {
                        if world.check_move(player.cell(), [0, 0, -1]) {
                            player.move_position([0, 0, -1], seconds);
                        }
                    }
                    keys[5] = state
                },
                VirtualKeyCode::Return => {
                    if state == ElementState::Pressed && player.solve.is_none() {
                        let mut delta = world.start;
                        for i in 0..3 {
                            delta[i] -= player.cell()[i];
                        }
                        player.move_position(delta, 0.0);
                        player.solve = Some((0, Instant::now() + Duration::from_secs(2)));
                        player.update();
                    }
                }
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
                    let mut p = player.get_position();
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
                pipeline::fs::ty::PlayerPositionData { pos: {
                    let mut pos = player.get_position();
                    pos[2] += 0.8;
                    pos
                } }
            ).unwrap();
            let player_descriptor_set = {
                let mut builder = desc_set_pool.next();
                builder.add_buffer(ppd_buffer.clone()).unwrap();
                builder.build().unwrap()
            };

            // Update game state
            player.update();
            let proj = player.camera.projection();
            let view_projection = linalg::mul(proj, player.camera.view());

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
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipeline.graphics_pipeline.layout().clone(),
                        0,
                        world_descriptor_set
                    );

                for level in 0..(player.cell()[2] + 1) as usize {
                    let [walls, floors, corners, ceilings] = &world_buffer[level];
                    builder
                        .push_constants(
                            pipeline.graphics_pipeline.layout().clone(),
                            0,
                            ViewProjectionData { vp: view_projection, pushColor: [0.0, 0.4, 0.8] })
                        .bind_vertex_buffers(0, (models["wall"].vertices.clone(), walls.clone()))
                        .draw(
                            models["wall"].vertices.len() as u32,
                            walls.len() as u32,
                            0,
                            0)
                        .unwrap()
                        .push_constants(
                            pipeline.graphics_pipeline.layout().clone(),
                            0,
                            ViewProjectionData { vp: view_projection, pushColor: [0.1, 0.6, 0.9] })
                        .bind_vertex_buffers(0, (models["floor"].vertices.clone(), floors.clone()))
                        .draw(
                            models["floor"].vertices.len() as u32,
                            floors.len() as u32,
                            0,
                            0)
                        .unwrap()
                        .push_constants(
                            pipeline.graphics_pipeline.layout().clone(),
                            0,
                            ViewProjectionData { vp: view_projection, pushColor: [0.0, 0.1, 0.3] })
                        .bind_vertex_buffers(0, (models["corner"].vertices.clone(), corners.clone()))
                        .draw(
                            models["corner"].vertices.len() as u32,
                            corners.len() as u32,
                            0,
                            0)
                        .unwrap()
                        .push_constants(
                            pipeline.graphics_pipeline.layout().clone(),
                            0,
                            ViewProjectionData { vp: view_projection, pushColor: [0.2, 0.8, 0.2] })
                        .bind_vertex_buffers(0, (models["ceiling"].vertices.clone(), ceilings.clone()))
                        .draw(
                            models["ceiling"].vertices.len() as u32,
                            ceilings.len() as u32,
                            0,
                            0)
                        .unwrap();
                }
                
                let player_instance = CpuAccessibleBuffer::from_iter(
                    device.clone(),
                    BufferUsage::vertex_buffer(),
                    false,
                    [InstanceModel {
                        m: linalg::model([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], player.get_position())
                    }].into_iter()
                ).unwrap();
                builder
                    .bind_vertex_buffers(0, (player_buffer.clone(), player_instance.clone()))
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipeline.graphics_pipeline.layout().clone(),
                        0,
                        player_descriptor_set
                    )
                    .push_constants(pipeline.graphics_pipeline.layout().clone(), 0, view_projection)
                    .draw(
                        player_buffer.len() as u32,
                        player_instance.len() as u32,
                        0,
                        0).unwrap()
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
