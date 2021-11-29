use std::borrow::Cow;
use std::collections::HashMap;
use std::vec;
use std::sync::Arc;
use std::time::Instant;

use vulkano::descriptor_set::{SingleLayoutDescSetPool};
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, KeyboardInput, VirtualKeyCode, WindowEvent, ElementState};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{WindowBuilder};
use winit::dpi::{PhysicalPosition, LogicalSize};

use vulkano::device::{Device, Features, DeviceExtensions};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::instance::{Instance, ApplicationInfo};
use vulkano::Version;
use vulkano::image::ImageUsage;
use vulkano::image::view::ImageView;
use vulkano::image::attachment::AttachmentImage;
use vulkano::swapchain;
use vulkano::swapchain::{Swapchain, AcquireError, SwapchainCreationError};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents};
use vulkano::pipeline::viewport::Viewport;
use vulkano::render_pass::{Framebuffer, FramebufferAbstract};
use vulkano::sync;
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::format::{ClearValue, Format};

use world::World;
use pipeline::cs::ty::Vertex;
use parameters::Params;
use player::{Player, GameState};
use model::Model;
use ui::UserInterface;
use ghost::Ghost;
use objects::Objects;
use texture::Texture;

mod world;
mod pipeline;
mod disjoint_set;
mod camera;
mod parameters;
mod player;
mod linalg;
mod model;
mod texture;
mod ui;
mod ghost;
mod objects;

const NAME: &str = "4D Pacman v0.2";

fn main() {
    // Create vulkan instance
    let app_infos = ApplicationInfo {
        application_name: Some(Cow::from(NAME)),
        application_version: Some(Version::V1_2),
        engine_name: None,
        engine_version: None };
    let instance_exts = vulkano_win::required_extensions();
    let instance = Instance::new(Some(&app_infos), Version::V1_2, &instance_exts, None).unwrap();

    // for layer in instance::layers_list().unwrap() {
    //     println!("Layer: {}", layer.name())
    // }

    let card = {
        let card_list = PhysicalDevice::enumerate(&instance).collect::<Vec<_>>();
        println!("Card list: {:?}", card_list.iter().map(|c| c.properties().device_name.clone()).collect::<Vec<_>>());
        let mut discrete_list = card_list.clone().into_iter().filter(|c| c.properties().device_type == PhysicalDeviceType::DiscreteGpu);
        discrete_list.next().unwrap_or(card_list[0])
    };
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
    let queues = [(draw_queue, 1.0)];
    let (device, mut qs) = Device::new(card, &features, &extensions, queues.iter().cloned()).unwrap();
    let draw_queue = qs.next().unwrap();
    println!("Created logical vulkan device {:?}", device);

    // Create window
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .with_inner_size(LogicalSize { width: 640, height: 640 })
        .with_position(PhysicalPosition { x : 300, y: 200 })
        .with_resizable(false)
        .with_title(NAME)
        .build_vk_surface(&event_loop, instance.clone()).unwrap();

    // Configure parameters
    let params = Params::new(device.clone());
    println!("{:?}", params);

    // Create swapchain
    let surface_caps = surface.capabilities(card).unwrap();
    let resolution = surface_caps.max_image_extent;
    let buffers = 2.clamp(surface_caps.min_image_count, surface_caps.max_image_count.unwrap_or(u32::MAX));
    println!("Using {} buffers. Min ({}) max {:?}", buffers, surface_caps.min_image_count, surface_caps.max_image_count);
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

    let mut init_futures = Vec::new();

    // Load models
    let models: HashMap<String, Box<Model>> = [
        Model::new(draw_queue.clone(), "wall.obj"),
        Model::new(draw_queue.clone(), "floor.obj"),
        Model::new(draw_queue.clone(), "corner.obj"),
        Model::new(draw_queue.clone(), "ceiling.obj")
    ].map(|(model, future)| {
        init_futures.push(future);
        (model.file.to_owned(), model)
    }).into_iter().collect();

    // Load textures
    let textures: HashMap<String, Texture> = [
        Texture::new(draw_queue.clone(), "controls.png"),
        Texture::new(draw_queue.clone(), "controls_dim.png"),
        Texture::new(draw_queue.clone(), "digits.png"),
        Texture::new(draw_queue.clone(), "win.png"),
        Texture::new(draw_queue.clone(), "lose.png")
    ].map(|(texture, future)| {
        init_futures.push(future);
        (texture.file.split(".").next().unwrap().to_owned(), texture)
    }).into_iter().collect();

    // Initialize game elements
    let (world, world_init_future) = World::new(&params, draw_queue.clone());
    let (mut player, player_init_future) = Player::new(device.clone(), draw_queue.clone(), world.clone());
    let (mut ghost, ghost_init_future) = Ghost::new(&params, draw_queue.clone(), world.clone(), [1.0, 1.0, 1.0]);
    let mut objects = Objects::new(draw_queue.clone(), &mut world.borrow_mut(), &params);
    let ui = UserInterface::new(draw_queue.clone(),pipeline.render_pass.clone(), &textures);
    init_futures.push(world_init_future);
    init_futures.push(player_init_future);
    init_futures.push(ghost_init_future);

    let init_future = init_futures.into_iter().fold(sync::now(device.clone()).boxed(), |acc, future| {
        acc.join(future).boxed()
    }).then_signal_fence_and_flush().expect("Flushing init commands failed");

    println!("---------------------------");
    println!("{0}", NAME);
    println!("WASD or Arrow Keys to move horizontally");
    println!("SPACE to move up, LeftControl to move down");
    println!("Q and E to move through left and right portals");
    println!("green screen = win");
    println!("Specify custom dimensions as command line arguments, eg:");
    println!("    maze.exe 10 10 10 10");

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
                    .add(dview.clone()
                ).unwrap().build().unwrap()
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        }).collect::<Vec<_>>();

    let mut previous_frame_end = Some (init_future.boxed());
    let mut previous_frame = Instant::now();
    let mut recreate_swapchain = false;
    let mut desc_set_pool = SingleLayoutDescSetPool::new(
        pipeline.graphics_pipeline.layout().descriptor_set_layouts()[0].clone()
    );

    // Up, down, left, right, ascend, descend, fourth dec, fourth inc
    let mut keys = [ElementState::Released; 8];

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
            if player.game_state != GameState::Playing {
                return; // ignore user input
            }
            let world = world.borrow();
            let seconds = 0.5;
            match keycode {
                VirtualKeyCode::W | VirtualKeyCode::Up => {
                    if state == ElementState::Pressed && keys[0] == ElementState::Released {
                        if world.check_move(player.cell(), [0, -1, 0, 0]) {
                            player.move_position([0, -1, 0, 0], seconds);
                        }
                    }
                    keys[0] = state;
                },
                VirtualKeyCode::S | VirtualKeyCode::Down => {
                    if state == ElementState::Pressed && keys[1] == ElementState::Released {
                        if world.check_move(player.cell(), [0, 1, 0, 0]) {
                            player.move_position([0, 1, 0, 0], seconds);
                        }
                    }
                    keys[1] = state
                },
                VirtualKeyCode::A | VirtualKeyCode::Left => {
                    if state == ElementState::Pressed && keys[2] == ElementState::Released {
                        if world.check_move(player.cell(), [-1, 0, 0, 0]) {
                            player.move_position([-1, 0, 0, 0], seconds);
                        }
                    }
                    keys[2] = state
                },
                VirtualKeyCode::D | VirtualKeyCode::Right => {
                    if state == ElementState::Pressed && keys[3] == ElementState::Released {
                        if world.check_move(player.cell(), [1, 0, 0, 0]) {
                            player.move_position([1, 0, 0, 0], seconds);
                        }
                    }
                    keys[3] = state
                },
                VirtualKeyCode::Space => {
                    if state == ElementState::Pressed && keys[4] == ElementState::Released {
                        if world.check_move(player.cell(), [0, 0, 1, 0]) {
                            player.move_position([0, 0, 1, 0], seconds);
                            objects.dirty_buffer = true;
                        }
                    }
                    keys[4] = state
                },
                VirtualKeyCode::LControl => {
                    if state == ElementState::Pressed && keys[5] == ElementState::Released {
                        if world.check_move(player.cell(), [0, 0, -1, 0]) {
                            player.move_position([0, 0, -1, 0], seconds);
                            objects.dirty_buffer = true;
                        }
                    }
                    keys[5] = state
                },
                VirtualKeyCode::Q => {
                    if state == ElementState::Pressed && keys[6] == ElementState::Released {
                        if world.check_move(player.cell(), [0, 0, 0, -1]) {
                            player.move_position([0, 0, 0, -1], seconds);
                            objects.dirty_buffer = true;
                        }
                    }
                },
                VirtualKeyCode::E => {
                    if state == ElementState::Pressed && keys[7] == ElementState::Released {
                        if world.check_move(player.cell(), [0, 0, 0, 1]) {
                            player.move_position([0, 0, 0, 1], seconds);
                            objects.dirty_buffer = true;
                        }
                    }
                }
                _ => {}
            }
        }
        Event::RedrawEventsCleared => {
            let now = Instant::now();
            if (now - previous_frame).as_secs_f32() < 1.0 / params.fps {
                return; // Don't render another frame yet
            }
            previous_frame = now;

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
            let mut builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                draw_queue.family(),
                CommandBufferUsage::OneTimeSubmit
            ).unwrap();

            // Update game state
            if player.game_state == GameState::Playing {
                player.update(&params, &mut objects);
                ghost.update(&mut player);
                objects.update(&player);
            }

            if player.game_state != GameState::Playing {
                builder
                    .begin_render_pass(
                        framebuffers[image_num].clone(),
                        SubpassContents::Inline,
                        clear_values
                    ).unwrap()
                    .set_viewport(0, [viewport.clone()])
                    .bind_pipeline_graphics(pipeline.graphics_pipeline.clone());
                
                // Game over; only render UI
                ui.render(&player, &world.borrow(), &params, &mut builder);

                builder.end_render_pass().unwrap();
            } else {
                builder
                    .begin_render_pass(
                        framebuffers[image_num].clone(),
                        SubpassContents::Inline,
                        clear_values
                    ).unwrap()
                    .set_viewport(0, [viewport.clone()])
                    .bind_pipeline_graphics(pipeline.graphics_pipeline.clone());

                world.borrow().render(&models, &player, &mut desc_set_pool, &mut builder, &pipeline);
                player.render(&mut desc_set_pool, &mut builder, &pipeline);
                ghost.render(&player, &mut desc_set_pool, &mut builder, &pipeline);
                objects.render(&player, &world.borrow(), &models, &mut builder, &pipeline);
                ui.render(&player, &world.borrow(), &params, &mut builder);
                
                builder.end_render_pass().unwrap();
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
