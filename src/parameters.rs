use std::sync::Arc;
use std::env;

use vulkano::image::SampleCount;
use vulkano::device::Device;

pub const RAINBOW: [[f32; 3]; 6] = [
    [ 1.000, 0.427, 0.416 ],
    [ 0.937, 0.745, 0.490 ],
    [ 0.914, 0.925, 0.420 ],
    [ 0.467, 0.867, 0.467 ],
    [ 0.545, 0.827, 0.902 ],
    [ 0.694, 0.635, 0.792 ]
];

#[derive(Debug)]
pub struct Params {
    pub samples: u32,
    pub sample_count: SampleCount,
    pub dimensions: [usize; 4],
    pub ghost_move_time: f32,
    pub fps: f32,
    pub food: usize
}

impl Params {
    pub fn new(device: Arc<Device>) -> Params {
        let dimensions: Vec<String> = env::args().collect();
        // First arg is path to executable
        let dimensions: [usize; 4] =
            if dimensions.len() != 5 {
                [5, 5, 5, 5]
            } else {
                [&dimensions[1], &dimensions[2], &dimensions[3], &dimensions[4]].map(|s| s.parse::<usize>().unwrap())
            };

        let (samples, sample_count) = [
                (device.physical_device().properties().framebuffer_color_sample_counts.sample1, 1, SampleCount::Sample1),
                (device.physical_device().properties().framebuffer_color_sample_counts.sample2, 2, SampleCount::Sample2),
                (device.physical_device().properties().framebuffer_color_sample_counts.sample4, 4, SampleCount::Sample4),
                (device.physical_device().properties().framebuffer_color_sample_counts.sample8, 8, SampleCount::Sample8),
                (device.physical_device().properties().framebuffer_color_sample_counts.sample16, 16, SampleCount::Sample16),
                (device.physical_device().properties().framebuffer_color_sample_counts.sample32, 32, SampleCount::Sample32),
                (device.physical_device().properties().framebuffer_color_sample_counts.sample64, 64, SampleCount::Sample64),
            ].iter()
            .filter_map(|(avail, i, sc)| if *avail { Some ((*i, *sc)) } else { None })
            .max_by_key(|(i, _sc)| *i)
            .expect("No framebuffer color sampling options available");
        Params {
            samples,
            sample_count,
            dimensions,
            ghost_move_time: 1.65,
            fps: 60.0,
            food: 10
        }
    }
}
