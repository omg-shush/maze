use std::sync::Arc;

use vulkano::image::SampleCount;
use vulkano::device::Device;

pub const RAINBOW: [[f32; 3]; 7] = [
    [ 0.8, 0.2, 0.2 ],
    [ 0.8, 0.4, 0.2 ],
    [ 0.4, 0.8, 0.2 ],
    [ 0.2, 0.8, 0.2 ],
    [ 0.2, 0.4, 0.8 ],
    [ 0.2, 0.2, 0.8 ],
    [ 0.4, 0.2, 0.8 ]
];

#[derive(Debug)]
pub struct Params {
    pub samples: u32,
    pub sample_count: SampleCount
}

impl Params {
    pub fn new(device: Arc<Device>) -> Params {
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
            samples: samples,
            sample_count: sample_count
        }
    }
}
