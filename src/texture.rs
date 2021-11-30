use std::fs::File;
use std::sync::Arc;

use png::{Decoder, Transformations};
use vulkano::image::view::ImageView;
use vulkano::image::{ImageDimensions, ImageViewAbstract, ImmutableImage, MipmapsCount};
use vulkano::sync::GpuFuture;
use vulkano::format::Format;
use vulkano::device::Queue;

pub struct Texture {
    pub file: String,
    pub image: Arc<ImmutableImage>
}

impl Texture {
    pub fn new(queue: Arc<Queue>, file: &str) -> (Texture, Box<dyn GpuFuture>) {
        let mut decoder = Decoder::new(File::open(file).expect("Failed to open file"));
        decoder.set_transformations(Transformations::empty());
        let mut reader = decoder.read_info().unwrap();
        let dimensions = ImageDimensions::Dim2d {
            width: reader.info().width,
            height: reader.info().height,
            array_layers: 1
        };
        let mut pixels = vec![0; reader.output_buffer_size()];
        reader.next_frame(&mut pixels).unwrap();
        let (image, future) = ImmutableImage::from_iter(
            pixels.into_iter(),
            dimensions,
            MipmapsCount::One,
            Format::R8G8B8A8_SRGB,
            queue).unwrap();
        println!("Loaded texture {}", file);
        (Texture { file: file.to_string(), image }, future.boxed())
    }

    pub fn access(&self) -> Arc<dyn ImageViewAbstract> {
        ImageView::new(self.image.clone()).unwrap()
    }
}
