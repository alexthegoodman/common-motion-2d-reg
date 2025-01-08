use crate::{
    inference::{self, CommonMotionInference},
    training,
};
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::{backend::Autodiff, tensor::backend::Backend};

pub fn load_common_motion_2d() -> CommonMotionInference<Wgpu> {
    let device = WgpuDevice::BestAvailable;

    let inference = load_model_wgpu::<Wgpu>(device);

    inference
}

pub fn load_model_wgpu<B: Backend>(device: B::Device) -> CommonMotionInference<B> {
    println!("Loading model...");
    let inference: CommonMotionInference<B> = CommonMotionInference::new(device);

    inference
}
