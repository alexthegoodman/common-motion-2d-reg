use burn::{backend::Autodiff, tensor::backend::Backend};
use common_motion_2d_reg::{
    inference::{self, CommonMotionInference},
    interface::load_common_motion_2d,
    training,
};

static ARTIFACT_DIR: &str = "/tmp/common-motion-2d-vae";

use burn::backend::wgpu::{Wgpu, WgpuDevice};

pub fn run_wgpu() {
    let device = WgpuDevice::DiscreteGpu(0);
    run::<Wgpu>(device);
}

/// Train a regression model and predict results on a number of samples.
pub fn run<B: Backend>(device: B::Device) {
    training::run::<Autodiff<B>>(ARTIFACT_DIR, device.clone());
    // println!("Loading model...");
    // let inference: CommonMotionInference<B> = CommonMotionInference::new(device);
    // println!("Running inference...");
    // inference.infer("0, 5, 354, 154, 239, 91, \n1, 5, 544, 244, 106, 240, ".to_string());
}

fn main() {
    // run_wgpu();
    let inference = load_common_motion_2d();
    println!("Running inference...");
    inference.infer("0, 5, 354, 154, 239, 91, \n1, 5, 544, 244, 106, 240, ".to_string());
}
