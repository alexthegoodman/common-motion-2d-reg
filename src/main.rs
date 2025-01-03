use burn::{backend::Autodiff, tensor::backend::Backend};
use common_motion_2d_reg::{inference, training};

static ARTIFACT_DIR: &str = "/tmp/common-motion-2d-reg";

use burn::backend::wgpu::{Wgpu, WgpuDevice};

pub fn run_wgpu() {
    let device = WgpuDevice::DiscreteGpu(0);
    run::<Wgpu>(device);
}

/// Train a regression model and predict results on a number of samples.
pub fn run<B: Backend>(device: B::Device) {
    training::run::<Autodiff<B>>(ARTIFACT_DIR, device.clone());
    inference::infer::<B>(ARTIFACT_DIR, device)
}

fn main() {
    run_wgpu();
}
