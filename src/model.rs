use crate::dataset::{KeyframeBatch, NUM_FEATURES};
use burn::{
    config::Config,
    module::Module,
    nn::{loss::MseLoss, Linear, LinearConfig, Relu},
    tensor::{
        backend::{AutodiffBackend, Backend},
        Tensor,
    },
    train::{
        metric::{Adaptor, LossInput},
        TrainOutput, TrainStep, ValidStep,
    },
};

#[derive(Module, Debug)]
pub struct RegressionModel<B: Backend> {
    input_layer: Linear<B>,
    output_layer: Linear<B>,
    activation: Relu,
}

#[derive(Config)]
pub struct RegressionModelConfig {
    #[config(default = 64)]
    pub hidden_size: usize,
}

#[derive(Debug)]
pub struct RegressionOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub output: Tensor<B, 3>,  // Changed to match sequence output
    pub targets: Tensor<B, 3>, // Changed to match sequence targets
}

impl<B: Backend> Adaptor<LossInput<B>> for RegressionOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

impl RegressionModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RegressionModel<B> {
        let input_layer = LinearConfig::new(NUM_FEATURES, self.hidden_size)
            .with_bias(true)
            .init(device);
        let output_layer = LinearConfig::new(self.hidden_size, NUM_FEATURES) // Output all features
            .with_bias(true)
            .init(device);

        RegressionModel {
            input_layer,
            output_layer,
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> RegressionModel<B> {
    /// Process a single timestep of the sequence
    fn forward_timestep(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.input_layer.forward(input);
        let x = self.activation.forward(x);
        self.output_layer.forward(x)
    }

    /// Forward pass handling the full sequence
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, features] = input.dims();

        // Reshape to (batch_size * seq_len, features)
        let flat_input = input.reshape([batch_size * seq_len, features]);

        // Process all timesteps at once
        let flat_output = self.forward_timestep(flat_input);

        // Reshape back to (batch_size, seq_len, features)
        flat_output.reshape([batch_size, seq_len, NUM_FEATURES])
    }

    pub fn forward_step(&self, item: KeyframeBatch<B>) -> RegressionOutput<B> {
        let output = self.forward(item.inputs.clone());

        let max_len = std::cmp::max(item.inputs.dims()[1], item.targets.dims()[1]);

        // Create mask for padding
        let mask = create_sequence_mask(&item.input_lengths, max_len, &item.inputs.device());

        // create target mask
        let target_mask =
            create_sequence_mask(&item.target_lengths, max_len, &item.targets.device());

        // Apply mask to both output and targets
        let masked_output = output.clone() * mask.clone();
        // let masked_targets = item.targets.clone() * mask;
        let masked_targets = item.targets.clone() * target_mask;

        // Compute loss only on valid (unmasked) elements
        let loss = MseLoss::new()
            .forward(
                masked_output.clone(),
                masked_targets.clone(),
                burn::nn::loss::Reduction::Mean,
            )
            .mean();

        RegressionOutput {
            loss,
            output,
            targets: item.targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<KeyframeBatch<B>, RegressionOutput<B>> for RegressionModel<B> {
    fn step(&self, item: KeyframeBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_step(item);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<KeyframeBatch<B>, RegressionOutput<B>> for RegressionModel<B> {
    fn step(&self, item: KeyframeBatch<B>) -> RegressionOutput<B> {
        self.forward_step(item)
    }
}

/// Creates a mask tensor for handling variable-length sequences
fn create_sequence_mask<B: Backend>(
    lengths: &Tensor<B, 1>,
    max_len: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let batch_size = lengths.dims()[0];

    // Create mask of shape [batch_size, max_len, 1]
    let mut mask_data = vec![0.0; batch_size * max_len * NUM_FEATURES];

    let lengths: Vec<f32> = lengths
        .to_data()
        .to_vec()
        .expect("Failed to convert lengths to Vec");

    // Set valid positions to 1.0
    for b in 0..batch_size {
        let length = *lengths.get(b).expect("Couldn't get length") as usize;
        for l in 0..length {
            mask_data[b * max_len + l] = 1.0;
        }
    }

    // Create and reshape mask tensor
    // fix like dataset pad_sequence?
    Tensor::<B, 1>::from_floats(mask_data.as_slice(), device)
        // .reshape([batch_size, max_len, 1])
        // .broadcast_like([batch_size, max_len, NUM_FEATURES]) // doesnt exist
        // why not this
        .reshape([batch_size, max_len, NUM_FEATURES])
}
