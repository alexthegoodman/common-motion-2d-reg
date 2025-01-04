use std::fs::OpenOptions;
use std::io::Write;

use crate::dataset::{KeyframeBatch, Normalizer, NUM_FEATURES};
use burn::{
    config::Config,
    module::Module,
    nn::{loss::MseLoss, Linear, LinearConfig, Lstm, LstmConfig, Relu},
    tensor::{
        backend::{AutodiffBackend, Backend},
        Tensor,
    },
    train::{
        metric::{Adaptor, LossInput},
        TrainOutput, TrainStep, ValidStep,
    },
};

// #[derive(Module, Debug)]
// pub struct RegressionModel<B: Backend> {
//     input_layer: Linear<B>,
//     output_layer: Linear<B>,
//     hidden_layers: Vec<Linear<B>>, // A vector of additional hidden layers
//     activation: Relu,
// }

// #[derive(Config)]
// pub struct RegressionModelConfig {
//     // #[config(default = 64)]
//     #[config(default = 128)]
//     pub hidden_size: usize,
// }

// #[derive(Debug)]
// pub struct RegressionOutput<B: Backend> {
//     pub loss: Tensor<B, 1>,
//     pub output: Tensor<B, 3>,  // Changed to match sequence output
//     pub targets: Tensor<B, 3>, // Changed to match sequence targets
// }

// impl<B: Backend> Adaptor<LossInput<B>> for RegressionOutput<B> {
//     fn adapt(&self) -> LossInput<B> {
//         LossInput::new(self.loss.clone())
//     }
// }

// const NUM_HIDDEN_LAYERS: usize = 2;

// impl RegressionModelConfig {
//     pub fn init<B: Backend>(&self, device: &B::Device) -> RegressionModel<B> {
//         let input_layer = LinearConfig::new(NUM_FEATURES, self.hidden_size)
//             .with_bias(true)
//             .init(device);

//         // Create multiple hidden layers
//         let hidden_layers: Vec<Linear<B>> = (0..NUM_HIDDEN_LAYERS)
//             .map(|_| {
//                 LinearConfig::new(self.hidden_size, self.hidden_size)
//                     .with_bias(true)
//                     .init(device)
//             })
//             .collect();

//         let output_layer = LinearConfig::new(self.hidden_size, NUM_FEATURES) // Output all features
//             .with_bias(true)
//             .init(device);

//         RegressionModel {
//             input_layer,
//             output_layer,
//             hidden_layers,
//             activation: Relu::new(),
//         }
//     }
// }

// impl<B: Backend> RegressionModel<B> {
//     /// Process a single timestep of the sequence
//     fn forward_timestep(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
//         let x = self.input_layer.forward(input);
//         let mut x = self.activation.forward(x);

//         // Pass through all hidden layers
//         for hidden_layer in &self.hidden_layers {
//             x = hidden_layer.forward(x);
//             x = self.activation.forward(x);
//         }

//         self.output_layer.forward(x)
//     }

//     /// Forward pass handling the full sequence
//     pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
//         let [batch_size, seq_len, features] = input.dims();

//         // Reshape to (batch_size * seq_len, features)
//         let flat_input = input.reshape([batch_size * seq_len, features]);

//         // Process all timesteps at once
//         let flat_output = self.forward_timestep(flat_input);

//         // Reshape back to (batch_size, seq_len, features)
//         flat_output.reshape([batch_size, seq_len, NUM_FEATURES])
//     }

//     pub fn forward_step(&self, item: KeyframeBatch<B>) -> RegressionOutput<B> {
//         let normalizer = Normalizer::new(&item.inputs.device());

//         // Normalize inputs and targets
//         let normalized_inputs = normalizer.normalize(item.inputs.clone());
//         let normalized_targets = normalizer.normalize(item.targets.clone());

//         let output = self.forward(normalized_inputs);

//         let max_len = std::cmp::max(item.inputs.dims()[1], item.targets.dims()[1]);
//         let mask = create_sequence_mask(&item.input_lengths, max_len, &item.inputs.device());
//         let target_mask =
//             create_sequence_mask(&item.target_lengths, max_len, &item.targets.device());

//         let masked_output = output.clone() * mask.clone();
//         let masked_targets = normalized_targets.clone() * target_mask;

//         let loss = MseLoss::new()
//             .forward(
//                 masked_output.clone(),
//                 masked_targets.clone(),
//                 burn::nn::loss::Reduction::Mean,
//             )
//             .mean();

//         // Denormalize for the actual predictions
//         let denormalized_output = normalizer.denormalize(output);

//         RegressionOutput {
//             loss,
//             output: denormalized_output,
//             targets: item.targets,
//         }
//     }
// }

// impl<B: AutodiffBackend> TrainStep<KeyframeBatch<B>, RegressionOutput<B>> for RegressionModel<B> {
//     fn step(&self, item: KeyframeBatch<B>) -> TrainOutput<RegressionOutput<B>> {
//         let item = self.forward_step(item);
//         TrainOutput::new(self, item.loss.backward(), item)
//     }
// }

// impl<B: Backend> ValidStep<KeyframeBatch<B>, RegressionOutput<B>> for RegressionModel<B> {
//     fn step(&self, item: KeyframeBatch<B>) -> RegressionOutput<B> {
//         self.forward_step(item)
//     }
// }

#[derive(Module, Debug)]
pub struct RnnModel<B: Backend> {
    // lstm: Lstm<B>,
    lstm_layers: Vec<Lstm<B>>,
    output_layer: Linear<B>,
    hidden_layers: Vec<Linear<B>>, // A vector of additional hidden layers
    activation: Relu,              // or try LeakyReLU
}

#[derive(Config)]
pub struct RnnModelConfig {
    #[config(default = 768)]
    // #[config(default = 64)]
    // #[config(default = 256)]
    pub hidden_size: usize,
}

#[derive(Debug)]
pub struct RnnOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub output: Tensor<B, 3>,
    pub targets: Tensor<B, 3>,
}

impl<B: Backend> Adaptor<LossInput<B>> for RnnOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

const NUM_HIDDEN_LAYERS: usize = 1;
const NUM_LSTM_LAYERS: usize = 2;

impl RnnModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RnnModel<B> {
        // Initialize LSTM
        // let lstm = LstmConfig::new(NUM_FEATURES, self.hidden_size, true).init(device);

        let mut lstm_layers: Vec<Lstm<B>> = Vec::with_capacity(NUM_LSTM_LAYERS);
        for _ in 0..NUM_LSTM_LAYERS {
            // first layer d_input = NUM_FEATURES, other layers d_input = self.hidden_size
            // lstm_layers.push(LstmConfig::new(NUM_FEATURES, self.hidden_size, true).init(device));

            if lstm_layers.is_empty() {
                lstm_layers.push(
                    LstmConfig::new(NUM_FEATURES, self.hidden_size, true)
                        .with_initializer(burn::nn::Initializer::XavierNormal { gain: 0.5 })
                        .init(device),
                )
            } else {
                lstm_layers.push(
                    LstmConfig::new(self.hidden_size, self.hidden_size, true)
                        .with_initializer(burn::nn::Initializer::XavierNormal { gain: 0.5 })
                        .init(device),
                );
            }
        }

        // Output layer to map from hidden size back to feature size
        let output_layer = LinearConfig::new(self.hidden_size, NUM_FEATURES)
            .with_bias(true)
            .with_initializer(burn::nn::Initializer::KaimingNormal {
                fan_out_only: true,
                gain: 1.0,
            })
            .init(device);

        // Create multiple hidden layers
        let hidden_layers: Vec<Linear<B>> = (0..NUM_HIDDEN_LAYERS)
            .map(|_| {
                LinearConfig::new(self.hidden_size, self.hidden_size)
                    .with_bias(true)
                    .with_initializer(burn::nn::Initializer::KaimingNormal {
                        fan_out_only: true,
                        gain: 1.0,
                    })
                    .init(device)
            })
            .collect();

        RnnModel {
            // lstm,
            lstm_layers,
            output_layer,
            hidden_layers,
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> RnnModel<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, features] = input.dims();

        // Process sequence through LSTM
        // LSTM output will be (batch_size, seq_len, hidden_size)
        // let lstm_out: (Tensor<B, 3>, burn::nn::LstmState<B, 2>) = self.lstm.forward(input, None);
        // Process sequence through LSTM layers
        // Create a file handle for logging (only create if it doesn't exist)
        let mut lstm_out = input;
        let mut lstm_state = None;
        // let mut log_file = OpenOptions::new()
        //     .write(true)
        //     .create(true)
        //     .append(true) // Append to existing file, if it exists
        //     .open("/tmp/common-motion-2d-reg/lstm_log.txt")
        //     .expect("Failed to create or open log file");

        for lstm in &self.lstm_layers {
            // Write log messages to the file
            // writeln!(
            //     log_file,
            //     "lstm_out shape: {:?} {:?}",
            //     lstm_out.dims(),
            //     lstm_out.shape()
            // )
            // .expect("Failed to write to log file");

            let (out, state) = lstm.forward(lstm_out, lstm_state);
            lstm_out = out;
            // TODO: how to save state for multiple iterations?
            lstm_state = Some(state);
        }

        // Close the log file
        // drop(log_file);

        // Reshape to (batch_size * seq_len, hidden_size) for the linear layer
        let mut flat_lstm_out = lstm_out
            // .0
            .reshape([batch_size * seq_len, self.lstm_layers[0].d_hidden]);

        // Pass through all hidden layers
        for hidden_layer in &self.hidden_layers {
            flat_lstm_out = hidden_layer.forward(flat_lstm_out);
            flat_lstm_out = self.activation.forward(flat_lstm_out);
        }

        // Pass through output layer
        let flat_output = self.output_layer.forward(flat_lstm_out);

        // Reshape back to (batch_size, seq_len, features)
        flat_output.reshape([batch_size, seq_len, NUM_FEATURES])
    }

    pub fn forward_step(&self, item: KeyframeBatch<B>) -> RnnOutput<B> {
        let normalizer = Normalizer::new(&item.inputs.device());

        // Normalize inputs and targets
        let normalized_inputs = normalizer.normalize(item.inputs.clone());
        let normalized_targets = normalizer.normalize(item.targets.clone());

        let output = self.forward(normalized_inputs);

        let max_len = std::cmp::max(item.inputs.dims()[1], item.targets.dims()[1]);
        let mask = create_sequence_mask(&item.input_lengths, max_len, &item.inputs.device());
        let target_mask =
            create_sequence_mask(&item.target_lengths, max_len, &item.targets.device());

        // // log to file
        // let mut log_file = OpenOptions::new()
        //     .write(true)
        //     .create(true)
        //     .append(true) // Append to existing file, if it exists
        //     .open("/tmp/common-motion-2d-reg/rnn_log.txt")
        //     .expect("Failed to create or open log file");

        // writeln!(
        //     log_file,
        //     "output shape: {:?} {:?}",
        //     output.dims(),
        //     output.shape()
        // )
        // .expect("Failed to write to log file");

        // writeln!(log_file, "mask shape: {:?} {:?}", mask.dims(), mask.shape())
        //     .expect("Failed to write to log file");

        // writeln!(
        //     log_file,
        //     "normalized_targets shape: {:?} {:?}",
        //     normalized_targets.dims(),
        //     normalized_targets.shape()
        // )
        // .expect("Failed to write to log file");

        // writeln!(
        //     log_file,
        //     "target_mask shape: {:?} {:?}",
        //     target_mask.dims(),
        //     target_mask.shape()
        // )
        // .expect("Failed to write to log file");

        let masked_output = output.clone() * mask.clone();
        let masked_targets = normalized_targets.clone() * target_mask;

        let loss = MseLoss::new()
            .forward(
                masked_output.clone(),
                masked_targets.clone(),
                burn::nn::loss::Reduction::Mean,
            )
            .mean();

        // Denormalize for the actual predictions
        let denormalized_output = normalizer.denormalize(output);

        // drop(log_file);

        RnnOutput {
            loss,
            output: denormalized_output,
            targets: item.targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<KeyframeBatch<B>, RnnOutput<B>> for RnnModel<B> {
    fn step(&self, item: KeyframeBatch<B>) -> TrainOutput<RnnOutput<B>> {
        let item = self.forward_step(item);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<KeyframeBatch<B>, RnnOutput<B>> for RnnModel<B> {
    fn step(&self, item: KeyframeBatch<B>) -> RnnOutput<B> {
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
