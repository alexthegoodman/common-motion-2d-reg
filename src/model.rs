use std::fs::OpenOptions;
use std::io::Write;

use crate::dataset::{KeyframeBatch, Normalizer, NUM_FEATURES};
use burn::{
    config::Config,
    module::Module,
    nn::{
        loss::MseLoss, Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, LeakyRelu,
        LeakyReluConfig, Linear, LinearConfig, Lstm, LstmConfig, Relu,
    },
    tensor::{
        backend::{AutodiffBackend, Backend},
        Distribution, Tensor,
    },
    train::{
        metric::{AccuracyInput, Adaptor, LossInput},
        TrainOutput, TrainStep, ValidStep,
    },
};

// #[derive(Module, Debug)]
// pub struct RnnModel<B: Backend> {
//     lstm_layers: Vec<Lstm<B>>,
//     layer_norms: Vec<LayerNorm<B>>,
//     output_norm: LayerNorm<B>, // Add specific normalization for output
//     dropout: Dropout,
//     hidden_layers: Vec<Linear<B>>,
//     output_layer: Linear<B>,
//     // activation: LeakyRelu,
//     activation: Gelu,
//     hidden_norms: Vec<LayerNorm<B>>, // Add normalization for hidden layers
// }

// #[derive(Config)]
// pub struct RnnModelConfig {
//     // #[config(default = 768)]
//     #[config(default = 64)]
//     // #[config(default = 256)]
//     pub hidden_size: usize,
// }

// #[derive(Debug)]
// pub struct RnnOutput<B: Backend> {
//     pub loss: Tensor<B, 1>,
//     pub output: Tensor<B, 3>,
//     pub targets: Tensor<B, 3>,
// }

// impl<B: Backend> Adaptor<LossInput<B>> for RnnOutput<B> {
//     fn adapt(&self) -> LossInput<B> {
//         LossInput::new(self.loss.clone())
//     }
// }

// // setup accuracy
// // impl<B: Backend> Adaptor<AccuracyInput<B>> for RnnOutput<B> {
// //     fn adapt(&self) -> AccuracyInput<B> {
// //         AccuracyInput::new(self.output.clone(), self.targets.clone())
// //     }
// // }

// const NUM_HIDDEN_LAYERS: usize = 1;
// const NUM_LSTM_LAYERS: usize = 1;
// const LEAKY_RELU_SLOPE: f64 = 0.1;
// const DROPOUT_RATE: f64 = 0.2;

// impl RnnModelConfig {
//     pub fn init<B: Backend>(&self, device: &B::Device) -> RnnModel<B> {
//         let mut lstm_layers: Vec<Lstm<B>> = Vec::with_capacity(NUM_LSTM_LAYERS);
//         let mut layer_norms: Vec<LayerNorm<B>> = Vec::with_capacity(NUM_LSTM_LAYERS);
//         let mut hidden_norms: Vec<LayerNorm<B>> = Vec::with_capacity(3); // For hidden layers

//         // Use scaled initialization to help with activation scaling
//         let initializer = burn::nn::Initializer::KaimingNormal {
//             fan_out_only: false,
//             gain: (1.0 + LEAKY_RELU_SLOPE * LEAKY_RELU_SLOPE).sqrt(), // Adjust gain for LeakyReLU
//         };

//         // LSTM layers
//         for i in 0..NUM_LSTM_LAYERS {
//             let input_size = if i == 0 {
//                 NUM_FEATURES
//             } else {
//                 self.hidden_size
//             };

//             lstm_layers.push(
//                 LstmConfig::new(input_size, self.hidden_size, true)
//                     .with_initializer(initializer.clone())
//                     .init(device),
//             );

//             layer_norms.push(
//                 LayerNormConfig::new(self.hidden_size)
//                     .with_epsilon(1e-5)
//                     .init(device),
//             );
//         }

//         // Hidden layers with their own normalizations
//         let hidden_layers: Vec<Linear<B>> = (0..3)
//             .map(|_| {
//                 hidden_norms.push(
//                     LayerNormConfig::new(self.hidden_size)
//                         .with_epsilon(1e-5)
//                         .init(device),
//                 );

//                 LinearConfig::new(self.hidden_size, self.hidden_size)
//                     .with_bias(true)
//                     .with_initializer(initializer.clone())
//                     .init(device)
//             })
//             .collect();

//         // Output layer with larger initialization to help with range
//         let output_layer = LinearConfig::new(self.hidden_size, NUM_FEATURES)
//             .with_bias(true)
//             .with_initializer(burn::nn::Initializer::KaimingNormal {
//                 fan_out_only: false,
//                 gain: 2.0, // Increased gain for output layer
//             })
//             .init(device);

//         // Special normalization for output
//         let output_norm = LayerNormConfig::new(NUM_FEATURES)
//             .with_epsilon(1e-5)
//             .init(device);

//         RnnModel {
//             lstm_layers,
//             layer_norms,
//             output_norm,
//             dropout: DropoutConfig::new(DROPOUT_RATE).init(),
//             hidden_layers,
//             output_layer,
//             // activation: LeakyReluConfig::new()
//             //     .with_negative_slope(LEAKY_RELU_SLOPE)
//             //     .init(),
//             activation: Gelu::new(),
//             hidden_norms,
//         }
//     }
// }

// impl<B: Backend> RnnModel<B> {
//     pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
//         let [batch_size, seq_len, _features] = input.dims();

//         let mut lstm_out = input;
//         let mut lstm_state = None;

//         // LSTM layers with careful normalization
//         for (i, ((lstm, layer_norm), prev_lstm)) in self
//             .lstm_layers
//             .iter()
//             .zip(self.layer_norms.iter())
//             .zip(std::iter::once(None).chain(self.lstm_layers.iter().map(Some)))
//             .enumerate()
//         {
//             let (out, state) = lstm.forward(lstm_out.clone(), lstm_state);
//             // NOTE: norms not needed as data is already normalized acceptably
//             // let normalized = layer_norm.forward(out);

//             // Residual connection if not first layer
//             // if i > 0 {
//             //     lstm_out = normalized + lstm_out;
//             // } else {
//             //     lstm_out = normalized;
//             // }

//             lstm_out = out;

//             // lstm_out = self.dropout.forward(lstm_out);
//             lstm_state = Some(state);
//         }

//         // Process through hidden layers
//         let mut hidden = lstm_out.reshape([batch_size * seq_len, self.lstm_layers[0].d_hidden]);

//         for (i, (hidden_layer, hidden_norm)) in self
//             .hidden_layers
//             .iter()
//             .zip(self.hidden_norms.iter())
//             .enumerate()
//         {
//             let layer_out = hidden_layer.forward(hidden.clone());
//             let activated = self.activation.forward(layer_out);
//             // NOTE: norms not needed as data is already normalized acceptably
//             // let normalized = hidden_norm.forward(activated);

//             // Residual connection
//             // hidden = normalized + hidden;

//             hidden = activated;

//             // Only apply dropout between layers, not after final layer
//             // if i < self.hidden_layers.len() - 1 {
//             //     hidden = self.dropout.forward(hidden);
//             // }
//         }

//         // Output layer with normalization to help with output range
//         let output = self.output_layer.forward(hidden);
//         // let normalized_output = self.output_norm.forward(output);

//         output.reshape([batch_size, seq_len, NUM_FEATURES])
//     }

//     pub fn forward_step(&self, item: KeyframeBatch<B>) -> RnnOutput<B> {
//         let normalizer = Normalizer::new(&item.inputs.device());

//         // Normalize inputs and targets
//         let normalized_inputs = normalizer.normalize(item.inputs.clone());
//         let normalized_targets = normalizer.normalize(item.targets.clone());

//         let output = self.forward(normalized_inputs);

//         // NOTE: already masked in batcher

//         let loss = MseLoss::new().forward(
//             output.clone(),
//             normalized_targets.clone(),
//             burn::nn::loss::Reduction::Mean,
//         );

//         // Denormalize for the actual predictions
//         let denormalized_output = normalizer.denormalize(output);

//         RnnOutput {
//             loss,
//             output: denormalized_output,
//             targets: item.targets,
//         }
//     }
// }

// impl<B: AutodiffBackend> TrainStep<KeyframeBatch<B>, RnnOutput<B>> for RnnModel<B> {
//     fn step(&self, item: KeyframeBatch<B>) -> TrainOutput<RnnOutput<B>> {
//         let item = self.forward_step(item);
//         TrainOutput::new(self, item.loss.backward(), item)
//     }
// }

// impl<B: Backend> ValidStep<KeyframeBatch<B>, RnnOutput<B>> for RnnModel<B> {
//     fn step(&self, item: KeyframeBatch<B>) -> RnnOutput<B> {
//         self.forward_step(item)
//     }
// }

// use ndarray::{Array, Array1, Array2, ArrayView, ArrayView1, ArrayView2, Axis, StrideShape};
// use ndarray_stats::EntropyExt;

// modfiied model approach:
#[derive(Module, Debug)]
pub struct RnnModel<B: Backend> {
    lstm_layers: Vec<Lstm<B>>,
    lstm_decoder: Lstm<B>,
    // layer_norms: Vec<LayerNorm<B>>,
    output_norm: LayerNorm<B>, // Add specific normalization for output
    dropout: Dropout,
    hidden_layers: Vec<Linear<B>>,
    output_layer: Linear<B>,
    activation: Gelu,
    // hidden_norms: Vec<LayerNorm<B>>, // Add normalization for hidden layers

    // VAE components
    mean_layer: Linear<B>,
    logvar_layer: Linear<B>,

    // for convenience
    pub hidden_size: usize,
    pub latent_dim: usize,
}

#[derive(Config)]
pub struct RnnModelConfig {
    #[config(default = 256)]
    // #[config(default = 64)]
    pub hidden_size: usize,
    #[config(default = 128)]
    // #[config(default = 32)] // Latent space dimension
    pub latent_dim: usize,
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
const NUM_LSTM_LAYERS: usize = 1;
const LEAKY_RELU_SLOPE: f64 = 0.1;
const DROPOUT_RATE: f64 = 0.2;

impl RnnModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RnnModel<B> {
        let mut lstm_layers: Vec<Lstm<B>> = Vec::with_capacity(NUM_LSTM_LAYERS);
        let mut layer_norms: Vec<LayerNorm<B>> = Vec::with_capacity(NUM_LSTM_LAYERS);
        let mut hidden_norms: Vec<LayerNorm<B>> = Vec::with_capacity(3); // For hidden layers

        // Use scaled initialization to help with activation scaling
        let initializer = burn::nn::Initializer::KaimingNormal {
            fan_out_only: false,
            gain: (1.0 + LEAKY_RELU_SLOPE * LEAKY_RELU_SLOPE).sqrt(), // Adjust gain for LeakyReLU
        };

        // LSTM layers
        for i in 0..NUM_LSTM_LAYERS {
            let input_size = if i == 0 {
                NUM_FEATURES
            } else {
                self.hidden_size
            };

            lstm_layers.push(
                LstmConfig::new(input_size, self.hidden_size, true)
                    .with_initializer(initializer.clone())
                    .init(device),
            );

            // layer_norms.push(
            //     LayerNormConfig::new(self.hidden_size)
            //         .with_epsilon(1e-5)
            //         .init(device),
            // );
        }

        // Hidden layers with their own normalizations
        let hidden_layers: Vec<Linear<B>> = (0..3)
            .map(|_| {
                // hidden_norms.push(
                //     LayerNormConfig::new(self.hidden_size)
                //         .with_epsilon(1e-5)
                //         .init(device),
                // );

                LinearConfig::new(self.hidden_size, self.hidden_size)
                    .with_bias(true)
                    .with_initializer(initializer.clone())
                    .init(device)
            })
            .collect();

        // Output layer with larger initialization to help with range
        let output_layer = LinearConfig::new(self.hidden_size, NUM_FEATURES)
            .with_bias(true)
            .with_initializer(burn::nn::Initializer::KaimingNormal {
                fan_out_only: false,
                gain: 2.0, // Increased gain for output layer
            })
            .init(device);

        // Special normalization for output
        let output_norm = LayerNormConfig::new(NUM_FEATURES)
            .with_epsilon(1e-5)
            .init(device);

        // Initialize mean and logvar layers
        let mean_layer = LinearConfig::new(self.hidden_size, self.latent_dim)
            .with_bias(true)
            .with_initializer(initializer.clone())
            .init(device);

        let logvar_layer = LinearConfig::new(self.hidden_size, self.latent_dim)
            .with_bias(true)
            .with_initializer(initializer.clone())
            .init(device);

        let lstm_decoder = LstmConfig::new(self.latent_dim, self.hidden_size, true)
            .with_initializer(initializer.clone())
            .init(device);

        RnnModel {
            lstm_layers,
            lstm_decoder,
            // layer_norms,
            output_norm,
            dropout: DropoutConfig::new(DROPOUT_RATE).init(),
            hidden_layers,
            output_layer,
            activation: Gelu::new(),
            // hidden_norms,
            mean_layer,
            logvar_layer,
            hidden_size: self.hidden_size,
            latent_dim: self.latent_dim,
        }
    }
}

impl<B: Backend> RnnModel<B> {
    fn sample_z(&self, mean: Tensor<B, 2>, logvar: Tensor<B, 2>) -> Tensor<B, 2> {
        let eps = mean.random_like(Distribution::Normal(0.0, 1.0));
        mean.add(eps) * (logvar.mul_scalar(0.5)).exp()
    }

    // fn kl_divergence(&self, mean: Tensor<B, 2>, logvar: Tensor<B, 2>) -> Tensor<B, 1> {
    //     // Calculate KL divergence between the learned distribution and a standard Gaussian
    //     (mean.powf_scalar(2.0) + (logvar.clone().neg()).exp() - logvar - 1.0)
    //         .sum_dim(1)
    //         .mean() // TODO: very suspicious, not so sure
    //         .mul_scalar(0.5)
    // }

    fn kl_divergence(&self, mean: Tensor<B, 2>, logvar: Tensor<B, 2>) -> Tensor<B, 1> {
        // Ensure the tensors have the same shape
        // if mean.shape() != logvar.shape() {
        //     panic!("Shape mismatch: mean and logvar tensors must have the same shape");
        // }

        // Compute KL divergence
        let kl = (logvar.clone().exp() + mean.powi_scalar(2) - 1.0 - logvar)
            .sum_dim(1) // Sum along the second dimension (columns)
            .squeeze(1); // Remove the reduced dimension

        kl
    }

    // fn kl_divergence(&self, mean_t: Tensor<B, 2>, logvar_t: Tensor<B, 2>) -> Tensor<B, 1> {
    //     let mean: Vec<f32> = mean_t
    //         .to_data()
    //         .to_vec()
    //         .expect("Couldn't convert to vector");
    //     let logvar: Vec<f32> = logvar_t
    //         .to_data()
    //         .to_vec()
    //         .expect("Couldn't convert to vector");

    //     let mean_arr = ArrayView2::from_shape(mean_t.shape().dims::<2>(), &mean)
    //         .expect("Couldn't crate arrayview");
    //     let logvar_arr = ArrayView2::from_shape(logvar_t.shape().dims::<2>(), &logvar)
    //         .expect("Couldn't crate arrayview");

    //     // Calculate standard deviation from log variance
    //     let std_dev_arr = logvar_arr.mapv(|x| (x / 2.0).exp());

    //     // Compute KL divergence for each dimension
    //     let kl_divs = mean_arr
    //         .outer_iter()
    //         .zip(std_dev_arr.outer_iter())
    //         .map(|(mean_row, std_dev_row)| {
    //             let p = mean_row.to_owned();
    //             let q = std_dev_row.to_owned();
    //             p.kl_divergence(&q)
    //                 .expect("KL divergence calculation failed")
    //         })
    //         .collect::<Vec<f32>>();

    //     // Sum the KL divergences across dimensions?

    //     let kl_vec = kl_divs.to_vec();
    //     let kl_vec = kl_vec.as_slice();

    //     // Convert back to Burn Tensor
    //     Tensor::from_floats(kl_vec, &mean_t.device())
    // }

    pub fn forward(&self, input: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let [batch_size, seq_len, _features] = input.dims();

        let mut lstm_out = input;
        let mut lstm_state = None;

        // LSTM layers with careful normalization
        for (i, lstm) in self
            .lstm_layers
            .iter()
            // .zip(self.layer_norms.iter())
            // .zip(std::iter::once(None).chain(self.lstm_layers.iter().map(Some)))
            .enumerate()
        {
            let (out, state) = lstm.forward(lstm_out.clone(), lstm_state);
            // NOTE: norms not needed as data is already normalized acceptably
            // let normalized = layer_norm.forward(out);

            // Residual connection if not first layer
            // if i > 0 {
            //     lstm_out = normalized + lstm_out;
            // } else {
            //     lstm_out = normalized;
            // }

            lstm_out = out;

            // lstm_out = self.dropout.forward(lstm_out);
            lstm_state = Some(state);
        }

        // Process through hidden layers
        let mut hidden = lstm_out.reshape([batch_size * seq_len, self.lstm_layers[0].d_hidden]);

        for (i, hidden_layer) in self
            .hidden_layers
            .iter()
            // .zip(self.hidden_norms.iter())
            .enumerate()
        {
            let layer_out = hidden_layer.forward(hidden.clone());
            let activated = self.activation.forward(layer_out);
            // NOTE: norms not needed as data is already normalized acceptably
            // let normalized = hidden_norm.forward(activated);

            // Residual connection
            // hidden = normalized + hidden;

            hidden = activated;

            // Only apply dropout between layers, not after final layer
            // if i < self.hidden_layers.len() - 1 {
            //     hidden = self.dropout.forward(hidden);
            // }
        }

        // Calculate mean and log variance
        let mean = self.mean_layer.forward(hidden.clone());
        let logvar = self.logvar_layer.forward(hidden);

        // Sample from the latent space
        let z = self.sample_z(mean.clone(), logvar.clone());

        // Calculate KL divergence loss
        let kl_loss = self.kl_divergence(mean, logvar);

        // Reshape z to match the sequence length
        let z_reshaped = z.reshape([batch_size, seq_len, self.latent_dim]);

        // Use z as input to the decoder (replace lstm_out with z_reshaped)
        let (out, state) = self.lstm_decoder.forward(z_reshaped.clone(), lstm_state);

        // Output layer
        let output = self.output_layer.forward(out);
        // let normalized_output = self.output_norm.forward(output);

        (output.reshape([batch_size, seq_len, NUM_FEATURES]), kl_loss)
    }

    pub fn forward_step(&self, item: KeyframeBatch<B>) -> RnnOutput<B> {
        let normalizer = Normalizer::new(&item.inputs.device());

        // Normalize inputs and targets
        let normalized_inputs: Tensor<B, 3> = normalizer.normalize(item.inputs.clone());
        let normalized_targets: Tensor<B, 3> = normalizer.normalize(item.targets.clone());

        let (output, kl_loss) = self.forward(normalized_inputs);

        let reconstruction_loss = MseLoss::new().forward(
            output.clone(),
            normalized_targets.clone(),
            burn::nn::loss::Reduction::Mean,
        );

        // Calculate KL divergence and actual loss?
        // Combine losses with beta parameter to control KL term
        // Beta typically starts small and is annealed up during training
        let beta = 0.01; // You might want to make this configurable
        let total_loss = reconstruction_loss + kl_loss.mul_scalar(beta);

        // Denormalize for the actual predictions
        let denormalized_output = normalizer.denormalize(output);

        RnnOutput {
            loss: total_loss,
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

// VAE specific model:
// const LATENT_SIZE: usize = 32;
// const HIDDEN_SIZE: usize = 256;
// const DROPOUT_RATE: f64 = 0.1;

// #[derive(Module, Debug)]
// pub struct VaeLstm<B: Backend> {
//     // Encoder
//     encoder_lstm: Lstm<B>,
//     encoder_norm: LayerNorm<B>,
//     mean_layer: Linear<B>,
//     logvar_layer: Linear<B>,

//     // Decoder
//     decoder_lstm: Lstm<B>,
//     decoder_norm: LayerNorm<B>,
//     output_layer: Linear<B>,

//     // Shared
//     dropout: Dropout,
// }

// #[derive(Debug)]
// pub struct VaeOutput<B: Backend> {
//     pub reconstruction: Tensor<B, 3>,
//     pub targets: Tensor<B, 3>,
//     pub mean: Tensor<B, 2>,
//     pub logvar: Tensor<B, 2>,
//     pub loss: Tensor<B, 1>,
// }

// impl<B: Backend> VaeLstm<B> {
//     pub fn new(device: &B::Device) -> Self {
//         let initializer = burn::nn::Initializer::KaimingNormal {
//             fan_out_only: false,
//             gain: 1.0,
//         };

//         // Encoder - processes prompt sequence
//         let encoder_lstm = LstmConfig::new(NUM_FEATURES, HIDDEN_SIZE, true)
//             .with_initializer(initializer.clone())
//             .init(device);

//         let encoder_norm = LayerNormConfig::new(HIDDEN_SIZE)
//             .with_epsilon(1e-5)
//             .init(device);

//         let mean_layer = LinearConfig::new(HIDDEN_SIZE, LATENT_SIZE)
//             .with_initializer(initializer.clone())
//             .init(device);

//         let logvar_layer = LinearConfig::new(HIDDEN_SIZE, LATENT_SIZE)
//             .with_initializer(initializer.clone())
//             .init(device);

//         // Decoder - generates completion sequence
//         let decoder_lstm = LstmConfig::new(LATENT_SIZE + NUM_FEATURES, HIDDEN_SIZE, true) // +NUM_FEATURES for conditioning
//             .with_initializer(initializer.clone())
//             .init(device);

//         let decoder_norm = LayerNormConfig::new(HIDDEN_SIZE)
//             .with_epsilon(1e-5)
//             .init(device);

//         let output_layer = LinearConfig::new(HIDDEN_SIZE, NUM_FEATURES)
//             .with_initializer(initializer)
//             .init(device);

//         Self {
//             encoder_lstm,
//             encoder_norm,
//             mean_layer,
//             logvar_layer,
//             decoder_lstm,
//             decoder_norm,
//             output_layer,
//             dropout: DropoutConfig::new(DROPOUT_RATE).init(),
//         }
//     }

//     fn sample(&self, mean: Tensor<B, 2>, logvar: Tensor<B, 2>, temperature: f64) -> Tensor<B, 2> {
//         if temperature > 0.0 {
//             let std = (logvar * 0.5).exp();
//             let eps = Tensor::rand_like(&mean);
//             mean + (eps * std * temperature)
//         } else {
//             mean // Deterministic
//         }
//     }

//     fn kl_divergence(&self, mean: Tensor<B, 2>, logvar: Tensor<B, 2>) -> Tensor<B, 1> {
//         let kl = -0.5 * (1.0 + logvar - mean.pow(2.0) - logvar.exp());
//         kl.mean([1])
//     }

//     pub fn forward_step(&self, item: KeyframeBatch<B>, temperature: f64) -> VaeOutput<B> {
//         let normalizer = Normalizer::new(&item.inputs.device());

//         // Normalize inputs and targets
//         let normalized_inputs = normalizer.normalize(item.inputs.clone());
//         let normalized_targets = normalizer.normalize(item.targets.clone());

//         // Encode prompt sequence
//         let (encoder_out, _) = self.encoder_lstm.forward(normalized_inputs.clone(), None);
//         let encoded = self.encoder_norm.forward(encoder_out);

//         // Get latent representation from last timestep
//         let last_hidden = encoded.slice([.., item.input_lengths.clone() - 1, ..]);
//         let mean = self.mean_layer.forward(last_hidden.clone());
//         let logvar = self.logvar_layer.forward(last_hidden);

//         // Sample latent vector
//         let z = self.sample(mean.clone(), logvar.clone(), temperature);

//         // Prepare decoder input: combine latent vector with prompt features
//         let [batch_size, target_len, _] = normalized_targets.dims();
//         let z_expanded = z.unsqueeze(1).repeat([1, target_len, 1]);

//         // Initial prompt state for conditioning
//         let prompt_state = normalized_inputs
//             .slice([.., 0, ..])
//             .unsqueeze(1)
//             .repeat([1, target_len, 1]);
//         let decoder_input = Tensor::cat([z_expanded, prompt_state], 2);

//         // Generate completion sequence
//         let (decoder_out, _) = self.decoder_lstm.forward(decoder_input, None);
//         let decoded = self.decoder_norm.forward(decoder_out);
//         let output = self.output_layer.forward(decoded);

//         // Denormalize for final output
//         let denormalized_output = normalizer.denormalize(output);

//         // Calculate losses
//         let recon_loss = MseLoss::new().forward(
//             denormalized_output.clone(),
//             item.targets.clone(),
//             burn::nn::loss::Reduction::Mean,
//         );

//         let kl_loss = self.kl_divergence(mean.clone(), logvar.clone());
//         let beta = 0.1;
//         let total_loss = recon_loss + (beta * kl_loss);

//         VaeOutput {
//             reconstruction: denormalized_output,
//             targets: item.targets,
//             mean,
//             logvar,
//             loss: total_loss,
//         }
//     }

//     pub fn generate(
//         &self,
//         prompt: Tensor<B, 3>,
//         target_len: usize,
//         temperature: f64,
//     ) -> Tensor<B, 3> {
//         let normalizer = Normalizer::new(&prompt.device());
//         let normalized_prompt = normalizer.normalize(prompt.clone());

//         // Encode prompt
//         let (encoder_out, _) = self.encoder_lstm.forward(normalized_prompt.clone(), None);
//         let encoded = self.encoder_norm.forward(encoder_out);

//         // Get latent vector
//         let last_hidden = encoded.slice([.., encoded.dims()[1] - 1, ..]);
//         let mean = self.mean_layer.forward(last_hidden);
//         let logvar = self.logvar_layer.forward(last_hidden);
//         let z = self.sample(mean, logvar, temperature);

//         // Prepare decoder input
//         let [batch_size, _, _] = prompt.dims();
//         let z_expanded = z.unsqueeze(1).repeat([1, target_len, 1]);

//         // Use initial prompt state for conditioning
//         let prompt_state = normalized_prompt
//             .slice([.., 0, ..])
//             .unsqueeze(1)
//             .repeat([1, target_len, 1]);
//         let decoder_input = Tensor::cat([z_expanded, prompt_state], 2);

//         // Generate sequence
//         let (decoder_out, _) = self.decoder_lstm.forward(decoder_input, None);
//         let decoded = self.decoder_norm.forward(decoder_out);
//         let output = self.output_layer.forward(decoded);

//         // Denormalize output
//         normalizer.denormalize(output)
//     }
// }

// impl<B: AutodiffBackend> TrainStep<KeyframeBatch<B>, VaeOutput<B>> for VaeLstm<B> {
//     fn step(&self, item: KeyframeBatch<B>) -> TrainOutput<VaeOutput<B>> {
//         let output = self.forward_step(item, 1.0); // Use temperature 1.0 during training
//         TrainOutput::new(self, output.loss.backward(), output)
//     }
// }

// impl<B: Backend> ValidStep<KeyframeBatch<B>, VaeOutput<B>> for VaeLstm<B> {
//     fn step(&self, item: KeyframeBatch<B>) -> VaeOutput<B> {
//         self.forward_step(item, 0.0) // Use temperature 0.0 for validation
//     }
// }
