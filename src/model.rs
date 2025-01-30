use std::fs::OpenOptions;
use std::io::Write;

use crate::dataset::{KeyframeBatch, Normalizer, NUM_FEATURES};
use burn::{
    config::Config,
    module::Module,
    nn::{
        attention::{MhaInput, MhaOutput, MultiHeadAttention, MultiHeadAttentionConfig},
        conv::{Conv1d, Conv1dConfig},
        loss::{BinaryCrossEntropyLoss, BinaryCrossEntropyLossConfig, MseLoss, Reduction},
        Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, LeakyRelu, LeakyReluConfig,
        Linear, LinearConfig, Lstm, LstmConfig, Relu, Sigmoid,
    },
    tensor::{
        backend::{AutodiffBackend, Backend},
        Distribution, Shape, Tensor, TensorData,
    },
    train::{
        metric::{AccuracyInput, Adaptor, LossInput},
        TrainOutput, TrainStep, ValidStep,
    },
};

// #[derive(Module, Debug)]
// pub struct Discriminator<B: Backend> {
//     conv_layers: Vec<Conv1d<B>>,
//     dense_layers: Vec<Linear<B>>,
//     output_layer: Linear<B>,
//     activation: Gelu,
// }

#[derive(Config)]
pub struct DiscriminatorConfig {
    // #[config(default = 64)]
    // #[config(default = 32)]
    #[config(default = 256)]
    pub hidden_size: usize,
    // #[config(default = 4)]
    // #[config(default = 3)]
    #[config(default = 1)]
    pub num_lstm_layers: usize,
    // #[config(default = 3)]
    #[config(default = 2)]
    pub num_dense_layers: usize,
    #[config(default = 256)]
    // #[config(default = 128)]
    pub dense_size: usize,
}

#[derive(Module, Debug)]
pub struct Discriminator<B: Backend> {
    lstm_layers: Vec<Lstm<B>>,
    dense_layers: Vec<Linear<B>>,
    output_layer: Linear<B>,
    activation: Gelu,
    sigmoid: Sigmoid,
}

impl DiscriminatorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Discriminator<B> {
        let initializer = burn::nn::Initializer::KaimingUniform {
            fan_out_only: false,
            // gain: (2.0f64).sqrt(), // For LeakyReLU
            gain: (1.0 + LEAKY_RELU_SLOPE * LEAKY_RELU_SLOPE).sqrt(),
        };

        // LSTM layers
        let mut lstm_layers = Vec::with_capacity(self.num_lstm_layers); // Reusing `num_lstm_layers` for LSTM layers
        for i in 0..self.num_lstm_layers {
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
        }

        // Dense layers with consistent size
        let mut dense_layers = Vec::with_capacity(self.num_dense_layers);
        for i in 0..self.num_dense_layers {
            let in_features = if i == 0 {
                self.hidden_size
            } else {
                self.dense_size
            };

            dense_layers.push(
                LinearConfig::new(in_features, self.dense_size)
                    .with_bias(true)
                    .with_initializer(initializer.clone())
                    .init(device),
            );
        }

        // Output layer for binary classification
        let output_layer = LinearConfig::new(self.dense_size, 1)
            .with_bias(true)
            .with_initializer(initializer)
            .init(device);

        let sigmoid = Sigmoid::new();

        Discriminator {
            lstm_layers,
            dense_layers,
            output_layer,
            // activation: LeakyReluConfig::new()
            //     .with_negative_slope(LEAKY_RELU_SLOPE)
            //     .init(),
            activation: Gelu::new(),
            sigmoid,
        }
    }
}

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

    // MultiHeadAttention
    encoder_attention: MultiHeadAttention<B>,
    query_proj: Linear<B>,
    key_proj: Linear<B>,
    value_proj: Linear<B>,

    // GAN capability
    discriminator: Discriminator<B>,

    // for convenience
    pub hidden_size: usize,
    pub latent_dim: usize,
}

#[derive(Config)]
pub struct RnnModelConfig {
    // #[config(default = 1024)]
    // #[config(default = 512)]
    #[config(default = 256)]
    // #[config(default = 64)]
    pub hidden_size: usize,
    // #[config(default = 512)]
    // #[config(default = 256)]
    // #[config(default = 128)]
    #[config(default = 32)] // Latent space dimension
    pub latent_dim: usize,
    #[config(default = 8)]
    pub n_heads: usize,
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
        // let mut layer_norms: Vec<LayerNorm<B>> = Vec::with_capacity(NUM_LSTM_LAYERS);
        // let mut hidden_norms: Vec<LayerNorm<B>> = Vec::with_capacity(3); // For hidden layers

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

        let encoder_attention =
            MultiHeadAttentionConfig::new(self.hidden_size, self.n_heads).init(device);

        // Add projection layers for Q, K, V
        let query_proj = LinearConfig::new(self.hidden_size, self.hidden_size)
            .with_bias(true)
            .with_initializer(initializer.clone())
            .init(device);

        let key_proj = LinearConfig::new(self.hidden_size, self.hidden_size)
            .with_bias(true)
            .with_initializer(initializer.clone())
            .init(device);

        let value_proj = LinearConfig::new(self.hidden_size, self.hidden_size)
            .with_bias(true)
            .with_initializer(initializer.clone())
            .init(device);

        // Initialize discriminator
        let discriminator_config = DiscriminatorConfig::new();
        let discriminator = discriminator_config.init(device);

        RnnModel {
            // lstm
            lstm_layers,
            lstm_decoder,
            // staandard
            output_norm,
            dropout: DropoutConfig::new(DROPOUT_RATE).init(),
            hidden_layers,
            output_layer,
            activation: Gelu::new(),
            // vae
            mean_layer,
            logvar_layer,
            // attn
            encoder_attention,
            query_proj,
            key_proj,
            value_proj,
            // gan
            discriminator,
            // config
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

    fn kl_divergence(&self, mean: Tensor<B, 2>, logvar: Tensor<B, 2>) -> Tensor<B, 1> {
        // Compute KL divergence
        let kl = (logvar.clone().exp() + mean.powi_scalar(2) - 1.0 - logvar)
            .sum_dim(1) // Sum along the second dimension (columns)
            .squeeze(1); // Remove the reduced dimension

        kl
    }

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

        // After LSTM processing, lstm_out contains your sequence information
        // Project lstm_out to get Q, K, V
        let query = self.query_proj.forward(lstm_out.clone());
        let key = self.key_proj.forward(lstm_out.clone());
        let value = self.value_proj.forward(lstm_out.clone());

        // Create attention input
        let attn_input = MhaInput::new(query, key, value);
        let attn_out: MhaOutput<B> = self.encoder_attention.forward(attn_input);

        // Combine attention output with LSTM output (residual connection)
        let combined = attn_out.context + lstm_out;

        // Continue with VAE encoding using the attention-enhanced representation
        let mut hidden = combined.reshape([batch_size * seq_len, self.hidden_size]);

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

    // pub fn forward_step(&self, item: KeyframeBatch<B>) -> RnnOutput<B> {
    //     let normalizer = Normalizer::new(&item.inputs.device());

    //     // Normalize inputs and targets
    //     let normalized_inputs: Tensor<B, 3> = normalizer.normalize(item.inputs.clone());
    //     let normalized_targets: Tensor<B, 3> = normalizer.normalize(item.targets.clone());

    //     let (output, kl_loss) = self.forward(normalized_inputs);

    //     let reconstruction_loss = MseLoss::new().forward(
    //         output.clone(),
    //         normalized_targets.clone(),
    //         burn::nn::loss::Reduction::Mean,
    //     );

    //     // *** just VAE ***
    //     let beta = 0.01; // loss stabilizes lower
    //                      // let beta = 1.0; // loss stabilizes higher
    //     let total_loss = reconstruction_loss + kl_loss.mul_scalar(beta);

    //     // Denormalize for the actual predictions
    //     let denormalized_output = normalizer.denormalize(output);

    //     RnnOutput {
    //         loss: total_loss,
    //         output: denormalized_output,
    //         targets: item.targets,
    //     }
    // }

    // fn discriminator_loss(
    //     &self,
    //     real_sequences: Tensor<B, 3>,
    //     generated_sequences: Tensor<B, 3>,
    // ) -> Tensor<B, 1> {
    //     // // Real sequences should be classified as 1
    //     // // Generated sequences should be classified as 0
    //     let real_labels = Tensor::<B, 3>::from_data([[1]], &real_sequences.device())
    //         .ones_like()
    //         .int();
    //     let fake_labels = Tensor::<B, 3>::from_data([[0]], &real_sequences.device())
    //         .zeros_like()
    //         .int();

    //     let real_scores = self.discriminator.forward(real_sequences);
    //     let fake_scores = self.discriminator.forward(generated_sequences);

    //     // Binary cross entropy loss
    //     let real_loss = BinaryCrossEntropyLossConfig::new().init(&real_labels.device());
    //     let real_loss = real_loss.forward(real_scores, real_labels);

    //     let fake_loss = BinaryCrossEntropyLossConfig::new().init(&fake_labels.device());
    //     let fake_loss = fake_loss.forward(fake_scores, fake_labels);

    //     real_loss + fake_loss
    // }

    fn discriminator_loss(
        &self,
        real_sequences: Tensor<B, 3>,
        generated_sequences: Tensor<B, 3>,
    ) -> Tensor<B, 1> {
        let device = &real_sequences.device();

        // Real sequences should be classified as 1, generated sequences as 0
        let batch_size = real_sequences.dims()[0];
        let seq_len = real_sequences.dims()[1];

        // Create labels with shape [batch_size, seq_len, 1]
        let real_labels = Tensor::<B, 3>::ones([batch_size, seq_len, 1], device).int();
        let fake_labels = Tensor::<B, 3>::zeros([batch_size, seq_len, 1], device).int();
        // let real_labels = Tensor::<B, 2>::ones([batch_size, 1], device).int(); // Shape: [batch_size, 1]
        // let fake_labels = Tensor::<B, 2>::zeros([batch_size, 1], device).int(); // Shape: [batch_size, 1]

        // Get discriminator scores
        let real_scores = self.discriminator.forward(real_sequences);
        let fake_scores = self.discriminator.forward(generated_sequences);

        // Binary cross entropy loss
        let bce_loss = BinaryCrossEntropyLossConfig::new().init(device);

        // Calculate loss for real and fake sequences
        let real_loss = bce_loss.forward(real_scores.clone(), real_labels);
        let fake_loss = bce_loss.forward(fake_scores.clone(), fake_labels);

        // Total loss
        real_loss + fake_loss
    }

    pub fn forward_step(&self, item: KeyframeBatch<B>) -> RnnOutput<B> {
        let normalizer = Normalizer::new(&item.inputs.device());

        // Normalize inputs and targets
        let normalized_inputs = normalizer.normalize(item.inputs.clone());
        let normalized_targets = normalizer.normalize(item.targets.clone());

        // Generator forward pass
        let (output, kl_loss) = self.forward(normalized_inputs);

        // Reconstruction loss (same as before)
        let reconstruction_loss =
            MseLoss::new().forward(output.clone(), normalized_targets.clone(), Reduction::Mean);

        // Discriminator loss
        // TODO: so is the officially d_loss or g_loss?
        let gen_loss = self.discriminator_loss(normalized_targets, output.clone());

        // Combined loss with weights
        // let beta = 0.01; // VAE KL weight
        // let lambda = 0.1; // GAN loss weight
        // let total_loss =
        //     reconstruction_loss + kl_loss.mul_scalar(beta) + gen_loss.mul_scalar(lambda);

        // Denormalize output
        let denormalized_output = normalizer.denormalize(output);

        RnnOutput {
            // loss: total_loss,
            loss: gen_loss, // smooth decrease in loss with right lr
            output: denormalized_output,
            targets: item.targets,
        }
    }
}

// // Discriminator implementation
// impl<B: Backend> Discriminator<B> {
//     fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
//         let [batch_size, seq_len, features] = input.dims();

//         // Reshape for 1D convolutions
//         // let mut x = input.transpose(1, 2); // [batch, features, seq_len]
//         let mut x = input.swap_dims(1, 2);
//         // let mut x = x.transpose();

//         // Conv layers
//         for lstm in &self.lstm_layers {
//             let (lstm_value, lstm_state) = lstm.forward(x, None);
//             x = lstm_value;
//             x = self.activation.forward(x);
//         }

//         // Global average pooling
//         let x = x.mean_dim(2); // [batch, features]

//         // Dense layers
//         let flattened_size = x.dims()[1]; // Get feature dimension size after conv layers
//         let mut x = x.reshape([batch_size, flattened_size]);
//         for dense in &self.dense_layers {
//             x = dense.forward(x);
//             x = self.activation.forward(x);
//         }

//         // Output layer with sigmoid for binary classification
//         let x = self.output_layer.forward(x);
//         self.sigmoid.forward(x)
//     }
// }

impl<B: Backend> Discriminator<B> {
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, features] = input.dims();

        // Transpose input for LSTM: [batch_size, seq_len, features] -> [batch_size, seq_len, features]
        // No need to transpose for LSTM, as it expects [batch_size, seq_len, features]
        let mut x = input;

        // LSTM layers
        for lstm in &self.lstm_layers {
            let (lstm_output, _) = lstm.forward(x, None); // lstm_output shape: [batch_size, seq_len, hidden_size]
            x = lstm_output;
            x = self.activation.forward(x); // Apply activation
        }

        // // Global average pooling along the sequence length dimension
        // let mut x = x.mean_dim(2);

        // // Flatten the tensor
        // let mut x = x.reshape([batch_size, seq_len]); // Shape: [batch_size, seq_len]

        // Dense layers
        for dense in &self.dense_layers {
            x = dense.forward(x); // Shape remains [batch_size, hidden_size] or [batch_size, dense_size]
            x = self.activation.forward(x); // Apply activation
        }

        // Output layer with sigmoid for binary classification
        let x = self.output_layer.forward(x); // Shape: [batch_size, 1]
        self.sigmoid.forward(x) // Apply sigmoid
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
