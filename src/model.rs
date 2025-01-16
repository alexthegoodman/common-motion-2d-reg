use std::fs::OpenOptions;
use std::io::Write;

use crate::dataset::{KeyframeBatch, Normalizer, NUM_FEATURES};
use burn::{
    config::Config,
    module::Module,
    nn::{
        attention::{MhaInput, MhaOutput, MultiHeadAttention, MultiHeadAttentionConfig},
        loss::MseLoss,
        Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, LeakyRelu, LeakyReluConfig,
        Linear, LinearConfig, Lstm, LstmConfig, Relu,
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

    // MultiHeadAttention
    encoder_attention: MultiHeadAttention<B>,
    query_proj: Linear<B>,
    key_proj: Linear<B>,
    value_proj: Linear<B>,

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

        // // do I need embeddings of some kind for q, k, and v? where do they come from?
        // let attn_input = MhaInput::new(query, key, value);
        // let attn_out: MhaOutput<B> = self.encoder_attention.forward(attn_input); // has context, weights

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

        // Process through hidden layers
        // let mut hidden = lstm_out.reshape([batch_size * seq_len, self.lstm_layers[0].d_hidden]);

        // TODO: experiement with 0 or 1 hidden layers as there are now plenty of Linear layers in use
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

        // *** just VAE ***
        let beta = 0.01; // loss stabilizes lower
                         // let beta = 1.0; // loss stabilizes higher
        let total_loss = reconstruction_loss + kl_loss.mul_scalar(beta);

        // Denormalize for the actual predictions
        let denormalized_output = normalizer.denormalize(output);

        RnnOutput {
            loss: total_loss,
            output: denormalized_output,
            targets: item.targets,
        }

        // *** with crisscross loss ***
        // // Denormalize outputs for crossing detection
        // let denormalized_output = normalizer.denormalize(output.clone());
        // let crossing_loss = self.calculate_crossing_loss(&denormalized_output);

        // // Combine losses with weights
        // let beta = 0.01; // KL loss weight
        // let gamma = 0.1; // Crossing loss weight
        // let total_loss =
        //     reconstruction_loss + kl_loss.mul_scalar(beta) + crossing_loss.mul_scalar(gamma);

        // RnnOutput {
        //     loss: total_loss,
        //     output: denormalized_output,
        //     targets: item.targets,
        // }

        // *** with endpoint loss ***
        // let denormalized_output = normalizer.denormalize(output.clone());
        // let endpoint_loss = self.calculate_endpoint_loss(&denormalized_output);
        // let beta = 0.01; // KL loss weight
        // let delta = 0.2; // Endpoint loss weight - adjust as needed

        // let total_loss =
        //     reconstruction_loss + kl_loss.mul_scalar(beta) + endpoint_loss.mul_scalar(delta);

        // RnnOutput {
        //     loss: total_loss,
        //     output: denormalized_output,
        //     targets: item.targets,
        // }
    }

    // fn segments_intersect(&self, s1: &Segment, s2: &Segment) -> bool {
    //     // Don't check segments from same polygon at adjacent timestamps
    //     if s1.polygon_id == s2.polygon_id && (s1.time2 == s2.time1 || s1.time1 == s2.time2) {
    //         return false;
    //     }

    //     // Line segment intersection math
    //     let dx1 = s1.x2 - s1.x1;
    //     let dy1 = s1.y2 - s1.y1;
    //     let dx2 = s2.x2 - s2.x1;
    //     let dy2 = s2.y2 - s2.y1;

    //     let determinant = dx1 * dy2 - dy1 * dx2;
    //     if determinant.abs() < 1e-8 {
    //         // Parallel lines
    //         return false;
    //     }

    //     let dx3 = s1.x1 - s2.x1;
    //     let dy3 = s1.y1 - s2.y1;

    //     let t = (dx3 * dy2 - dy3 * dx2) / determinant;
    //     let u = (dx1 * dy3 - dy1 * dx3) / determinant;

    //     // Check if intersection point lies within both line segments
    //     t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0
    // }

    // fn calculate_crossing_loss(&self, output: &Tensor<B, 3>) -> Tensor<B, 1> {
    //     let device = output.device();
    //     let [batch_size, seq_len, _features] = output.dims();

    //     // Convert tensor to flat vec for easier processing
    //     let output_data: Vec<f32> = output
    //         .to_data()
    //         .to_vec()
    //         .expect("Couldn't convert output to vec");

    //     // Will store crossing loss for each batch
    //     let mut crossing_losses = Vec::with_capacity(batch_size);

    //     // Process each batch
    //     for b in 0..batch_size {
    //         let mut segments = Vec::new();
    //         let mut current_polygon = -1;
    //         let mut polygon_segments = Vec::new();

    //         // Create line segments from consecutive points
    //         for i in 0..seq_len - 1 {
    //             // Calculate indices for accessing the flat vector
    //             let base_idx = (b * seq_len * 6 + i * 6) as usize;
    //             let next_base_idx = (b * seq_len * 6 + (i + 1) * 6) as usize;

    //             let polygon_id = output_data[base_idx] as i32;
    //             let time = output_data[base_idx + 1];
    //             let x1 = output_data[base_idx + 4];
    //             let y1 = output_data[base_idx + 5];
    //             let x2 = output_data[next_base_idx + 4];
    //             let y2 = output_data[next_base_idx + 5];
    //             let time2 = output_data[next_base_idx + 1];

    //             // If same polygon, add segment
    //             if polygon_id == current_polygon {
    //                 polygon_segments.push(Segment::new(x1, y1, x2, y2, polygon_id, time, time2));
    //             } else {
    //                 // New polygon started, add accumulated segments
    //                 if !polygon_segments.is_empty() {
    //                     segments.extend(polygon_segments.drain(..));
    //                 }
    //                 current_polygon = polygon_id;
    //                 polygon_segments = vec![Segment::new(x1, y1, x2, y2, polygon_id, time, time2)];
    //             }
    //         }
    //         // Add remaining segments from last polygon
    //         segments.extend(polygon_segments);

    //         // Count intersections
    //         let mut intersections = 0;
    //         for i in 0..segments.len() {
    //             for j in i + 1..segments.len() {
    //                 if self.segments_intersect(&segments[i], &segments[j]) {
    //                     intersections += 1;
    //                 }
    //             }
    //         }

    //         // Convert to tensor loss
    //         // add for each row in the sequence to apply loss to whole sequence and match shape of other loss values
    //         for i in 0..seq_len {
    //             crossing_losses.push(intersections as f32);
    //         }
    //     }

    //     // Create tensor from crossing losses
    //     // I repeat the crossing_loss over each row in the sequence to match the shape size
    //     let data = TensorData::new(crossing_losses, Shape::new([batch_size * seq_len]));
    //     Tensor::<B, 1>::from_data(data, &output.device())
    // }

    // /// Calculate loss for whether start or end keyframe is off the canvas as it should be
    // fn calculate_endpoint_loss(&self, output: &Tensor<B, 3>) -> Tensor<B, 1> {
    //     let device = output.device();
    //     let [batch_size, seq_len, _features] = output.dims();
    //     let output_data: Vec<f32> = output
    //         .to_data()
    //         .to_vec()
    //         .expect("Couldn't convert output to vec");

    //     let mut endpoint_losses = Vec::with_capacity(batch_size);

    //     // Process each batch
    //     for b in 0..batch_size {
    //         let mut current_polygon = -1;
    //         let mut polygon_start_idx = 0;
    //         let mut batch_loss = 0.0f32;

    //         // Examine each point to find polygon boundaries
    //         for i in 0..seq_len {
    //             let base_idx = (b * seq_len * 6 + i * 6) as usize;
    //             let polygon_id = output_data[base_idx] as i32;

    //             // If we've found a new polygon or reached the end
    //             if polygon_id != current_polygon || i == seq_len - 1 {
    //                 if current_polygon != -1 {
    //                     // Check end point of previous polygon
    //                     let end_idx = base_idx - 6;
    //                     let end_x = output_data[end_idx + 4];
    //                     let end_y = output_data[end_idx + 5];

    //                     // Add loss if end point is inside canvas
    //                     if end_x >= 0.0 && end_x <= 800.0 && end_y >= 0.0 && end_y <= 450.0 {
    //                         batch_loss += 1.0;
    //                     }
    //                 }

    //                 if i < seq_len - 1 {
    //                     // Check start point of new polygon
    //                     let start_x = output_data[base_idx + 4];
    //                     let start_y = output_data[base_idx + 5];

    //                     // Add loss if start point is inside canvas
    //                     if start_x >= 0.0 && start_x <= 800.0 && start_y >= 0.0 && start_y <= 450.0
    //                     {
    //                         batch_loss += 1.0;
    //                     }

    //                     current_polygon = polygon_id;
    //                     polygon_start_idx = i;
    //                 }
    //             }
    //         }

    //         for i in 0..seq_len {
    //             endpoint_losses.push(batch_loss);
    //         }
    //     }

    //     // Create tensor from endpoint losses
    //     let data = TensorData::new(endpoint_losses, Shape::new([batch_size * seq_len]));
    //     Tensor::<B, 1>::from_data(data, &output.device())
    // }
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

// Helper struct to represent a line segment
struct Segment {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    polygon_id: i32,
    time1: f32,
    time2: f32,
}

impl Segment {
    fn new(x1: f32, y1: f32, x2: f32, y2: f32, polygon_id: i32, time1: f32, time2: f32) -> Self {
        Self {
            x1,
            y1,
            x2,
            y2,
            polygon_id,
            time1,
            time2,
        }
    }
}
