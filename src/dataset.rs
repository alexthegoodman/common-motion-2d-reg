use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::backend::Backend,
    tensor::Tensor,
};

pub const NUM_FEATURES: usize = 6;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KeyframeItem {
    pub polygon_index: f32,
    pub time: f32,
    pub width: f32,
    pub height: f32,
    pub x: f32,
    pub y: f32,
}

pub struct MotionDataset {
    sequences: Vec<(Vec<KeyframeItem>, Vec<KeyframeItem>)>, // (inputs, targets) pairs
}

impl Dataset<(Vec<KeyframeItem>, Vec<KeyframeItem>)> for MotionDataset {
    fn get(&self, index: usize) -> Option<(Vec<KeyframeItem>, Vec<KeyframeItem>)> {
        self.sequences.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.sequences.len()
    }
}

impl MotionDataset {
    pub fn from_string(data: &str) -> Self {
        let mut sequences = Vec::new();
        let sequences_raw = data.split("---").filter(|s| !s.trim().is_empty());

        for sequence in sequences_raw {
            let parts: Vec<&str> = sequence.split("!!!").collect();
            if parts.len() != 2 {
                continue;
            }

            let parse_items = |s: &str| -> Vec<KeyframeItem> {
                s.lines()
                    .filter(|line| !line.trim().is_empty())
                    .filter_map(|line| {
                        let values: Vec<f32> = line
                            .split(',')
                            .filter_map(|v| v.trim().parse().ok())
                            .collect();

                        if values.len() == 6 {
                            Some(KeyframeItem {
                                polygon_index: values[0],
                                time: values[1],
                                width: values[2],
                                height: values[3],
                                x: values[4],
                                y: values[5],
                            })
                        } else {
                            None
                        }
                    })
                    .collect()
            };

            let inputs = parse_items(parts[0]);
            let targets = parse_items(parts[1]);

            if !inputs.is_empty() && !targets.is_empty() {
                sequences.push((inputs, targets));
            }
        }

        // print 5 sequences
        // for i in 0..5 {
        //     println!("Sequence {}", i);
        //     let (inputs, targets) = &sequences[i];
        //     for (input, target) in inputs.iter().zip(targets.iter()) {
        //         println!("{:?} -> {:?}", input, target);
        //     }
        // }

        Self { sequences }
    }
}

#[derive(Clone, Debug)]
pub struct KeyframeBatch<B: Backend> {
    pub inputs: Tensor<B, 3>,         // [batch_size, seq_len, features]
    pub targets: Tensor<B, 3>,        // [batch_size, seq_len, features]
    pub input_lengths: Tensor<B, 1>,  // [batch_size]
    pub target_lengths: Tensor<B, 1>, // [batch_size]
}

#[derive(Clone, Debug)]
pub struct KeyframeBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> KeyframeBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    /// actually pads multiple sequences
    fn pad_sequence(
        &self,
        items: &[Vec<KeyframeItem>],
        max_len: usize,
    ) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let batch_size = items.len();

        // Calculate the actual number of elements needed
        // let total_elements = items.iter().map(|seq| seq.len() * NUM_FEATURES).sum();

        // Create padded tensor with the correct size
        // let mut data = vec![0.0; total_elements];
        let mut data = vec![0.0; batch_size * max_len * NUM_FEATURES];
        let mut lengths = vec![0; batch_size];

        let mut data_idx = 0;
        for (batch_idx, sequence) in items.iter().enumerate() {
            if data_idx + NUM_FEATURES + 1 < data.len() {
                lengths[batch_idx] = sequence.len();

                for item in sequence {
                    data[data_idx] = item.polygon_index;
                    data[data_idx + 1] = item.time;
                    data[data_idx + 2] = item.width;
                    data[data_idx + 3] = item.height;
                    data[data_idx + 4] = item.x;
                    data[data_idx + 5] = item.y;
                    data_idx += NUM_FEATURES;
                }
            }
            // Pad with zeros for sequences shorter than max_len
            for _ in 0..(max_len - sequence.len()) {
                for _ in 0..NUM_FEATURES {
                    // check has data_idx first

                    data[data_idx] = 0.0;

                    data_idx += 1;
                }
            }
        }

        let tensor = Tensor::<B, 1>::from_floats(data.as_slice(), &self.device).reshape([
            batch_size,
            max_len,
            NUM_FEATURES,
        ]);
        let lengths = Tensor::from_ints(lengths.as_slice(), &self.device).float();

        (tensor, lengths)
    }
}

impl<B: Backend> Batcher<(Vec<KeyframeItem>, Vec<KeyframeItem>), KeyframeBatch<B>>
    for KeyframeBatcher<B>
{
    fn batch(&self, items: Vec<(Vec<KeyframeItem>, Vec<KeyframeItem>)>) -> KeyframeBatch<B> {
        // Unzip the inputs and targets for the entire batch
        let (input_sequences, target_sequences): (Vec<_>, Vec<_>) = items.into_iter().unzip();

        let max_input_len = input_sequences
            .iter()
            .map(|seq| seq.len())
            .max()
            .unwrap_or(0);

        let max_target_len = target_sequences
            .iter()
            .map(|seq| seq.len())
            .max()
            .unwrap_or(0);

        let max_len = std::cmp::max(max_input_len, max_target_len);

        let (inputs, input_lengths) = self.pad_sequence(&input_sequences, max_len);
        let (targets, target_lengths) = self.pad_sequence(&target_sequences, max_len);

        KeyframeBatch {
            inputs,
            targets,
            input_lengths,
            target_lengths,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Normalizer<B: Backend> {
    pub means: Tensor<B, 1>,
    pub stds: Tensor<B, 1>,
}

impl<B: Backend> Normalizer<B> {
    pub fn new(device: &B::Device) -> Self {
        // Based on your data format: [polygon_index, time, width, height, x, y]
        // Column 0: Mean = 1.265, Std = 1.171
        // Column 1: Mean = 9.286, Std = 7.406
        // Column 2: Mean = 192.398, Std = 104.484
        // Column 3: Mean = 153.433, Std = 71.826
        // Column 4: Mean = 286.305, Std = 205.695
        // Column 5: Mean = 199.321, Std = 136.268
        let means = Tensor::from_floats([1.265, 9.286, 192.398, 153.433, 286.305, 199.321], device);
        let stds = Tensor::from_floats([1.171, 7.406, 104.484, 71.826, 205.695, 136.268], device);

        Self { means, stds }
    }

    pub fn normalize(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = input.dims();

        // Reshape and repeat means and stds to match input dimensions
        let means = self
            .means
            .clone()
            .reshape([1, 1, NUM_FEATURES])
            .repeat(&[batch_size, seq_len, 1]);
        let stds = self
            .stds
            .clone()
            .reshape([1, 1, NUM_FEATURES])
            .repeat(&[batch_size, seq_len, 1]);

        (input - means) / stds
    }

    pub fn denormalize(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = input.dims();

        let means = self
            .means
            .clone()
            .reshape([1, 1, NUM_FEATURES])
            .repeat(&[batch_size, seq_len, 1]);
        let stds = self
            .stds
            .clone()
            .reshape([1, 1, NUM_FEATURES])
            .repeat(&[batch_size, seq_len, 1]);

        (input * stds) + means
    }
}
