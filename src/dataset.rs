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

    fn pad_sequence(
        &self,
        items: &[Vec<KeyframeItem>],
        max_len: usize,
    ) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let batch_size = items.len();

        // Create padded tensor
        let mut data = vec![0.0; batch_size * max_len * NUM_FEATURES];
        let mut lengths = vec![0; batch_size];

        for (batch_idx, sequence) in items.iter().enumerate() {
            lengths[batch_idx] = sequence.len();

            for (seq_idx, item) in sequence.iter().enumerate() {
                let base_idx = (batch_idx * max_len + seq_idx) * NUM_FEATURES;
                data[base_idx] = item.polygon_index;
                data[base_idx + 1] = item.time;
                data[base_idx + 2] = item.width;
                data[base_idx + 3] = item.height;
                data[base_idx + 4] = item.x;
                data[base_idx + 5] = item.y;
            }
        }

        let tensor = Tensor::<B, 3>::from_floats(data.as_slice(), &self.device).reshape([
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

        let (inputs, input_lengths) = self.pad_sequence(&input_sequences, max_input_len);
        let (targets, target_lengths) = self.pad_sequence(&target_sequences, max_target_len);

        KeyframeBatch {
            inputs,
            targets,
            input_lengths,
            target_lengths,
        }
    }
}
