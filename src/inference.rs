use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    module::Module,
    record::{BinBytesRecorder, FullPrecisionSettings, NoStdTrainingRecorder, Recorder},
    tensor::backend::Backend,
};
// use rgb::RGB8;
// use textplots::{Chart, ColorPlot, Shape};

use crate::{
    dataset::{KeyframeBatcher, KeyframeItem, MotionDataset, Normalizer},
    model::{RnnModel, RnnModelConfig, RnnModelRecord},
};

pub struct CommonMotionInference<B: Backend> {
    pub model: RnnModel<B>,
    pub batcher: KeyframeBatcher<B>,
}

impl<B: Backend> CommonMotionInference<B> {
    pub fn new(device: B::Device) -> CommonMotionInference<B> {
        // Embed the model file directly in the binary
        const MODEL_BYTES: &[u8] = include_bytes!("D:/tmp/common-motion-2d-reg/model.bin");

        // let record: RnnModelRecord<B> = NoStdTrainingRecorder::new()
        //     .load(format!("{artifact_dir}/model").into(), &device)
        //     .expect("Trained model should exist; run train first");
        let record: RnnModelRecord<B> = BinBytesRecorder::<FullPrecisionSettings>::default()
            .load(MODEL_BYTES.to_vec(), &device)
            // .load(format!("{artifact_dir}/model").into(), &device)
            .expect("Trained model should exist; run train first");

        let model = RnnModelConfig::new().init(&device).load_record(record);

        let batcher = KeyframeBatcher::new(device);

        Self { model, batcher }
    }

    pub fn infer(&self, user_prompt: String) {
        // Use a sample of 1000 items from the test split
        // let dataset = MotionDataset::test();
        // let items: Vec<KeyframeItem> = dataset.iter().take(1000).collect();

        // inputs / prompts
        let mut prompts = Vec::new();
        // prompts.push(
        //     "0, 5, 361, 161, 305, 217, \n1, 5, 232, 332, 50, 70, \n2, 5, 149, 149, 304, 116, "
        //         .to_string(),
        // );
        // prompts.push("0, 5, 354, 154, 239, 91, \n1, 5, 544, 244, 106, 240, ".to_string());
        // prompts.push(
        //     "0, 5, 161, 161, 210, 168, \n1, 5, 165, 265, 189, 262, \n2, 5, 112, 212, 439, 266, \n3, 5, 152, 152, 462, 163, ".to_string()
        // );
        prompts.push(user_prompt);

        let mut items: Vec<Vec<KeyframeItem>> = Vec::new();

        for prompt in prompts {
            let mut sequence = Vec::new();
            for line in prompt.lines() {
                let values: Vec<f32> = line
                    .split(',')
                    .filter_map(|v| v.trim().parse().ok())
                    .collect();

                if values.len() == 6 {
                    sequence.push(KeyframeItem {
                        polygon_index: values[0],
                        time: values[1],
                        width: values[2],
                        height: values[3],
                        x: values[4],
                        y: values[5],
                    });
                }
            }
            items.push(sequence);
        }

        let mut target_items: Vec<Vec<KeyframeItem>> = Vec::new();

        // Generate target sequences based on prompt length
        for sequence in &items {
            let prompt_len = sequence.len();
            let target_len = prompt_len * 6; // Each input row generates 6 target rows

            let mut target_sequence = Vec::with_capacity(target_len);
            for _ in 0..target_len {
                target_sequence.push(KeyframeItem {
                    polygon_index: 0.0,
                    time: 0.0,
                    width: 0.0,
                    height: 0.0,
                    x: 0.0,
                    y: 0.0,
                });
            }
            target_items.push(target_sequence);
        }

        let combined_for_batcher = items
            .iter()
            .cloned()
            .zip(target_items.iter().cloned())
            .collect::<Vec<_>>();

        let batch = self.batcher.batch(combined_for_batcher.clone());

        let targets = batch.targets;

        let normalizer = Normalizer::new(&targets.device());

        let normalized_inputs = normalizer.normalize(batch.inputs.clone());
        let predicted = self.model.forward(normalized_inputs);
        let predicted = normalizer.denormalize(predicted);

        // Display the predicted vs expected values
        let predicted_data = predicted.clone().into_data();
        // let expected_data = targets.clone().into_data();

        // normalize values to see differential in numbers
        // let normalized_predicted = normalizer.normalize(predicted);
        // let normalized_expected = normalizer.normalize(targets);
        // let normalized_predicted_data = normalized_predicted.into_data();
        // let normalized_expected_data = normalized_expected.into_data();

        // let points = predicted_data
        //     .iter::<f32>()
        //     .zip(expected_data.iter::<f32>())
        //     .collect::<Vec<_>>();

        // let normalized_points = normalized_predicted_data
        //     .iter::<f32>()
        //     .zip(normalized_expected_data.iter::<f32>())
        //     .collect::<Vec<_>>();

        println!("Predicted Motion Paths:");

        // println!("Denormalized...");
        // // Print all values
        // for (predicted, expected) in points {
        //     println!("Predicted {} Expected {}", predicted, expected);
        // }

        // print predicted values in lines of 6 columns
        for (i, predicted) in predicted_data.iter::<f32>().enumerate() {
            if i % 6 == 0 {
                println!();
            }
            print!("{}, ", predicted);
        }

        // println!("Normalized...");
        // // Print all values
        // for (predicted, expected) in normalized_points {
        //     println!("Predicted {} Expected {}", predicted, expected);
        // }
    }
}
