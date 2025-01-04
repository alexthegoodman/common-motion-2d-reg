use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    module::Module,
    record::{NoStdTrainingRecorder, Recorder},
    tensor::backend::Backend,
};
// use rgb::RGB8;
// use textplots::{Chart, ColorPlot, Shape};

use crate::{
    dataset::{KeyframeBatcher, KeyframeItem, MotionDataset, Normalizer},
    model::{RnnModelConfig, RnnModelRecord},
};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device) {
    let record: RnnModelRecord<B> = NoStdTrainingRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    let model = RnnModelConfig::new().init(&device).load_record(record);

    // Use a sample of 1000 items from the test split
    // let dataset = MotionDataset::test();
    // let items: Vec<KeyframeItem> = dataset.iter().take(1000).collect();

    // inputs / prompts
    let mut prompts = Vec::new();
    prompts.push(
        "0, 5, 361, 161, 305, 217, \n1, 5, 232, 332, 50, 70, \n2, 5, 149, 149, 304, 116, "
            .to_string(),
    );

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

    // prepare as Vec<(Vec<KeyframeItem>, Vec<KeyframeItem>)> for batcher?
    // targets only included here to compare predictions, not used in inference
    let mut targets = Vec::new();
    targets.push(
        "0, 0, 361, 161, 330, -13
0, 2.5, 361, 161, 309, 90
0, 5, 361, 161, 305, 217
0, 15, 361, 161, 305, 217
0, 17.5, 361, 161, 312, 83
0, 20, 361, 161, 298, -22
1, 0, 232, 332, -17, 101
1, 2.5, 232, 332, 37, 86
1, 5, 232, 332, 50, 70
1, 15, 232, 332, 50, 70
1, 17.5, 232, 332, -5, 69
1, 20, 232, 332, -28, 106
2, 0, 149, 149, 305, -6
2, 2.5, 149, 149, 304, 57
2, 5, 149, 149, 304, 116
2, 15, 149, 149, 304, 116
2, 17.5, 149, 149, 306, 77
2, 20, 149, 149, 303, -11"
            .to_string(),
    );

    let mut target_items: Vec<Vec<KeyframeItem>> = Vec::new();

    for target in targets {
        let mut sequence = Vec::new();
        for line in target.lines() {
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
        target_items.push(sequence);
    }

    let combined_for_batcher = items
        .iter()
        .cloned()
        .zip(target_items.iter().cloned())
        .collect::<Vec<_>>();

    let batcher = KeyframeBatcher::new(device);
    let batch = batcher.batch(combined_for_batcher.clone());
    // let predicted = model.forward(batch.inputs);
    let predicted = model.forward_step(batch.clone()).output;
    let targets = batch.targets;

    // Display the predicted vs expected values
    let predicted_data = predicted.clone().into_data();
    let expected_data = targets.clone().into_data();

    let normalizer = Normalizer::new(&targets.device());
    // normalize values to see differential in numbers
    let normalized_predicted = normalizer.normalize(predicted);
    let normalized_expected = normalizer.normalize(targets);
    let normalized_predicted_data = normalized_predicted.into_data();
    let normalized_expected_data = normalized_expected.into_data();

    let points = predicted_data
        .iter::<f32>()
        .zip(expected_data.iter::<f32>())
        .collect::<Vec<_>>();

    let normalized_points = normalized_predicted_data
        .iter::<f32>()
        .zip(normalized_expected_data.iter::<f32>())
        .collect::<Vec<_>>();

    println!("Predicted Motion Paths:");

    println!("Denormalized...");
    // Print all values
    for (predicted, expected) in points {
        println!("Predicted {} Expected {}", predicted, expected);
    }

    println!("Normalized...");
    // Print all values
    for (predicted, expected) in normalized_points {
        println!("Predicted {} Expected {}", predicted, expected);
    }
}
