use crate::dataset::{KeyframeBatcher, MotionDataset};
use crate::model::RnnModelConfig;
use burn::lr_scheduler;
use burn::lr_scheduler::constant::ConstantLr;
use burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig;
use burn::lr_scheduler::linear::{LinearLrScheduler, LinearLrSchedulerConfig};
use burn::lr_scheduler::noam::NoamLrSchedulerConfig;
use burn::optim::{AdamConfig, AdamWConfig};
use burn::train::metric::{AccuracyMetric, CudaMetric, LearningRateMetric};
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    prelude::*,
    record::{CompactRecorder, NoStdTrainingRecorder},
    tensor::backend::AutodiffBackend,
    train::{metric::LossMetric, LearnerBuilder},
};

#[derive(Config)]
pub struct ExpConfig {
    #[config(default = 500)]
    pub num_epochs: usize,

    #[config(default = 4)]
    pub num_workers: usize,

    #[config(default = 1337)]
    pub seed: u64,

    pub optimizer: AdamWConfig,

    // #[config(default = 16)]
    #[config(default = 256)]
    // #[config(default = 128)]
    // #[config(default = 64)]
    // #[config(default = 2)]
    // #[config(default = 4)]
    // #[config(default = 4)]
    // targets large so batch size 1? possible vanishing gradients in lstm with larger batch size?
    pub batch_size: usize,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn run<B: AutodiffBackend>(artifact_dir: &str, device: B::Device) {
    create_artifact_dir(artifact_dir);

    // Config
    // let optimizer = AdamConfig::new();
    let optimizer = AdamWConfig::new().with_weight_decay(1.0e-8);
    let config = ExpConfig::new(optimizer);
    let model = RnnModelConfig::new().init(&device);
    B::seed(config.seed);

    // Define train/valid datasets and dataloaders
    let train_dataset =
        MotionDataset::from_string(include_str!("../backup/augmented_perc_stretched.txt"));
    let valid_dataset =
        MotionDataset::from_string(include_str!("../backup/test_perc_stretched.txt"));

    println!("Train Dataset Size: {}", train_dataset.len());
    println!("Valid Dataset Size: {}", valid_dataset.len());

    let batcher_train = KeyframeBatcher::<B>::new(device.clone());

    let batcher_test = KeyframeBatcher::<B::InnerBackend>::new(device.clone());

    // print 5 sequences from train
    // for i in 0..5 {
    //     let (inputs, targets) = train_dataset.get(i).unwrap();
    //     println!("Train Sequence inputs {}: {:?}", i, inputs.len());
    //     // print input data
    //     for input in inputs.iter() {
    //         println!("{:?}", input);
    //     }
    //     println!("Train Sequence targets {}: {:?}", i, targets.len());
    //     // print input data
    //     for target in targets.iter() {
    //         println!("{:?}", target);
    //     }
    // }

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(valid_dataset);

    // let lr_scheduler = NoamLrSchedulerConfig::new(1e-1 as f64)
    //     .with_warmup_steps(2000)
    //     // .with_model_size(config.transformer.d_model)
    //     .init();

    // let lr_scheduler = LinearLrSchedulerConfig::new(1e-3, 1e-4, 200).init();

    let lr_scheduler = ConstantLr::new(1e-3);
    // let lr_scheduler = ConstantLr::new(0.1);

    // let lr_scheduler = CosineAnnealingLrSchedulerConfig::new(1e-3, 100).init();

    // Model
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        // .metric_train_numeric(AccuracyMetric::new())
        // .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, config.optimizer.init(), lr_scheduler);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    println!("Saving model and config to {}", artifact_dir);

    config
        .save(format!("{artifact_dir}/config.json").as_str())
        .unwrap();

    model_trained
        .save_file(
            format!("{artifact_dir}/model"),
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save trained model");
}
