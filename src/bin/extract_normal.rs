use std::fs::File;
use std::io::{self, BufRead, BufReader};

fn main() -> io::Result<()> {
    let file = File::open("D:/projects/common/common-motion-2d-reg/backup/augmented.txt")?;
    let reader = BufReader::new(file);

    // Store all values for each column
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); 6];

    for line in reader.lines() {
        let line = line?;

        // Skip separator lines
        if line.contains("---") || line.contains("!!!") {
            continue;
        }

        // Parse the line
        let values: Vec<f64> = line
            .split(',')
            .map(|s| s.trim().parse::<f64>().unwrap_or(0.0))
            .collect();

        // Add each value to its respective column
        for (i, &value) in values.iter().enumerate() {
            if i < columns.len() {
                columns[i].push(value);
            }
        }
    }

    // Calculate mean and std for each column
    for (i, column) in columns.iter().enumerate() {
        let mean = calculate_mean(column);
        let std = calculate_std(column, mean);

        println!("Column {}: Mean = {:.3}, Std = {:.3}", i, mean, std);
    }

    Ok(())
}

fn calculate_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn calculate_std(values: &[f64], mean: f64) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }

    let variance =
        values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

    variance.sqrt()
}
