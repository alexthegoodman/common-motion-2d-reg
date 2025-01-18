use std::fs::File;
use std::io::{BufRead, BufReader, Write};

fn main() {
    let input_file_path = "D:/projects/common/common-motion-2d-reg/backup/test_backup.txt"; // input file
    let output_file_path = "D:/projects/common/common-motion-2d-reg/backup/test_perc.txt"; // output file
    let canvas_width = 800;
    let canvas_height = 450;

    let input_file = File::open(input_file_path).expect("Could not open input file");
    let reader = BufReader::new(input_file);

    let mut output_file = File::create(output_file_path).expect("Could not create output file");

    for line in reader.lines() {
        let line = line.expect("Could not read line");

        if line.is_empty() || line == "---" {
            writeln!(output_file, "{}", line).expect("Could not write to output file");
            continue;
        }

        if line == "!!!" {
            writeln!(output_file, "{}", line).expect("Could not write to output file");
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != 6 {
            writeln!(output_file, "{}", line).expect("Could not write to output file");
            continue;
        }

        // Extract x and y coordinates
        let x: f64 = parts[4].parse().unwrap();
        let y: f64 = parts[5].parse().unwrap();

        // Calculate percentage coordinates
        let x_percent = (x / canvas_width as f64) * 100.0;
        let y_percent = (y / canvas_height as f64) * 100.0;

        // Reconstruct the line with percentage coordinates
        let mut new_line = line.clone();
        new_line = new_line.replace(&parts[4], &format!("{}", x_percent.round() as u32));
        new_line = new_line.replace(&parts[5], &format!("{}", y_percent.round() as u32));

        writeln!(output_file, "{}", new_line).expect("Could not write to output file");
    }
}
