use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;

fn process_sequence(sequence: &[String]) -> Vec<String> {
    if sequence.is_empty() {
        return Vec::new();
    }

    // Find max x and y values
    let mut max_x: f64 = 0.0;
    let mut max_y: f64 = 0.0;

    for line in sequence {
        if line == "!!!" {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 6 {
            let x: f64 = parts[4].parse().unwrap_or(0.0);
            let y: f64 = parts[5].parse().unwrap_or(0.0);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }

    // Calculate scaling factors
    let scale_x = if max_x > 0.0 { 100.0 / max_x } else { 1.0 };
    let scale_y = if max_y > 0.0 { 100.0 / max_y } else { 1.0 };

    // Process each line with NDC-style stretching
    sequence
        .iter()
        .map(|line| {
            if line == "!!!" {
                line.clone()
            } else {
                let mut parts: Vec<&str> = line.split(',').collect();
                if parts.len() >= 6 {
                    let x: f64 = parts[4].parse().unwrap_or(0.0);
                    let y: f64 = parts[5].parse().unwrap_or(0.0);

                    // NDC-style stretching
                    let new_x = ndc_stretch(x, max_x, scale_x);
                    let new_y = ndc_stretch(y, max_y, scale_y);

                    let part_x = format!("{:.0}", new_x.round() as i32);
                    let part_y = format!("{:.0}", new_y.round() as i32);

                    // Update the parts with new values
                    parts[4] = &part_x;
                    parts[5] = &part_y;
                    parts.join(",")
                } else {
                    line.clone()
                }
            }
        })
        .collect()
}

fn ndc_stretch(value: f64, max: f64, scale: f64) -> f64 {
    if max == 0.0 {
        return value;
    }

    // Calculate how close the value is to the maximum (0.0 to 1.0)
    let normalized = value / max;

    // Apply non-linear stretching: values closer to max stretch more towards 100,
    // values closer to 0 stretch less
    let stretched = normalized * normalized * 100.0;

    // Ensure we don't exceed 100
    stretched.min(100.0)
}

fn main() -> io::Result<()> {
    let input_path = Path::new("D:/projects/common/common-motion-2d-reg/backup/test_perc.txt");
    let output_path =
        Path::new("D:/projects/common/common-motion-2d-reg/backup/test_perc_stretched.txt");

    let file = File::open(input_path)?;
    let reader = BufReader::new(file);
    let mut output = File::create(output_path)?;

    let mut current_sequence: Vec<String> = Vec::new();

    for line in reader.lines() {
        let line = line?;

        if line == "---" {
            // Process and write the current sequence
            let processed = process_sequence(&current_sequence);
            for processed_line in processed {
                writeln!(output, "{}", processed_line)?;
            }
            writeln!(output, "---")?;
            current_sequence.clear();
        } else {
            current_sequence.push(line);
        }
    }

    // Process any remaining sequence
    if !current_sequence.is_empty() {
        let processed = process_sequence(&current_sequence);
        for processed_line in processed {
            writeln!(output, "{}", processed_line)?;
        }
    }

    Ok(())
}
