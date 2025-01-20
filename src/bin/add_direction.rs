use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};
use std::io::{BufRead, BufReader};
use std::path::Path;

fn main() {
    let file_path = "D:/projects/common/common-motion-2d-reg/backup/test_perc_stretched.txt";
    let mut output_file = File::create(
        "D:/projects/common/common-motion-2d-reg/backup/test_perc_stretched_with_direction.txt",
    )
    .unwrap();

    if let Ok(lines) = read_lines(file_path) {
        let mut prev_points: HashMap<i32, (f64, f64)> = HashMap::new();
        let mut is_completion = false;

        for line in lines {
            if let Ok(line_str) = line {
                if line_str.trim() == "---" {
                    writeln!(output_file, "---").unwrap();
                    prev_points.clear();
                    is_completion = false;
                    continue;
                }
                if line_str.trim() == "!!!" {
                    writeln!(output_file, "!!!").unwrap();
                    prev_points.clear();
                    is_completion = true;
                    continue;
                }

                let parts: Vec<&str> = line_str.split(',').collect();
                if parts.len() < 6 {
                    eprintln!("Error: Invalid line format: {}", line_str);
                    continue;
                }

                let object_id: i32 = parts[0].trim().parse().unwrap_or_default();
                let x: f64 = parts[4].trim().parse().unwrap_or_default();
                let y: f64 = parts[5].trim().parse().unwrap_or_default();

                let direction = if let Some(&(prev_x, prev_y)) = prev_points.get(&object_id) {
                    calculate_direction(prev_x, prev_y, x, y)
                } else {
                    0.0 // First point for this object, no direction yet
                };

                // Update previous point for this object
                prev_points.insert(object_id, (x, y));

                // Create new line with direction
                let mut new_parts = parts.to_vec();
                let new_part = format!(" {:.3}", direction);
                new_parts.push(&new_part);
                let new_line = new_parts.join(",");
                writeln!(output_file, "{}", new_line).unwrap();
            }
        }
    }
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn calculate_direction(prev_x: f64, prev_y: f64, curr_x: f64, curr_y: f64) -> f64 {
    // Calculate direction vector between previous and current point
    let dx = curr_x - prev_x;
    let dy = curr_y - prev_y;

    // Calculate angle in radians
    dy.atan2(dx)
}
