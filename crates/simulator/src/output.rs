use std::fs::File;
use std::io::{self, Write};

pub fn write_csv(path: &str, rows: &[(f64, f64)]) -> io::Result<()> {
    let mut f = File::create(path)?;
    writeln!(f, "theta,energy")?;
    for (theta, energy) in rows {
        writeln!(f, "{},{}", theta, energy)?;
    }
    Ok(())
}
