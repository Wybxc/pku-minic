use miette::{IntoDiagnostic, Result};
use std::io::Read;

struct Args {
    mode: Mode,
    input: clio::Input,
    output: clio::Output,
}

enum Mode {
    Koopa,
    Riscv,
}

impl Args {
    fn help() -> String {
        let help = r#"_Usage_: *pku-minic* <mode> <input> [-o output]

_Arguments_:
    *mode*        Mode of the compiler, can be one of:
                    *-koopa*      Generate koopa IR
                    *-riscv*      Generate riscv assembly
                    *-help*       Print this help message
    *input*       Input file, use - for stdin
    *-o output*   Output file, use - or omit for stdout"#;

        markup(help)
    }

    /// Parse command line arguments, return Err if failed.
    fn try_parse() -> Result<Self, String> {
        let mut args = std::env::args_os();
        args.next(); // skip program name

        // Parse mode
        let mode = args.next().ok_or("missing argument `mode`")?;
        let mode = match mode.to_str() {
            Some("-koopa") => Mode::Koopa,
            Some("-riscv") => Mode::Riscv,
            Some("-help") => {
                // Print help message and exit
                println!("{}", Self::help());
                std::process::exit(0);
            }
            _ => return Err(format!("invalid mode: {}", mode.to_string_lossy())),
        };

        // Parse input
        let input = args.next().ok_or("missing argument `input`")?;
        let input = clio::Input::new(&input).map_err(|err| err.to_string())?;

        // Parse output. Missing output is allowed.
        let output = if args.next().is_none() {
            "-".into()
        } else {
            args.next().unwrap_or_else(|| "-".into())
        };
        let output = clio::Output::new(&output).map_err(|err| err.to_string())?;

        Ok(Self {
            mode,
            input,
            output,
        })
    }

    /// Parse command line arguments, print help message and exit if failed.
    fn parse() -> Self {
        Self::try_parse().unwrap_or_else(|err| {
            eprintln!("{}: {}", markup("_Error_"), err);
            eprintln!("{}", Self::help());
            std::process::exit(1);
        })
    }
}

fn main() -> Result<()> {
    // Setup panic hook
    human_panic::setup_panic!();

    // Setup logger
    env_logger::init();

    // Parse command line arguments
    let mut args = Args::parse();

    // Read input
    let mut input = String::new();
    args.input.read_to_string(&mut input).into_diagnostic()?;

    // Compile
    let program = match pku_minic::compile(&input) {
        Ok(program) => program,
        Err(diagnostic) => Err(diagnostic.with_source_code(input))?,
    };

    // Generate output
    match args.mode {
        Mode::Koopa => {
            let mut gen = koopa::back::KoopaGenerator::new(args.output);
            gen.generate_on(&program).into_diagnostic()?;
        }
        Mode::Riscv => {
            pku_minic::codegen(program, args.output).into_diagnostic()?;
        }
    }

    Ok(())
}

/// Simple markup for help message.
///
/// * `_underline_`
/// * `*bold*`
fn markup(s: &str) -> String {
    use regex::{Captures, Regex};
    use colored::*;

    Regex::new(r"_(?P<underline>.*?)_|\*(?P<bold>.*?)\*")
        .unwrap()
        .replace_all(s, |caps: &Captures| {
            if let Some(s) = caps.name("bold") {
                return s.as_str().bold().to_string();
            }
            if let Some(s) = caps.name("underline") {
                return s.as_str().bold().underline().to_string();
            }
            unreachable!()
        })
        .into_owned()
}
