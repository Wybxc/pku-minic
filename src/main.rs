use std::io::{Read, Write};

use miette::{IntoDiagnostic, Result};

struct Args {
    mode: Mode,
    input: clio::Input,
    output: clio::Output,
    opt_level: u8,
}

enum Mode {
    Koopa,
    Riscv,
    Perf,
}

impl Args {
    fn help() -> String {
        let help = r#"_Usage_: *pku-minic* <mode> <input> [-o output] [-Olevel]

_Arguments_:
    *mode*        Mode of the compiler, can be one of:
                    *-koopa*      Generate koopa IR
                    *-riscv*      Generate riscv assembly
                    *-perf*       Generate riscv assembly with optimizations, implies *-O3*
                    *-help*       Print this help message
    *input*       Input file, use - for stdin
    *-o output*   Output file, use - or omit for stdout
    *-Olevel*     Optimization level, can be *-O0*, *-O1*, *-O2* or *-O3*"#;

        markup(help)
    }

    /// Parse command line arguments, return Err if failed.
    fn try_parse() -> Result<Self, String> {
        let mut args = std::env::args_os();
        args.next(); // skip program name

        let mut output = None;
        let mut opt_level = 0;

        // Parse mode
        let mode = args.next().ok_or("missing argument `mode`")?;
        let mode = match mode.to_str() {
            Some("-koopa") => Mode::Koopa,
            Some("-riscv") => Mode::Riscv,
            Some("-perf") => {
                opt_level = 3;
                Mode::Perf
            }
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

        // Parse other arguments

        while let Some(arg) = args.next() {
            match arg.to_str() {
                Some("-o") => {
                    output = Some(args.next().unwrap_or_else(|| "-".into()));
                }
                Some(s) if s.starts_with("-O") => {
                    opt_level = s[2..]
                        .parse()
                        .map_err(|_| format!("invalid optimization level: {}", &s[2..]))?;
                }
                Some(s) => return Err(format!("invalid argument: {}", s)),
                None => {}
            }
        }
        // No output specified, use stdout
        let output = output.unwrap_or_else(|| "-".into());

        // Open output file
        let output = clio::Output::new(&output).map_err(|err| err.to_string())?;

        Ok(Self {
            mode,
            input,
            output,
            opt_level,
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
    pretty_env_logger::init();

    // Parse command line arguments
    let mut args = Args::parse();

    // Read input
    let mut input = String::new();
    args.input.read_to_string(&mut input).into_diagnostic()?;

    // Optimization level
    let opt_level = args.opt_level;

    // Compile
    let (program, metadata) = match pku_minic::compile(&input, opt_level) {
        Ok(program) => program,
        Err(diagnostic) => Err(diagnostic.with_source_code(input))?,
    };

    // Generate output
    match args.mode {
        Mode::Koopa => {
            let mut gen = koopa::back::KoopaGenerator::new(args.output);
            gen.generate_on(&program).into_diagnostic()?;
        }
        Mode::Riscv | Mode::Perf => {
            let program = pku_minic::codegen(program, &metadata, opt_level)?;
            write!(args.output, "{}", program).into_diagnostic()?;
        }
    }

    Ok(())
}

/// Simple markup for help message.
///
/// * `_underline_`
/// * `*bold*`
fn markup(s: &str) -> String {
    use owo_colors::*;
    use regex::{Captures, Regex};

    Regex::new(r"_(?P<underline>.*?)_|\*(?P<bold>.*?)\*")
        .unwrap()
        .replace_all(s, |caps: &Captures| {
            if let Some(s) = caps.name("bold") {
                return s
                    .as_str()
                    .if_supports_color(Stream::Stdout, |s| s.bold())
                    .to_string();
            }
            if let Some(s) = caps.name("underline") {
                return s
                    .as_str()
                    .if_supports_color(Stream::Stdout, |&s| s.bold().underline().to_string())
                    .to_string();
            }
            unreachable!()
        })
        .into_owned()
}
