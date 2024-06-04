fn main() {
    #[cfg(feature = "fuzz")]
    {
        afl::fuzz!(|data: &[u8]| {
            if let Ok(input) = std::str::from_utf8(data) {
                if fuzz(input.to_string()).is_err() {
                    println!("Fuzzing failed");
                } else {
                    println!("Fuzzing succeeded");
                }
            }
        });
    }
}

#[cfg(feature = "fuzz")]
fn fuzz(input: String) -> miette::Result<()> {
    let (program, metadata) =
        pku_minic::compile(&input, 3).map_err(|err| err.with_source_code(input.clone()))?;
    let program = pku_minic::codegen(program, &metadata, 3)
        .map_err(|err| err.with_source_code(input.clone()))?;
    let program = format!("{}", program);

    drop(program);
    Ok(())
}
