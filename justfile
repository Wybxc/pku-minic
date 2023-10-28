default: koopa riscv

export RUST_BACKTRACE := "1"

koopa:
    cargo run -- -koopa hello.c

riscv:
    cargo run -- -riscv hello.c

autotest: autotest-koopa autotest-riscv

autotest-koopa:
    autotest -koopa -s lv4 .

autotest-riscv:
    autotest -riscv -s lv4 .

doc:
    cargo doc --no-deps --document-private-items
    dufs
