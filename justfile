export RUST_BACKTRACE := "1"

test_case := "hello.c"
level := "lv4"

default: koopa riscv

koopa:
    cargo run -- -koopa {{test_case}}

riscv:
    cargo run -- -riscv {{test_case}}

perf:
    cargo run -- -perf {{test_case}}

autotest: autotest-koopa autotest-riscv

autotest-koopa:
    autotest -koopa -s {{level}} .

autotest-riscv:
    autotest -riscv -s {{level}} .

autotest-perf:
    autotest -perf -s {{level}} .

doc:
    cargo doc --no-deps --document-private-items
    dufs
