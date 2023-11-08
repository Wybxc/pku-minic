export RUST_BACKTRACE := "1"

test_c := "hello.c"
level := "lv5"

test_koopa := replace_regex(test_c, '\.c$', ".kp")
test_riscv := replace_regex(test_c, '\.c$', ".s")
test_llvm := replace_regex(test_c, '\.c$', ".ll")
test_llvm_riscv := replace_regex(test_c, '\.c$', ".ll.s")

default: koopa riscv perf

dump-ast:
    cargo run -- -dump-ast {{test_c}}

koopa:
    cargo run -- -koopa {{test_c}} -o {{test_koopa}}
    cat {{test_koopa}}

riscv:
    cargo run -- -riscv {{test_c}} -o {{test_riscv}}
    cat {{test_riscv}}

perf:
    cargo run -- -perf {{test_c}} -o {{test_riscv}}
    cat {{test_riscv}}

trace:
    cargo run --features=trace -- -perf {{test_c}} -o {{test_riscv}}
    cat {{test_riscv}}

llvm args="":
    clang -S -emit-llvm {{test_c}} -O0 -Xclang -disable-O0-optnone --target=riscv32-unknown-unknown
    opt -S -mem2reg {{test_llvm}} -o {{test_llvm}}
    llc {{test_llvm}} -o {{test_llvm_riscv}} -O0 --frame-pointer=none -march=riscv32 -mattr=+m,+relax {{args}}
    cat {{test_llvm_riscv}}

autotest: autotest-koopa autotest-riscv autotest-perf

autotest-koopa:
    autotest -koopa -s {{level}} .

autotest-riscv:
    autotest -riscv -s {{level}} .

autotest-perf:
    autotest -perf -s {{level}} .

doc:
    cargo doc --no-deps --document-private-items
    dufs

test:
    RUST_BACKTRACE=0 cargo test

test-arbitrary:
    cargo test --features proptest,arb-coverage -- --nocapture

gen-test-case:
    cargo run --features proptest -- -gen-test-case > hello.c
    cat hello.c

build-timings:
    cargo clean
    cargo build --release --timings

bloat:
    cargo bloat --release --crates --split-std
