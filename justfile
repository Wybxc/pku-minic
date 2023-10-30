export RUST_BACKTRACE := "1"

test_c := "hello.c"
level := "lv4"

test_koopa := replace_regex(test_c, '\.c$', ".kp")
test_riscv := replace_regex(test_c, '\.c$', ".s")
test_llvm := replace_regex(test_c, '\.c$', ".ll")
test_llvm_riscv := replace_regex(test_c, '\.c$', ".ll.s")

default: koopa riscv

koopa:
    cargo run -- -koopa {{test_c}} -o {{test_koopa}}
    cat {{test_koopa}}

riscv:
    cargo run -- -riscv {{test_c}} -o {{test_riscv}}
    cat {{test_riscv}}

perf:
    cargo run -- -perf {{test_c}} -o {{test_riscv}}
    cat {{test_riscv}}

llvm args="": koopa
    koopac {{test_koopa}} -o {{test_llvm}}
    llc {{test_llvm}} -o {{test_llvm_riscv}} -march=riscv32 -mattr=+m,+relax {{args}}
    cat {{test_llvm_riscv}}

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
