export RUST_BACKTRACE := "1"

test_c := "hello.c"
level := "lv9"

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

llvm:
    clang -S -emit-llvm {{test_c}} -O0 -Xclang -disable-O0-optnone --target=riscv32-unknown-unknown
    opt -S -p=mem2reg {{test_llvm}} -o {{test_llvm}}
    llc {{test_llvm}} -o {{test_llvm_riscv}} -O0 --frame-pointer=none -march=riscv32 -mattr=+m,+relax
    cat {{test_llvm_riscv}}

koopac: koopa
    koopac {{test_koopa}} -o {{test_koopa}}.ll
    llc {{test_koopa}}.ll -o {{test_koopa}}.s -O0 --frame-pointer=none -march=riscv32 -mattr=+m,+relax
    cat {{test_koopa}}.s

autotest: autotest-koopa autotest-riscv autotest-perf

autotest-koopa:
    docker run -it --rm -v .:/root/compiler maxxing/compiler-dev autotest -koopa -s {{level}} /root/compiler/

autotest-riscv:
    docker run -it --rm -v .:/root/compiler maxxing/compiler-dev autotest -riscv -s {{level}} /root/compiler/

autotest-perf:
    docker run -it --rm -v .:/root/compiler maxxing/compiler-dev autotest -perf -s {{level}} /root/compiler/

doc:
    cargo doc --no-deps --document-private-items
    dufs

build-timings:
    cargo clean
    cargo build --release --timings

bloat:
    cargo bloat --release --crates --split-std
