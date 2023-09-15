fn main() {
    println!("cargo:rerun-if-changed=src/sysy.lalrpop");
    lalrpop::process_root().unwrap();
}
