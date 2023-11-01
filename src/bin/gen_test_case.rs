use proptest::{
    strategy::{Strategy, ValueTree},
    test_runner::TestRunner,
};

fn main() {
    let mut runner = TestRunner::default();
    let gen = pku_minic::ast::arbitrary::arb_comp_unit();
    let ast = gen.new_tree(&mut runner).unwrap();
    let ast = ast.current();
    println!("{}", ast);
}
