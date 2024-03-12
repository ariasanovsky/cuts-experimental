use cuts::inplace_sct::cut_mat;
use faer::{Col, Mat, Row};

fn main() {
    const OUTPUTS: usize = 32;
    const INPUTS: usize = 64;
    let mut leftover = Mat::zeros(32, 64);
    for i in 0..OUTPUTS {
        for j in 0..INPUTS {
            leftover[(i, j)] = (i + j) as f64;
        }
    }
    let mut optimal_input = Col::zeros(INPUTS);
    let mut optimal_output = Row::zeros(OUTPUTS);
    let mut test_input = Col::zeros(INPUTS);
    let mut test_output = Row::zeros(OUTPUTS);
    let mut rng = rand::thread_rng();
    println!("{}", leftover.norm_l2());
    for i in 0..500 {
        let new_cut = cut_mat(
            leftover.as_mut(),
            optimal_input.as_mut(),
            optimal_output.as_mut(),
            test_input.as_mut(),
            test_output.as_mut(),
            &mut rng,
        );
        println!("i = {i}: {}, {}", leftover.norm_l2(), new_cut.cut_value)
    }
}