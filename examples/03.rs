use cuts::inplace_sct::cut_mat;
use faer::{Col, Mat};

fn main() {
    const OUTPUTS: usize = 3; //32;
    const INPUTS: usize = 2; //64;
    let mut leftover = Mat::zeros(OUTPUTS, INPUTS);
    for i in 0..OUTPUTS {
        for j in 0..INPUTS {
            leftover[(i, j)] = 1.0; //(i + j) as f64;
        }
    }
    let mut optimal_input = Col::zeros(INPUTS);
    let mut optimal_output = Col::zeros(OUTPUTS);
    let mut test_input = Col::zeros(INPUTS);
    let mut test_output = Col::zeros(OUTPUTS);
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
        ).unwrap();
        println!("i = {i}: {}, {}", leftover.norm_l2(), new_cut.cut_value);
        println!("{leftover:?}");
    }
}