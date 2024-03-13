use cuts::inplace_sct_signed::cut_mat_signed;
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
    for i in 0..5 {
        let new_cut = cut_mat_signed(
            leftover.as_mut(),
            optimal_input.as_mut(),
            optimal_output.as_mut(),
            test_input.as_mut(),
            test_output.as_mut(),
            &mut rng,
            1.0,
            10,
        ).unwrap();
        println!("i = {i}: {}, {}", leftover.norm_l2(), new_cut.cut_value);
        println!("{leftover:?}");
    }
}
