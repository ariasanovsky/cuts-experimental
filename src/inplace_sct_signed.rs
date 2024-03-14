use faer::{reborrow::{Reborrow, ReborrowMut}, scale, ColMut, ColRef, MatMut, MatRef};

#[derive(Debug)]
pub struct SignedCut {
    pub pos_inputs: usize,
    pub pos_outputs: usize,
    pub cut_value: f64,
}

pub fn cut_mat_signed(
    mut remainder: MatMut<f64>,
    mut approximat: MatMut<f64>,
    mut optimal_input: ColMut<f64>,
    mut optimal_output: ColMut<f64>,
    mut test_input: ColMut<f64>,
    mut test_output: ColMut<f64>,
    rng: &mut impl rand::Rng,
    adjustment_factor: f64,
    trials: usize,
    max_loops: usize,
) -> Option<SignedCut> {
    optimal_input.fill_zero();
    optimal_output.fill_zero();
    let mut optimal_cut = SignedCut {
        pos_inputs: 0,
        pos_outputs: 0,
        cut_value: 0.0,
    };
    for trial in 0..trials {
        let mut pos_inputs = 0;
        for i in 0..test_input.nrows() {
            if rng.gen() {
                test_input.write(i, 1.0);
                pos_inputs += 1;
            } else {
                test_input.write(i, -1.0);
            }
        }
        test_output.fill(0.0);
        let mut test_cut = SignedCut {
            pos_inputs,
            pos_outputs: 0,
            cut_value: 0.0,
        };
        for _ in 0..max_loops {
            let improved_output = improve_output(
                remainder.rb(),
                test_input.rb(),
                test_output.rb_mut(),
                &mut test_cut,
            );
            // let cut_value = test_output.rb().transpose() * mat.rb() * test_input.rb();
            // assert_eq!(test_cut.cut_value, cut_value);
            if !improved_output {
                break
            }
            let improved_input = improve_input(
                remainder.rb(),
                test_input.rb_mut(),
                test_output.rb(),
                &mut test_cut,
            );
            // let cut_value = test_output.rb().transpose() * mat.rb() * test_input.rb();
            // assert_eq!(test_cut.cut_value, cut_value);

            if !improved_input {
                break
            }
        }
        if test_cut.cut_value.abs() > optimal_cut.cut_value.abs() {
            // dbg!(&test_cut);
            optimal_cut = test_cut;
            optimal_input.copy_from(test_input.rb());
            optimal_output.copy_from(test_output.rb());
        }
    }
    let normalization = remainder.nrows() * remainder.ncols();
    if normalization == 0 {
        unreachable!();
        return None
    }
    let normalized_cut = optimal_cut.cut_value / normalization as f64;
    let optimal_input = optimal_input.transpose_mut();
    let cut_matrix = scale(adjustment_factor * normalized_cut) * optimal_output * optimal_input;

    remainder -= &cut_matrix;
    approximat += cut_matrix;
    Some(optimal_cut)
}

fn improve_output(
    mat: MatRef<f64>,
    test_input: ColRef<f64>,
    mut test_output: ColMut<f64>,
    test_cut: &mut SignedCut,
) -> bool {
    let new_output = mat * test_input;
    let new_cut = new_output.norm_l1();
    if new_cut <= test_cut.cut_value {
        return false
    } else {
        test_cut.cut_value = new_cut
    }
    // let mut pos_sum = 0.0;
    // let mut neg_sum = 0.0;
    let mut pos_count = 0;
    // let mut neg_count = 0;
    for i in 0..new_output.nrows() {
        let out_i = new_output.read(i);
        if out_i < 0.0 {
            test_output.write(i, -1.0);
            // neg_count += 1;
            // neg_sum += out_i;
        } else {
            test_output.write(i, 1.0);
            pos_count += 1;
            // pos_sum += out_i;
        }
    }
    test_cut.pos_outputs = pos_count;
    // let new_cut = new_output.norm_l1();
    // let (new_count, new_cut) = if pos_sum > -neg_sum {
    //     if pos_sum < test_cut.cut_value.abs() {
    //         return false
    //     }
    //     for i in 0..new_output.nrows() {
    //         if new_output.read(i) <= 0.0 {
    //             test_output.write(i, 0.0)
    //         } else {
    //             test_output.write(i, 1.0)
    //         }
    //     }
    //     (pos_count, pos_sum)
    // } else {
    //     if -neg_sum < test_cut.cut_value.abs() {
    //         return false
    //     }
    //     for i in 0..new_output.nrows() {
    //         if new_output.read(i) >= 0.0 {
    //             test_output.write(i, 0.0)
    //         } else {
    //             test_output.write(i, 1.0)
    //         }
    //     }
    //     (neg_count, neg_sum)
    // };
    // let SignedCut {
    //     input_size: _,
    //     output_size,
    //     cut_value,
    // } = test_cut;
    // *cut_value = new_cut;
    // *output_size = new_count;
    true
}

fn improve_input(
    mat: MatRef<f64>,
    mut test_input: ColMut<f64>,
    test_output: ColRef<f64>,
    test_cut: &mut SignedCut,
) -> bool {
    let new_input = test_output.transpose() * mat;
    let new_cut = new_input.norm_l1();
    if new_cut <= test_cut.cut_value {
        return false
    } else {
        test_cut.cut_value = new_cut
    }
    // let mut pos_sum = 0.0;
    // let mut neg_sum = 0.0;
    let mut pos_count = 0;
    // let mut neg_count = 0;
    for j in 0..new_input.ncols() {
        let in_j = new_input.read(j);
        if in_j < 0.0 {
            test_input.write(j, -1.0)
            // neg_count += 1;
            // neg_sum += in_j;

        } else {
            test_input.write(j, 1.0);
            pos_count += 1;
            // pos_sum += in_j;
        }
    }
    test_cut.pos_inputs = pos_count;
    // let (new_count, new_cut) = if pos_sum > -neg_sum {
    //     if pos_sum < test_cut.cut_value.abs() {
    //         return false
    //     }
    //     for i in 0..new_input.ncols() {
    //         if new_input.read(i) <= 0.0 {
    //             test_input.write(i, 0.0)
    //         } else {
    //             test_input.write(i, 1.0)
    //         }
    //     }
    //     (pos_count, pos_sum)
    // } else {
    //     if -neg_sum < test_cut.cut_value.abs() {
    //         return false
    //     }
    //     for j in 0..new_input.ncols() {
    //         if new_input.read(j) >= 0.0 {
    //             test_input.write(j, 0.0)
    //         } else {
    //             test_input.write(j, 1.0)
    //         }
    //     }
    //     (neg_count, neg_sum)
    // };
    // let SignedCut {
    //     input_size,
    //     output_size: _,
    //     cut_value,
    // } = test_cut;
    // *cut_value = new_cut;
    // *input_size = new_count;
    true
}
