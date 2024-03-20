use faer::{
    reborrow::{Reborrow, ReborrowMut},
    scale, ColMut, ColRef, MatMut, MatRef,
};

#[derive(Debug)]
pub struct Cut {
    pub input_size: usize,
    pub output_size: usize,
    pub cut_value: f64,
}

pub fn cut_mat(
    mut mat: MatMut<f64>,
    mut optimal_input: ColMut<f64>,
    mut optimal_output: ColMut<f64>,
    mut test_input: ColMut<f64>,
    mut test_output: ColMut<f64>,
    rng: &mut impl rand::Rng,
    adjustment_factor: f64,
    trials: usize,
) -> Option<Cut> {
    optimal_input.fill_zero();
    optimal_output.fill_zero();
    let mut optimal_cut = Cut {
        input_size: 0,
        output_size: 0,
        cut_value: 0.0,
    };
    for trial in 0..trials {
        let mut input_size = 0;
        for i in 0..test_input.nrows() {
            if rng.gen() {
                test_input.write(i, 1.0);
                input_size += 1;
            } else {
                test_input.write(i, 0.0);
            }
        }
        test_output.fill_zero();
        let mut test_cut = Cut {
            input_size,
            output_size: 0,
            cut_value: 0.0,
        };
        for _ in 0..1_000 {
            let improved_output = improve_output(
                mat.rb(),
                test_input.rb(),
                test_output.rb_mut(),
                &mut test_cut,
            );
            // let cut_value = test_output.rb().transpose() * mat.rb() * test_input.rb();
            // assert_eq!(test_cut.cut_value, cut_value);
            if !improved_output {
                break;
            }
            let improved_input = improve_input(
                mat.rb(),
                test_input.rb_mut(),
                test_output.rb(),
                &mut test_cut,
            );
            // let cut_value = test_output.rb().transpose() * mat.rb() * test_input.rb();
            // assert_eq!(test_cut.cut_value, cut_value);
            if !improved_input {
                break;
            }
        }
        if test_cut.cut_value.abs() > optimal_cut.cut_value.abs() {
            dbg!(&test_cut);
            optimal_cut = test_cut;
            optimal_input.copy_from(test_input.rb());
            optimal_output.copy_from(test_output.rb());
        }
    }
    let normalization = optimal_cut.input_size * optimal_cut.output_size;
    if normalization == 0 {
        return None;
    }
    let normalized_cut = optimal_cut.cut_value / normalization as f64;
    let optimal_input = optimal_input.transpose_mut();
    let cut_matrix = scale(adjustment_factor * normalized_cut) * optimal_output * optimal_input;

    mat -= cut_matrix;
    Some(optimal_cut)
}

fn improve_output(
    mat: MatRef<f64>,
    test_input: ColRef<f64>,
    mut test_output: ColMut<f64>,
    test_cut: &mut Cut,
) -> bool {
    let new_output = mat * test_input;
    let mut pos_sum = 0.0;
    let mut neg_sum = 0.0;
    let mut pos_count = 0;
    let mut neg_count = 0;
    for i in 0..new_output.nrows() {
        let out_i = new_output.read(i);
        if out_i < 0.0 {
            neg_count += 1;
            neg_sum += out_i;
        } else if out_i > 0.0 {
            pos_count += 1;
            pos_sum += out_i;
        }
    }
    let (new_count, new_cut) = if pos_sum > -neg_sum {
        if pos_sum < test_cut.cut_value.abs() {
            return false;
        }
        for i in 0..new_output.nrows() {
            if new_output.read(i) <= 0.0 {
                test_output.write(i, 0.0)
            } else {
                test_output.write(i, 1.0)
            }
        }
        (pos_count, pos_sum)
    } else {
        if -neg_sum < test_cut.cut_value.abs() {
            return false;
        }
        for i in 0..new_output.nrows() {
            if new_output.read(i) >= 0.0 {
                test_output.write(i, 0.0)
            } else {
                test_output.write(i, 1.0)
            }
        }
        (neg_count, neg_sum)
    };
    let Cut {
        input_size: _,
        output_size,
        cut_value,
    } = test_cut;
    *cut_value = new_cut;
    *output_size = new_count;
    true
}

fn improve_input(
    mat: MatRef<f64>,
    mut test_input: ColMut<f64>,
    test_output: ColRef<f64>,
    test_cut: &mut Cut,
) -> bool {
    let new_input = test_output.transpose() * mat;
    let mut pos_sum = 0.0;
    let mut neg_sum = 0.0;
    let mut pos_count = 0;
    let mut neg_count = 0;
    for j in 0..new_input.ncols() {
        let in_j = new_input.read(j);
        if in_j < 0.0 {
            neg_count += 1;
            neg_sum += in_j;
        } else if in_j > 0.0 {
            pos_count += 1;
            pos_sum += in_j;
        }
    }
    let (new_count, new_cut) = if pos_sum > -neg_sum {
        if pos_sum < test_cut.cut_value.abs() {
            return false;
        }
        for i in 0..new_input.ncols() {
            if new_input.read(i) <= 0.0 {
                test_input.write(i, 0.0)
            } else {
                test_input.write(i, 1.0)
            }
        }
        (pos_count, pos_sum)
    } else {
        if -neg_sum < test_cut.cut_value.abs() {
            return false;
        }
        for j in 0..new_input.ncols() {
            if new_input.read(j) >= 0.0 {
                test_input.write(j, 0.0)
            } else {
                test_input.write(j, 1.0)
            }
        }
        (neg_count, neg_sum)
    };
    let Cut {
        input_size,
        output_size: _,
        cut_value,
    } = test_cut;
    *cut_value = new_cut;
    *input_size = new_count;
    true
}
