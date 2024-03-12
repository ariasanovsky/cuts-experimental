use faer::{reborrow::{Reborrow, ReborrowMut}, ColMut, ColRef, MatMut, MatRef, RowMut, RowRef};

pub struct Cut {
    pub input_size: usize,
    pub output_size: usize,
    pub cut_value: f64,
}

pub fn cut_mat(
    mut mat: MatMut<f64>,
    mut optimal_input: ColMut<f64>,
    mut optimal_output: RowMut<f64>,
    mut test_input: ColMut<f64>,
    mut test_output: RowMut<f64>,
    rng: &mut impl rand::Rng
) -> Cut {
    optimal_input.fill_zero();
    optimal_output.fill_zero();
    let mut optimal_cut = Cut {
        input_size: 0,
        output_size: 0,
        cut_value: 0.0,
    };
    for trial in 0..10 {
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
        loop {
            let improved_output = improve_output(
                mat.rb(),
                test_input.rb(),
                test_output.rb_mut(),
                &mut test_cut,
            );
            if !improved_output {
                break
            }
            let improved_input = improve_input(
                mat.rb(),
                test_input.rb_mut(),
                test_output.rb(),
                &mut test_cut,
            );
            if !improved_input {
                break
            }
        }
        if test_cut.cut_value.abs() > optimal_cut.cut_value.abs() {
            optimal_cut = test_cut
        }
    }
    optimal_cut
}

fn improve_output(
    mat: MatRef<f64>,
    mut test_input: ColRef<f64>,
    mut test_output: RowMut<f64>,
    test_cut: &mut Cut,
) -> bool {
    todo!()
}

fn improve_input(
    mat: MatRef<f64>,
    mut test_input: ColMut<f64>,
    mut test_output: RowRef<f64>,
    test_cut: &mut Cut,
) -> bool {
    todo!()
}
