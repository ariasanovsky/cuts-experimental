use faer::{col, linalg::matmul::matmul, reborrow::{Reborrow, ReborrowMut}, row, scale, Col, ColMut, ColRef, MatMut, MatRef};

pub struct FrobeniusNormTracker {
    first_squared_norm: f64,
    current_squared_norm: f64,
}

impl FrobeniusNormTracker {
    pub fn new(squared_norm: f64) -> Self {
        assert!(squared_norm >= 0.);
        Self {
            first_squared_norm: squared_norm,
            current_squared_norm: squared_norm,
        }
    }

    pub fn decrement_by_signed_cut(&mut self, cut_value: f64, nrows: usize, ncols: usize) {
        self.current_squared_norm -= cut_value * cut_value / (nrows * ncols) as f64
    }

    pub fn norm(&self) -> f64 {
        self.current_squared_norm.sqrt()
    }

    pub fn squared_norm(&self) -> f64 {
        self.current_squared_norm
    }

    pub fn ratio(&self) -> f64 {
        self.current_squared_norm / self.first_squared_norm
    }
    
    pub fn squared_ratio(&self) -> f64 {
        (self.current_squared_norm / self.first_squared_norm).sqrt()
    }
}

#[derive(Debug)]
pub struct SignedCut {
    pub s_sizes: (usize, usize),
    pub t_sizes: (usize, usize),
    pub value: f64,
}

pub struct CutHelper {
    t_signs: Col<f64>,
    t_image: Col<f64>,
    s_signs: Col<f64>,
    s_image: Col<f64>,
}

impl CutHelper {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            t_signs: Col::zeros(ncols),
            t_image: Col::zeros(nrows),
            s_signs: Col::zeros(nrows),
            s_image: Col::zeros(ncols),
        }
    }

    pub fn cut_mat(
        &mut self,
        mut remainder: MatMut<f64>,
        mut approximat: MatMut<f64>,
        rng: &mut impl rand::Rng,
        max_iterations: usize,
    ) -> SignedCut {
        let Self {
            t_signs,
            t_image,
            s_signs,
            s_image,
        } = self;
        let mut s_pos = 0;
        t_signs.as_slice_mut().iter_mut().for_each(|t_sign| {
            *t_sign = if rng.gen() {
                s_pos += 1;
                1.0
            } else {
                -1.0
            }
        });
        let mut cut = SignedCut {
            s_sizes: (s_pos, t_signs.nrows() - s_pos),
            t_sizes: (0, 0),
            value: 0.0,
        };
        for _iter in 0..max_iterations {
            let improved_s: bool = improve_s(
                remainder.rb(),
                t_signs.as_ref(),
                t_image.as_mut(),
                s_signs.as_mut(),
                &mut cut,
            );
            let prod = remainder.as_ref() * t_signs.as_ref();
            let _cut = s_signs.transpose() * prod;
            if !improved_s {
                break;
            }
            let improved_input: bool = improve_t(
                remainder.rb(),
                s_signs.as_ref(),
                s_image.as_mut(),
                t_signs.as_mut(),
                &mut cut,
            );
            if !improved_input {
                break;
            }
        }
        // `acc = alpha * acc + beta * lhs * rhs`
        // `rem = 1.0 * rem - q * u * v`
        // `alpha: f64 = 1.0`
        // `beta = -q`, `q: f64`
        // `lhs = u`, `u` a `[f64; m]` column as a `(m x 1)` matrix
        // `rhs = v`, `v` a `[f64; n]` column as a `(1 x n)` matrix
        // TODO! rewrite like
        // matmul(
        //     rem.rb_mut(),
        //     u.rb().as_2d(),
        //     v.rb().as_2d(),
        //     Some(1.0),
        //     -q,
        //     faer::Parallelism::None,
        // );

        let normalization = remainder.nrows() * remainder.ncols();
        let normalized_cut = cut.value / normalization as f64;
        let cut_matrix = scale(normalized_cut) * s_signs.as_ref() * t_signs.transpose();
        remainder -= &cut_matrix;
        approximat += cut_matrix;
        cut
    }
}

// pub fn cut_mat_signed(
//     mut remainder: MatMut<f64>,
//     mut approximat: MatMut<f64>,
//     mut optimal_input: ColMut<f64>,
//     mut optimal_output: ColMut<f64>,
//     mut test_input: ColMut<f64>,
//     mut test_output: ColMut<f64>,
//     rng: &mut impl rand::Rng,
//     trials: usize,
//     max_loops: usize,
// ) -> SignedCut {
//     todo!();
    // optimal_input.fill_zero();
    // optimal_output.fill_zero();
    // let mut optimal_cut = SignedCut {
    //     pos_inputs: 0,
    //     pos_outputs: 0,
    //     cut_value: 0.0,
    // };
    // for _trial in 0..trials {
    //     let mut pos_inputs = 0;
    //     for i in 0..test_input.nrows() {
    //         if rng.gen() {
    //             test_input.write(i, 1.0);
    //             pos_inputs += 1;
    //         } else {
    //             test_input.write(i, -1.0);
    //         }
    //     }
    //     test_output.fill(0.0);
    //     let mut test_cut = SignedCut {
    //         pos_inputs,
    //         pos_outputs: 0,
    //         cut_value: 0.0,
    //     };
    //     for _ in 0..max_loops {
    //         let improved_output = improve_output(
    //             remainder.rb(),
    //             test_input.rb(),
    //             test_output.rb_mut(),
    //             &mut test_cut,
    //         );
    //         // let cut_value = test_output.rb().transpose() * mat.rb() * test_input.rb();
    //         // assert_eq!(test_cut.cut_value, cut_value);
    //         if !improved_output {
    //             break
    //         }
    //         let improved_input = improve_input(
    //             remainder.rb(),
    //             test_input.rb_mut(),
    //             test_output.rb(),
    //             &mut test_cut,
    //         );
    //         // let cut_value = test_output.rb().transpose() * mat.rb() * test_input.rb();
    //         // assert_eq!(test_cut.cut_value, cut_value);

    //         if !improved_input {
    //             break
    //         }
    //     }
    //     if test_cut.cut_value.abs() > optimal_cut.cut_value.abs() {
    //         // dbg!(&test_cut);
    //         optimal_cut = test_cut;
    //         optimal_input.copy_from(test_input.rb());
    //         optimal_output.copy_from(test_output.rb());
    //     }
    // }
    // let normalization = remainder.nrows() * remainder.ncols();
    // let normalized_cut = optimal_cut.cut_value / normalization as f64;
    // let optimal_input = optimal_input.transpose_mut();
    // let cut_matrix = scale(normalized_cut) * optimal_output * optimal_input;

    // remainder -= &cut_matrix;
    // approximat += cut_matrix;
    // optimal_cut
// }

fn improve_s(
    mat: MatRef<f64>,
    t_signs: ColRef<f64>,
    mut t_image: ColMut<f64>,
    mut s_signs: ColMut<f64>,
    cut: &mut SignedCut,
) -> bool {
    matmul(
        t_image.rb_mut().as_2d_mut(),
        mat,
        t_signs.as_2d(),
        None,
        1.0,
        faer::Parallelism::None,
    );
    let new_cut = t_image.norm_l1();
    if new_cut <= cut.value {
        return false
    } else {
        cut.value = new_cut
    }
    let mut pos_count = 0;
    for i in 0..t_image.nrows() {
        let out_i = t_image.read(i);
        if out_i <= 0.0 {
            s_signs.write(i, -1.0);
        } else {
            s_signs.write(i, 1.0);
            pos_count += 1;
        }
    }
    cut.s_sizes = (pos_count, t_signs.nrows() - pos_count);
    true
}

fn improve_t(
    mat: MatRef<f64>,
    s_signs: ColRef<f64>,
    mut s_image: ColMut<f64>,
    mut t_signs: ColMut<f64>,
    test_cut: &mut SignedCut,
) -> bool {
    matmul(
        s_image.rb_mut().as_2d_mut(),
        mat.transpose(),
        s_signs.as_2d(),
        None,
        1.0,
        faer::Parallelism::None,
    );
    let new_cut = s_image.norm_l1();
    if new_cut <= test_cut.value {
        return false
    } else {
        test_cut.value = new_cut
    }
    let mut pos_count = 0;
    for j in 0..s_image.nrows() {
        let in_j = s_image.read(j);
        if in_j <= 0.0 {
            t_signs.write(j, -1.0)
        } else {
            t_signs.write(j, 1.0);
            pos_count += 1;
        }
    }
    test_cut.t_sizes = (pos_count, t_signs.nrows() - pos_count);
    true
}
