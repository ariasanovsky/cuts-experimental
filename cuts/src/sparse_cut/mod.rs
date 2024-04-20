use dyn_stack::ReborrowMut;
use faer::{linalg::matmul::matmul, reborrow::Reborrow, ColMut, ColRef, MatMut, MatRef};

use crate::inplace_sct_signed::CutHelper;

pub mod helpers;

#[derive(Debug)]
pub struct SparseCut {
    pub s_sizes: (usize, usize),
    pub t_sizes: (usize, usize),
    pub value: f64,
}

impl SparseCut {
    pub fn num_bits(&self) -> usize {
        16 * (self.s_sizes.0 + self.s_sizes.1 + self.t_sizes.0 + self.t_sizes.1)
    }
}

impl CutHelper {
    pub fn sparse_cut_mat(
        &mut self,
        remainder: MatMut<f64>,
        rng: &mut impl rand::Rng,
        max_iterations: usize,
        parallelism: faer::Parallelism,
    ) -> SparseCut {
        let Self {
            t_signs,
            t_image,
            s_signs,
            s_image,
        } = self;
        let mut t_pos = 0;
        let mut t_neg = 0;
        t_signs.as_slice_mut().iter_mut().for_each(|t_sign| {
            let i = rng.gen_range(0..3);
            match i {
                0 => {
                    t_pos += 1;
                    *t_sign = 1.0;
                },
                1 => {
                    t_neg += 1;
                    *t_sign = -1.0;
                },
                2 => {
                    *t_sign = 0.0;
                },
                _ => unreachable!(),
            }
        });
        let mut cut: SparseCut = SparseCut {
            s_sizes: (0, 0),
            t_sizes: (t_pos, t_neg),
            value: 0.0,
        };

        for _iter in 0..max_iterations {
            let improved_s: bool = improve_s(
                remainder.rb(),
                t_signs.as_ref(),
                t_image.as_mut(),
                s_signs.as_mut(),
                &mut cut,
                parallelism,
            );
            if !improved_s {
                println!("{cut:?}");
                break;
            }
            let improved_input: bool = improve_t(
                remainder.rb(),
                s_signs.as_ref(),
                s_image.as_mut(),
                t_signs.as_mut(),
                &mut cut,
                parallelism,
            );
            if !improved_input {
                println!("{cut:?}");
                break;
            }
        }

        let s_cardinality = cut.s_sizes.0 + cut.s_sizes.1;
        let t_cardinality = cut.t_sizes.0 + cut.t_sizes.1;
        let normalization = s_cardinality * t_cardinality;
        let normalized_cut = cut.value / normalization as f64;
        // let cut_matrix = scale(normalized_cut) * s_signs.as_ref() * t_signs.transpose();
        // remainder -= &cut_matrix;
        // `acc = alpha * acc + beta *     lhs *                 rhs`
        // `rem =   1.0 * rem + (-q) * s_signs * t_signs.transpose()`
        matmul(
            remainder,                   // acc
            s_signs.as_ref().as_2d(),    // lhs
            t_signs.transpose().as_2d(), // rhs
            Some(1.0),                   // alpha
            -normalized_cut,             // beta
            parallelism,     // parallelism
        );
        cut
    }
}

pub(crate) fn improve_s(
    mat: MatRef<f64>,
    t_signs: ColRef<f64>,
    mut t_image: ColMut<f64>,
    mut s_signs: ColMut<f64>,
    cut: &mut SparseCut,
    parallelism: faer::Parallelism,
) -> bool {
    matmul(
        t_image.rb_mut().as_2d_mut(),
        mat,
        t_signs.as_2d(),
        None,
        1.0,
        parallelism,
    );
    let mut entries: Vec<(usize, f64)> = t_image.try_as_slice().unwrap().iter().enumerate().map(|(i, x)| {
        (i, *x)
    }).collect();
    entries.sort_by(|(i, xi), (j, xj)| {
        xj.abs().partial_cmp(&xi.abs()).unwrap().then(i.cmp(j))
    });
    let mut new_cut = 0.0;
    let mut num_pos = 0;
    let mut num_neg = 0;
    let SparseCut {
        s_sizes,
        t_sizes,
        value,
    } = cut;
    let t_cardinality = t_sizes.0 + t_sizes.1;
    s_signs.fill_zero();
    for (i, xi) in entries {
        if xi.abs() * (t_cardinality + num_pos + num_neg) as f64 <= new_cut {
            break;
        }
        if xi > 0.0 {
            num_pos += 1;
            s_signs.write(i, 1.0);
            new_cut += xi;
        } else {
            num_neg += 1;
            s_signs.write(i, -1.0);
            new_cut -= xi;
        }
    }
    let normalized_value = *value / (s_sizes.0 + s_sizes.1 + t_sizes.0 + t_sizes.1) as f64;
    let normalized_new_cut = new_cut / (t_cardinality + num_pos + num_neg) as f64;
    match normalized_value.partial_cmp(&normalized_new_cut).unwrap() {
        std::cmp::Ordering::Greater => false, //unreachable!("{new_cut}\n-\n{value}\n={}", new_cut - *value),
        std::cmp::Ordering::Equal => false,
        std::cmp::Ordering::Less => {
            *s_sizes = (num_pos, num_neg);
            *value = new_cut;
            true
        },
    }
}

pub(crate) fn improve_t(
    mat: MatRef<f64>,
    s_signs: ColRef<f64>,
    mut s_image: ColMut<f64>,
    mut t_signs: ColMut<f64>,
    cut: &mut SparseCut,
    parallelism: faer::Parallelism,
) -> bool {
    matmul(
        s_image.rb_mut().as_2d_mut(),
        mat.transpose(),
        s_signs.as_2d(),
        None,
        1.0,
        parallelism,
    );
    let mut entries: Vec<(usize, f64)> = s_image.try_as_slice().unwrap().iter().enumerate().map(|(i, x)| {
        (i, *x)
    }).collect();
    entries.sort_by(|(i, xi), (j, xj)| {
        xj.abs().partial_cmp(&xi.abs()).unwrap().then(i.cmp(j))
    });
    let mut new_cut = 0.0;
    let mut num_pos = 0;
    let mut num_neg = 0;
    let SparseCut {
        s_sizes,
        t_sizes,
        value,
    } = cut;
    let s_cardinality = s_sizes.0 + s_sizes.1;
    t_signs.fill_zero();
    for (i, xi) in entries {
        if xi.abs() * (s_cardinality + num_pos + num_neg) as f64 <= new_cut {
            break;
        }
        if xi > 0.0 {
            num_pos += 1;
            t_signs.write(i, 1.0);
            new_cut += xi;
        } else {
            num_neg += 1;
            t_signs.write(i, -1.0);
            new_cut -= xi;
        }
    }
    let normalized_value = *value / (s_sizes.0 + s_sizes.1 + t_sizes.0 + t_sizes.1) as f64;
    let normalized_new_cut = new_cut / (s_cardinality + num_pos + num_neg) as f64;
    match normalized_value.partial_cmp(&normalized_new_cut).unwrap() {
        std::cmp::Ordering::Greater => false, //unreachable!("{new_cut}\n-\n{value}\n={}", new_cut - *value),
        std::cmp::Ordering::Equal => false,
        std::cmp::Ordering::Less => {
            *t_sizes = (num_pos, num_neg);
            *value = new_cut;
            true
        },
    }
}