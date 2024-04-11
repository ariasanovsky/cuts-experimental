use faer::{
    linalg::matmul::matmul,
    reborrow::{Reborrow, ReborrowMut},
    Col, ColMut, ColRef, MatMut, MatRef,
};

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
        remainder: MatMut<f64>,
        // approximat: MatMut<f64>,
        rng: &mut impl rand::Rng,
        max_iterations: usize,
        parallelism: faer::Parallelism,
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
                parallelism,
            );
            if !improved_s {
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
                break;
            }
        }

        let normalization = remainder.nrows() * remainder.ncols();
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
        // approximat += cut_matrix;
        // `acc = alpha * acc + beta *     lhs *                 rhs`
        // `rem =   1.0 * rem + (-q) * s_signs * t_signs.transpose()`
        // matmul(
        //     approximat,                  // acc
        //     s_signs.as_ref().as_2d(),    // lhs
        //     t_signs.transpose().as_2d(), // rhs
        //     Some(1.0),                   // alpha
        //     normalized_cut,              // beta
        //     faer::Parallelism::None,     // parallelism
        // );
        cut
    }

    pub fn s_signs(&self) -> &Col<f64> {
        &self.s_signs
    }

    pub fn t_signs(&self) -> &Col<f64> {
        &self.t_signs
    }
}

pub(crate) fn improve_s(
    mat: MatRef<f64>,
    t_signs: ColRef<f64>,
    mut t_image: ColMut<f64>,
    mut s_signs: ColMut<f64>,
    cut: &mut SignedCut,
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
    let new_cut = t_image.norm_l1();
    if new_cut <= cut.value {
        return false;
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
    cut.s_sizes = (pos_count, s_signs.nrows() - pos_count);
    true
}

pub(crate) fn improve_t(
    mat: MatRef<f64>,
    s_signs: ColRef<f64>,
    mut s_image: ColMut<f64>,
    mut t_signs: ColMut<f64>,
    test_cut: &mut SignedCut,
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
    let new_cut = s_image.norm_l1();
    if new_cut <= test_cut.value {
        return false;
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
