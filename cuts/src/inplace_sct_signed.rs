use equator::assert;
use faer::{
    linalg::matmul::matmul,
    reborrow::{Reborrow, ReborrowMut},
    Col, ColMut, ColRef, MatMut, MatRef,
};
use std::iter::zip;

#[derive(Debug)]
pub struct SignedCut {
    pub s_sizes: (usize, usize),
    pub t_sizes: (usize, usize),
    pub value: f64,
}

pub struct CutHelper {
    pub(crate) t_signs: Col<f64>,
    pub(crate) t_image: Col<f64>,
    pub(crate) s_signs: Col<f64>,
    pub(crate) s_image: Col<f64>,
}

pub struct CutHelperV2 {
    pub(crate) t_signs_old: Col<f64>,
    pub(crate) s_signs_old: Col<f64>,
    pub(crate) t_signs: Col<f64>,
    pub(crate) t_image: Col<f64>,
    pub(crate) s_signs: Col<f64>,
    pub(crate) s_image: Col<f64>,
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
        // TODO! these are backwards
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
        matmul(
            remainder,                   // acc
            s_signs.as_ref().as_2d(),    // lhs
            t_signs.transpose().as_2d(), // rhs
            Some(1.0),                   // alpha
            -normalized_cut,             // beta
            parallelism,                 // parallelism
        );
        cut
    }

    pub fn s_signs(&self) -> &Col<f64> {
        &self.s_signs
    }

    pub fn t_signs(&self) -> &Col<f64> {
        &self.t_signs
    }
}

impl CutHelperV2 {
    pub fn new(mat: MatRef<'_, f64>) -> Self {
        let nrows = mat.nrows();
        let ncols = mat.ncols();

        let s_ones = Col::from_fn(nrows, |_| 1.0);
        let t_ones = Col::from_fn(ncols, |_| 1.0);

        let s_image = mat.transpose() * &s_ones;
        let t_image = mat * &t_ones;

        Self {
            t_signs: t_ones.clone(),
            t_image,
            s_signs: s_ones.clone(),
            s_image,
            t_signs_old: t_ones,
            s_signs_old: s_ones,
        }
    }

    pub fn cut_mat(
        &mut self,
        remainder: MatMut<f64>,
        remainder_transposed: MatMut<f64>,
        rng: &mut impl rand::Rng,
        max_iterations: usize,
        parallelism: faer::Parallelism,
    ) -> f64 {
        let Self {
            t_signs,
            t_image,
            s_signs,
            s_image,
            t_signs_old,
            s_signs_old,
        } = self;
        assert!(all(
            remainder.row_stride() == 1,
            remainder_transposed.row_stride() == 1,
        ));

        t_signs_old.copy_from(&t_signs);
        t_signs
            .as_slice_mut()
            .iter_mut()
            .for_each(|t_sign| *t_sign = if rng.gen() { 1.0 } else { -1.0 });
        // PERF
        t_image.copy_from(&remainder * &*t_signs);
        let mut cut = &s_signs.as_ref().transpose() * &*t_image;

        for _iter in 0..max_iterations {
            let improved_s = improve_v2(
                remainder_transposed.rb(),
                t_image.as_ref(),
                s_signs_old.as_mut(),
                s_signs.as_mut(),
                s_image.as_mut(),
                &mut cut,
                parallelism,
            );
            if !improved_s {
                break;
            }
            let improved_t = improve_v2(
                remainder.rb(),
                s_image.as_ref(),
                t_signs_old.as_mut(),
                t_signs.as_mut(),
                t_image.as_mut(),
                &mut cut,
                parallelism,
            );
            if !improved_t {
                break;
            }
        }

        let normalization = remainder.nrows() * remainder.ncols();
        let normalized_cut = cut / normalization as f64;
        // remainder <- remainder - S * c * T^top
        // s_image <- s_image - T * c * S^top * S
        // t_image <- t_image - S * c * T^top * T
        *s_image -= faer::scale(cut / remainder.nrows() as f64) * &*t_signs;
        *t_image -= faer::scale(cut / remainder.ncols() as f64) * &*s_signs;

        // PERF
        matmul(
            remainder,                   // acc
            s_signs.as_ref().as_2d(),    // lhs
            t_signs.transpose().as_2d(), // rhs
            Some(1.0),                   // alpha
            -normalized_cut,             // beta
            parallelism,                 // parallelism
        );
        // PERF
        matmul(
            remainder_transposed,        // acc
            t_signs.as_ref().as_2d(),    // lhs
            s_signs.transpose().as_2d(), // rhs
            Some(1.0),                   // alpha
            -normalized_cut,             // beta
            parallelism,                 // parallelism
        );
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

pub(crate) fn improve_v2(
    mat_transposed: MatRef<f64>,
    t_image: ColRef<f64>,
    mut s_signs_old: ColMut<f64>,
    mut s_signs: ColMut<f64>,
    mut s_image: ColMut<f64>,
    cut: &mut f64,
    parallelism: faer::Parallelism,
) -> bool {
    assert!(mat_transposed.row_stride() == 1);
    let _ = parallelism;
    let new_cut = t_image.norm_l1();
    if new_cut <= *cut {
        return false;
    } else {
        *cut = new_cut
    }

    s_signs_old.copy_from(&s_signs.rb());
    for i in 0..t_image.nrows() {
        let t = t_image.read(i);
        if t <= 0.0 {
            s_signs.write(i, -1.0);
        } else {
            s_signs.write(i, 1.0);
        }
    }

    {
        let s_image = s_image.rb_mut().try_as_slice_mut().unwrap();
        let s_signs = s_signs.try_as_slice().unwrap();
        let s_signs_old = s_signs_old.try_as_slice().unwrap();

        struct Impl<'a> {
            mat_transposed: MatRef<'a, f64>,
            s_signs: &'a [f64],
            s_signs_old: &'a [f64],
            s_image: &'a mut [f64],
        }

        impl pulp::NullaryFnOnce for Impl<'_> {
            type Output = ();

            #[inline(always)]
            fn call(self) -> Self::Output {
                let Self {
                    mat_transposed,
                    s_signs,
                    s_signs_old,
                    s_image,
                } = self;

                for j in 0..mat_transposed.ncols() {
                    let s = s_signs[j];
                    let s_old = s_signs_old[j];
                    let col = mat_transposed.col(j).try_as_slice().unwrap();

                    if s != s_old {
                        let delta = s - s_old;
                        for (acc, &src) in zip(&mut *s_image, col) {
                            *acc += delta * src;
                        }
                    }
                }
            }
        }

        pulp::Arch::new().dispatch(Impl {
            mat_transposed,
            s_signs,
            s_signs_old,
            s_image,
        })
    }
    true
}
