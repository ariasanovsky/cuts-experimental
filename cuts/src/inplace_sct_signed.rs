use dyn_stack::PodStack;
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

    // remainder (+ S * C * T^top)
    pub fn cut_mat(
        &mut self,
        two_remainder: MatRef<'_, f64>,
        two_remainder_transposed: MatRef<'_, f64>,
        S: MatMut<'_, f64>,
        C: ColMut<'_, f64>,
        T: MatMut<'_, f64>,
        rng: &mut impl rand::Rng,
        max_iterations: usize,
        parallelism: faer::Parallelism,
        mut stack: PodStack<'_>,
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
            two_remainder.row_stride() == 1,
            two_remainder_transposed.row_stride() == 1,
            S.ncols() > 0,
            T.ncols() > 0,
            C.nrows() > 0,
        ));

        let width = S.ncols() - 1;
        let (S, S_next) = S.split_at_col_mut(width);
        let (T, T_next) = T.split_at_col_mut(width);
        let (C, C_next) = C.split_at_mut(width);
        let S = S.rb();
        let T = T.rb();
        let C = C.rb();
        let mut S_next = S_next.col_mut(0);
        let mut T_next = T_next.col_mut(0);
        let C_next = C_next.get_mut(0);

        t_signs_old.copy_from(&t_signs);
        t_signs
            .as_slice_mut()
            .iter_mut()
            .for_each(|t_sign| *t_sign = if rng.gen() { 1.0 } else { -1.0 });
        mul_add_with_rank_update(
            t_image.as_mut().try_as_slice_mut().unwrap(),
            two_remainder.rb(),
            S,
            C,
            T,
            t_signs.as_ref().try_as_slice().unwrap(),
            t_signs_old.as_ref().try_as_slice().unwrap(),
            stack.rb_mut(),
        );
        let mut cut = &s_signs.as_ref().transpose() * &*t_image;

        for _ in 0..max_iterations {
            let improved_s = improve_with_rank_update(
                two_remainder_transposed.rb(),
                T,
                C,
                S,
                t_image.as_ref(),
                s_signs_old.as_mut(),
                s_signs.as_mut(),
                s_image.as_mut(),
                &mut cut,
                parallelism,
                stack.rb_mut(),
            );
            if !improved_s {
                break;
            }
            let improved_t = improve_with_rank_update(
                two_remainder.rb(),
                S,
                C,
                T,
                s_image.as_ref(),
                t_signs_old.as_mut(),
                t_signs.as_mut(),
                t_image.as_mut(),
                &mut cut,
                parallelism,
                stack.rb_mut(),
            );
            if !improved_t {
                break;
            }
        }

        let normalization = two_remainder.nrows() * two_remainder.ncols();
        let normalized_cut = cut / normalization as f64;
        // remainder <- remainder - S * c * T^top
        // s_image <- s_image - T * c * S^top * S
        // t_image <- t_image - S * c * T^top * T
        *s_image -= faer::scale(cut / two_remainder.nrows() as f64) * &*t_signs;
        *t_image -= faer::scale(cut / two_remainder.ncols() as f64) * &*s_signs;

        *C_next = -normalized_cut;
        S_next.copy_from(&s_signs);
        T_next.copy_from(&t_signs);

        cut
    }

    pub fn s_signs(&self) -> &Col<f64> {
        &self.s_signs
    }

    pub fn t_signs(&self) -> &Col<f64> {
        &self.t_signs
    }
}

// acc += two * (mat + S * C * T^top) * (x_new - x_old)
// 1. acc += two * mat * (x_new - x_old)
// 2. acc += two * S * C * T^top * (x_new - x_old)
fn mul_add_with_rank_update(
    acc: &mut [f64],
    two_mat: MatRef<'_, f64>,
    S: MatRef<'_, f64>,
    C: ColRef<'_, f64>,
    T: MatRef<'_, f64>,
    x_new: &[f64],
    x_old: &[f64],
    stack: PodStack<'_>,
) {
    struct Impl<'a> {
        two_mat: MatRef<'a, f64>,
        S: MatRef<'a, f64>,
        C: ColRef<'a, f64>,
        T: MatRef<'a, f64>,
        x_new: &'a [f64],
        x_old: &'a [f64],
        acc: &'a mut [f64],
        stack: PodStack<'a>,
    }

    impl pulp::NullaryFnOnce for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                two_mat,
                S,
                C,
                T,
                x_new,
                x_old,
                acc,
                stack,
            } = self;

            let n = two_mat.ncols();
            let width = S.ncols();

            let (diff_indices, stack) = stack.make_raw::<u64>(n);

            let mut pos = 0usize;
            for j in 0..n {
                let s = x_new[j];
                let s_old = x_old[j];
                let col = two_mat.col(j).try_as_slice().unwrap();

                if s != s_old {
                    assert!((s - s_old).abs() == 2.0);
                    diff_indices[pos] = (((s < s_old) as u64) << 63) | j as u64;
                    pos += 1;

                    let delta = s - s_old;
                    if delta == 2.0 {
                        for (acc, &src) in zip(&mut *acc, col) {
                            *acc += src;
                        }
                    } else {
                        for (acc, &src) in zip(&mut *acc, col) {
                            *acc -= src;
                        }
                    }
                }
            }
            let diff_indices = &diff_indices[..pos];

            // y = T^top * (x_new - x_old)
            let (y, _) = faer::linalg::temp_mat_uninit::<f64>(width, 1, stack);
            let y = y.col_mut(0).try_as_slice_mut().unwrap();
            for j in 0..width {
                let col = T.col(j).try_as_slice().unwrap();
                let mut acc = 0.0;
                for &i in diff_indices {
                    let sign_bit = i & (1 << 63);
                    let i = i & ((1 << 63) - 1);
                    let i = i as usize;
                    acc += f64::from_bits(sign_bit ^ col[i].to_bits());
                }
                y[j] = acc;
            }
            // y = C * y
            for (y, &c) in zip(&mut *y, C.try_as_slice().unwrap()) {
                *y *= 2.0 * c;
            }
            // acc += S * y
            matmul(
                faer::col::from_slice_mut::<f64>(acc).as_2d_mut(),
                S,
                faer::col::from_slice::<f64>(y).as_2d(),
                Some(1.0),
                1.0,
                faer::Parallelism::None,
            );
        }
    }

    pulp::Arch::new().dispatch(Impl {
        two_mat,
        S,
        C,
        T,
        x_new,
        x_old,
        acc,
        stack,
    })
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

pub(crate) fn improve_with_rank_update(
    two_mat: MatRef<'_, f64>,
    S: MatRef<'_, f64>,
    C: ColRef<'_, f64>,
    T: MatRef<'_, f64>,
    s_image: ColRef<'_, f64>,
    mut t_signs_old: ColMut<'_, f64>,
    mut t_signs: ColMut<'_, f64>,
    mut t_image: ColMut<'_, f64>,
    cut: &mut f64,
    parallelism: faer::Parallelism,
    stack: PodStack<'_>,
) -> bool {
    let _ = parallelism;
    let new_cut = s_image.norm_l1();
    if new_cut <= *cut {
        return false;
    } else {
        *cut = new_cut
    }

    t_signs_old.copy_from(&t_signs.rb());
    for i in 0..s_image.nrows() {
        let t = s_image.read(i);
        let one = 1.0f64.to_bits();
        let sign_bit = t.to_bits();
        let signed_one = one | (sign_bit & (1u64 << 63));
        t_signs.write(i, f64::from_bits(signed_one));
    }

    let t_image = t_image.rb_mut().try_as_slice_mut().unwrap();
    let t_signs = t_signs.try_as_slice().unwrap();
    let t_signs_old = t_signs_old.try_as_slice().unwrap();

    mul_add_with_rank_update(
        t_image.as_mut(),
        two_mat,
        S,
        C,
        T,
        t_signs.rb(),
        t_signs_old.rb(),
        stack,
    );

    true
}
