use dyn_stack::PodStack;
use equator::assert;
use faer::{
    linalg::matmul::matmul,
    reborrow::{Reborrow, ReborrowMut},
    Col, ColMut, ColRef, MatMut, MatRef,
};
use std::iter::zip;

use crate::bit_magic;

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
    pub(crate) t_signs_old: Box<[u8]>,
    pub(crate) s_signs_old: Box<[u8]>,
    pub(crate) t_signs: Box<[u8]>,
    pub(crate) s_signs: Box<[u8]>,
    pub(crate) t_image: Col<f64>,
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
    pub fn new(two_mat: MatRef<'_, f64>, two_mat_transposed: MatRef<'_, f64>) -> Self {
        let nrows = two_mat.nrows();
        let ncols = two_mat.ncols();

        let s_ones = vec![u8::MAX; nrows / 8].into_boxed_slice();
        let t_ones = vec![u8::MAX; ncols / 8].into_boxed_slice();

        let mut s_image = Col::<f64>::zeros(ncols);
        let mut t_image = Col::<f64>::zeros(nrows);

        bit_magic::matvec_bit(
            ncols,
            nrows,
            s_image.as_slice_mut(),
            two_mat_transposed,
            &s_ones,
        );
        bit_magic::matvec_bit(nrows, ncols, t_image.as_slice_mut(), two_mat, &t_ones);
        s_image *= faer::scale(0.5);
        t_image *= faer::scale(0.5);

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
        S: &mut [u8],
        C: ColMut<'_, f64>,
        T: &mut [u8],
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
            C.nrows() > 0,
        ));

        let width = C.nrows() - 1;
        let (S, S_next) = S.split_at_mut(s_signs.len() * width);
        let (T, T_next) = T.split_at_mut(t_signs.len() * width);
        let (C, C_next) = C.split_at_mut(width);
        let S = S.rb();
        let T = T.rb();
        let C = C.rb();
        let C_next = C_next.get_mut(0);

        t_signs_old.copy_from_slice(&t_signs);
        t_signs.iter_mut().for_each(|t_sign| *t_sign = rng.gen());
        mul_add_with_rank_update(
            t_image.as_mut().try_as_slice_mut().unwrap(),
            two_remainder.rb(),
            S,
            C,
            T,
            t_signs,
            t_signs_old,
            stack.rb_mut(),
        );
        let mut cut = 0.0;
        for (&s, &t) in zip(&**s_signs, pulp::as_arrays::<8, _>(t_image.as_slice()).0) {
            for idx in 0..8 {
                let sign = (((s >> idx) & 1 == 0) as u64) << 63;
                cut += f64::from_bits(t[idx].to_bits() ^ sign);
            }
        }

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

        let k = cut / two_remainder.ncols() as f64;
        for (s, &t) in zip(
            pulp::as_arrays_mut::<8, _>(s_image.as_slice_mut()).0,
            &**t_signs,
        ) {
            for idx in 0..8 {
                let sign = (((t >> idx) & 1 == 0) as u64) << 63;
                s[idx] -= f64::from_bits(k.to_bits() ^ sign)
            }
        }
        let k = cut / two_remainder.nrows() as f64;
        for (t, &s) in zip(
            pulp::as_arrays_mut::<8, _>(t_image.as_slice_mut()).0,
            &**s_signs,
        ) {
            for idx in 0..8 {
                let sign = (((s >> idx) & 1 == 0) as u64) << 63;
                t[idx] -= f64::from_bits(k.to_bits() ^ sign)
            }
        }

        *C_next = -2.0 * normalized_cut;
        S_next.copy_from_slice(&s_signs);
        T_next.copy_from_slice(&t_signs);

        cut
    }

    pub fn s_signs(&self) -> &[u8] {
        &self.s_signs
    }

    pub fn t_signs(&self) -> &[u8] {
        &self.t_signs
    }
}

// acc += two * (mat + S * C * T^top) * (x_new - x_old)
// 1. acc += two * mat * (x_new - x_old)
// 2. acc += two * S * C * T^top * (x_new - x_old)
fn mul_add_with_rank_update(
    acc: &mut [f64],
    two_mat: MatRef<'_, f64>,
    S: &[u8],
    C: ColRef<'_, f64>,
    T: &[u8],
    x_new: &[u8],
    x_old: &[u8],
    stack: PodStack<'_>,
) {
    struct Impl<'a> {
        two_mat: MatRef<'a, f64>,
        S: &'a [u8],
        C: ColRef<'a, f64>,
        T: &'a [u8],
        x_new: &'a [u8],
        x_old: &'a [u8],
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
            let width = T.len() / x_new.len();

            let (diff_indices, stack) = stack.make_raw::<u64>(n);

            let mut pos = 0usize;
            for j in 0..n {
                let s_pos = (x_new[j / 8] >> (j % 8)) & 1 == 1;
                let s_pos_old = (x_old[j / 8] >> (j % 8)) & 1 == 1;
                let col = two_mat.col(j).try_as_slice().unwrap();

                if s_pos != s_pos_old {
                    diff_indices[pos] = ((s_pos as u64) << 63) | j as u64;
                    pos += 1;

                    if s_pos {
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
                let col = &T[j * n / 8..][..n / 8];
                let mut acc = 0i64;
                for &i in diff_indices {
                    let positive = (i >> 63) == 1;
                    let i = i & ((1 << 63) - 1);
                    let i = i as usize;
                    let positive2 = (col[i / 8] >> (i % 8)) & 1 == 1;
                    if positive == positive2 {
                        acc += 1;
                    } else {
                        acc -= 1;
                    }
                }
                y[j] = acc as f64;
            }
            // y = C * y
            for (y, &c) in zip(&mut *y, C.try_as_slice().unwrap()) {
                *y *= c;
            }
            // acc += S * y
            bit_magic::matvec(acc.len(), y.len(), acc, S, y);
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
    S: &[u8],
    C: ColRef<'_, f64>,
    T: &[u8],
    s_image: ColRef<'_, f64>,
    t_signs_old: &mut [u8],
    t_signs: &mut [u8],
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

    t_signs_old.copy_from_slice(&t_signs.rb());
    let s_image = s_image.try_as_slice().unwrap();
    for (i, t) in t_signs.iter_mut().enumerate() {
        let mut sign = 0u8;
        sign |= ((s_image[8 * i + 0].to_bits() >> 63) as u8) << 0;
        sign |= ((s_image[8 * i + 1].to_bits() >> 63) as u8) << 1;
        sign |= ((s_image[8 * i + 2].to_bits() >> 63) as u8) << 2;
        sign |= ((s_image[8 * i + 3].to_bits() >> 63) as u8) << 3;
        sign |= ((s_image[8 * i + 4].to_bits() >> 63) as u8) << 4;
        sign |= ((s_image[8 * i + 5].to_bits() >> 63) as u8) << 5;
        sign |= ((s_image[8 * i + 6].to_bits() >> 63) as u8) << 6;
        sign |= ((s_image[8 * i + 7].to_bits() >> 63) as u8) << 7;
        *t = !sign;
    }

    let t_image = t_image.rb_mut().try_as_slice_mut().unwrap();

    mul_add_with_rank_update(
        t_image.as_mut(),
        two_mat,
        S,
        C,
        T,
        t_signs,
        t_signs_old,
        stack,
    );

    true
}
