use faer::{linalg::zip::MatShape, reborrow::Reborrow, Col, Mat, MatMut, MatRef};

use crate::{inplace_sct_signed::{improve_s, improve_t, SignedCut}, sct_helper::Sct};

pub struct CutSetLdl {
    // `-s` where `s = nrows * ncols`, the flat dimension of `H`, our space of tensors
    negative_s: f64,
    // `- s * M^T` where `M = L^{-1}` and `X^TX = s * LL^T` (ldl w/ `D = s * I`)
    m_t: Mat<f64>,
}

impl CutSetLdl {
    pub fn new(nrows: usize, ncols: usize, rank: usize) -> Self {
        let s = (nrows * ncols) as f64;
        let m_t = Mat::with_capacity(rank, rank);
        Self { negative_s: s, m_t }
    }

    pub fn add_column(&mut self, sct: &Sct, cut: &SignedCut, remainder: MatMut<f64>) -> f64 {
        let Self { negative_s, m_t } = self;
        // we are writing into a new column of `m_t`
        // the previous uninitialized data is completely overwritten
        let old_rank = m_t.nrows();
        debug_assert_eq!(old_rank, m_t.ncols());
        let new_rank = old_rank + 1;
        debug_assert!(m_t.col_capacity() >= new_rank);
        debug_assert!(m_t.row_capacity() >= new_rank);
        unsafe {
            m_t.set_dims(new_rank, new_rank);
            m_t.write_unchecked(old_rank, old_rank, *negative_s);
        };
        let (m_t, col) = m_t.as_mut().split_at_col_mut(old_rank);
        debug_assert_eq!(m_t.ncols(), old_rank);
        debug_assert_eq!(m_t.nrows(), new_rank);
        debug_assert_eq!(col.ncols(), 1);
        debug_assert_eq!(col.nrows(), new_rank);
        let m_t = m_t.as_ref().subrows(0, old_rank);
        debug_assert_eq!(m_t.ncols(), old_rank);
        debug_assert_eq!(m_t.nrows(), old_rank);
        let col = col.subrows_mut(0, old_rank);
        debug_assert_eq!(col.ncols(), 1);
        debug_assert_eq!(col.nrows(), old_rank);
        todo!()
    }
}

pub struct CutSetBuffers {
    s_signs: Col<f64>,
    s_image: Col<f64>,
    t_signs: Col<f64>,
    t_image: Col<f64>,
}

impl CutSetBuffers {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            s_signs: Col::zeros(nrows),
            s_image: Col::zeros(ncols),
            t_signs: Col::zeros(ncols),
            t_image: Col::zeros(nrows),
        }
    }

    pub fn write_cut(
        &mut self,
        remainder: MatRef<f64>,
        rng: &mut impl rand::Rng,
        max_iterations: usize,
    ) -> SignedCut {
        let Self {
            s_signs,
            s_image,
            t_signs,
            t_image,
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
                // parallelism,
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
                // parallelism,
            );
            if !improved_input {
                break;
            }
        }
        cut
    }

    pub fn s_signs(&self) -> &Col<f64> {
        &self.s_signs
    }

    pub fn t_signs(&self) -> &Col<f64> {
        &self.t_signs
    }
}