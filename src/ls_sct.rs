use dyn_stack::ReborrowMut;
use faer::{mat::As2D, reborrow::Reborrow, Col, Mat, MatRef};

use crate::{bit_magic::{cache_parameters_avx2, lazy_matmul_avx2}, inplace_sct_signed::{improve_s, improve_t, SignedCut}, sct_helper::Sct};

pub struct CutSetLdl {
    // `-s` where `s = nrows * ncols`, the flat dimension of `H`, our space of tensors
    s_recip: f64,
    // `- s * M^T` where `M = L^{-1}` and `X^TX = s * LL^T` (ldl w/ `D = s * I`)
    m_t: Mat<f64>,
}

impl CutSetLdl {
    pub fn new(nrows: usize, ncols: usize, rank: usize) -> Self {
        let s = (nrows * ncols) as f64;
        let m_t = Mat::with_capacity(rank, rank);
        Self { s_recip: s.recip(), m_t }
    }

    pub fn add_column(&mut self, sct: &Sct, cut: &SignedCut, remainder: &mut Mat<f64>) -> f64 {
        let Self { s_recip, m_t } = self;
        // we are writing into a new column of `m_t`
        // the previous uninitialized data is completely overwritten
        let old_rank = m_t.nrows();
        let new_rank = old_rank + 1;
        unsafe { m_t.set_dims(new_rank, new_rank) };
        // z{k+1}       = X{k}^T * x{k+1}
        let col_top = m_t.col_as_slice_mut(old_rank);
        let it = sct.latest_inner_products(old_rank);
        let it = it.chain(core::iter::once(1.0));
        col_top
            .iter_mut()
            .zip(it)
            .for_each(|(a, b)| {
                *a = b
            });
        let (m_t_old, mut col) = m_t.as_mut().split_at_col_mut(old_rank);
        let m_t_old = m_t_old.as_ref().subrows(0, old_rank);
        // l{k+1}'      = M{k}   * z{k+1}
        use faer::linalg::matmul::triangular::{BlockStructure, matmul, matmul_with_conj};
        // TODO! allocates, use `matmul` instead
        let l = m_t_old.as_ref() * col.rb().subrows(0, old_rank);
        // m{k+1}       = -s^{-1} * M{k}^T * l{k+1}'
        // n{k+1}       = m{k+1} (+) 1
        matmul_with_conj(
            col.rb_mut().subrows_mut(0, old_rank),
            BlockStructure::Rectangular,
            m_t_old.as_ref(),
            BlockStructure::UnitTriangularUpper,
            faer::Conj::Yes,
            l.as_2d_ref(),
            BlockStructure::Rectangular,
            faer::Conj::No,
            None,
            -*s_recip,
            faer::Parallelism::None,
        );
        let n = m_t.as_ref().col(old_rank);
        // alpha{k+1}  = <x{k+1}, r{k}>
        let alpha = cut.value;
        // alpha{k+1}' = -alpha{k+1} / s
        let alpha = -alpha * *s_recip;
        // n{k+1}'     = alpha{k+1}' * n{k+1}
        // TODO! allocates
        let diag = faer::scale(alpha) * n;
        let diag = diag.as_slice();
        // r{k+1}      = r{k} + alpha{k+1}' * X{k+1}^T * n{k+1}
        let nrows = remainder.nrows();
        let ncols = remainder.ncols();
        let cache_params = cache_parameters_avx2(nrows, ncols, new_rank);
        // check majorization requirements
        lazy_matmul_avx2(
            cache_params,
            nrows,
            ncols,
            old_rank,
            todo!(),
            sct.s(),
            diag,
            sct.t(),
            crate::bit_magic::Layout::RowMajor,
            dyn_stack::PodStack::new(&mut []),
        );
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
