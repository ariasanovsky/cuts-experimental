use dyn_stack::ReborrowMut;
use faer::{linalg::zip::MatShape, mat::As2D, reborrow::Reborrow, Col, Mat, MatRef};

use crate::{
    bit_magic::{cache_parameters_avx2, lazy_matmul_avx2},
    faer::FlatMat,
    inplace_sct_signed::{improve_s, improve_t, SignedCut},
    sct_helper::Sct,
};

mod tests;

pub struct CutSetLdl {
    // `-s` where `s = nrows * ncols`, the flat dimension of `H`, our space of tensors
    s_recip: f64,
    // `M^T` where `M = L^{-1}` and `X^TX = s * LL^T` (ldl w/ `D = s * I`)
    m_t: Mat<f64>,
}

impl CutSetLdl {
    pub fn new(nrows: usize, ncols: usize, rank: usize) -> Self {
        let s = (nrows * ncols) as f64;
        let m_t = Mat::with_capacity(rank, rank);
        Self {
            s_recip: s.recip(),
            m_t,
        }
    }

    pub fn add_column(&mut self, sct: &Sct, cut: &SignedCut, remainder: &mut FlatMat) -> f64 {
        let faer_remainder = remainder.as_mut();
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
        col_top.iter_mut().zip(it).for_each(|(a, b)| *a = b);
        eprintln!("extended w/ inner products:\n{m_t:?}");
        let (m_t_old, mut col) = m_t.as_mut().split_at_col_mut(old_rank);
        let m_t_old = m_t_old.as_ref().subrows(0, old_rank);
        // l{k+1}'      = M{k}   * z{k+1}
        use faer::linalg::matmul::triangular::{matmul, BlockStructure};
        // TODO! allocates, use `matmul` instead
        let l = m_t_old.transpose() * col.rb().subrows(0, old_rank);
        eprintln!("l' = {l:?}");
        // m{k+1}       = -s^{-1} * M{k}^T * l{k+1}'
        // n{k+1}       = m{k+1} (+) 1
        matmul(
            col.rb_mut().subrows_mut(0, old_rank),
            BlockStructure::Rectangular,
            m_t_old.as_ref(),
            BlockStructure::UnitTriangularUpper,
            l.as_2d_ref(),
            BlockStructure::Rectangular,
            None,
            -*s_recip,
            faer::Parallelism::None,
        );
        // dbg!(col.rb().subrows(0, old_rank).squared_norm_l2());
        eprintln!("filled w/ m:\n{m_t:?}");
        let n = m_t.as_ref().col(old_rank);
        assert_eq!(n.nrows(), new_rank);
        eprintln!("n = {n:?}");
        // alpha{k+1}  = <x{k+1}, r{k}>
        let (k, s, t) = sct.padded_slices();
        eprintln!("s = {s:?}");
        eprintln!("t = {t:?}");
        // TODO! allocates
        let mut diag_padded = Col::zeros(k);
        let mut diag = diag_padded.as_mut().subrows_mut(0, new_rank);
        let scale = faer::scale(- cut.value * *s_recip);
        dbg!(cut.value, scale);
        let this_allocates = scale * n;
        diag.copy_from(this_allocates);
        let diag = diag_padded.as_slice();
        eprintln!("diag = {diag:?}");
        // r{k+1}      = r{k} + alpha{k+1}' * X{k+1}^T * n{k+1}
        let nrows = faer_remainder.nrows();
        let ncols = faer_remainder.ncols();
        let cache_params = cache_parameters_avx2(nrows, ncols, k);
        // check majorization requirements
        // TODO! pad everything and manage the pointer math better'
        // TODO! this requires a few refactors in `Sct`: `0`-initialize buffers and manually manage rank, no Vecs
        // dbg!(k);
        lazy_matmul_avx2(
            cache_params,
            nrows,
            ncols,
            k,
            remainder.slice_mut(),
            s,
            diag,
            t,
            crate::bit_magic::Layout::RowMajor,
            dyn_stack::PodStack::new(&mut []),
        );
        0.0
        // todo!()
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
