use dyn_stack::ReborrowMut;
use faer::{linalg::zip::MatShape, mat::{As2D, As2DMut, AsMatRef}, reborrow::Reborrow, Col, ColMut, Mat, MatRef};

use crate::{
    bit_magic::{cache_parameters_avx2, lazy_matmul_avx2},
    faer::FlatMat,
    inplace_sct_signed::{improve_s, improve_t, SignedCut},
    sct_helper::Sct,
};

#[cfg(test)]
mod tests;

pub struct CutSetLdl {
    // `-s` where `s = nrows * ncols`, the flat dimension of `H`, our space of tensors
    pub(crate) s: f64,
    // `M^T` where `M = L^{-1}` and `X^TX = s * LL^T` (ldl w/ `D = s * I`)
    pub(crate) m_t: Mat<f64>,
    pub(crate) d_recip: Col<f64>,
}

pub struct LdlCut {
    pub squared_frob_decrease: f64,
}

impl CutSetLdl {
    pub fn new(nrows: usize, ncols: usize, rank: usize) -> Self {
        let s = (nrows * ncols) as f64;
        let m_t = Mat::with_capacity(rank, rank);
        let d_recip = Col::zeros(rank);
        Self { s, m_t, d_recip }
    }

    pub fn add_column(
        &mut self,
        projection_buffer: &mut Col<f64>,
        sct: &Sct,
        cut: &SignedCut,
        remainder: &mut FlatMat,
    ) -> LdlCut {
        // unsafe { self.write_temp_column(sct) };
        // TODO! no allocation
        let new_diagonal = unsafe { self.write_temp_columns(sct, projection_buffer) };
        // dbg!(new_diagonal);
        unsafe { self.fill_column(projection_buffer) };
        unsafe { self.adjust_remainder(remainder, sct, cut.value, new_diagonal) };
        LdlCut { squared_frob_decrease: cut.value * cut.value * new_diagonal }
    }

    pub(crate) unsafe fn write_temp_columns(&mut self, sct: &Sct, buffer: &mut Col<f64>) -> f64 {
        let old_rank = self.m_t.nrows();
        let new_rank = old_rank + 1;
        // write z_{k+1} = X_k^T * x_{k+1} to scratch space
        let it = sct.latest_inner_products(old_rank);
        buffer.as_slice_mut().iter_mut().zip(it).for_each(|(b, z)| {
            *b = z
        });
        // write l_{k+1}' = M_k * z_{k+1} to column (scratch space)
        unsafe { self.m_t.set_dims(new_rank, new_rank) };
        let (m_t, col) = self.m_t.as_mut().split_at_col_mut(old_rank);
        let m = m_t.as_ref().subrows(0, old_rank).transpose();
        let (mut col, mut diag) = col.split_at_row_mut(old_rank);
        use faer::linalg::matmul::triangular::{BlockStructure, matmul};
        matmul(
            col.rb_mut(),
            BlockStructure::Rectangular,
            m,
            BlockStructure::UnitTriangularLower,
            buffer.as_2d_ref(),
            BlockStructure::Rectangular,
            None,
            1.0,
            faer::Parallelism::None,
        );
        diag.write(0, 0, 1.0);
        // write l_{k+1} = D_k^{-1} l_{k+1}' to scratch space
        let col = col.as_ref().col(0);
        let diag_recip = self.d_recip.as_ref().subrows(0, old_rank).column_vector_as_diagonal();
        // TODO! allocates
        let l = diag_recip * col;
        buffer.copy_from(l);
        // d_{k+1} = <l_{k+1}', l_{k+1}>
        let d = col.transpose() * buffer.as_ref();
        let s = self.s;
        let d = s - d;
        let d = d.recip();
        self.d_recip.write(old_rank, d);
        d
    }

    pub(crate) unsafe fn fill_column(&mut self, projection: &Col<f64>) {
        let old_rank = self.m_t.nrows() - 1;
        let m_t_top = self.m_t.as_mut().subrows_mut(0, old_rank);
        let (m_t, col) = m_t_top.split_at_col_mut(old_rank);
        let m_t = m_t.as_ref();
        use faer::linalg::matmul::triangular::{BlockStructure, matmul};
        matmul(
            col,
            BlockStructure::Rectangular,
            m_t,
            BlockStructure::UnitTriangularUpper,
            projection.as_2d_ref(),
            BlockStructure::Rectangular,
            None,
            -1.0,
            faer::Parallelism::None,
        );
    }

    pub(crate) unsafe fn adjust_remainder(
        &self,
        remainder: &mut FlatMat,
        sct: &Sct,
        cut: f64,
        new_diagonal: f64,
    ) {
        let nrows = remainder.as_ref().nrows();
        let ncols = remainder.as_ref().ncols();
        let old_rank = self.m_t.nrows() - 1;
        let (k, s, t) = sct.padded_slices();
        // TODO! allocates
        let mut diag: Col<f64> = Col::zeros(k);
        let n = self.m_t.col_as_slice(old_rank);
        let scale = -cut * new_diagonal;
        diag.as_slice_mut().iter_mut().zip(n.iter()).for_each(|(d, n)| {
            *d = *n * scale
        });
        let cache_params = cache_parameters_avx2(nrows, ncols, k);
        lazy_matmul_avx2(
            cache_params,
            nrows,
            ncols,
            k,
            remainder.slice_mut(),
            s,
            diag.as_slice(),
            t,
            crate::bit_magic::Layout::RowMajor,
            dyn_stack::PodStack::new(&mut []),
        );
    }

    // pub fn add_column(&mut self, sct: &Sct, cut: &SignedCut, remainder: &mut FlatMat) -> f64 {
    //     todo!("use `add_column_in_steps`");
    //     let faer_remainder = remainder.as_mut();
    //     let Self { s_recip, m_t } = self;
    //     // we are writing into a new column of `m_t`
    //     // the previous uninitialized data is completely overwritten
    //     let old_rank = m_t.nrows();
    //     let new_rank = old_rank + 1;
    //     unsafe { m_t.set_dims(new_rank, new_rank) };
    //     // z{k+1}       = X{k}^T * x{k+1}
    //     let col_top = m_t.col_as_slice_mut(old_rank);
    //     let it = sct.latest_inner_products(old_rank);
    //     let it = it.chain(core::iter::once(1.0));
    //     col_top.iter_mut().zip(it).for_each(|(a, b)| *a = b);
    //     eprintln!("extended w/ inner products:\n{m_t:?}");
    //     let (m_t_old, mut col) = m_t.as_mut().split_at_col_mut(old_rank);
    //     let m_t_old = m_t_old.as_ref().subrows(0, old_rank);
    //     // l{k+1}'      = M{k}   * z{k+1}
    //     use faer::linalg::matmul::triangular::{matmul, BlockStructure};
    //     // TODO! allocates, use `matmul` instead
    //     let l = m_t_old.transpose() * col.rb().subrows(0, old_rank);
    //     let projection_norm_squared = l.squared_norm_l2() * *s_recip;
    //     let improvement = (cut.value * cut.value) * *s_recip * (1.0 + projection_norm_squared);
    //     dbg!(improvement);
    //     eprintln!("l' = {l:?}");
    //     // m{k+1}       = -s^{-1} * M{k}^T * l{k+1}'
    //     // n{k+1}       = m{k+1} (+) 1
    //     matmul(
    //         col.rb_mut().subrows_mut(0, old_rank),
    //         BlockStructure::Rectangular,
    //         m_t_old.as_ref(),
    //         BlockStructure::UnitTriangularUpper,
    //         l.as_2d_ref(),
    //         BlockStructure::Rectangular,
    //         None,
    //         -*s_recip,
    //         faer::Parallelism::None,
    //     );
    //     // dbg!(col.rb().subrows(0, old_rank).squared_norm_l2());
    //     eprintln!("filled w/ m:\n{m_t:?}");
    //     let n = m_t.as_ref().col(old_rank);
    //     assert_eq!(n.nrows(), new_rank);
    //     eprintln!("n = {n:?}");
    //     // alpha{k+1}  = <x{k+1}, r{k}>
    //     let (k, s, t) = sct.padded_slices();
    //     eprintln!("s = {s:?}");
    //     eprintln!("t = {t:?}");
    //     // TODO! allocates
    //     let mut diag_padded = Col::zeros(k);
    //     let mut diag = diag_padded.as_mut().subrows_mut(0, new_rank);
    //     let scale = faer::scale(- cut.value * *s_recip);
    //     dbg!(cut.value, scale);
    //     let this_allocates = scale * n;
    //     diag.copy_from(this_allocates);
    //     let diag = diag_padded.as_slice();
    //     eprintln!("diag = {diag:?}");
    //     // r{k+1} = r{k} + X{k+1} * n{k+1}'
    //     let nrows = faer_remainder.nrows();
    //     let ncols = faer_remainder.ncols();
    //     let cache_params = cache_parameters_avx2(nrows, ncols, k);
    //     // check majorization requirements
    //     // TODO! pad everything and manage the pointer math better'
    //     // TODO! this requires a few refactors in `Sct`: `0`-initialize buffers and manually manage rank, no Vecs
    //     // dbg!(k);
    //     lazy_matmul_avx2(
    //         cache_params,
    //         nrows,
    //         ncols,
    //         k,
    //         remainder.slice_mut(),
    //         s,
    //         diag,
    //         t,
    //         crate::bit_magic::Layout::RowMajor,
    //         dyn_stack::PodStack::new(&mut []),
    //     );
    //     improvement
    // }
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
