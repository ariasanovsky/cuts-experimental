use faer::{Col, Mat, MatMut};

use crate::{inplace_sct_signed::SignedCut, sct_helper::Sct};

pub struct CutSetLdl {
    // `nrows * ncols`, the flat dimension of `H`, our space of tensors
    s: f64,
    // `- s * M^T` where `M = L^{-1}` and `(X^TX) = s * LL^T`
    m_t: Mat<f64>,
}

impl CutSetLdl {
    pub fn new(nrows: usize, ncols: usize, rank: usize) -> Self {
        todo!()
    }

    pub fn add_column(&mut self, sct: &Sct) -> f64 {
        todo!()
    }
}

pub struct CutSetBuffers {
    
}

impl CutSetBuffers {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        todo!()
    }

    pub fn write_cut(
        &mut self,
        remainder: MatMut<f64>,
        rng: &mut impl rand::Rng,
    ) -> SignedCut {
        todo!()
    }

    pub fn s_signs(&self) -> &Col<f64> {
        todo!()
    }

    pub fn t_signs(&self) -> &Col<f64> {
        todo!()
    }
}