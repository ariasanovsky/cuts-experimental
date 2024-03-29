use faer::{Col, Mat, MatMut};

use crate::{inplace_sct_signed::SignedCut, sct_helper::Sct};

pub struct CutSetLdl {
    // `nrows * ncols`, the flat dimension of `H`, our space of tensors
    s: f64,
    // `- s * M^T` where `M = L^{-1}` and `X^TX = s * LL^T` (ldl w/ `D = s * I`)
    m_t: Mat<f64>,
}

impl CutSetLdl {
    pub fn new(nrows: usize, ncols: usize, rank: usize) -> Self {
        let s = (nrows * ncols) as f64;
        let m_t = Mat::with_capacity(rank, rank);
        Self { s, m_t }
    }

    pub fn add_column(&mut self, sct: &Sct) -> f64 {
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