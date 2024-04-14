use faer::MatMut;

use crate::inplace_sct_signed::CutHelper;

pub mod helpers;

impl CutHelper {
    pub fn sparse_cut_mat(
        &mut self,
        remainder: MatMut<f64>,
        rng: &mut impl rand::Rng,
        max_iters: usize,
        parallelism: faer::Parallelism,
    ) -> SparseCut {
        todo!()
    }
}

pub struct SparseCut {
    pub value: f64,
}

impl SparseCut {
    pub fn num_bits(&self) -> usize {
        todo!()
    }
}
