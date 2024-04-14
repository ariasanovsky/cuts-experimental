use faer::Col;

pub struct SparseParameters {
    pub max_iters: usize,
    pub max_bits: usize,
    pub parallelism: faer::Parallelism,
}

impl core::fmt::Display for SparseParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

pub struct SparseSct {

}

impl SparseSct {
    pub fn new(nrows: usize, ncols: usize, max_u16s: usize) -> Self {
        todo!()
    }

    pub fn extend_with(&mut self, s_signs: &Col<f64>, t_signs: &Col<f64>, cut: f64) {
        todo!()
    }
}
