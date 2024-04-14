use faer::Col;

pub struct SparseParameters {
    pub max_iters: usize,
    pub max_bits: usize,
    pub parallelism: faer::Parallelism,
}

impl core::fmt::Display for SparseParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "iters.{}.bits.{}", self.max_iters, self.max_bits)
    }
}

pub struct SparseSct {
    s: Vec<Vec<(usize, Sign)>>,
    t: Vec<Vec<(usize, Sign)>>,
    c: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
pub enum Sign {
    Positive,
    Negative,
}

impl SparseSct {
    pub fn new(nrows: usize, ncols: usize, max_u16s: usize) -> Self {
        Self {
            s: Vec::with_capacity(max_u16s),
            t: Vec::with_capacity(max_u16s),
            c: Vec::with_capacity(max_u16s),
        }
    }

    pub fn extend_with(&mut self, s_signs: &Col<f64>, t_signs: &Col<f64>, cut: f64) {
        todo!()
    }
}
