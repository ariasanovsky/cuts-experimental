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
    pub fn new(rank: usize) -> Self {
        Self {
            s: Vec::with_capacity(rank),
            t: Vec::with_capacity(rank),
            c: Vec::with_capacity(rank),
        }
    }

    pub fn extend_with(&mut self, s_signs: &Col<f64>, t_signs: &Col<f64>, cut: f64) {
        let Self {
            s,
            t,
            c,
        } = self;
        let s_signs = s_signs.as_slice().iter().enumerate().filter_map(|(i, s)| {
            match s.partial_cmp(&0.0).unwrap() {
                std::cmp::Ordering::Less => Some((i, Sign::Negative)),
                std::cmp::Ordering::Equal => None,
                std::cmp::Ordering::Greater => Some((i, Sign::Positive)),
            }
        }).collect();
        s.push(s_signs);
        let t_signs = t_signs.as_slice().iter().enumerate().filter_map(|(i, t)| {
            match t.partial_cmp(&0.0).unwrap() {
                std::cmp::Ordering::Less => Some((i, Sign::Negative)),
                std::cmp::Ordering::Equal => None,
                std::cmp::Ordering::Greater => Some((i, Sign::Positive)),
            }
        }).collect();
        t.push(t_signs);
        c.push(cut);
    }
}
