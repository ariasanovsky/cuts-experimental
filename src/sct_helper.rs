use faer::Col;

pub struct Sct {
    s: Vec<u8>,
    t: Vec<u8>,
    c: Vec<f64>,
    dims: SctDims,
}

pub struct SctDims {
    nrows: usize,
    ncols: usize,
    rank: usize,
}

impl SctDims {
    pub fn num_s_bytes(&self) -> usize {
        self.nrows.div_ceil(8)
    }

    pub fn num_t_bytes(&self) -> usize {
        self.ncols.div_ceil(8)
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn rank(&self) -> usize {
        self.rank
    }
}

impl Sct {
    pub fn new(nrows: usize, ncols: usize, rank: usize) -> Self {
        let dims = SctDims { nrows, ncols, rank };
        Self {
            s: Vec::with_capacity(dims.num_s_bytes() * rank),
            t: Vec::with_capacity(dims.num_t_bytes() * rank),
            c: Vec::with_capacity(rank),
            dims: SctDims { nrows, ncols, rank },
        }
    }

    /// Encodes both column vectors `[x{i}]_{i}` of floats as a binary sequence [b{i}]_{i}.
    /// For each `i`, `b{i} = 1` iff `x{i} == 1.0`, and otherwise `b{i} = 0`.
    pub fn extend_with(&mut self, s_signs: &Col<f64>, t_signs: &Col<f64>, cut: f64) {
        let s_chunks = s_signs.as_slice().chunks(8);
        self.s.extend(s_chunks.map(|s_chunk| {
            let mut byte = 0;
            for (i, sign) in s_chunk.iter().enumerate() {
                if *sign == -1.0 {
                    byte ^= 1 << i
                }
            }
            byte
        }));
        let t_chunks = t_signs.as_slice().chunks(8);
        self.t.extend(t_chunks.map(|t_chunk| {
            let mut byte = 0;
            for (i, sign) in t_chunk.iter().enumerate() {
                if *sign == -1.0 {
                    byte ^= 1 << i
                }
            }
            byte
        }));
        self.c.push(cut);
    }

    pub fn s(&self) -> &[u8] {
        self.s.as_slice()
    }

    pub fn t(&self) -> &[u8] {
        self.t.as_slice()
    }

    pub fn c(&self) -> &[f64] {
        self.c.as_slice()
    }

    pub fn dimensions(&self) -> &SctDims {
        &self.dims
    }
}
