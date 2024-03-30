use faer::Col;

pub struct Sct {
    s: Vec<u8>,
    t: Vec<u8>,
    c: Vec<f64>,
    dims: SctDims,
}

#[derive(Debug)]
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

    pub(crate) fn latest_inner_products(&self, old_rank: usize) -> impl Iterator<Item = f64> + '_ {
        dbg!(self.s.len(), self.t.len(), &self.dims);
        // pointer math for s
        let s_chunk = self.dims.num_s_bytes();
        let s_split = s_chunk * old_rank;
        let (s_old, s_new) = self.s.as_slice().split_at(s_split);
        let s_old = s_old.chunks_exact(s_chunk);
        // pointer math for t
        let t_chunk = self.dims.num_t_bytes();
        let t_split = t_chunk * old_rank;
        let (t_old, t_new) = self.t.as_slice().split_at(t_split);
        let t_old = t_old.chunks_exact(t_chunk);
        let num_s_bits = self.dims.nrows as isize;
        let num_t_bits = self.dims.ncols as isize;
        let foo = s_old.zip(t_old).map(move |(s_old, t_old)| {
            let s_negative_ones = s_old.iter().zip(s_new.iter()).map(|(s_old, s_new)| {
                (*s_old ^ *s_new).count_ones()
            }).sum::<u32>() as isize;
            let s_dot = num_s_bits - 2 * s_negative_ones;
            let t_negative_ones = t_old.iter().zip(t_new.iter()).map(|(t_old, t_new)| {
                (*t_old ^ *t_new).count_ones()
            }).sum::<u32>() as isize;
            let t_dot = num_t_bits - 2 * t_negative_ones;
            dbg!(s_negative_ones, s_dot, t_negative_ones, t_old);
            todo!();
            (s_dot * t_dot) as f64
        });
        // let it = core::iter::empty();
        // todo!();
        foo
    }
}
