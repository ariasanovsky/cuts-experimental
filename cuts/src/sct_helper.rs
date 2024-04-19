use std::ops::AddAssign;

use faer::Col;

pub struct Sct {
    s: Box<[u8]>,
    t: Box<[u8]>,
    c: Box<[f64]>,
    dims: SctDims,
    current_rank: usize,
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
            s: vec![0; (dims.num_s_bytes() * rank).next_multiple_of(4)].into_boxed_slice(),
            t: vec![0; (dims.num_t_bytes() * rank).next_multiple_of(4)].into_boxed_slice(),
            c: vec![0.0; rank].into_boxed_slice(),
            dims: SctDims { nrows, ncols, rank },
            current_rank: 0,
        }
    }

    /// Encodes both column vectors `[x{i}]_{i}` of floats as a binary sequence [b{i}]_{i}.
    /// For each `i`, `b{i} = 1` iff `x{i} == 1.0`, and otherwise `b{i} = 0`.
    pub fn extend_with(&mut self, s_signs: &Col<f64>, t_signs: &Col<f64>, cut: f64) {
        let s_chunks = s_signs.as_slice().chunks(8);
        let s_bytes = s_chunks.map(|s_chunk| {
            let mut byte = 0;
            for (i, sign) in s_chunk.iter().enumerate() {
                if *sign == -1.0 {
                    byte ^= 1 << i
                }
            }
            byte
        });
        let s_next = &mut self.s[self.current_rank * self.dims.num_s_bytes()..];
        s_bytes.zip(s_next.iter_mut()).for_each(|(s_new, s_empty)| {
            *s_empty = s_new
        });
        // let t_chunks = t_signs.as_slice().chunks(8);
        // self.t.extend(t_chunks.map(|t_chunk| {
        //     let mut byte = 0;
        //     for (i, sign) in t_chunk.iter().enumerate() {
        //         if *sign == -1.0 {
        //             byte ^= 1 << i
        //         }
        //     }
        //     byte
        // }));
        let t_chunks = t_signs.as_slice().chunks(8);
        let t_bytes = t_chunks.map(|t_chunk| {
            let mut byte = 0;
            for (i, sign) in t_chunk.iter().enumerate() {
                if *sign == -1.0 {
                    byte ^= 1 << i
                }
            }
            byte
        });
        let t_next = &mut self.t[self.current_rank * self.dims.num_t_bytes()..];
        t_bytes.zip(t_next.iter_mut()).for_each(|(t_new, t_empty)| {
            *t_empty = t_new
        });
        
        // self.c.push(cut);
        self.c[self.current_rank] = cut;
        self.current_rank.add_assign(1);
    }

    pub fn s(&self) -> &[u8] {
        &self.s[..self.current_rank * self.dims.num_s_bytes()]
    }

    pub fn t(&self) -> &[u8] {
        &self.t[..self.current_rank * self.dims.num_t_bytes()]
        // todo!();
        // self.t.as_slice()
    }

    pub fn c(&self) -> &[f64] {
        &self.c[..self.current_rank]
        // todo!();
        // self.c.as_slice()
    }

    pub fn padded_slices(&self) -> (usize, &[u8], &[u8]) {
        let k = self.current_rank.next_multiple_of(4);
        (
            k,
            &self.s[..k * self.dims.num_s_bytes()],
            &self.t[..k * self.dims.num_t_bytes()],
        )
    }
    
    pub fn dimensions(&self) -> &SctDims {
        &self.dims
    }

    pub(crate) fn latest_inner_products(&self, old_rank: usize) -> impl Iterator<Item = f64> + '_ {
        assert_eq!(old_rank + 1, self.current_rank);
        // dbg!(self.s.len(), self.t.len(), &self.dims);
        // pointer math for s
        let s_chunk = self.dims.num_s_bytes();
        let s_split = s_chunk * old_rank;
        let (s_old, s_new) = self.s.split_at(s_split);
        let s_old = s_old.chunks_exact(s_chunk);
        // pointer math for t
        let t_chunk = self.dims.num_t_bytes();
        let t_split = t_chunk * old_rank;
        let (t_old, t_new) = self.t.split_at(t_split);
        let t_old = t_old.chunks_exact(t_chunk);
        let num_s_bits = self.dims.nrows as isize;
        let num_t_bits = self.dims.ncols as isize;
        let foo = s_old.zip(t_old).map(move |(s_old, t_old)| {
            let s_negative_ones = s_old
                .iter()
                .zip(s_new.iter())
                .map(|(s_old, s_new)| (*s_old ^ *s_new).count_ones())
                .sum::<u32>() as isize;
            let s_dot = num_s_bits - 2 * s_negative_ones;
            let t_negative_ones = t_old
                .iter()
                .zip(t_new.iter())
                .map(|(t_old, t_new)| (*t_old ^ *t_new).count_ones())
                .sum::<u32>() as isize;
            let t_dot = num_t_bits - 2 * t_negative_ones;
            // dbg!(s_dot, t_dot);
            // todo!();
            (s_dot * t_dot) as f64
        });
        // let it = core::iter::empty();
        // todo!();
        foo
    }
}
