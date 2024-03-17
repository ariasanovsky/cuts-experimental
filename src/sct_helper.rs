use faer::Col;

pub struct SctHelper {
    s: Vec<u8>,
    t: Vec<u8>,
    c: Vec<f64>,
}

impl SctHelper {
    pub fn new(nrows: usize, ncols: usize, rank: usize) -> Self {
        let s_length = nrows.div_ceil(8);
        let t_length = ncols.div_ceil(8);
        Self {
            s: Vec::with_capacity(s_length * rank),
            t: Vec::with_capacity(t_length * rank),
            c: Vec::with_capacity(rank),
        }
    }

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
}
