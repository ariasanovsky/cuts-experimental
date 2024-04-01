use faer::{linalg::zip::MatShape, reborrow::Reborrow, Col, ColRef, Mat, MatRef};
use rand::thread_rng;

use crate::{faer::FlatMat, sct_helper::Sct};

use super::{CutSetBuffers, CutSetLdl};

use equator::assert;

#[test]
fn ldl_sct_with_identity_matrix_of_size_8() {
    let mut remainder = identity_matrix(8);
    test_decompose_matrix(&mut remainder, 5);
}


struct SignMatrices {
    s_signs: Vec<Col<f64>>,
    t_signs: Vec<Col<f64>>,
}

impl SignMatrices {
    fn new() -> Self {
        Self {
            s_signs: vec![],
            t_signs: vec![],
        }
    }

    fn push(&mut self, cutset: &CutSetBuffers) {
        self.s_signs.push(cutset.s_signs().clone());
        self.t_signs.push(cutset.t_signs().clone());
    }

    fn inner_products(&self) -> Mat<f64> {
        let Self { s_signs, t_signs } = self;
        let rank = self.s_signs.len();
        Mat::from_fn(rank, rank, |row, col| {
            let s_prod = s_signs[row].transpose() * &s_signs[col];
            let t_prod = t_signs[row].transpose() * &t_signs[col];
            s_prod * t_prod
        })
    }

    fn iter(&self) -> impl Iterator<Item = (&Col<f64>, &Col<f64>)> + '_ {
        let s_signs = self.s_signs.iter();
        let t_signs = self.t_signs.iter();
        s_signs.zip(t_signs)
    }
}

fn identity_matrix(size: usize) -> FlatMat {
    let mut remainder = FlatMat::zeros(size, size);
    for i in 0..size {
        remainder.write(i, i, 1.0)
    }
    remainder
}

fn test_decompose_matrix(remainder: &mut FlatMat, rank: usize) {
    let nrows = remainder.as_ref().nrows();
    let ncols = remainder.as_ref().ncols();
    let mut ldl = CutSetLdl::new(nrows, ncols, rank);
    let mut sct = Sct::new(nrows, ncols, rank);
    let mut cutset = CutSetBuffers::new(nrows, ncols);
    let mut rng = thread_rng();
    let mut signs = SignMatrices::new();
    let s = (nrows * ncols) as f64;
    for r in 1..=rank {
        println!("r = {r}");
        println!("{:?}", remainder.as_ref());
        let mut cut = cutset.write_cut(remainder.as_ref(), &mut rng, 1_000);
        let mut pos_s = vec![];
        let mut neg_s = vec![];
        for (i, &s) in cutset.s_signs().as_slice().iter().enumerate() {
            if s == 1.0 {
                pos_s.push(i)
            } else if s == -1.0 {
                neg_s.push(i)
            } else {
                unreachable!("s_{r}[{i}] != +/- 1.0")
            }
        }
        let mut pos_t = vec![];
        let mut neg_t = vec![];
        for (i, &t) in cutset.t_signs().as_slice().iter().enumerate() {
            if t == 1.0 {
                pos_t.push(i)
            } else if t == -1.0 {
                neg_t.push(i)
            } else {
                unreachable!("t_{r}[{i}] != +/- 1.0")
            }
        }
        println!("s+, s- = {pos_s:?}, {neg_s:?}");
        println!("t+, t- = {pos_t:?}, {neg_t:?}");
        signs.push(&cutset);
        let sllt = signs.inner_products();
        println!("{sllt:?}");
        sct.extend_with(cutset.s_signs(), cutset.t_signs(), cut.value);
        let improvement = ldl.add_column(&sct, &cut, remainder);
        let m = ldl.m();
        assert!(all(
            m.nrows() == r,
            m.ncols() == r,
        ));
        let mtm = m.transpose() * m;
        for (i, (s, t)) in signs.iter().enumerate() {
            let prod = s.transpose() * remainder.as_ref() * t;
            assert!(prod == 0.0, "i = {i}")
        }
        let mtm = m.transpose() * m;
        let prod = &mtm * &sllt;
        // dbg!(&prod);
        test_is_identity_scaled(prod.as_ref(), s);
    }
}

fn test_is_identity_scaled(mat: MatRef<f64>, scale: f64) {
    assert!(mat.nrows() == mat.ncols());
    for row in 0..mat.nrows() {
        for col in 0..mat.ncols() {
            let value = if row == col {
                scale
            } else {
                0.0
            };
            assert!(mat[(row, col)] == value);
        }
    }
}

impl CutSetLdl {
    fn m(&self) -> MatRef<f64> {
        let Self { s_recip, m_t } = self;
        m_t.as_ref()
    }
}