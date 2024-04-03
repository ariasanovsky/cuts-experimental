use faer::{Mat, MatMut, MatRef};

#[derive(Clone)]
pub struct FlatMat {
    nums: aligned_vec::AVec<f64>,
    nrows: usize,
    ncols: usize,
}

impl FlatMat {
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        assert!(nrows % 8 == 0);
        assert!(ncols % 8 == 0);
        Self {
            nums: aligned_vec::avec![0.0; nrows * ncols],
            nrows,
            ncols,
        }
    }

    pub fn write(&mut self, row: usize, col: usize, value: f64) {
        let pos = self.nrows * col + row;
        self.nums[pos] = value;
    }

    pub fn as_ref(&self) -> MatRef<f64> {
        faer::mat::from_column_major_slice(self.nums.as_slice(), self.nrows, self.ncols)
    }

    pub fn as_mut(&mut self) -> MatMut<f64> {
        faer::mat::from_column_major_slice_mut(self.nums.as_mut_slice(), self.nrows, self.ncols)
    }

    pub fn slice_mut(&mut self) -> &mut [f64] {
        &mut self.nums
    }

    pub fn to_faer(self) -> Mat<f64> {
        // TODO! ?allocates
        faer::mat::from_column_major_slice::<f64>(self.nums.as_slice(), self.nrows, self.ncols).to_owned()
    }
}
