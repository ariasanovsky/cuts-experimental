pub struct FlatMat<'a> {
    nums: &'a mut [f64],
    nrows: usize,
    ncols: usize,
}