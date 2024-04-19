pub mod faer;
pub mod hira_helpers;
pub mod safetensors;
#[cfg(feature = "experimenting_with_safetensors_view")]
pub mod safetensors_view;
pub mod sct_map;
pub mod sparse_safetensors;

pub fn half_point<F>(nrows: usize, ncols: usize) -> usize {
    let f_size: usize = core::mem::size_of::<F>();
    let numerator = f_size * nrows * ncols;
    let denominator = 2 * (nrows.div_ceil(8) + ncols.div_ceil(8) + f_size);
    numerator / denominator
}
