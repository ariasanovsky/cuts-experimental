use cuts::faer::FlatMat;
// use faer::Mat;
use half::bf16;
use safetensors::{tensor::TensorView, Dtype};

pub fn flat_mat(view: &TensorView) -> FlatMat {
    let &[nrows, ncols]: &[usize; 2] = view.shape().try_into().unwrap();
    assert_eq!(view.dtype(), Dtype::BF16);
    let mut mat = FlatMat::zeros(nrows, ncols);
    let data = view.data();
    let data: &[u16] = bytemuck::try_cast_slice(data).unwrap();
    let data: &[bf16] = unsafe { core::mem::transmute(data) };
    assert_eq!(data.len(), nrows * ncols);
    let rows = data.chunks_exact(ncols);
    rows.enumerate().for_each(|(i, row)| {
        row.iter().enumerate().for_each(|(j, x)| {
            // mat[(i, j)] = bf16::to_f64(*x)
            mat.write(i, j, bf16::to_f64(*x))
        })
    });
    mat
}
