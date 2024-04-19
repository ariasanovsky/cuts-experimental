use safetensors::View;

pub struct ColumnMajorSlice<T> {
    array: Vec<T>,
    shape: [usize; 2],
}

impl<T> ColumnMajorSlice<T> {
    pub fn new(array: Vec<T>, nrows: usize, ncols: usize) -> Self {
        assert_eq!(array.len(), nrows * ncols);
        Self {
            array,
            shape: [nrows, ncols],
        }
    }
}

impl<T: bytemuck::Pod> View for ColumnMajorSlice<T> {
    fn dtype(&self) -> safetensors::Dtype {
        safetensors::Dtype::BF16
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> std::borrow::Cow<[u8]> {
        bytemuck::cast_slice(self.array.as_slice()).into()
    }

    fn data_len(&self) -> usize {
        self.array.len()
    }
}
