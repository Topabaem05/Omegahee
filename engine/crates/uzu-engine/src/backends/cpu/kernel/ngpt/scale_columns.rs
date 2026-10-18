use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::data_type::DataTypeRepr;

/// In place `data[row * row_stride + column_offset + column] *= scales[column]` for `column < num_cols`.
#[kernel(ScaleColumns)]
#[variants(T, f32, f16, bf16)]
pub fn scale_columns<T: DataTypeRepr + Float>(
    data: *mut T,
    scales: *const f32,
    num_rows: u32,
    num_cols: u32,
    row_stride: u32,
    column_offset: u32,
) {
    assert!(column_offset + num_cols <= row_stride, "scaled columns exceed the row stride");
    for row in 0..num_rows as usize {
        for column in 0..num_cols as usize {
            unsafe {
                let element = data.add(row * row_stride as usize + column_offset as usize + column);
                let scaled = (*element).to_f32().unwrap() * *scales.add(column);
                *element = T::from(scaled).unwrap();
            }
        }
    }
}
