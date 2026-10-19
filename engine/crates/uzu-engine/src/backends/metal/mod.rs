mod backend;
mod buffer;
mod capture;
mod command_buffer;
mod context;
mod decompression;
mod error;
mod gpu_types;
mod heaps;
mod kernel;
#[cfg(test)]
pub(crate) use kernel::matmul::{GemvTile, TILE_OVERRIDE};
mod metal_extensions;

pub use backend::Metal;
