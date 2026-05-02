pub mod mandelbrot;
pub mod newton;

pub use mandelbrot::MandelbrotMap;
pub use newton::{find_roots, iterate_newton, newton_tile_pixel, NewtonMap, NewtonResult};
