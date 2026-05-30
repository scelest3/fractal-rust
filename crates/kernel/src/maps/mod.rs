pub mod julia;
pub mod mandelbrot;
pub mod multibrot;
pub mod newton;

pub use julia::JuliaMap;
pub use mandelbrot::MandelbrotMap;
pub use multibrot::MultibroMap;
pub use newton::{find_roots, iterate_newton, newton_tile_pixel, NewtonMap, NewtonResult};
