use num_traits::float::Float;

/// maps the given parameter to a coefficient describing how far along it is between min and max.
pub(crate) fn normalize<V: Float>(p: V, min: V, max: V) -> V {
    (p - min) / (max - min)
}

mod spline;
pub use spline::*;

#[macro_export]
macro_rules! assert_near {
    ($left: expr, $right: expr $(,)?) => {
        assert!(($left - $right).abs() <= TEST_TOLERANCE, "assertion failed: `left` is near `right`
left: {:?},
right: {:?}", $left, $right)
    };
    ($left: expr, $right: expr, $($arg: tt)+) => {
        assert!(($left - $right).abs() <= TEST_TOLERANCE, "assertion failed: `left` is near `right`
left: {:?},
right: {:?}: {}", $left, $right, format_args!($($arg)+))
    };
}
