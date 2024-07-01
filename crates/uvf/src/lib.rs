use num_traits::{float::Float, FromPrimitive};

/// maps the given parameter to a coefficient describing how far along it is between min and max.
pub(crate) fn normalize<V: Float>(p: V, min: V, max: V) -> V {
    (p - min) / (max - min)
}

/// derivative of the normalize function.
pub(crate) fn normalize_dpdt<V: Float>(min: V, max: V) -> V {
    V::one().div(max - min)
}

/// Describes the hyperparameters when training a spline.
pub struct Params {
    pub learning_rate: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
        }
    }
}

/// A function able to learn through backpropagation.
pub trait Trainable<V>
where
    V: Float + std::fmt::Debug + FromPrimitive,
{
    /// Computes the output for the given input parameter.
    fn eval(&self, t: V) -> V;
    /// Computes the derivative of the output with respect to the input.
    fn dtdy(&self, t: V) -> V;
    /// Updates the function given learning parameters, an input parameter value and the error
    /// or loss of the output.
    fn adjust(&mut self, params: &Params, t: V, error: V);
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
