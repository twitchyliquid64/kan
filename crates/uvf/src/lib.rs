use num_traits::float::Float;
use smallvec::{smallvec, SmallVec};

const DEFAULT_MAX_GRIDS: usize = 6;

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

/// Spline implements quadratic interpolation between a series of control points,
/// defined in input/parameter space (denoted `t`) and output space (denoted `y`).
///
/// An empty spline is considered invalid and may result in panics.
#[derive(Debug, Clone)]
pub struct Spline<V: Float = f32, const M: usize = DEFAULT_MAX_GRIDS> {
    /// cp_t describes the t value of the ith control point.
    /// The control points relevant to computing a curve are the ith control point,
    /// the previous control point, and the subsequent control point.
    ///
    /// Values of cp_t must always be ascending.
    cp_t: SmallVec<[V; M]>,

    /// cp_y describes the output value of the ith control point.
    cp_y: SmallVec<[V; M]>,
}

impl<V: Float + std::fmt::Debug, const M: usize> Spline<V, M> {
    pub fn identity() -> Self {
        let max = V::one();// (V::one() + V::one()).powi(2); // 2^2

        Self {
            cp_t: smallvec![-max, V::zero(), max],
            cp_y: smallvec![-max, V::zero(), max],
        }
    }

    /// returns the index of the control point preceeding the given
    /// value in input space, if any.
    pub(crate) fn ith_floor(&self, t: V) -> Option<usize> {
        self.cp_t.iter().rposition(|cp_t| *cp_t <= t)
    }

    /// computes the output value of the spline for the given input.
    pub fn eval(&self, t: V) -> V {
        println!();
        let reciprocal_or_zero = |v: V| {
            if !v.is_normal() || v.abs() <= (V::epsilon() + V::epsilon()) {
                V::zero()
            } else {
                V::one() / v
            }
        };

        let i = self.ith_floor(t);
        match i {
            // NOTE: clamping when out of bounds
            None => self.cp_y[0],
            Some(i) => {
                let t_center = self.cp_t[i];
                let y_center = self.cp_y[i];

                let (t_prev, y_prev) = if i == 0 {
                    (t_center, y_center)
                } else {
                    (self.cp_t[i - 1], self.cp_y[i - 1])
                };
                let (t_next, y_next) = if i == self.cp_t.len() - 1 {
                    (t_center, y_center)
                } else {
                    (self.cp_t[i + 1], self.cp_y[i + 1])
                };

                println!("t: ({:?}, {:?}, {:?}) - {:?}", t_prev, t_center, t_next, t);

                let b_prev = (t - t_center).powi(2)
                    * reciprocal_or_zero((t_prev - t_center) * (t_prev - t_next));
                let b_center = ((t - t_prev) * (t - t_next))
                    * reciprocal_or_zero((t_center - t_prev) * (t_center - t_next));
                let b_next = (t - t_center).powi(2)
                    * reciprocal_or_zero((t_next - t_center) * (t_next - t_prev + V::one()));

                println!("b: ({:?}, {:?}, {:?})", b_prev, b_center, b_next);
                println!("y: ({:?}, {:?}, {:?})", y_prev, self.cp_y[i], y_next);

                (y_prev * b_prev) + (y_center * b_center) + (y_next * b_next)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_TOLERANCE: f32 = 1.0e-6;

    #[test]
    fn ith_floor() {
        let spline = Spline::<f32>::identity();
        assert_eq!(spline.ith_floor(0.0), Some(1));
        assert_eq!(spline.ith_floor(-0.1), Some(0));
        assert_eq!(spline.ith_floor(0.1), Some(1));
    }

    #[test]
    fn eval_identity_zero() {
        assert_near!(Spline::<f32>::identity().eval(0.0), 0.0);
    }

    #[test]
    fn eval_identity_one() {
        assert_near!(Spline::<f32>::identity().eval(1.0), 1.0);
    }
}
