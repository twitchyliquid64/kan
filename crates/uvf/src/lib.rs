use num_traits::float::Float;
use splines::{Interpolate, Interpolation, Key, Spline};

const POW_DOMAIN: i32 = 10;

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

/// S implements interpolation between a series of control points,
/// defined in input/parameter space (denoted `t`) and output space (denoted `y`).
///
/// An empty spline is considered invalid and may result in panics.
#[derive(Debug, Clone)]
pub struct S<V: Interpolate<V> + Float = f32> {
    s: Spline<V, V>,
}

impl<V: Float + std::fmt::Debug + Interpolate<V> + splines::interpolate::Interpolator> S<V> {
    /// identity returns a spline where the output value is the same as the input value.
    ///
    /// To keep numeric stability, this is defined over the domain -2^POW_DOMAIN~2^POW_DOMAIN.
    pub fn identity() -> Self {
        let max = (V::one() + V::one()).powi(POW_DOMAIN); // 2^10

        let start = Key::new(-max, -max, Interpolation::Bezier(V::zero()));
        let end = Key::new(max, max, Interpolation::Linear);

        Self {
            s: Spline::from_vec(vec![start, end]),
        }
    }

    /// non_linear returns a spline with non-linear input to output characteristics.
    ///
    /// To keep numeric stability, this is defined over the domain -2^POW_DOMAIN~2^POW_DOMAIN.
    pub fn non_linear() -> Self {
        let two = V::one() + V::one();
        let max = two.powi(POW_DOMAIN); // 2^10

        let start = Key::new(-max, -max, Interpolation::Bezier(-two.powi(POW_DOMAIN - 1)));
        let end = Key::new(max, max, Interpolation::Linear);

        Self {
            s: Spline::from_vec(vec![start, end]),
        }
    }

    /// computes the output value of the spline for the given input.
    pub fn eval(&self, t: V) -> V {
        self.s.clamped_sample(t).unwrap()
    }

    /// invert flips the sign of the parameter space.
    ///
    /// For instance, an identity function would become a negation function.
    pub fn invert(&mut self) {
        self.s = Spline::from_iter(self.s.into_iter().map(|k| {
            let mut k = *k;
            k.t = k.t.neg();
            if let Interpolation::Bezier(v) = &mut k.interpolation {
                *v = v.neg();
            }
            k
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_TOLERANCE: f32 = 1.0e-5;

    #[test]
    fn eval_identity_zero() {
        assert_near!(S::<f32>::identity().eval(0.0), 0.0);
        assert_near!(S::<f32>::identity().eval(-0.0), -0.0);
    }

    #[test]
    fn eval_identity_one() {
        assert_near!(S::<f32>::identity().eval(1.0), 1.0);
        assert_near!(S::<f32>::identity().eval(-1.0), -1.0);
    }

    #[test]
    fn eval_identity_five() {
        assert_near!(S::<f32>::identity().eval(5.0), 5.0);
        assert_near!(S::<f32>::identity().eval(-5.0), -5.0);
    }

    #[test]
    fn eval_identity_clamped() {
        assert_near!(S::<f32>::identity().eval(5000.0), 1024.0);
        assert_near!(S::<f32>::identity().eval(-5000.0), -1024.0);
    }

    #[test]
    fn eval_non_linear() {
        let mut s = S::<f32>::non_linear();
        assert_near!(s.eval(0.0), -256.0);
        assert_near!(s.eval(256.0), 16.0);
        assert_near!(s.eval(512.0), 320.0);
        s.invert();
        assert_near!(s.eval(0.0), 0.);
        assert_near!(s.eval(256.0), -256.0);
        assert_near!(s.eval(512.0), -512.0);
    }

    #[test]
    fn invert_identity() {
        let mut s = S::<f32>::identity();
        assert_near!(s.eval(1.0), 1.0);
        s.invert();
        assert_near!(s.eval(-1.0), 1.0);
        assert_near!(s.eval(1.0), -1.0);
        assert_near!(s.eval(3.5), -3.5);
    }
}
