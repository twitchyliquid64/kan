use num_traits::float::Float;
use smallvec::{smallvec, SmallVec};

const DEFAULT_MAX_GRIDS: usize = 6;
const POW_DOMAIN: i32 = 12;

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

/// maps the given parameter to a coefficient describing how far along it is between min and max.
fn normalize<V: Float>(p: V, min: V, max: V) -> V {
    (p - min) / (max - min)
}

/// S implements interpolation between a series of control points,
/// defined in input/parameter space (denoted `t`) and output space (denoted `y`).
///
/// An empty spline is considered invalid and may result in panics.
#[derive(Debug, Clone)]
pub struct S<V: Float = f32, const M: usize = DEFAULT_MAX_GRIDS> {
    /// cp_t describes the t value of the ith control point.
    /// The control points relevant to computing a curve are the ith control point,
    /// the previous control point, and the subsequent control point.
    ///
    /// Values of cp_t must always be ascending.
    cp_t: SmallVec<[V; M]>,

    /// cp_y describes the output value of the ith control point.
    cp_y: SmallVec<[V; M]>,
}

impl<V: Float + std::fmt::Debug> S<V> {
    /// identity returns a spline where the output value is the same as the input value.
    ///
    /// To keep numeric stability, this is defined over the domain -2^POW_DOMAIN~2^POW_DOMAIN.
    pub fn identity() -> Self {
        let two = V::one() + V::one();
        let max = (two).powi(POW_DOMAIN); // 2^12
        let mid = max.div(two);

        Self {
            cp_t: smallvec![-max, -mid, V::zero(), mid, max],
            cp_y: smallvec![-max, -mid, V::zero(), mid, max],
        }
    }

    /// returns the index of the control point preceeding the given
    /// value in input space, if any.
    pub(crate) fn ith_floor(&self, t: V) -> Option<usize> {
        self.cp_t.iter().rposition(|cp_t| *cp_t <= t)
    }

    /// computes the output value of the spline for the given input.
    pub fn eval(&self, t: V) -> V {
        let i = self.ith_floor(t);
        match i {
            // NOTE: clamping when out of bounds
            None => self.cp_y[0],
            Some(i) => {
                let n = self.cp_t.len() - 1;
                let (t0, y0) = (self.cp_t[i], self.cp_y[i]);
                let (tc, yc) = if i + 1 <= n {
                    (self.cp_t[i + 1], self.cp_y[i + 1])
                } else {
                    return y0; // NOTE: clamping when out of bounds
                };
                let (t1, y1) = if i + 2 <= n {
                    (self.cp_t[i + 2], self.cp_y[i + 2])
                } else {
                    (tc, yc)
                };

                let t = normalize(t, t0, t1);
                // println!(
                //     "[{:?}]\n T:\t{:?} => {:?} => {:?}\tnorm: {:?}\n Y:\t{:?} => {:?} => {:?}",
                //     i, t0, tc, t1, t, y0, yc, y1
                // );

                let one_t = V::one() - t;
                let one_t2 = one_t * one_t;
                // Le quadratic formular-ayy
                yc + (y0 - yc) * one_t2 + (y1 - yc) * t * t
            }
        }
    }

    /// The parameter space this function is defined for
    pub fn t_domain(&self) -> (V, V) {
        (self.cp_t[0], self.cp_t[self.cp_t.len() - 1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_TOLERANCE: f32 = 1.0e-6;

    #[test]
    fn _normalize() {
        assert_near!(normalize(1f32, 1.0, 2.0), 0.0);
        assert_near!(normalize(2f32, 1.0, 2.0), 1.0);
        assert_near!(normalize(2f32, 1.0, 3.0), 0.5);
        assert_near!(normalize(3f32, 1.0, 5.0), 0.5);
        assert_near!(normalize(4f32, 1.0, 5.0), 0.75);
    }

    #[test]
    fn t_domain_identity() {
        let max = 2f32.powi(POW_DOMAIN);
        assert_near!(S::<f32>::identity().t_domain().0, -max);
        assert_near!(S::<f32>::identity().t_domain().1, max);
    }

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
    fn eval_identity_about_bounds() {
        let max = 2f32.powi(POW_DOMAIN);
        // at the bounds
        assert_near!(S::<f32>::identity().eval(max), max);
        assert_near!(S::<f32>::identity().eval(-max), -max);

        // TODO: non-linear near the bounds, is that okay?
        // assert_near!(S::<f32>::identity().eval(max - 10.75), max - 10.75);
        // assert_near!(S::<f32>::identity().eval(10.75 - max), 10.75 - max);
    }

    #[test]
    fn eval_identity_clamped() {
        let max = 2f32.powi(POW_DOMAIN);
        assert_near!(S::<f32>::identity().eval(500000.0), max);
        assert_near!(S::<f32>::identity().eval(-500000.0), -max);
    }

    // #[test]
    // fn eval_non_linear() {
    //     let mut s = S::<f32>::non_linear();
    //     assert_near!(s.eval(0.0), -256.0);
    //     assert_near!(s.eval(256.0), 16.0);
    //     assert_near!(s.eval(512.0), 320.0);
    //     s.invert();
    //     assert_near!(s.eval(0.0), 0.);
    //     assert_near!(s.eval(256.0), -256.0);
    //     assert_near!(s.eval(512.0), -512.0);
    // }
}
