use num_traits::float::Float;
use smallvec::{smallvec, SmallVec};

const DEFAULT_MAX_GRIDS: usize = 6;
const POW_DOMAIN: i32 = 15;

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
    /// lower_t describes the lower t value of the ith interval.
    ///
    /// Values of cp_t must always be ascending.
    lower_t: SmallVec<[V; M]>,
    /// lower_y describes the lower value of the ith interval.
    lower_y: SmallVec<[V; M]>,
    /// cp_t describes the control point of the ith interval.
    cp_t: SmallVec<[V; M]>,
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
            lower_t: smallvec![-max, V::zero(), max],
            lower_y: smallvec![-max, V::zero(), max],
            cp_t: smallvec![-mid, mid],
        }
    }

    /// returns the index of the control point preceeding the given
    pub(crate) fn ith_floor(&self, t: V) -> Option<usize> {
        self.lower_t.iter().rposition(|lower_t| *lower_t <= t)
    }

    /// computes the output value of the spline for the given input.
    pub fn eval(&self, t: V) -> V {
        let i = self.ith_floor(t);
        match i {
            // NOTE: clamping when out of bounds
            None => self.lower_y[0],
            Some(i) => {
                let n = self.lower_t.len() - 1;
                let (t0, y0) = (self.lower_t[i], self.lower_y[i]);
                let (t1, y1) = if i + 1 <= n {
                    (self.lower_t[i + 1], self.lower_y[i + 1])
                } else {
                    return y0; // NOTE: clamping when out of bounds
                };

                let yc = self.cp_t[i];

                let t = normalize(t, t0, t1);
                println!(
                    "[{:?}]\n T:\t{:?} => {:?}\tnorm: {:?}\n Y:\t{:?} => {:?} => {:?}",
                    i, t0, t1, t, y0, yc, y1
                );

                // let lerp1 = y0 + (yc - y0) * t;
                // let lerp2 = yc + (y1 - yc) * t;
                // lerp1 + (lerp2 - lerp1) * t

                // let one_t = V::one() - t;
                // let one_t2 = one_t * one_t;
                // // Le quadratic formular-ayy
                // let out = yc + (y0 - yc) * one_t2 + (y1 - yc) * t * t;

                // Bernstein polynomials of degree 3
                let b0 = (V::one() - t).powi(2);
                let b1 = (V::one() + V::one()) * t * (V::one() - t);
                let b2 = t.powi(2);
                let out = (b0 * y0) + (b1 * yc) + (b2 * y1);

                println!(" Out:\t{:?}", out);
                out
            }
        }
    }

    /// The parameter space this function is defined for
    pub fn t_domain(&self) -> (V, V) {
        (self.lower_t[0], self.lower_t[self.lower_t.len() - 1])
    }

    /// The number of points in the spline.
    pub fn num_points(&self) -> usize {
        self.lower_t.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_TOLERANCE: f32 = 1.0e-7;

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
