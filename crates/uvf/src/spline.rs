use crate::{normalize, normalize_dpdt};
use num_traits::{float::Float, FromPrimitive};
use smallvec::{smallvec, SmallVec};

const DEFAULT_MAX_CURVES: usize = 6;
const POW_DOMAIN: i32 = 15;

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

/// S implements interpolation between a series of control points,
/// defined in input/parameter space (denoted `t`) and output space (denoted `y`).
///
/// An empty spline is considered invalid and may result in panics.
#[derive(Debug, Clone)]
pub struct S<V: Float = f32, const M: usize = DEFAULT_MAX_CURVES> {
    /// lower_t describes the lower t value of the ith interval.
    ///
    /// Values of cp_t must always be ascending.
    lower_t: SmallVec<[V; M]>,
    /// lower_y describes the lower value of the ith interval.
    lower_y: SmallVec<[V; M]>,
    /// cp_t describes the control points of the ith interval.
    cp_t: SmallVec<[(V, V); M]>,
}

impl<V: Float + std::fmt::Debug + std::ops::SubAssign + FromPrimitive> S<V> {
    /// identity returns a spline where the output value is the same as the input value.
    ///
    /// To keep numeric stability, this is defined over the domain -2^POW_DOMAIN~2^POW_DOMAIN.
    pub fn identity() -> Self {
        let two = V::one() + V::one();
        let three = two + V::one();
        let max = (two).powi(POW_DOMAIN); // 2^15

        Self {
            lower_t: smallvec![-max, V::zero(), max],
            lower_y: smallvec![-max, V::zero(), max],
            cp_t: smallvec![
                (max.div(three) - max, (two * max.div(three) - max)),
                (max.div(three), two * max.div(three))
            ],
        }
    }

    /// identity_smol returns a spline where the output value is the same as the input value.
    /// This spline has a single curve segment.
    pub fn identity_smol() -> Self {
        let two = V::one() + V::one();
        let max = (two).powi(POW_DOMAIN); // 2^15

        Self {
            lower_t: smallvec![-max, max],
            lower_y: smallvec![-max, max],
            cp_t: smallvec![(-max.div(two), max.div(two))],
        }
    }

    /// returns the index of the control point preceeding the given
    #[inline(always)]
    pub(crate) fn ith_floor(&self, t: V) -> Option<usize> {
        self.lower_t.iter().rposition(|lower_t| *lower_t <= t)
    }

    /// the parameter t must be normalized within a segment (i.e.: 0 <= t <= 1).
    #[inline(always)]
    fn basis_functions(t: V) -> (V, V, V, V) {
        let two = V::one() + V::one();
        let three = two + V::one();

        // Bernstein polynomials of degree 4
        let b0 = (V::one() - t).powi(3);
        let b1 = three * t * (V::one() - t).powi(2);
        let b2 = three * t.powi(2) * (V::one() - t);
        let b3 = t.powi(3);
        (b0, b1, b2, b3)
    }

    /// computes the output value of the spline for the given input.
    pub fn eval(&self, t: V) -> V {
        let i = self.ith_floor(t);
        match i {
            // NOTE: clamping when out of bounds
            None => self.lower_y[0],
            Some(i) => {
                if i >= self.lower_t.len() - 1 {
                    return self.lower_y[self.lower_t.len() - 1]; // clamping when out of bounds
                };

                let ((y0, y1, y2, y3), (t0, t3)) = self.coeffs_for_interval(i);

                let t = normalize(t, t0, t3);
                // println!(
                //     "[{:?}]\n T:\t{:?} => {:?}\tnorm: {:?}\n Y:\t{:?} => {:?} => {:?} => {:?}",
                //     i, t0, t3, t, y0, y1, y2, y3
                // );

                let (b0, b1, b2, b3) = S::basis_functions(t);
                let out = (b0 * y0) + (b1 * y1) + (b2 * y2) + (b3 * y3);

                // println!(" Out:\t{:?}", out);
                out
            }
        }
    }

    /// computes the derivative of the spline's output (y) with respect to t.
    pub fn dydt(&self, t: V) -> V {
        let i = self.ith_floor(t);
        match i {
            None => V::zero(), // OOB
            Some(i) => {
                if i >= self.lower_t.len() - 1 {
                    return V::zero(); // OOB
                };

                // The partial derivative of a bezier spline with respect to its input
                // is itself a bezier spline of one degree less. This spline is:
                //
                // sum( Bx-1(t) * (Yx+1 - Yx) )
                let ((y0, y1, y2, y3), (t0, t3)) = self.coeffs_for_interval(i);
                let y0 = y1 - y0;
                let y1 = y2 - y1;
                let y2 = y3 - y2;

                let two = V::one() + V::one();
                let three = two + V::one();

                let norm_dt = normalize_dpdt(t0, t3);
                let t = normalize(t, t0, t3);
                // Bernstein polynomials of degree 3
                let b0 = (V::one() - t).powi(2);
                let b1 = two * t * (V::one() - t);
                let b2 = t.powi(2);

                three * (b0 * y0 + b1 * y1 + b2 * y2) * norm_dt
            }
        }
    }

    /// Returns value for each control point, and the bounds of the interval in input space.
    ///
    /// SAFETY: Will panic if the interval is out of bounds or is the last interval.
    #[inline(always)]
    fn coeffs_for_interval(&self, i: usize) -> ((V, V, V, V), (V, V)) {
        let (t0, y0) = (self.lower_t[i], self.lower_y[i]);
        let (t3, y3) = (self.lower_t[i + 1], self.lower_y[i + 1]);
        let (y1, y2) = self.cp_t[i];

        ((y0, y1, y2, y3), (t0, t3))
    }

    /// The parameter space this function is defined for
    #[inline(always)]
    pub fn t_domain(&self) -> (V, V) {
        (self.lower_t[0], self.lower_t[self.lower_t.len() - 1])
    }

    /// The number of points in the spline.
    pub fn num_points(&self) -> usize {
        self.lower_t.len()
    }

    /// Scales the output values of the spline by the given scalar.
    pub fn scale_y(&mut self, amt: V) {
        self.lower_y.iter_mut().for_each(|y| *y = amt * *y);
        self.cp_t
            .iter_mut()
            .for_each(|p| *p = (amt * p.0, amt * p.1));
    }

    /// Adjusts the spline based on some error, and the input + observed output value.
    pub fn adjust(&mut self, params: &Params, t: V, error: V) {
        // TODO: avoid searching for correct interval?
        if let Some(i) = self.ith_floor(t) {
            let (_, (t0, t3)) = self.coeffs_for_interval(i);
            let (b0, b1, b2, b3) = S::basis_functions(normalize(t, t0, t3)); // TODO: Avoid normalizing?

            let lr = V::from_f32(params.learning_rate).unwrap();

            self.lower_y[i] -= error * lr * b0;
            self.lower_y[i + 1] -= error * lr * b3;
            self.cp_t[i].0 -= error * lr * b1;
            self.cp_t[i].1 -= error * lr * b2;

            // For the sake of learning smooth functions, normalize the sister control point
            // across the interval to have the same tangent as well.
            let two = V::one() + V::one();
            if i + 1 < self.cp_t.len() {
                let mix = ((self.lower_y[i + 1] - self.cp_t[i].1)
                    + (self.cp_t[i + 1].0 - self.lower_y[i + 1]))
                    / two;
                self.cp_t[i].1 = -mix;
                self.cp_t[i + 1].0 = mix;
            }
            if i > 0 {
                let mix = ((self.lower_y[i] - self.cp_t[i - 1].1)
                    + (self.cp_t[i].0 - self.lower_y[i]))
                    / two;
                self.cp_t[i - 1].1 = -mix;
                self.cp_t[i].0 = mix;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_near;

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

        assert_near!(S::<f32>::identity_smol().eval(0.0), 0.0);
        assert_near!(S::<f32>::identity_smol().eval(-0.0), -0.0);
    }

    #[test]
    fn eval_identity_one() {
        assert_near!(S::<f32>::identity().eval(1.0), 1.0);
        assert_near!(S::<f32>::identity().eval(-1.0), -1.0);

        const TEST_TOLERANCE: f32 = 0.5;
        assert_near!(S::<f32>::identity_smol().eval(1.0), 1.0);
        assert_near!(S::<f32>::identity_smol().eval(-1.0), -1.0);
    }

    #[test]
    fn eval_identity_five() {
        assert_near!(S::<f32>::identity().eval(5.0), 5.0);
        assert_near!(S::<f32>::identity().eval(-5.0), -5.0);

        const TEST_TOLERANCE: f32 = 0.8;
        assert_near!(S::<f32>::identity_smol().eval(5.0), 5.0);
        assert_near!(S::<f32>::identity_smol().eval(-5.0), -5.0);
    }

    #[test]
    fn eval_identity_about_bounds() {
        let max = 2f32.powi(POW_DOMAIN);
        // at the bounds
        assert_near!(S::<f32>::identity().eval(max), max);
        assert_near!(S::<f32>::identity().eval(-max), -max);

        assert_near!(S::<f32>::identity().eval(max - 10.73), max - 10.729);
        assert_near!(S::<f32>::identity().eval(10.75 - max), 10.748 - max);
    }

    #[test]
    fn eval_identity_clamped() {
        let max = 2f32.powi(POW_DOMAIN);
        assert_near!(S::<f32>::identity().eval(500000.0), max);
        assert_near!(S::<f32>::identity().eval(-500000.0), -max);
    }

    #[test]
    fn adjust_train_loop() {
        let mut s = S::<f32>::identity();
        let p = Params {
            learning_rate: 0.2,
            ..Params::default()
        };

        // Toy training loop: f(-10000) = 10000, f(0) = 0, f(10000) = 10000
        let pos = -10000.0;
        for _ in 0..1000 {
            let out = s.eval(pos);
            s.adjust(&p, pos, out - 10000.0);
            let pos = 0.0;
            let out = s.eval(pos);
            s.adjust(&p, pos, out);
            let pos = 10000.0;
            let out = s.eval(pos);
            s.adjust(&p, pos, out + 10000.0);
        }

        const TEST_TOLERANCE: f32 = 2.0;
        assert_near!(s.eval(-10000.0), 10000.0);
        assert_near!(s.eval(0.0), 0.0);
        assert_near!(s.eval(10000.0), -10000.0);
    }

    #[test]
    fn dydt_identity() {
        assert_near!(S::<f32>::identity().dydt(1.0), 1.0);
        assert_near!(S::<f32>::identity().dydt(-1.0), 1.0);

        const TEST_TOLERANCE: f32 = 0.5;
        assert_near!(S::<f32>::identity_smol().dydt(1.0), 1.0);
        assert_near!(S::<f32>::identity_smol().dydt(-1.0), 1.0);
    }

    #[test]
    fn dydt_linear() {
        // Make the spline double the input: f(x) = 2x.
        let mut s = S::<f32>::identity();
        s.scale_y(2.0);
        assert_near!(s.dydt(1.0), 2.0);
        assert_near!(s.dydt(0.0), 2.0);
        assert_near!(s.dydt(-1.0), 2.0);
        const TEST_TOLERANCE: f32 = 0.5;
        let mut s = S::<f32>::identity_smol();
        s.scale_y(2.0);
        assert_near!(s.dydt(1.0), 2.0);
        assert_near!(s.dydt(0.0), 2.0);
        assert_near!(s.dydt(-1.0), 2.0);
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
