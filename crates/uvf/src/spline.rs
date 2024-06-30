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

/// Bez implements bezier interpolation between a series of control points,
/// defined in input/parameter space (denoted `t`) and output space (denoted `y`).
///
/// An empty spline is considered invalid and may result in panics.
#[derive(Debug, Clone)]
pub struct Bez<V: Float = f32, const M: usize = DEFAULT_MAX_CURVES> {
    /// lower_t describes the lower t value of the ith interval.
    ///
    /// Values of cp_t must always be ascending.
    lower_t: SmallVec<[V; M]>,
    /// lower_y describes the lower value of the ith interval.
    lower_y: SmallVec<[V; M]>,
    /// cp_t describes the control points of the ith interval.
    cp_t: SmallVec<[(V, V); M]>,
}

impl<V: Float + std::fmt::Debug + std::ops::AddAssign + std::ops::SubAssign + FromPrimitive>
    Bez<V>
{
    // TODO: TESTS
    pub fn new(min: V, max: V, segments: usize) -> Self {
        let per_segment = (max - min) / V::from(segments).unwrap();
        let mut lower_t = SmallVec::with_capacity(segments + 1);
        let mut lower_y = SmallVec::with_capacity(segments + 1);
        let mut cp_t = SmallVec::with_capacity(segments);

        for n in 0..segments {
            let l = min + per_segment * V::from(n).unwrap();
            lower_t.push(l);
            lower_y.push(l);

            let s3 = per_segment / V::from(3usize).unwrap();
            cp_t.push((l + s3, l + s3 + s3));
        }

        lower_t.push(max);
        lower_y.push(max);

        Self {
            lower_t,
            lower_y,
            cp_t,
        }
    }

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

                let (b0, b1, b2, b3) = Bez::basis_functions(t);
                let out = (b0 * y0) + (b1 * y1) + (b2 * y2) + (b3 * y3);

                // println!(" Out:\t{:?}", out);
                out
            }
        }
    }

    /// computes the derivative of the spline's output (y) with respect to t.
    pub fn dtdy(&self, mut t: V) -> V {
        let i = match self.ith_floor(t) {
            None => {
                t = V::zero();
                0
            }
            Some(i) => {
                if i >= self.lower_t.len() - 1 {
                    t = V::one();
                    self.lower_t.len() - 2
                } else {
                    i
                }
            }
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

    /// The output space this function is defined for
    #[inline(always)]
    pub fn y_domain(&self) -> (V, V) {
        (self.lower_y[0], self.lower_y[self.lower_y.len() - 1])
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

    /// the parameter t must be normalized within a segment (i.e.: 0 <= t <= 1).
    #[allow(dead_code)]
    #[inline(always)]
    fn dt_basis_functions(t: V) -> (V, V, V, V) {
        let two = V::one() + V::one();
        let three = two + V::one();
        let four = three + V::one();

        let b0 = -three * (t - V::one()).powi(2);
        let b1 = three * (three * t.powi(2) - four * t + V::one());
        let b2 = -three * (three * t.powi(2) - two * t);
        let b3 = three * t.powi(2);
        (b0, b1, b2, b3)
    }

    #[inline(always)]
    fn grad_clamp(v: V) -> V {
        let max_spread = V::from(10.).unwrap();
        v.max(-max_spread).min(max_spread)
    }

    /// Adjusts the spline based on some error, and the input value.
    pub fn adjust(&mut self, params: &Params, t: V, error: V) {
        if !error.is_normal() {
            println!("skipping non-normal error {:?}", error);
            return;
        }

        // TODO: avoid searching for correct interval?
        if let Some(i) = self.ith_floor(t) {
            if i >= self.lower_t.len() - 1 {
                // self.lower_y[i] -= error * V::from_f32(params.learning_rate).unwrap();
                return;
            };
            let (_, (t0, t3)) = self.coeffs_for_interval(i);

            // Use the basis function to apply the error to each parameter.
            let lr = V::from_f32(params.learning_rate).unwrap();
            let t = normalize(t, t0, t3);
            let (b0, b1, b2, b3) = Bez::basis_functions(t);
            let calc = |b: V| error * Bez::grad_clamp(b * lr);

            let two = V::one() + V::one(); // CP's on interval boundaries get updated twice as frequently
            self.lower_y[i] -= calc(b0) / two;
            self.lower_y[i + 1] -= calc(b3) / two;
            self.cp_t[i].0 -= calc(b1);
            self.cp_t[i].1 -= calc(b2);

            // let two = V::one() + V::one();

            // Lets try moving the adjacent control point too.
            // if i + 1 < self.cp_t.len() {
            //     let amt = self.lower_y[i + 1] - self.cp_t[i].1;
            //     self.cp_t[i + 1].0 = (self.cp_t[i + 1].0 + amt) / two;
            // }
            // if i > 0 {
            //     let amt = self.cp_t[i].0 - self.lower_y[i];
            //     self.cp_t[i - 1].1 = (self.cp_t[i - 1].1 - amt) / two;
            // }

            // // For the sake of learning smooth functions, normalize the sister control point
            // // across the interval to have the same tangent as well.
            // if i + 1 < self.cp_t.len() {
            //     let mix = ((self.lower_y[i + 1] - self.cp_t[i].1)
            //         + (self.cp_t[i + 1].0 - self.lower_y[i + 1]))
            //         / two;
            //     self.cp_t[i].1 = self.lower_y[i] - mix;
            //     self.cp_t[i + 1].0 = self.lower_y[i] + mix;
            // }
            // if i > 0 {
            //     let mix = ((self.lower_y[i] - self.cp_t[i - 1].1)
            //         + (self.cp_t[i].0 - self.lower_y[i]))
            //         / two;
            //     self.cp_t[i - 1].1 = self.lower_y[i] - mix;
            //     self.cp_t[i].0 = self.lower_y[i] + mix;
            // }
        }
    }

    /// Adjusts each parameter based on the value returned from the given callback.
    pub fn dither_y<F: FnMut() -> V>(&mut self, mut cb: F) {
        for y in self.lower_y.iter_mut() {
            *y = *y + cb();
        }
        for y in self.cp_t.iter_mut() {
            y.0 = y.0 + cb();
            y.1 = y.1 + cb();
        }
    }

    pub fn control_points(&self) -> Vec<(V, V)> {
        let mut out = Vec::with_capacity(self.lower_t.len() * 3 + 1);
        for (i, lt) in self.lower_t.iter().enumerate() {
            out.push((*lt, self.lower_y[i]));
            if i < self.lower_t.len() - 1 {
                let t3 = self.lower_t[i + 1];
                out.push((
                    *lt + V::from(1.0 / 3.0).unwrap() * (t3 - *lt),
                    self.cp_t[i].0,
                ));
                out.push((
                    *lt + V::from(2.0 / 3.0).unwrap() * (t3 - *lt),
                    self.cp_t[i].1,
                ));
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_near;
    use rand::{rngs::StdRng, Rng, SeedableRng};

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
        assert_near!(Bez::<f32>::identity().t_domain().0, -max);
        assert_near!(Bez::<f32>::identity().t_domain().1, max);
    }

    #[test]
    fn eval_identity_zero() {
        assert_near!(Bez::<f32>::identity().eval(0.0), 0.0);
        assert_near!(Bez::<f32>::identity().eval(-0.0), -0.0);

        assert_near!(Bez::<f32>::identity_smol().eval(0.0), 0.0);
        assert_near!(Bez::<f32>::identity_smol().eval(-0.0), -0.0);
    }

    #[test]
    fn eval_identity_one() {
        assert_near!(Bez::<f32>::identity().eval(1.0), 1.0);
        assert_near!(Bez::<f32>::identity().eval(-1.0), -1.0);

        const TEST_TOLERANCE: f32 = 0.5;
        assert_near!(Bez::<f32>::identity_smol().eval(1.0), 1.0);
        assert_near!(Bez::<f32>::identity_smol().eval(-1.0), -1.0);
    }

    #[test]
    fn eval_identity_five() {
        assert_near!(Bez::<f32>::identity().eval(5.0), 5.0);
        assert_near!(Bez::<f32>::identity().eval(-5.0), -5.0);

        const TEST_TOLERANCE: f32 = 0.8;
        assert_near!(Bez::<f32>::identity_smol().eval(5.0), 5.0);
        assert_near!(Bez::<f32>::identity_smol().eval(-5.0), -5.0);
    }

    #[test]
    fn eval_identity_about_bounds() {
        let max = 2f32.powi(POW_DOMAIN);
        // at the bounds
        assert_near!(Bez::<f32>::identity().eval(max), max);
        assert_near!(Bez::<f32>::identity().eval(-max), -max);

        assert_near!(Bez::<f32>::identity().eval(max - 10.73), max - 10.729);
        assert_near!(Bez::<f32>::identity().eval(10.75 - max), 10.748 - max);
    }

    #[test]
    fn eval_identity_clamped() {
        let max = 2f32.powi(POW_DOMAIN);
        assert_near!(Bez::<f32>::identity().eval(500000.0), max);
        assert_near!(Bez::<f32>::identity().eval(-500000.0), -max);
    }

    #[test]
    fn adjust_trivial() {
        let mut s = Bez::<f32>::new(-5.0, 5.0, 2);
        let p = Params {
            learning_rate: 0.05,
            ..Params::default()
        };

        println!("{:?}", s);
        let target = 15.0;
        for _ in 0..3000 {
            let out = s.eval(0.0);
            let delta = out - target;
            s.adjust(&p, 0.0, delta);
        }
        println!("{:?}", s);

        const TEST_TOLERANCE: f32 = 1.0e-4;
        assert_near!(s.eval(0.0), 15.0);
    }

    #[test]
    fn adjust_trivial_invert() {
        let mut s = Bez::<f32>::new(-5.0, 5.0, 1);
        s.scale_y(-1.0);
        let p = Params {
            learning_rate: 0.4,
            ..Params::default()
        };

        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..2000 {
            let target: f32 = rng.gen_range(-5.0..=5.0);
            let out = s.eval(target);
            let delta = out - target;
            s.adjust(&p, target, delta);
        }

        const TEST_TOLERANCE: f32 = 0.02;
        assert_near!(s.eval(0.0), 0.0);
        assert_near!(s.eval(5.0), 5.0);
        assert_near!(s.eval(-3.0), -3.0);
    }

    #[test]
    fn adjust_nontrivial() {
        let mut s = Bez::<f32>::new(-20000.0, 20000.0, 4);
        let p = Params {
            learning_rate: 0.1,
            ..Params::default()
        };

        // Toy training loop: f(-10000) = 10000, f(0) = 0, f(10000) = 10000
        for _ in 0..500 {
            let pos = -10000.0;
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
    /// Configures a network like this:
    ///   Input x--> Spline 1 (learnable) -->
    ///   Input y--> Spline 2 (learnable) -->
    ///                                       Spline 3 (learnable) --> Output
    ///
    /// With a desired function of: f(x, y) = x/y, domain [1, 3]
    fn adjust_div_network() {
        let mut rng = StdRng::seed_from_u64(4);
        let mut x = Bez::new(0.7, 3.5, 3);
        let mut y = Bez::new(0.7, 3.5, 3);
        let mut c = Bez::new(0.5, 7.5, 3);

        let params = &Params { learning_rate: 0.1 };

        // Train
        for _ in 0..95000 {
            let x_input: f32 = rng.gen_range(1.0..3.0);
            let y_input: f32 = rng.gen_range(1.0..3.0);
            let x_output = x.eval(x_input);
            let y_output = y.eval(y_input);
            let output = c.eval(x_output + y_output);

            let target = x_input / y_input;
            let error_der = output - target;
            let c_der = c.dtdy(x_output + y_output);

            c.adjust(params, x_output + y_output, error_der);
            x.adjust(params, x_input, error_der * c_der);
            y.adjust(params, y_input, error_der * c_der);
        }

        const TEST_TOLERANCE: f32 = 0.1;
        assert_near!(c.eval(x.eval(2.0) + y.eval(2.0)), 1.0);
        assert_near!(c.eval(x.eval(3.0) + y.eval(3.0)), 1.0);
        assert_near!(c.eval(x.eval(1.0) + y.eval(1.0)), 1.0);
        assert_near!(c.eval(x.eval(3.0) + y.eval(2.0)), 1.5);
        assert_near!(c.eval(x.eval(2.0) + y.eval(1.0)), 2.0);
        assert_near!(c.eval(x.eval(3.0) + y.eval(1.0)), 3.0);
        assert_near!(c.eval(x.eval(1.0) + y.eval(2.0)), 0.5);
    }

    #[test]
    fn dtdy_identity() {
        assert_near!(Bez::<f32>::identity().dtdy(1.0), 1.0);
        assert_near!(Bez::<f32>::identity().dtdy(-1.0), 1.0);

        const TEST_TOLERANCE: f32 = 0.5;
        assert_near!(Bez::<f32>::identity_smol().dtdy(1.0), 1.0);
        assert_near!(Bez::<f32>::identity_smol().dtdy(-1.0), 1.0);
    }

    #[test]
    fn dtdy_linear() {
        // Make the spline double the input: f(x) = 2x.
        let mut s = Bez::<f32>::identity();
        s.scale_y(2.0);
        assert_near!(s.dtdy(1.0), 2.0);
        assert_near!(s.dtdy(0.0), 2.0);
        assert_near!(s.dtdy(-1.0), 2.0);
        const TEST_TOLERANCE: f32 = 0.5;
        let mut s = Bez::<f32>::identity_smol();
        s.scale_y(2.0);
        assert_near!(s.dtdy(1.0), 2.0);
        assert_near!(s.dtdy(0.0), 2.0);
        assert_near!(s.dtdy(-1.0), 2.0);
    }

    #[test]
    fn scale_y() {
        // Make the spline negate the input: f(x) = -x.
        let mut s = Bez::<f32>::identity();
        s.scale_y(-1.0);
        assert_near!(s.eval(1.0), -1.0);
        assert_near!(s.eval(0.0), 0.0);
        assert_near!(s.eval(-1.0), 1.0);
    }

    #[test]
    fn dt_basis_functions() {
        let (b0, b1, b2, b3) = Bez::dt_basis_functions(0.0);
        assert_near!(b0, -3.0);
        assert_near!(b1, 3.0);
        assert_near!(b2, 0.0);
        assert_near!(b3, 0.0);

        let (b0, b1, b2, b3) = Bez::dt_basis_functions(0.5);
        assert_near!(b0, -0.75);
        assert_near!(b1, -0.75);
        assert_near!(b2, 0.75);
        assert_near!(b3, 0.75);

        let (b0, b1, b2, b3) = Bez::dt_basis_functions(1.0);
        assert_near!(b0, 0.0);
        assert_near!(b1, 0.0);
        assert_near!(b2, -3.0);
        assert_near!(b3, 3.0);
    }
}
