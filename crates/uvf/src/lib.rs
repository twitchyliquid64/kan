//! Trainable univariate functions.
//!
//! Provides an abstraction over different kinds of univariate functions
//! and a way to train them.

use num_traits::{float::Float, FromPrimitive};

/// The numeric base type a uvf is defined for.
pub trait BaseNum:
    Float + std::fmt::Debug + std::ops::AddAssign + std::ops::SubAssign + FromPrimitive
{
}

impl<T: Float + std::fmt::Debug + std::ops::AddAssign + std::ops::SubAssign + FromPrimitive> BaseNum
    for T
{
}

/// maps the given parameter to a coefficient describing how far along it is between min and max.
pub(crate) fn normalize<V: Float>(p: V, min: V, max: V) -> V {
    (p - min) / (max - min)
}

/// derivative of the normalize function.
pub(crate) fn normalize_dpdt<V: Float>(min: V, max: V) -> V {
    V::one().div(max - min)
}

/// Describes the hyperparameters when training a learnable function.
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

/// Some uvf able to learn through backpropagation.
pub trait Trainable<V>
where
    V: BaseNum,
{
    /// Computes the output for the given input parameter.
    fn eval(&self, t: V) -> V;
    /// Computes the derivative of the output with respect to the input.
    fn dtdy(&self, t: V) -> V;
    /// Updates the function given learning parameters, an input parameter value and the error
    /// or loss of the output.
    fn adjust(&mut self, params: &Params, t: V, error: V);

    // TODO: move out of trait
    fn scale_y(&mut self, amt: V);
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

#[derive(Debug)]
pub struct Layer<E, V>
where
    V: BaseNum,
    E: Trainable<V> + std::fmt::Debug,
{
    edges: Vec<Vec<E>>, // [input idx][output idx]
    outputs: Vec<V>,    // [output idx]
}

impl<E, V> Layer<E, V>
where
    V: BaseNum,
    E: Trainable<V> + std::default::Default + std::fmt::Debug,
{
    /// creates a new layer with the given number of inputs and outputs.
    ///
    /// Safety: Must have more than zero inputs + outputs.
    pub fn new(inputs: usize, outputs: usize) -> Self {
        assert!(inputs > 0);
        assert!(outputs > 0);
        let edges = (0..inputs)
            .map(|_ix| (0..outputs).map(|_ox| E::default()).collect())
            .collect();
        let outputs = (0..outputs).map(|_| V::zero()).collect();

        Self { edges, outputs }
    }
}

impl<E, V> Layer<E, V>
where
    V: BaseNum,
    E: Trainable<V> + std::fmt::Debug,
{
    /// creates a new layer with the given number of inputs and outputs. Each
    /// spline is generated by calling the given function.
    ///
    /// Safety: Must have more than zero inputs + outputs.
    pub fn new_with_init<F: Fn(usize, usize) -> E>(inputs: usize, outputs: usize, func: F) -> Self {
        assert!(inputs > 0);
        assert!(outputs > 0);
        let edges = (0..inputs)
            .map(|ix| (0..outputs).map(|ox| func(ix, ox)).collect())
            .collect();
        let outputs = (0..outputs).map(|_| V::zero()).collect();

        Self { edges, outputs }
    }

    /// computes the outputs for this layer of functions, based on the given inputs.
    ///
    /// Safety: This method will panic if the input or output vectors are too
    /// short for the requisite number of inputs and outputs
    pub fn eval(&mut self, inputs: &Vec<V>) -> &Vec<V> {
        assert!(inputs.len() >= self.edges.len());

        self.outputs.iter_mut().for_each(|v| *v = V::zero());
        for (i, edges) in self.edges.iter().enumerate() {
            let inp = inputs[i];

            for (o, e) in edges.iter().enumerate() {
                self.outputs[o] += e.eval(inp);
            }
        }

        &self.outputs
    }

    /// performs a single backpropergation step for this layer.
    ///
    /// Safety: This method will panic if the number of elements in the inputs and/or
    /// output_loss array doesn't match the size of the layer.
    pub fn adjust(
        &mut self,
        params: &Params,
        inputs: &Vec<V>,
        output_loss: &Vec<V>,
        mut propergate_input_loss: Option<&mut Vec<V>>,
    ) {
        assert_eq!(inputs.len(), self.edges.len());
        for (i, edges) in self.edges.iter_mut().enumerate() {
            assert_eq!(edges.len(), output_loss.len());

            if let Some(upper_loss) = propergate_input_loss.as_mut() {
                let upper_loss = &mut upper_loss[i];
                for (o, edge) in edges.iter_mut().enumerate() {
                    *upper_loss += edge.dtdy(inputs[i]) * output_loss[o];
                }
            }

            for (o, edge) in edges.iter_mut().enumerate() {
                edge.adjust(params, inputs[i], output_loss[o]);
            }
        }
    }

    pub fn scale_y(&mut self, y: V) {
        self.edges.iter_mut().flatten().for_each(|s| s.scale_y(y));
    }
}

pub fn layered_eval<E, V>(layers: &mut Vec<Layer<E, V>>, mut inputs: Vec<V>) -> &Vec<V>
where
    V: BaseNum,
    E: Trainable<V> + std::fmt::Debug,
{
    for l in layers.iter_mut() {
        let outputs = l.eval(&inputs);
        inputs.clear();
        inputs.extend_from_slice(&outputs[..]);
    }
    &layers.last().unwrap().outputs
}

pub fn layered_adjust<E, V>(
    params: &Params,
    layers: &mut Vec<Layer<E, V>>,
    inputs: Vec<V>,
    mut outputs_loss: Vec<V>,
) where
    V: BaseNum,
    E: Trainable<V> + std::fmt::Debug,
{
    let iter = (0..layers.len()).rev();
    for (c_idx, p_idx) in iter.clone().zip(iter.skip(1)) {
        let inputs = layers[p_idx].outputs.clone(); // TODO: eliminate clone.
        let mut next_loss = (0..layers[c_idx].edges.len()).map(|_| V::zero()).collect();

        layers[c_idx].adjust(params, &inputs, &outputs_loss, Some(&mut next_loss));
        outputs_loss = next_loss;
    }

    layers[0].adjust(params, &inputs, &outputs_loss, None);
}

// TODO: Refactor to store the inputs for each spline, as we need that for backprop.

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    const TEST_TOLERANCE: f32 = 1.0e-5;

    #[test]
    fn _normalize() {
        assert_near!(normalize(1f32, 1.0, 2.0), 0.0);
        assert_near!(normalize(2f32, 1.0, 2.0), 1.0);
        assert_near!(normalize(2f32, 1.0, 3.0), 0.5);
        assert_near!(normalize(3f32, 1.0, 5.0), 0.5);
        assert_near!(normalize(4f32, 1.0, 5.0), 0.75);
    }

    #[test]
    fn layer_new() {
        let _l: Layer<spline::Bez, f32> = Layer::new(2, 1);
    }

    #[test]
    fn layer_eval() {
        let mut l: Layer<spline::Bez, f32> = Layer::new(2, 1);
        assert_near!(l.eval(&vec![1., 1.])[0], 2.0);

        let mut l: Layer<spline::Bez, f32> = Layer::new(3, 2);
        assert_near!(l.eval(&vec![1., 1., 2.])[0], 4.0);
        assert_near!(l.eval(&vec![1., 1., 2.])[1], 4.0);
    }

    #[test]
    fn layer_adjust_trivial() {
        let mut l: Layer<spline::Bez, f32> =
            Layer::new_with_init(1, 1, |_, _| Bez::new(-5.0, 5.0, 1));

        let p = Params {
            learning_rate: 0.8,
            ..Params::default()
        };

        let mut rng = StdRng::seed_from_u64(11);
        for _ in 0..500 {
            let input: f32 = rng.gen_range(-5.0..=5.0);
            let delta = l.eval(&vec![input])[0] - (-input);
            l.adjust(&p, &vec![input], &vec![delta], None);
        }

        const TEST_TOLERANCE: f32 = 0.25;
        let mut test_point = |in_val: f32, out_val: f32| {
            assert_near!(l.eval(&vec![in_val])[0], out_val);
        };

        test_point(0.0, 0.0);
        test_point(-3.0, 3.0);
        test_point(5.0, -5.0);
        test_point(3.0, -3.0);
    }

    #[test]
    fn layer_adjust_sq_difference() {
        let mut l: Layer<spline::Bez, f32> =
            Layer::new_with_init(3, 1, |_, _| Bez::new(-5.0, 5.0, 1));

        let p = Params {
            learning_rate: 0.1,
            ..Params::default()
        };

        let mut rng = StdRng::seed_from_u64(29);
        for _ in 0..4000 {
            let input_x: f32 = rng.gen_range(-5.0..=5.0);
            let input_y: f32 = rng.gen_range(-5.0..=5.0);
            let input_z: f32 = rng.gen_range(-5.0..=5.0);
            let inputs = vec![input_x, input_y, input_z];
            let target = input_x.powi(2) + input_y.powi(2) + (-input_z.powi(2));

            let delta = l.eval(&inputs)[0] - target;
            l.adjust(&p, &inputs, &vec![delta], None);
        }

        const TEST_TOLERANCE: f32 = 0.15;
        let mut test_points = |in_val: [f32; 3], out_val: f32| {
            assert_near!(l.eval(&in_val.into())[0], out_val);
        };

        test_points([0.0, 0.0, 0.0], 0.0);
        test_points([1.0, 0.0, 1.0], 0.0);
        test_points([2.0, 1.0, 1.0], 4.0);
        test_points([4.0, 0.0, 2.0], 12.0);
        test_points([3.0, 2.0, 2.0], 9.0);
        test_points([5.0, 5.0, 4.0], 34.0);
        test_points([1.0, 0.0, 2.0], -3.0);
        test_points([4.0, 0.0, 4.0], 0.0);
        test_points([0.0, 4.0, 4.0], 0.0);
        test_points([1.0, 0.0, 5.0], -24.0);
    }

    #[test]
    fn layer_adjust_output_loss() {
        let mut l: Layer<spline::Bez, f32> =
            Layer::new_with_init(2, 1, |_, _| Bez::new(-5.0, 5.0, 1));
        l.edges[0][0].scale_y(-1.0);

        let p = Params {
            learning_rate: 0.1,
            ..Params::default()
        };
        let inputs = vec![1.0, 1.0];
        let mut next_layer = vec![0.0, 0.0];
        l.adjust(&p, &inputs, &vec![1.0], Some(&mut next_layer));

        assert_near!(next_layer[0], -1.0);
        assert_near!(next_layer[1], 1.0);
    }

    #[test]
    fn _layered_eval() {
        assert_near!(
            layered_eval(
                &mut vec![Layer::<spline::Bez, f32>::new_with_init(2, 1, |_, _| {
                    Bez::new(-5.0, 5.0, 1)
                })],
                vec![2.0, 1.0]
            )[0],
            3.0
        );

        let mut layers = vec![
            Layer::new_with_init(2, 2, |_, _| Bez::new(-15.0, 15.0, 1)),
            Layer::new_with_init(2, 2, |_, _| Bez::new(-15.0, 15.0, 1)),
            Layer::new_with_init(2, 1, |_, _| Bez::new(-15.0, 15.0, 1)),
        ];
        assert_near!(layered_eval(&mut layers, vec![2.0, 1.0])[0], 12.0);
        assert_near!(layered_eval(&mut layers, vec![2.0, -1.5])[0], 2.0);

        layered_adjust(&Params::default(), &mut layers, vec![2.0, 1.0], vec![1.0]);
    }

    #[test]
    fn layered_train_trivial() {
        let p = Params {
            learning_rate: 0.3,
            ..Params::default()
        };
        let mut layers = vec![
            Layer::new_with_init(2, 1, |_, _| Bez::new(-5.0, 5.0, 3)),
            Layer::new_with_init(1, 1, |_, _| Bez::new(-50.0, 50.0, 1)),
            Layer::new_with_init(1, 1, |_, _| Bez::new(-50.0, 50.0, 1)),
        ];
        layers[1].scale_y(-1.0);

        let mut rng = StdRng::seed_from_u64(11);
        for _ in 0..2000 {
            let input_x: f32 = rng.gen_range(-5.0..=5.0);
            let input_y: f32 = rng.gen_range(-5.0..=5.0);
            let inputs = vec![input_x, input_y];

            let delta =
                layered_eval(&mut layers, inputs.clone())[0] - (input_x.powi(2) + input_y.powi(2));

            layered_adjust(&p, &mut layers, inputs, vec![delta]);
        }

        const TEST_TOLERANCE: f32 = 0.22;
        assert_near!(layered_eval(&mut layers, vec![2.0, 2.0])[0], 8.0);
        assert_near!(layered_eval(&mut layers, vec![0.0, 0.0])[0], 0.0);
        assert_near!(layered_eval(&mut layers, vec![3.0, 0.0])[0], 9.0);
        assert_near!(layered_eval(&mut layers, vec![0.0, 3.0])[0], 9.0);
    }

    #[test]
    fn layered_train_div() {
        let p = Params {
            learning_rate: 0.05,
            ..Params::default()
        };
        let mut layers = vec![
            Layer::new_with_init(2, 4, |_, _| Bez::new(0.0, 6.0, 3)),
            Layer::new_with_init(4, 2, |_, _| Bez::new(0.0, 15.0, 2)),
            Layer::new_with_init(2, 1, |_, _| Bez::new(0.0, 15.0, 1)),
        ];

        let mut rng = StdRng::seed_from_u64(11);
        for _ in 0..20000 {
            let input_x: f32 = rng.gen_range(1.0..=4.0);
            let input_y: f32 = rng.gen_range(0.5..=4.0);
            let inputs = vec![input_x, input_y];

            let delta = layered_eval(&mut layers, inputs.clone())[0] - (input_x / input_y);

            layered_adjust(&p, &mut layers, inputs, vec![delta]);
        }

        const TEST_TOLERANCE: f32 = 0.22;
        assert_near!(layered_eval(&mut layers, vec![2.0, 2.0])[0], 1.0);
        assert_near!(layered_eval(&mut layers, vec![1.0, 1.0])[0], 1.0);
        assert_near!(layered_eval(&mut layers, vec![3.0, 2.0])[0], 1.5);
        assert_near!(layered_eval(&mut layers, vec![2.0, 4.0])[0], 0.5);
        assert_near!(layered_eval(&mut layers, vec![2.0, 1.0])[0], 2.0);
        assert_near!(layered_eval(&mut layers, vec![2.0, 0.5])[0], 4.0);
    }
}
