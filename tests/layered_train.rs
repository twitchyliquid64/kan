use plotters::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};

use uvf::S;
use viz::{make_video, Spline};

use uvf::assert_near;

// struct UnaryScenario<'a, IF, LF, TF, const IN: usize = 1>
// where
//     IF: FnMut(&mut Self),
//     LF: FnMut(&mut Self, [f32; IN], f32) -> f32,
//     TF: FnMut(&mut Self) -> ([f32; IN], f32),
// {
//     name: &'a str,
//     splines: Vec<S>,

//     init: IF,
//     calc_loss: LF,
//     training_datapoint: TF,
// }

#[test]
#[ignore]
/// Configures a network like this:
///   Input --> Spline 1 (learnable) --> Spline 2 --> Output
///   where:
///     - Spline 1 is the identity spline
///     - Spline 2 is a mostly-linear scaling function with a little bit of randomness.
///     - Desired function is f(x) = x
///
/// hence, spline 1 needs to learn the inverse of spline 2, backpropergating
/// through spline 1.
fn transfer_through_spline() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut s = S::identity_smol();
    let mut last = S::identity_smol();
    last.dither_y(|| rng.gen_range(-260.0..260.0));
    last.scale_y(8.0);

    let (min, max) = s.t_domain();

    make_video(
        (1080, 720),
        "/tmp/vid_transfer_through_spline.mp4",
        |buff, n| {
            // Render
            let root = BitMapBackend::<plotters::backend::RGBPixel>::with_buffer_and_format(
                buff,
                (1080, 720),
            )
            .unwrap();
            let d = root.into_drawing_area().split_evenly((1, 2));
            Spline::viz(s.clone())
                .title("trainable")
                .render(&d[0])
                .unwrap();
            Spline::viz(last.clone())
                .title("fixed")
                .render(&d[1])
                .unwrap();

            // Train
            for _ in 0..1200 {
                let input: f32 = rng.gen_range(min..=max);
                let intermediate = s.eval(input);
                let out = last.eval(intermediate);
                let error = out - input;

                s.adjust(
                    &uvf::Params {
                        learning_rate: 0.001,
                    },
                    input,
                    error * last.dydt(intermediate),
                );
            }

            n < 150
        },
    )
    .unwrap();

    let bmb = BitMapBackend::new("/tmp/transfer_through_spline.png", (1080, 720));
    let d = bmb.into_drawing_area().split_evenly((1, 2));

    Spline::viz(s.clone()).render(&d[0]).unwrap();
    Spline::viz(last.clone()).render(&d[1]).unwrap();

    const TEST_TOLERANCE: f32 = 0.5;
    assert_near!(last.eval(s.eval(1.0)), 1.0);
    assert_near!(last.eval(s.eval(0.0)), 0.0);
    assert_near!(last.eval(s.eval(-1.0)), -1.0);
    assert_near!(last.eval(s.eval(100.0)), 100.0);
    assert_near!(last.eval(s.eval(-100.0)), -100.0);

    // Quick general eval
    for _ in 0..16 {
        let input: f32 = rng.gen_range(min..=max);
        let intermediate = s.eval(input);
        let out = last.eval(intermediate);
        assert_near!(out, input);
    }
}

#[test]
#[ignore]
/// Configures a network like this:
///   Input x--> Spline 1 (learnable) -->
///   Input y--> Spline 2 (learnable) -->
///                                       Spline 3 (learnable) --> Output
///
/// With a desired function of: f(x, y) = xy.
fn mul_network() {
    let mut rng = StdRng::seed_from_u64(1);
    let mut x = S::new(-60.0, 60.0, 3);
    let mut y = S::new(-60.0, 60.0, 3);
    let mut c = S::new(-250.0, 250.0, 4);
    // x.dither_y(|| rng.gen_range(-2.0..2.0));
    // y.dither_y(|| rng.gen_range(-2.0..2.0));
    // c.dither_y(|| rng.gen_range(-2.0..2.0));

    let params = &uvf::Params {
        learning_rate: 0.00000001,
    };
    make_video((1080, 720), "/tmp/vid_mul_network.mp4", |buff, n| {
        // Render
        let root =
            BitMapBackend::<plotters::backend::RGBPixel>::with_buffer_and_format(buff, (1080, 720))
                .unwrap();
        let r = root.into_drawing_area().split_evenly((2, 1));
        let d = r[0].split_evenly((1, 2));
        Spline::viz(x.clone()).title("x").render(&d[0]).unwrap();
        Spline::viz(y.clone()).title("y").render(&d[1]).unwrap();
        Spline::viz(c.clone()).title("c").render(&r[1]).unwrap();

        // Train
        for _ in 0..24000 {
            let x_input: f32 = rng.gen_range(-45.0..45.0);
            let y_input: f32 = rng.gen_range(-45.0..45.0);
            let x_output = x.eval(x_input);
            let y_output = y.eval(y_input);
            let output = c.eval(x_output + y_output);
            let error = output - (x_input * y_input);
            let error = error.signum() * error.powi(2);

            let gradient = c.dydt(x_output + y_output);
            c.adjust(params, x_output + y_output, error);

            x.adjust(params, x_input, error * gradient);
            y.adjust(params, y_input, error * gradient);
        }

        n < 250
    })
    .unwrap();

    const TEST_TOLERANCE: f32 = 0.5;
    assert_near!(c.eval(x.eval(4.0) + y.eval(4.0)), 16.0);
    assert_near!(c.eval(x.eval(2.0) + y.eval(1.0)), 2.0);
}
