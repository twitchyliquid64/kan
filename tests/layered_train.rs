use plotters::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};

use uvf::Bez;
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
    let mut s = Bez::new(-2000.0, 2000.0, 1);
    let mut last = Bez::identity_smol();
    last.dither_y(|| rng.gen_range(-260.0..260.0));
    last.scale_y(18.0);

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
            for _ in 0..100 {
                let input: f32 = rng.gen_range(min..=max);
                let intermediate = s.eval(input);
                let out = last.eval(intermediate);
                let target = input;

                // let error = 0.5 * (target - out).powi(2); // Would be summed if there was more outputs
                let error_der = out - target; // 1/2 * (t - o)^2 => 1/2 * 2(t-o) * -1

                // println!(
                //     "i={}=>{}, dydt={}",
                //     input,
                //     intermediate,
                //     last.dtdy(out) / out
                // );

                s.adjust(
                    &uvf::Params {
                        learning_rate: 0.0005,
                    },
                    input,
                    error_der * last.dtdy(intermediate),
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
/// With a desired function of: f(x, y) = x^2 + y^2.
fn sq_sum_network() {
    let mut rng = StdRng::seed_from_u64(4);
    let mut x = Bez::new(-60.0, 60.0, 5);
    let mut y = Bez::new(-60.0, 60.0, 5);
    let mut c = Bez::new(-7200.0, 7200.0, 1);
    c.scale_y(-1.0);
    x.dither_y(|| rng.gen_range(-2.0..2.0));
    y.dither_y(|| rng.gen_range(-2.0..2.0));
    c.dither_y(|| rng.gen_range(-2.0..2.0));

    let params = &uvf::Params {
        learning_rate: 0.07,
    };
    make_video((1080, 720), "/tmp/vid_sq_sum_network.mp4", |buff, n| {
        let root =
            BitMapBackend::<plotters::backend::RGBPixel>::with_buffer_and_format(buff, (1080, 720))
                .unwrap();
        let r = root.into_drawing_area().split_evenly((2, 1));
        let d = r[0].split_evenly((1, 2));
        Spline::viz(x.clone()).title("x").render(&d[0]).unwrap();
        Spline::viz(y.clone()).title("y").render(&d[1]).unwrap();
        Spline::viz(c.clone()).title("c").render(&r[1]).unwrap();

        // Train
        for _ in 0..17000 {
            let x_input: f32 = rng.gen_range(-60.0..60.0);
            let y_input: f32 = rng.gen_range(-60.0..60.0);
            let x_output = x.eval(x_input);
            let y_output = y.eval(y_input);
            let output = c.eval(x_output + y_output);

            let target = x_input.powi(2) + y_input.powi(2);
            let error_der = output - target;
            let c_der = c.dtdy(x_output + y_output);
            c.adjust(params, x_output + y_output, error_der);

            x.adjust(params, x_input, error_der * c_der);
            y.adjust(params, y_input, error_der * c_der);
        }

        n < 200
    })
    .unwrap();

    const TEST_TOLERANCE: f32 = 0.5;
    assert_near!(c.eval(x.eval(4.0) + y.eval(4.0)), 32.0);
    assert_near!(c.eval(x.eval(2.0) + y.eval(1.0)), 5.0);
}

#[test]
#[ignore]
/// Configures a network like this:
///   Input x--> Spline 1 (learnable) -->
///   Input y--> Spline 2 (learnable) -->
///                                       Spline 3 (learnable) --> Output
///
/// With a desired function of: f(x, y) = x/y
///
/// Lessons so far:
///  - If domain constrains inputs then nothing will work - clamping breaks learning
///  - If not enough parameters exist to express problem then also nothing will work
///  - When the values + dtdy yeet off into forever, it doesnt have enough grid points to learn.
fn div_network() {
    let mut rng = StdRng::seed_from_u64(4);
    let mut x = Bez::new(0.7, 3.5, 3);
    let mut y = Bez::new(0.7, 3.5, 3);
    let mut c = Bez::new(0.5, 7.5, 3);

    let params = &uvf::Params {
        learning_rate: 0.02,
    };
    let mut last: Option<(f32, f32, f32, f32)> = None;
    make_video((1080, 720), "/tmp/vid_div_network.mp4", |buff, n| {
        let root =
            BitMapBackend::<plotters::backend::RGBPixel>::with_buffer_and_format(buff, (1080, 720))
                .unwrap();
        let r = root.into_drawing_area().split_evenly((2, 1));
        let d = r[0].split_evenly((1, 2));
        Spline::viz(x.clone())
            .title("x")
            .dtdy("dtdy")
            .highlight(last.map(|(t, _, e, c_der)| (t, x.eval(t) - e * c_der)))
            .render(&d[0])
            .unwrap();
        Spline::viz(y.clone())
            .title("y")
            .dtdy("dtdy")
            .highlight(last.map(|(_, t, e, c_der)| (t, y.eval(t) - e * c_der)))
            .render(&d[1])
            .unwrap();
        Spline::viz(c.clone())
            .title("c")
            .dtdy("dtdy")
            .highlight(last.map(|(xt, yt, e, c_der)| (xt + yt, c.eval(xt + yt) - e)))
            .render(&r[1])
            .unwrap();

        // Train
        for _ in 0..15000 {
            let x_input: f32 = rng.gen_range(1.0..3.0);
            let y_input: f32 = rng.gen_range(1.0..3.0);
            if y_input < 0.002 && y_input > -0.002 {
                continue;
            }
            let x_output = x.eval(x_input);
            let y_output = y.eval(y_input);
            let output = c.eval(x_output + y_output);

            let target = x_input / y_input;
            let error_der = output - target;
            let c_der = c.dtdy(x_output + y_output);

            // println!(
            //     "{:.2} / {:.2} = {:.2} ({:.2})\n\tder:   {:.5}\n\tc_der: {:.5}",
            //     x_input,
            //     y_input,
            //     output,
            //     target,
            //     error_der,
            //     error_der * c_der
            // );
            c.adjust(params, x_output + y_output, error_der);

            x.adjust(params, x_input, error_der * c_der);
            y.adjust(params, y_input, error_der * c_der);

            last = Some((x_input, y_input, error_der, c_der))
        }

        n < 250
    })
    .unwrap();

    const TEST_TOLERANCE: f32 = 0.1;
    assert_near!(c.eval(x.eval(2.0) + y.eval(2.0)), 1.0);
    assert_near!(c.eval(x.eval(3.0) + y.eval(3.0)), 1.0);
    assert_near!(c.eval(x.eval(1.0) + y.eval(1.0)), 1.0);
    assert_near!(c.eval(x.eval(3.0) + y.eval(2.0)), 1.5);
    assert_near!(c.eval(x.eval(2.0) + y.eval(1.0)), 2.0);
    assert_near!(c.eval(x.eval(3.0) + y.eval(1.0)), 3.0);
    assert_near!(c.eval(x.eval(1.0) + y.eval(2.0)), 0.5);
}
