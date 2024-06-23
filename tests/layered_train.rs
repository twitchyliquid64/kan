use plotters::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};

use uvf::S;
use viz::{make_video, Spline};

use uvf::assert_near;

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
    last.dither_y(|| rng.gen_range(-2600.0..2600.0));
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
            Spline::viz(s.clone()).render(&d[0]).unwrap();
            Spline::viz(last.clone()).render(&d[1]).unwrap();

            // Train
            for _ in 0..65 {
                let input: f32 = rng.gen_range(min..=max);
                let intermediate = s.eval(input);
                let out = last.eval(intermediate);
                let error = out - input;

                s.adjust(
                    &uvf::Params {
                        learning_rate: 0.004,
                    },
                    input,
                    error * last.dydt(intermediate),
                );
            }

            n < 100
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
