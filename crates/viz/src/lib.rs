use plotters::prelude::*;
use uvf::S;

/// Visualizes a spline.
pub struct Spline {
    pub spline: S,
    pub extend_by: Option<f32>,
}

impl Spline {
    pub fn viz(spline: S) -> Self {
        Self {
            spline,
            extend_by: None,
        }
    }

    pub fn render(self, bb: BitMapBackend) -> Result<(), Box<dyn std::error::Error>> {
        const STEPS: usize = 150;

        let (mut t0, mut t1) = self.spline.t_domain();
        if let Some(e) = self.extend_by {
            t0 -= e;
            t1 += e;
        }

        // Populate an array with 1024 (x, y) pairs over the entire domain,
        // keeping track of the min/max y value observed.
        let (mut y_min, mut y_max) = (f32::MAX, f32::MIN);
        let step_dist = (t1 - t0) / STEPS as f32;
        let mut points = vec![(0f32, 0f32); STEPS];
        for (i, p) in points.iter_mut().enumerate() {
            let t = t0 + i as f32 * step_dist;
            let y = self.spline.eval(t);
            y_max = y_max.max(y);
            y_min = y_min.min(y);
            *p = (t, y);
        }

        let canvas = bb.into_drawing_area();
        canvas.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&canvas)
            .caption(
                format!("spline<N={}, O=4>", self.spline.num_points()),
                ("sans-serif", 50).into_font(),
            )
            .margin(12)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(t0..t1, y_min..y_max)?;

        chart.configure_mesh().x_desc("t").y_desc("y").draw()?;

        // Insert the data into the chart, add a legend
        chart
            .draw_series(LineSeries::new(points, &RED))?
            .label("uwu spline")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        canvas.present()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn spline_viz_smoketest() {
        let bmb = BitMapBackend::new("/tmp/spline_viz_smoketest.png", (1080, 720));

        Spline {
            spline: uvf::S::identity(),
            extend_by: None,// Some(1500.),
        }
        .render(bmb)
        .unwrap();
    }
}
