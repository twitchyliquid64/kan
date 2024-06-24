use plotters::coord::Shift;
use plotters::prelude::*;
use uvf::S;

const DEFAULT_NUM_GRAPH_POINTS: usize = 115;

/// Renders a video file by calling the given function to generate every frame.
///
/// The callback function should write the frame in rgb (3x8 bits a pixel) format,
/// and return true if the callback should be called again to generate the next
/// frame.
///
/// The video is encoded as an MP4 at 30 fps.
#[cfg(target_os = "linux")]
pub fn make_video<F: FnMut(&mut Vec<u8>, usize) -> bool>(
    size: (usize, usize),
    out_path: &str,
    mut frame_cb: F,
) -> Result<(), Box<dyn std::error::Error>> {
    let (w, h) = size;
    let mut buf = vec![0u8; w * h * 3];

    let (reader, mut writer) = os_pipe::pipe()?;

    use command_fds::CommandFdExt;
    let mut command = std::process::Command::new("ffmpeg");
    command
        .args([
            "-f",
            "rawvideo",
            "-video_size",
            &format!("{}x{}", w, h),
            "-pixel_format",
            "rgb24",
            "-framerate",
            "30/1",
        ])
        .arg("-i")
        .arg("pipe:3")
        .args([
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-profile:v",
            "baseline",
            "-level",
            "3.0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "faststart",
        ])
        .arg("-y")
        .arg(out_path);

    command
        .fd_mappings(vec![command_fds::FdMapping {
            parent_fd: reader.into(),
            child_fd: 3,
        }])
        .unwrap();

    let mut child = command.spawn().unwrap();

    let mut n = 0;
    while frame_cb(&mut buf, n) {
        use std::io::Write;
        writer.write_all(&buf)?;
        n += 1;
        if n == 4 {
            writer.flush()?; // get ffmpeg encoding early on
        }
    }

    drop(writer);
    child.wait().unwrap();
    Ok(())
}

#[derive(Debug, Default)]
pub enum Title {
    #[default]
    Default,
    Str(&'static str),
}

/// Visualizes a spline.
pub struct Spline {
    pub spline: S,
    pub extend_by: Option<f32>,
    pub title: Title,
}

impl Spline {
    pub fn viz(spline: S) -> Self {
        Self {
            spline,
            extend_by: None,
            title: Title::Default,
        }
    }

    pub fn title(mut self, title: &'static str) -> Self {
        self.title = Title::Str(title);
        self
    }

    pub fn datapoints(&self, t0: f32, t1: f32, points: &mut Vec<(f32, f32)>) -> (f32, f32) {
        let (mut y_min, mut y_max) = (f32::MAX, f32::MIN);
        let step_dist = (t1 - t0) / points.len() as f32;
        for (i, p) in points.iter_mut().enumerate() {
            let t = t0 + i as f32 * step_dist;
            let y = self.spline.eval(t);
            y_max = y_max.max(y);
            y_min = y_min.min(y);
            *p = (t, y);
        }

        (y_min, y_max)
    }

    pub(crate) fn title_str(&self) -> String {
        match self.title {
            Title::Default => {
                let (t0, t1) = self.spline.t_domain();
                format!(
                    "spline<N={}, {:.2}..{:.2}>",
                    self.spline.num_points(),
                    t0,
                    t1
                )
            }
            Title::Str(s) => s.into(),
        }
    }

    pub(crate) fn label(&self) -> String {
        "uwu spline".to_string()
    }

    pub fn render<'a, DB: DrawingBackend>(
        self,
        canvas: &'a DrawingArea<DB, Shift>,
    ) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
        let (mut t0, mut t1) = self.spline.t_domain();
        if let Some(e) = self.extend_by {
            t0 -= e;
            t1 += e;
        }

        // Populate an array with 1024 (x, y) pairs over the entire domain,
        // keeping track of the min/max y value observed.
        let mut points = vec![(0f32, 0f32); DEFAULT_NUM_GRAPH_POINTS];
        let (mut y_min, mut y_max) = self.datapoints(t0, t1, &mut points);

        let control_points = self.spline.control_points();
        control_points.iter().for_each(|(_x, y)| {
            y_min = y_min.min(*y);
            y_max = y_max.max(*y);
        });

        canvas.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&canvas)
            .caption(self.title_str(), ("sans-serif", 50).into_font())
            .margin(12)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(t0..t1, y_min..y_max)?;

        chart.configure_mesh().x_desc("t").y_desc("y").draw()?;

        // Insert the data into the chart, add a legend
        chart
            .draw_series(LineSeries::new(points, &RED))?
            .label(self.label())
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        // Plot the control points into the chart
        chart.draw_series(
            control_points
                .iter()
                .map(|(t, y)| Circle::new((*t, *y), 1, BLACK.filled())),
        )?;

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
    fn spline_viz_smoketest() {
        let mut buffer = vec![0u8; 512 * 384 * 3];
        let bmb = BitMapBackend::<plotters::backend::RGBPixel>::with_buffer_and_format(
            &mut buffer,
            (512, 384),
        )
        .unwrap();

        Spline::viz(uvf::S::identity())
            .render(&bmb.into_drawing_area())
            .unwrap();
    }

    #[test]
    #[ignore]
    fn spline_viz_training() {
        let mut s = uvf::S::identity();

        // Toy training loop: f(-10000) = 10000, f(0) = 0, f(10000) = 10000
        for _ in 0..5000 {
            let pos = -1000.0;
            let out = s.eval(pos);
            s.adjust(&uvf::Params::default(), pos, out - 1000.0);
            let pos = 0.0;
            let out = s.eval(pos);
            s.adjust(&uvf::Params::default(), pos, out);
            let pos = 1000.0;
            let out = s.eval(pos);
            s.adjust(&uvf::Params::default(), pos, out + 1000.0);
        }

        let bmb = BitMapBackend::new("/tmp/spline_viz_training.png", (1080, 720));

        Spline {
            spline: s,
            extend_by: None,
            title: Title::Default,
        }
        .render(&bmb.into_drawing_area())
        .unwrap();
    }

    #[test]
    #[ignore]
    fn spline_viz_video_smoketest() {
        let mut s = uvf::S::new(-15000.0, 15000.0, 3);
        let p = uvf::Params {
            learning_rate: 0.006,
            ..uvf::Params::default()
        };

        make_video((1080, 720), "/tmp/vid_spline_smoketest.mp4", |buff, n| {
            // Render
            let root = BitMapBackend::<plotters::backend::RGBPixel>::with_buffer_and_format(
                buff,
                (1080, 720),
            )
            .unwrap();
            Spline {
                spline: s.clone(),
                extend_by: None,
                title: Title::Default,
            }
            .render(&root.into_drawing_area())
            .unwrap();

            // Train
            for _ in 0..325 {
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

            n < 300 // && (s.eval(-10000.0) - 10000.0).abs() > 1.0
        })
        .unwrap();
    }
}
