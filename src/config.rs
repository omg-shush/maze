use std::fs::read_to_string;

pub enum Card {
    Discrete,
    Number (usize)
}

impl Default for Card {
    fn default() -> Self {
        Card::Discrete
    }
}

#[derive(PartialEq, Eq)]
pub enum Window {
    Borderless,
    Exclusive,
    Size (u32, u32)
}

impl Default for Window {
    fn default() -> Self {
        Window::Size (640, 480)
    }
}

pub enum TargetFps {
    Unlimited,
    Fixed (usize)
}

impl Default for TargetFps {
    fn default() -> Self {
        TargetFps::Unlimited
    }
}

pub enum Resolution {
    Max,
    Fixed (u32, u32)
}

impl Default for Resolution {
    fn default() -> Self {
        Resolution::Max
    }
}

pub struct Config {
    pub card: Card,
    pub resource_path: String,
    pub window: Window,
    pub resolution: Resolution,
    pub target_fps: TargetFps,
    pub fov: u32,
    pub ui_scale: f32,
    pub display_controls: bool,
    pub display_stopwatch: bool,
    pub dimensions: [usize; 4],
    pub ghost_move_time: f32,
    pub food_count: usize
}

impl Default for Config {
    fn default() -> Self {
        Config {
            card: Card::Discrete,
            resource_path: "res/".to_string(),
            window: Window::Size(1280, 720),
            resolution: Resolution::Max,
            target_fps: TargetFps::Fixed(60),
            fov: 90,
            ui_scale: 1.0,
            display_controls: true,
            display_stopwatch: false,
            dimensions: [5, 5, 5, 3],
            ghost_move_time: 1.65,
            food_count: 10
        }
    }
}

impl Config {
    pub fn new(file: &str) -> Config {
        let contents = read_to_string(file).expect("Couldn't find config file");
        contents.lines().fold(Default::default(), |mut acc, line| {
            let line = line.split("#").next().unwrap_or_default().trim();
            if line.is_empty() {
                return acc; // Skip empty/comment line
            }
            let (key, value) = line.split_once(":").expect("Invalid config line");
            let (key, value) = (key.trim(), value.trim());
            match key {
                "card" => acc.card = if value == "discrete" { Card::Discrete } else { Card::Number (value.parse().expect("Expected integer")) },
                "resources" => acc.resource_path = value.to_string(),
                "window" => acc.window = match value {
                    "borderless" => Window::Borderless,
                    "exclusive" => Window::Exclusive,
                    _ => {
                        let (x, y) = value.split_once("x").expect("Expected window size of the form 640x480");
                        Window::Size (x.parse().expect("Expected integer"), y.parse().expect("Expected integer"))
                    }
                },
                "resolution" => acc.resolution = if value == "max" { Resolution::Max } else {
                    let (x, y) = value.split_once("x").expect("Expected resolution of the form 640x640");
                    Resolution::Fixed (x.parse().expect("Expected integer"), y.parse().expect("Expected integer"))
                },
                "target-fps" => acc.target_fps = if value == "unlimited" { TargetFps::Unlimited } else { TargetFps::Fixed (value.parse().expect("Expected integer")) },
                "fov" => acc.fov = value.parse().expect("Expected integer"),
                "ui-scale" => acc.ui_scale = value.parse().expect("Expected decimal value"),
                "display-controls" => acc.display_controls = value.parse().expect("Expected true or false"),
                "display-stopwatch" => acc.display_stopwatch = value.parse().expect("Expected true or false"),
                "dimensions" => acc.dimensions = value.split("x").map(|s| s.parse::<usize>().unwrap()).collect::<Vec<_>>().try_into().unwrap(),
                "ghost-move-time" => acc.ghost_move_time = value.parse().expect("Expected decimal value"),
                "food-count" => acc.food_count = value.parse().expect("Expected integer"),
                _ => panic!("Invalid config line: {}", line)
            }
            acc
        })
    }
}
