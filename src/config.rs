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

#[derive(Default)]
pub struct Config {
    pub card: Card,
    pub resource_path: String,
    pub resolution: (usize, usize)
}

impl Config {
    pub fn new(file: &str) -> Config {
        let contents = read_to_string(file).expect("Couldn't find config file");
        contents.lines().fold(Default::default(), |mut acc, line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with("#") {
                return acc; // Skip empty/comment line
            }
            let (key, value) = line.split_once(":").expect("Invalid config line");
            let (key, value) = (key.trim(), value.trim());
            match key {
                "card" => acc.card = if value == "discrete" { Card::Discrete } else { Card::Number (value.parse().unwrap()) },
                "resources" => acc.resource_path = value.to_owned(),
                "resolution" => {
                    let (x, y) = value.split_once("x").expect("Expected resolution of the form 640x640");
                    acc.resolution = (x.parse().expect("Expected integer"), y.parse().expect("Expected integer"));
                },
                _ => panic!("Invalid config line: {}", line)
            }
            acc
        })
    }
}
