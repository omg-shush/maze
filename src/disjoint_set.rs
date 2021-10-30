use std::collections::hash_map::HashMap;
use std::hash::Hash;
use core::fmt::Debug;

enum DSPtr<T> {
    Ptr (T),
    Top
}

pub struct DisjointSet<T> {
    table: HashMap<T, DSPtr<T>>
}

impl<T: Eq + Hash + Copy + Debug> DisjointSet<T> {
    pub fn new() -> DisjointSet<T> {
        DisjointSet { table: HashMap::new() }
    }

    // Add
    pub fn add(&mut self, val: &T) {
        // println!(">>> Adding {:?}", val);
        self.table.insert(val.clone(), DSPtr::Top);
    }

    // Union: add a to b's set
    pub fn union(&mut self, a: &T, b: &T) {
        // println!(">>> Unionizing {:?} and {:?}", a, b);
        let a_top = self.find(&a);
        let b_top = self.find(&b);
        self.table.insert(a_top.clone(), DSPtr::Ptr(b_top.clone()));
    }

    // Find
    pub fn find(&self, val: &T) -> T {
        // println!(">>> Finding {:?}", val);
        match self.table.get(val).expect("Value not in disjoint set") {
            DSPtr::Ptr (above) => self.find(above),
            DSPtr::Top => val.clone()
        }
        // TODO need to optimize find()!
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }
}
