use std::collections::hash_map::HashMap;
use std::hash::Hash;
use core::fmt::Debug;

#[derive(PartialEq, Debug)]
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
        self.table.insert(val.clone(), DSPtr::Top);
    }

    // Union: add a to b's set
    pub fn union(&mut self, a: &T, b: &T) {
        let a_top = self.find(&a);
        let b_top = self.find(&b);
        self.table.insert(a_top.clone(), DSPtr::Ptr(b_top.clone()));
    }

    // Find
    pub fn find(&mut self, val: &T) -> T {
        let mut current = val;
        while let DSPtr::Ptr (above) = self.table.get(current).expect("Value not in disjoint set") {
            current = above;
        }
        let top = current.clone();
        if current != val {
            self.table.insert(val.clone(), DSPtr::Ptr(top));
        }
        top
    }
}
