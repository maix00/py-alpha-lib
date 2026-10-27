//! CURRENT/CANDIDATE search kernel shared by batch, incremental and bridge.

use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub enum MatchExpr {
    Current,
    Candidate,
    Input(usize),
    Unary(String,Box<MatchExpr>),
    Binary(String,Box<MatchExpr>,Box<MatchExpr>),
    Where(Box<MatchExpr>,Box<MatchExpr>,Box<MatchExpr>),
    Const(f64),
    Add(Box<MatchExpr>, Box<MatchExpr>),
    Sub(Box<MatchExpr>, Box<MatchExpr>),
    Mul(Box<MatchExpr>, Box<MatchExpr>),
    Div(Box<MatchExpr>, Box<MatchExpr>),
    Abs(Box<MatchExpr>),
    Gt(Box<MatchExpr>, Box<MatchExpr>),
    Lt(Box<MatchExpr>, Box<MatchExpr>),
    Ge(Box<MatchExpr>, Box<MatchExpr>),
    Le(Box<MatchExpr>, Box<MatchExpr>),
    Eq(Box<MatchExpr>, Box<MatchExpr>),
    Ne(Box<MatchExpr>, Box<MatchExpr>),
}

impl MatchExpr {
    pub fn validate(&self,width:usize)->bool {
        match self {
            Self::Input(index)=>*index<width,
            Self::Unary(op,a)=>matches!(op.as_str(),"neg"|"abs"|"sign"|"sqrt"|"log"|"tanh"|"not") && a.validate(width),
            Self::Binary(op,a,b)=>matches!(op.as_str(),"add"|"sub"|"mul"|"div"|"pow"|"max"|"min"|"gt"|"lt"|"ge"|"le"|"eq"|"ne"|"and"|"or") && a.validate(width) && b.validate(width),
            Self::Where(c,a,b)=>c.validate(width)&&a.validate(width)&&b.validate(width),
            Self::Abs(a)=>a.validate(width),
            Self::Add(a,b)|Self::Sub(a,b)|Self::Mul(a,b)|Self::Div(a,b)|Self::Gt(a,b)|Self::Lt(a,b)|Self::Ge(a,b)|Self::Le(a,b)|Self::Eq(a,b)|Self::Ne(a,b)=>a.validate(width)&&b.validate(width),
            _=>true,
        }
    }
    pub fn eval(&self,current:f64,candidate:f64)->f64 {self.eval_with(current,candidate,&[])}
    pub fn eval_with(&self, current: f64, candidate: f64, inputs:&[f64]) -> f64 {
        match self {
            Self::Current => current,
            Self::Candidate => candidate,
            Self::Input(index)=>inputs.get(*index).copied().unwrap_or(f64::NAN),
            Self::Unary(op,a)=>crate::algo::panel_plan::unary(op,a.eval_with(current,candidate,inputs)),
            Self::Binary(op,a,b)=>crate::algo::panel_plan::binary(op,a.eval_with(current,candidate,inputs),b.eval_with(current,candidate,inputs)),
            Self::Where(c,a,b)=>if c.eval_with(current,candidate,inputs)!=0.0 {a.eval_with(current,candidate,inputs)} else {b.eval_with(current,candidate,inputs)},
            Self::Const(value) => *value,
            Self::Add(a, b) => a.eval_with(current, candidate,inputs) + b.eval_with(current, candidate,inputs),
            Self::Sub(a, b) => a.eval_with(current, candidate,inputs) - b.eval_with(current, candidate,inputs),
            Self::Mul(a, b) => a.eval_with(current, candidate,inputs) * b.eval_with(current, candidate,inputs),
            Self::Div(a, b) => a.eval_with(current, candidate,inputs) / b.eval_with(current, candidate,inputs),
            Self::Abs(a) => a.eval_with(current, candidate,inputs).abs(),
            Self::Gt(a, b) => f64::from(a.eval_with(current, candidate,inputs) > b.eval_with(current, candidate,inputs)),
            Self::Lt(a, b) => f64::from(a.eval_with(current, candidate,inputs) < b.eval_with(current, candidate,inputs)),
            Self::Ge(a, b) => f64::from(a.eval_with(current, candidate,inputs) >= b.eval_with(current, candidate,inputs)),
            Self::Le(a, b) => f64::from(a.eval_with(current, candidate,inputs) <= b.eval_with(current, candidate,inputs)),
            Self::Eq(a, b) => f64::from(a.eval_with(current, candidate,inputs) == b.eval_with(current, candidate,inputs)),
            Self::Ne(a, b) => f64::from(a.eval_with(current, candidate,inputs) != b.eval_with(current, candidate,inputs)),
        }
    }
}

/// Explicit partition and independent lookback. Neither is inferred from the other.
#[derive(Clone,Debug)]
pub struct SearchConfig {
    pub partition:Option<usize>,pub lookback:Option<usize>,pub gap_ns:Option<i64>,
    pub nearest:bool,pub default_scope:bool,pub default:f64,pub include_current:bool,pub propagate_nan:bool,
}
#[derive(Default)]
pub struct BarSearchState {
    ordinal:u64,base:u64,last_time:Option<i64>,
    history:VecDeque<(u64,f64)>,events:VecDeque<u64>,
}
impl BarSearchState {
    pub fn update(&mut self,time:i64,value:f64,inputs:&[f64],predicate:Option<&MatchExpr>,config:&SearchConfig,reset:bool)->f64 {
        let now=self.ordinal;
        if reset || config.partition.is_some_and(|n|now%n as u64==0)
            || config.gap_ns.is_some_and(|gap|self.last_time.is_some_and(|last|time-last>=gap)) {
            self.base=now;self.history.clear();self.events.clear();
        }
        self.last_time=Some(time);
        let age=now-self.base;
        let age=config.lookback.map_or(age,|n|age.min(n as u64));
        let fallback=if config.default_scope {age as f64}else{config.default};
        let minimum=config.lookback.map_or(self.base,|n|self.base.max(now.saturating_sub(n as u64)));
        while self.history.front().is_some_and(|(i,_)|*i<minimum) {self.history.pop_front();}
        while self.events.front().is_some_and(|i|*i<minimum) {self.events.pop_front();}
        let output=if let Some(predicate)=predicate {
            let mut result=fallback;
            if value.is_finite() {
                let check=|candidate:f64| {
                    let truth=predicate.eval_with(value,candidate,inputs);
                    candidate.is_finite() && truth.is_finite() && truth!=0.0
                };
                if config.include_current && config.nearest && check(value) {result=0.0;}
                else {
                    let hit=if config.nearest {self.history.iter().rev().find(|(_,v)|check(*v))}
                            else {self.history.iter().find(|(_,v)|check(*v))};
                    if let Some((at,_))=hit {result=(now-at) as f64;}
                    else if config.include_current && check(value) {result=0.0;}
                }
            }
            self.history.push_back((now,value));
            // Next update can need at most lookback previous observations.
            if let Some(n)=config.lookback {while self.history.len()>n {self.history.pop_front();}}
            result
        } else {
            let truth=value.is_finite() && value!=0.0;
            if truth && config.include_current {self.events.push_back(now);}
            let hit=if config.nearest {self.events.back()}else{self.events.front()};
            let result=hit.map_or(fallback,|at|(now-at) as f64);
            if truth && !config.include_current {self.events.push_back(now);}
            // Without a sliding limit, first/last suffice for event age.
            if config.lookback.is_none() && self.events.len()>2 {
                let last=self.events.pop_back().unwrap();self.events.truncate(1);self.events.push_back(last);
            }
            if config.propagate_nan && value.is_nan() {f64::NAN}else{result}
        };
        self.ordinal+=1;output
    }
}


#[cfg(test)]
mod unified_search_tests {
    use super::*;
    fn config()->SearchConfig {SearchConfig{partition:Some(3),lookback:None,gap_ns:None,nearest:true,
        default_scope:false,default:9.0,include_current:false,propagate_nan:true}}
    #[test]
    fn partition_and_lookback_are_independent() {
        let predicate=MatchExpr::Lt(Box::new(MatchExpr::Candidate),Box::new(MatchExpr::Current));
        let mut partition=BarSearchState::default();let mut sliding=BarSearchState::default();
        let p=config();let mut s=p.clone();s.partition=None;s.lookback=Some(3);
        let pv:Vec<_>=(0..6).map(|t|partition.update(t,t as f64,&[],Some(&predicate),&p,false)).collect();
        let sv:Vec<_>=(0..6).map(|t|sliding.update(t,t as f64,&[],Some(&predicate),&s,false)).collect();
        assert_eq!(pv,vec![9.0,1.0,1.0,9.0,1.0,1.0]);
        assert_eq!(sv,vec![9.0,1.0,1.0,1.0,1.0,1.0]);
    }
    #[test]
    fn long_nearest_self_hit_keeps_a_bounded_history() {
        let mut state=BarSearchState::default();let mut c=config();c.partition=None;c.lookback=Some(3);c.include_current=true;
        for t in 0..10000 {assert_eq!(state.update(t,1.0,&[],Some(&MatchExpr::Const(1.0)),&c,false),0.0);}
        assert_eq!(state.history.len(),3);
    }
    #[test]
    fn search_capacity_stabilizes_after_warmup() {
        let mut matched=BarSearchState::default();let mut events=BarSearchState::default();
        let mut c=config();c.partition=Some(120);c.lookback=Some(60);
        let predicate=MatchExpr::Lt(Box::new(MatchExpr::Candidate),Box::new(MatchExpr::Current));
        let mut capacities=(0,0);
        for t in 0..100000 {
            matched.update(t,(t%197) as f64,&[],Some(&predicate),&c,false);
            events.update(t,1.0,&[],None,&c,false);
            if t==999 {capacities=(matched.history.capacity(),events.events.capacity());}
            if t>=1000 {assert_eq!((matched.history.capacity(),events.events.capacity()),capacities);}
            assert!(matched.history.len()<=60 && events.events.len()<=61);
        }
    }

}
