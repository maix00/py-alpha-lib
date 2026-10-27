//! Installs a shared native factor DAG and its consumers on an existing engine.
use crate::{
    FactorOutputSink, FactorUpdate,
    family::{ConsumerSpec, FamilyRule, FamilyStrategy, SharedRankings},
    nautilus::FactorActor,
    output::JsonFactorSink,
    replay::FactorReplayHandle,
};
use nautilus_backtest::engine::BacktestEngine;
use nautilus_common::msgbus;
use nautilus_model::data::BarType;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::{cell::RefCell, rc::Rc};

struct SharedPublication {
    topic: nautilus_common::msgbus::mstr::MStr<nautilus_common::msgbus::mstr::Topic>,
    count: Arc<AtomicUsize>,
    trace: Option<JsonFactorSink>,
    rankings: Rc<RefCell<SharedRankings>>,
}
impl FactorOutputSink for SharedPublication {
    fn write_update(&mut self, update: &FactorUpdate) -> anyhow::Result<()> {
        if let Some(trace) = self.trace.as_mut() {
            trace.write_update(update)?;
        }
        self.rankings.borrow_mut().update(update)?;
        msgbus::publish_any(self.topic, update);
        self.count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn finish(&mut self) -> anyhow::Result<()> {
        if let Some(trace) = self.trace.as_mut() {
            trace.finish()?;
        }
        Ok(())
    }
}

pub struct FamilyRuntimeHandle {
    outcome: FactorReplayHandle,
    slices: Arc<AtomicUsize>,
    consumers: Vec<Arc<AtomicUsize>>,
    pub node_count: usize,
    pub consumer_strategy_ids: Vec<(String,String)>,
    rankings: Rc<RefCell<SharedRankings>>,
    actor: std::rc::Weak<std::cell::UnsafeCell<dyn nautilus_common::actor::Actor>>,
}

impl FamilyRuntimeHandle {
    /// Read one existing output from the Actor's single latest frame.
    /// No computation, history buffer or per-publication copy is introduced.
    pub fn latest_output(&self, output: usize) -> anyhow::Result<(i64, i64, Vec<f64>)> {
        use nautilus_common::actor::registry::{get_actor, try_get_actor_unchecked};
        let id = ustr::Ustr::from("shared-factors");
        let installed = self.actor.upgrade().ok_or_else(|| anyhow::anyhow!("factor runtime released"))?;
        let current = get_actor(&id).ok_or_else(|| anyhow::anyhow!("factor runtime unavailable"))?;
        anyhow::ensure!(Rc::ptr_eq(&installed, &current), "factor runtime superseded by another run");
        let actor = try_get_actor_unchecked::<FactorActor<crate::plan::FactorPlanEngine>>(&id)
            .ok_or_else(|| anyhow::anyhow!("factor runtime type mismatch"))?;
        let update = actor.last_update().ok_or_else(|| anyhow::anyhow!("no factor output yet"))?;
        let width = update.products.len();
        let start = output.checked_mul(width).ok_or_else(|| anyhow::anyhow!("output overflow"))?;
        let end = start.checked_add(width).ok_or_else(|| anyhow::anyhow!("output overflow"))?;
        let values = update.values.get(start..end).ok_or_else(|| anyhow::anyhow!("output outside plan"))?;
        Ok((update.timestamp_ns, update.available_at_ns, values.to_vec()))
    }

    pub fn raise_if_failed(&self) -> anyhow::Result<()> {
        self.outcome.ensure_success()
    }
    pub fn ensure_success(&self) -> anyhow::Result<()> {
        self.outcome.ensure_success()?;
        let count = self.slice_count();
        anyhow::ensure!(count > 0, "factor runtime published no slices");
        anyhow::ensure!(
            self.consumers
                .iter()
                .all(|c| c.load(Ordering::Relaxed) == count),
            "a strategy missed shared factor publications"
        );
        Ok(())
    }
    pub fn rank_passes(&self) -> usize {
        self.rankings.borrow().passes
    }
    pub fn slice_count(&self) -> usize {
        self.slices.load(Ordering::Relaxed)
    }
    pub fn consumer_counts(&self) -> Vec<usize> {
        self.consumers
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect()
    }
}

pub fn install_family_runtime(
    engine: &mut BacktestEngine,
    plan_json: &str,
    products: &[String],
    bar_types: &[BarType],
    specs: Vec<ConsumerSpec>,
    calendar: Vec<(i64, i64)>,
    trace_path: Option<&str>,
) -> anyhow::Result<FamilyRuntimeHandle> {
    anyhow::ensure!(
        !specs.is_empty() && !products.is_empty(),
        "empty factor consumers or products"
    );
    anyhow::ensure!(
        products
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            == products.len(),
        "duplicate frozen factor product"
    );
    let mut actor = FactorActor::from_plan_json("shared-factors", products.len(), plan_json)?;
    let outputs = actor.output_count();
    let node_count = actor.node_count();
    anyhow::ensure!(
        actor.curve_channels() == 0,
        "term-curve event stream is required for this plan"
    );
    let boundaries = actor.boundary_names().to_vec();
    actor.configure_calendar(&boundaries, calendar)?;
    let columns = actor.input_columns().to_vec();
    let fields = columns.iter().map(String::as_str).collect::<Vec<_>>();
    actor.validate_bar_columns(&fields)?;
    actor.expect_products(&products.iter().map(String::as_str).collect::<Vec<_>>());
    actor.configure_bar_inputs(bar_types, &fields)?;
    let error = Arc::new(Mutex::new(None));
    let instruments: Vec<_> = bar_types.iter().map(|t| t.instrument_id()).collect();
    anyhow::ensure!(
        instruments
            .iter()
            .all(|id| engine.kernel().cache.borrow().instrument(id).is_some()),
        "factor instrument is absent from configured engine"
    );
    let topic = "factor.shared-factors".to_string();
    let rankings = Rc::new(RefCell::new(SharedRankings::new(
        products.to_vec(),
        specs
            .iter()
            .filter(|spec| matches!(spec.rule, FamilyRule::CrossSectionalGroup { .. }))
            .map(|spec| spec.signal_output)
            .collect(),
    )));
    let mut strategies = Vec::new();
    let mut consumers = Vec::new();
    let mut ids = std::collections::HashSet::new();
    let reserved_tags:std::collections::HashSet<String>=specs.iter().map(|s|s.strategy_id.rsplit('-').next().unwrap().to_owned()).collect();
    let mut used_tags=std::collections::HashSet::new();
    let mut consumer_strategy_ids=Vec::with_capacity(specs.len());
    for (ordinal,mut spec) in specs.into_iter().enumerate() {
        anyhow::ensure!(
            ids.insert(spec.strategy_id.clone()),
            "duplicate strategy consumer id"
        );
        anyhow::ensure!(
            spec.signal_output < outputs,
            "consumer signal output is outside the plan"
        );
        let logical_id=spec.strategy_id.clone();
        let tag=spec.strategy_id.rsplit('-').next().unwrap().to_owned();
        if !used_tags.insert(tag) {
            let mut suffix=ordinal;
            loop {
                let tag=format!("factor{suffix}");
                if !reserved_tags.contains(&tag) && used_tags.insert(tag.clone()) {
                    spec.strategy_id=format!("{}-{tag}",spec.strategy_id);
                    break;
                }
                suffix+=1;
            }
        }
        consumer_strategy_ids.push((logical_id,spec.strategy_id.clone()));
        let strategy = FamilyStrategy::new(
            spec,
            instruments.clone(),
            rankings.clone(),
            topic.clone(),
            error.clone(),
        )?;
        consumers.push(strategy.publications.clone());
        strategies.push(strategy);
    }
    let slices = Arc::new(AtomicUsize::new(0));
    let trace = trace_path
        .map(|path| {
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .map(|file| JsonFactorSink::new(file, None))
        })
        .transpose()?;
    actor.set_output_sink(SharedPublication {
        topic: topic.as_str().into(),
        count: slices.clone(),
        trace,
        rankings: rankings.clone(),
    });
    let failure = error.clone();
    actor.set_error_sink(move |message| {
        let mut failure = failure.lock().unwrap();
        if failure.is_none() {
            *failure = Some(message.to_owned());
        }
    });
    for strategy in strategies {
        engine.add_strategy(strategy)?;
    }
    engine.add_actor(actor)?;
    let actor = nautilus_common::actor::registry::get_actor(&ustr::Ustr::from("shared-factors"))
        .ok_or_else(|| anyhow::anyhow!("factor actor was not registered"))?;
    Ok(FamilyRuntimeHandle {
        actor: Rc::downgrade(&actor),
        outcome: FactorReplayHandle { error },
        slices,
        consumers,
        node_count,
        consumer_strategy_ids,
        rankings,
    })
}
