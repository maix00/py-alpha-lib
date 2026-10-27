# alpha Nautilus adapter

Independent Rust adapter for `alpha-algo`, owned and versioned by alpha.
It contains the shared factor Actor/Plan integration and the public threshold
and cross-sectional group strategies. It does not contain a strategy library,
user permissions, frozen strategy registrations, RunSpec or job orchestration;
those are host responsibilities (for example TradeResearch).

## Dependency boundary

The core alpha workspace never depends on Nautilus. This adapter is a separate
Cargo workspace, with optional `nautilus` (Actor) and `nautilus-backtest`
(Catalog replay and public strategies) features. The current API contract pins
Nautilus Rust packages to 0.67.0. A host must supply the compatible source
revision, including the configured-engine replay interface; package version
alone does not prove interface compatibility. TradeResearch pins its Nautilus
submodule and patches these dependencies to that SAME checkout at its composition
root. No second engine, message bus, raw market cache or factor implementation
is created. This is a source-integrated adapter, not a claim that every released
Nautilus wheel contains these interfaces.

The host can test this adapter through its composed Cargo dependency graph:

```sh
python scripts/test_alpha_nautilus_adapter.py
```

This host-side test launcher reads source patches from the composition manifest
and runs the adapter workspace (including its dev-dependencies) against them.
It does not duplicate the dependency list or any adapter implementation. Framework-independent adapter/kernel tests can also run from
this crate without Nautilus features if the pinned dependencies are resolvable.

## Shared state and memory

Compile factors with identical inputs, product order, clock and availability
semantics into one Plan, then install one FactorActor. Consumers borrow the
same publication. Group consumers of the same signal share a frame-local
ranking; they do not independently compute factors or re-sort each group.
Orders and positions remain owned by Nautilus. Buffers and rolling state are
bounded by configured windows/pending slices; the adapter does not retain full
market or output histories. Optional JSONL diagnostics use a bounded writer.
Separate causal clocks or account/order state must not be incorrectly merged.

Session boundaries require the frozen calendar or authoritative custom events.
Sparse clock/curve support and end-to-end deployment are separate acceptance
items; this move does not relax the existing explicit failures.

Native consumers accept an optional frozen `product_indices` scope. They share
one factor publication/ranking, filter the ranking through a preallocated buffer
and never trade instruments outside that scope. Product-local factor ancestors
can be shared across scopes; cross-sectional ancestors must be specialized by
universe via alpha's plan compiler before installing the runtime. A host must
honor Nautilus OMS/account semantics: independent overlapping target strategies
require hedging positions rather than inventing per-strategy netting balances.
