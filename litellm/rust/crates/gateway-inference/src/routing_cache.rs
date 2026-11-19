use std::{collections::HashMap, time::Duration};

use litellm_cache::Error;
use litellm_error::Source;
use litellm_redis_store::{
    Number, Remote, TierPolicy,
    router_usage::{COUNTER_TTL, Counter, Kind, RouterUsage, Scope},
};
use litellm_router_core::{
    HealthState, InMemoryRouterCache, Outcomes, RouterCache, UsageCounts, UsageWindow,
    health::CooldownValue,
};

/// The router's usage store as the proxy builds it: `DualCache`'s `redis_batch_cache_expiry`
/// default, `DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE` and no `default_in_memory_ttl`.
const USAGE_POLICY: TierPolicy = TierPolicy {
    refresh_after: Duration::from_secs(10),
    tracked_reads: 1000,
    default_in_memory_ttl: None,
};

/// The deployments' routing state: per-minute usage in the router's `global_router:` counters,
/// which Python's router counts in too, and the rest in this process.
pub struct RoutingCache {
    memory: InMemoryRouterCache,
    usage: RouterUsage,
}

impl RoutingCache {
    /// Usage counted in `remote` when there is one, else in this process only.
    pub fn new(remote: Option<Remote>) -> Self {
        Self {
            memory: InMemoryRouterCache::default(),
            usage: RouterUsage::new(remote, USAGE_POLICY),
        }
    }

    /// Each counter's value, `None` for one nothing counted.
    pub(crate) async fn counted(
        &self,
        counters: &[Counter],
    ) -> Result<Vec<Option<Number>>, litellm_redis_store::Error> {
        self.usage.get_many(counters).await
    }
}

impl Default for RoutingCache {
    fn default() -> Self {
        Self::new(None)
    }
}

pub(crate) fn counter(window: &UsageWindow, kind: Kind) -> Counter {
    Counter {
        scope: Scope::GlobalRouter,
        deployment_id: window.deployment_id.clone(),
        model: Some(window.model.clone()),
        window: window.minute.clone(),
        kind,
    }
}

/// The `int` Python passes for a whole count.
fn amount(value: f64) -> Number {
    match value.fract() == 0.0 && value.abs() < i64::MAX as f64 {
        true => Number::Int(value as i64),
        false => Number::Float(value),
    }
}

impl RouterCache for RoutingCache {
    async fn cooldowns(&self, ids: &[String]) -> Result<Vec<Option<CooldownValue>>, Error> {
        self.memory.cooldowns(ids).await
    }

    async fn put_cooldown(
        &self,
        id: &str,
        cooldown: &CooldownValue,
        ttl: Duration,
    ) -> Result<(), Error> {
        self.memory.put_cooldown(id, cooldown, ttl).await
    }

    async fn increment_allowed_fails(
        &self,
        id: &str,
        suffix: Option<&str>,
        ttl: Duration,
    ) -> Result<f64, Error> {
        self.memory.increment_allowed_fails(id, suffix, ttl).await
    }

    /// Usage that cannot be written never fails the call, as on the Python router.
    async fn add_usage(
        &self,
        window: &UsageWindow,
        tokens: f64,
        requests: f64,
    ) -> Result<(), Error> {
        let written = self
            .usage
            .increment_deployment(
                &window.deployment_id,
                Some(&window.model),
                &window.minute,
                amount(tokens),
                requests as i64,
            )
            .await;
        if let Err(error) = written {
            tracing::debug!(%error, "_increment_deployment_usage: usage was not recorded");
        }
        Ok(())
    }

    async fn add_request(&self, window: &UsageWindow) -> Result<(), Error> {
        self.usage
            .increment(&counter(window, Kind::Rpm), Number::Int(1), COUNTER_TTL)
            .await;
        Ok(())
    }

    async fn usage(&self, window: &UsageWindow) -> Result<UsageCounts, Error> {
        let counted = self
            .counted(&[counter(window, Kind::Tpm), counter(window, Kind::Rpm)])
            .await
            .map_err(|error| Error::Unavailable(Some(Source::new(error))))?;
        let count = |index: usize| counted.get(index).copied().flatten().map(Number::as_f64);
        Ok(UsageCounts {
            tokens: count(0),
            requests: count(1),
        })
    }

    async fn health_states(&self) -> Result<HashMap<String, HealthState>, Error> {
        self.memory.health_states().await
    }

    async fn outcomes(&self, id: &str) -> Result<Outcomes, Error> {
        self.memory.outcomes(id).await
    }

    async fn count_success(&self, id: &str) -> Result<(), Error> {
        self.memory.count_success(id).await
    }

    async fn count_failure(&self, id: &str) -> Result<(), Error> {
        self.memory.count_failure(id).await
    }
}
