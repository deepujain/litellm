//! The proxy's `dynamic_rate_limiter_v3` callback as a callback of the stage: a model group's
//! `tpm` and `rpm` are shared out by priority, and once the group is saturated each priority is
//! held to the share `litellm_settings.priority_reservation` reserves it.
//!
//! Every decision is the engine's in `litellm-callbacks-model-capacity`, which the proxy runs too.
//! This file only tells it who is calling and for which model, and settles the tokens the call
//! used. It counts on the key limiter's counters, so the proxy and the gateway see one another's
//! calls on a shared Redis. It counts every caller of a model group that sets `tpm` or `rpm`, the
//! master key included, and the statuses it publishes are the answer the key limiter, registered
//! after it, starts its own from.
//!
//! A key's own per-model limit is a share that borrows the group's idle capacity when
//! `priority_reservation_settings.model_limit_borrowing_config.borrow_idle_capacity` lists `key`:
//! the key limiter leaves that limit alone, and here the key is held to it only once the group is
//! saturated, and to `max_borrow_multiple` times it before. A share that holds tokens reserves
//! what the body asks for before the call and settles it to what the call used.
//!
//! Accepted differences: a route that tells no usage settles no tokens, so what a share reserved
//! for it is given back. `/v1/messages`, chat completions and Responses tell theirs, OCR counts
//! none and transcription tells none yet. The share and throttle metrics are Prometheus's, which
//! is a callback of its own. A refusal carries its message once: the proxy's
//! error body repeats it under `provider_specific_fields`, which is only how its `HTTPException`
//! is converted.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Abort, Activation, Callback, Field, FieldSet, Hook, HookInput, HookOutput, Manifest, Outcome,
    Patch, Scalar, Selection, Subscription, UsageRecord, Verdict, typed,
};
use litellm_callbacks_model_capacity::{
    BorrowerKind, BorrowingLimits, CapacityError, ClassShareEntity, Engine, Fact, Facts, GroupInfo,
    KeyFacts, LoggedGroup, NAME, PreCall, Priority, Request as CapacityRequest, SaturationFallback,
    Settings as EngineSettings, Settlement, SharedStash, StoreLimiter, TokenRateLimitType,
    UsageTokens,
};
use litellm_callbacks_parallel_request_limiter::{
    PolicyFacts, TokenRateLimitType as LimiterTokenType,
};
use litellm_config::{BorrowingEntity, Config, PriorityReservation};
use litellm_gateway_auth::{
    AuthenticatedRequest,
    keys::{KeyLimits, ModelLimits},
};
use litellm_gateway_inference::group_limits;
use litellm_python_compat::json::loaded_str;
use litellm_redis_store::{RequestRedis, active_teams::ActiveTeams, rate_limits::RateLimits};
use litellm_router_core::Router;
use serde_json::{Map, Value, json};

use super::{Registered, parallel_request_limiter};
use crate::host::{Asked, NativeCapacityHost, cap_output};

/// The settlement flushes its own post-call batch, so this deadline never sends it.
const POST_CALL_DEADLINE: Duration = Duration::from_secs(1);

/// What the config says of the dynamic rate limiter, as the proxy's `CapacityHost.settings` hands
/// it to the engine.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    engine: EngineSettings,
    limits: parallel_request_limiter::Settings,
    borrowing: Borrowing,
}

/// `model_limit_borrowing_config` as it reaches a caller the gateway serves: whether a key's own
/// per-model limits are shares that borrow idle capacity, and how far past one it may run. A team
/// or a customer organization that borrows is no caller the gateway authenticates.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Borrowing {
    keys: bool,
    max_borrow_multiple: Option<f64>,
}

impl Borrowing {
    /// Nobody borrows unless `litellm_settings.callbacks` names `dynamic_rate_limiter_v3`, since
    /// the proxy's `dynamic_rate_limiter_covers` covers no model without it.
    pub fn read(config: &Config) -> Self {
        let litellm = &config.litellm_settings;
        match &litellm.priority_reservation_settings {
            Some(settings) if Settings::named(config) => {
                let written = &settings.model_limit_borrowing_config;
                Self {
                    keys: written.borrow_idle_capacity.contains(&BorrowingEntity::Key),
                    max_borrow_multiple: written.max_borrow_multiple,
                }
            }
            _ => Self::default(),
        }
    }

    pub(crate) fn keys(self) -> bool {
        self.keys
    }

    /// `dynamic_rate_limiter_covers`: whether a key's limit on `group` is a share held here and
    /// not by the key limiter, which is so for a model group that sets `tpm` or `rpm`.
    pub(crate) fn covers(self, router: &Router, group: &str) -> bool {
        self.keys
            && group_limits(router, group)
                .is_some_and(|limits| limits.tpm.is_some() || limits.rpm.is_some())
    }
}

fn reserved(reservation: &PriorityReservation) -> Value {
    match reservation {
        PriorityReservation::Share(share) => json!(share),
        PriorityReservation::Sized { unit, value } => {
            json!({"type": <&'static str>::from(*unit), "value": value})
        }
    }
}

fn class_share_entity(entity: litellm_config::ClassShareEntity) -> ClassShareEntity {
    match entity {
        litellm_config::ClassShareEntity::Team => ClassShareEntity::Team,
        litellm_config::ClassShareEntity::Key => ClassShareEntity::Key,
        litellm_config::ClassShareEntity::CustomerOrganization => {
            ClassShareEntity::CustomerOrganization
        }
        litellm_config::ClassShareEntity::CustomerMember => ClassShareEntity::CustomerMember,
    }
}

impl Settings {
    /// Whether `litellm_settings.callbacks` names `dynamic_rate_limiter_v3`.
    pub(crate) fn named(config: &Config) -> bool {
        config
            .litellm_settings
            .callbacks
            .iter()
            .flatten()
            .any(|callback| callback.as_str() == Some(NAME))
    }

    /// `None` unless `litellm_settings.callbacks` names `dynamic_rate_limiter_v3`. The window and
    /// the tokens a limit counts are the key limiter's. `licensed` is whether `LITELLM_LICENSE` is
    /// set at all, which is all the proxy asks before it honors a reserved priority.
    pub fn read(
        config: &Config,
        limits: parallel_request_limiter::Settings,
        licensed: bool,
    ) -> Option<Self> {
        let litellm = &config.litellm_settings;
        let written = litellm
            .priority_reservation_settings
            .clone()
            .unwrap_or_default();
        let engine = EngineSettings {
            priority_reservation: litellm.priority_reservation.as_ref().map(|reservations| {
                reservations
                    .iter()
                    .map(|(priority, reservation)| (json!(priority), reserved(reservation)))
                    .collect()
            }),
            default_priority: written.default_priority,
            saturation_threshold: written.saturation_threshold,
            class_split: written
                .priority_class_share_config
                .split_equally_among_active
                .iter()
                .copied()
                .map(class_share_entity)
                .collect(),
            saturation_check_cache_ttl: json!(written.saturation_check_cache_ttl),
            window_size: limits.window_size(),
            rate_limit_type: match limits.token_type() {
                LimiterTokenType::Input => TokenRateLimitType::Input,
                LimiterTokenType::Output => TokenRateLimitType::Output,
                LimiterTokenType::Total => TokenRateLimitType::Total,
            },
            licensed,
            policy: PolicyFacts::default(),
        };
        Self::named(config).then_some(Self {
            engine,
            limits,
            borrowing: Borrowing::read(config),
        })
    }
}

/// The proxy's `dynamic_rate_limiter_v3` over the limiter's counters.
pub struct ModelCapacity {
    manifest: Manifest,
    limiter: Arc<StoreLimiter>,
    /// The registry of callers a priority class's share is split among.
    active: Arc<ActiveTeams>,
    settings: EngineSettings,
    borrowing: Borrowing,
    /// Tells what a model group's deployments serve a minute.
    router: Router,
}

fn manifest() -> Manifest {
    Manifest {
        selection: Selection {
            activation: Activation::Required,
            ..Selection::named(NAME)
        },
        subscriptions: vec![
            Subscription {
                reads: FieldSet::of(&[Field::Model, Field::RequestBody]),
                writes: FieldSet::of(&[Field::RateLimitStatuses, Field::OptionalParams]),
                may_abort: true,
                ..Subscription::on(Hook::PreCall)
            },
            Subscription::on(Hook::PostCallSuccess),
            Subscription {
                writes: FieldSet::of(&[Field::ResponseHeaders]),
                ..Subscription::on(Hook::ResponseHeaders)
            },
            Subscription::on(Hook::PostCallFailure),
            Subscription {
                reads: FieldSet::of(&[Field::Outcome, Field::Usage, Field::LoggedModelGroup]),
                ..Subscription::on(Hook::AttemptEnded)
            },
        ],
        has_request_state: true,
    }
}

/// One request as the engine counts it: who it counts under, and what the hooks it ran left for
/// the ones after them.
pub struct Counted {
    capacity: Arc<ModelCapacity>,
    key: KeyFacts,
    priority: Option<Arc<str>>,
    /// The key's own per-model limits, which are its shares when keys borrow.
    limits: KeyLimits,
    stash: SharedStash,
    /// The pre-call hook counted the call against a model group, so its tokens settle there.
    admitted: AtomicBool,
    /// The answer is complete, so it names the priority it was counted under.
    complete: AtomicBool,
}

impl Counted {
    fn of(capacity: Arc<ModelCapacity>, identity: &AuthenticatedRequest) -> Self {
        let priority = identity.caller().authentication().priority.clone();
        let caller = parallel_request_limiter::Caller::of(identity);
        Self {
            capacity,
            limits: caller.limits,
            key: KeyFacts {
                api_key: Some(caller.api_key),
                priority: priority.as_deref().map(|priority| Priority {
                    value: json!(priority),
                    hash_error: None,
                }),
                ..KeyFacts::default()
            },
            priority,
            stash: SharedStash::default(),
            admitted: AtomicBool::new(false),
            complete: AtomicBool::new(false),
        }
    }
}

/// A stream the client left tells no hook, so what its share still holds for it is given back
/// here.
impl Drop for Counted {
    fn drop(&mut self) {
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        if self.stash.lock().holds_tokens() {
            let (capacity, stash) = (self.capacity.clone(), self.stash.clone());
            runtime.spawn(async move { capacity.give_back(&stash).await });
        }
    }
}

/// `RateLimitedModel.limit_in`: the limit under the name asked for, else under its group.
fn limit_in(limits: &ModelLimits, requested: &str, group: &str) -> Option<i64> {
    limits.get(requested).or_else(|| limits.get(group)).copied()
}

/// The members of `capped` that `sent` does not hold the same, which is the output cap written.
fn written(sent: &Map<String, Value>, capped: Value) -> Map<String, Value> {
    match capped {
        Value::Object(capped) => capped
            .into_iter()
            .filter(|(name, value)| sent.get(name) != Some(value))
            .collect(),
        _ => Map::new(),
    }
}

fn tokens(count: &Option<Scalar>) -> i64 {
    match count {
        Some(Scalar::Int(count)) => i64::try_from(*count).unwrap_or(i64::MAX),
        _ => 0,
    }
}

/// The proxy's 503 when `fail_closed_rate_limit_enforcement` could not verify a counter. The
/// gateway only serves a config that leaves it off, so nothing else is ever raised here.
fn unverifiable(error: &CapacityError) -> Abort {
    Abort::new(503, error.to_string())
}

impl ModelCapacity {
    /// `counters` are the ones the key limits count on, and `active` is the registry of callers a
    /// priority class's share is split among.
    pub fn new(
        settings: Settings,
        counters: Arc<RateLimits>,
        active: ActiveTeams,
        router: Router,
    ) -> Arc<Self> {
        Arc::new(Self {
            manifest: manifest(),
            limiter: Arc::new(StoreLimiter::new(settings.limits.engine(counters), None)),
            active: Arc::new(active),
            settings: settings.engine,
            borrowing: settings.borrowing,
            router,
        })
    }

    pub(crate) fn registered(self: &Arc<Self>) -> Registered {
        let capacity = self.clone();
        Registered::for_caller(self.clone(), move |identity| {
            Counted::of(capacity.clone(), identity)
        })
    }

    /// The engine for one hook, whose host reads the call that hook is about. `asked` is the
    /// body a share that holds tokens estimates.
    fn engine(&self, asked: Option<Asked>) -> Engine {
        Engine {
            limiter: self.limiter.clone(),
            host: Arc::new(NativeCapacityHost::new(self.active.clone(), asked)),
        }
    }

    /// `model_limit_borrowers`: the key's own limit on `model` as a share that borrows, when keys
    /// borrow and the dynamic rate limiter covers the model's group.
    fn borrowers(&self, counted: &Counted, model: &str) -> Vec<BorrowingLimits> {
        let registry = self.router.registry();
        let group = registry.alias_of(model).unwrap_or(model);
        let limit = |limits: &ModelLimits| limit_in(limits, model, group);
        let (tpm, rpm) = (
            limit(&counted.limits.model_tpm),
            limit(&counted.limits.model_rpm),
        );
        let Some(api_key) = &counted.key.api_key else {
            return Vec::new();
        };
        if (tpm.is_none() && rpm.is_none()) || !self.borrowing.covers(&self.router, group) {
            return Vec::new();
        }
        vec![BorrowingLimits {
            kind: BorrowerKind::Key,
            entity_id: api_key.clone(),
            alias: None,
            group: group.to_owned(),
            tpm,
            rpm,
            max_borrow_multiple: self.borrowing.max_borrow_multiple,
        }]
    }

    /// The output cap the proxy writes into the body of an admitted call whose share holds
    /// tokens, as the params it adds.
    fn output_cap(counted: &Counted, sent: Option<&Value>) -> Map<String, Value> {
        let cap = counted.stash.lock().output_cap.take();
        let (Some((tpm_share, call_type)), Some(Value::Object(sent))) = (cap, sent) else {
            return Map::new();
        };
        let mut capped = Value::Object(sent.clone());
        match cap_output(
            &mut capped,
            call_type.as_deref().unwrap_or_default(),
            tpm_share,
        ) {
            Ok(()) => written(sent, capped),
            Err(_) => Map::new(),
        }
    }

    /// Gives back what the call's shares still hold for it: `release_admission`, on the
    /// post-call batch the proxy's failure hook rides.
    async fn give_back(&self, stash: &SharedStash) {
        if !stash.lock().holds_tokens() {
            return;
        }
        let request = RequestRedis::new(tokio::runtime::Handle::current(), POST_CALL_DEADLINE);
        let released = self
            .engine(None)
            .release_admission(&self.settings, stash, Some(&request))
            .await;
        request.flush_post().await;
        if let Err(error) = released {
            tracing::error!(
                "[Non-Blocking] Error in async_post_call_failure_hook callback: {}",
                error.python_message()
            );
        }
    }

    /// What `model`'s group serves a minute. `None` for a name that is no model group, which the
    /// proxy's hook leaves alone.
    fn group(&self, model: &str) -> Option<GroupInfo> {
        let limits = group_limits(&self.router, model)?;
        Some(GroupInfo {
            tpm: limits.tpm,
            rpm: limits.rpm,
            input_cost_per_token: None,
            output_cost_per_token: None,
        })
    }

    async fn pre_call(&self, input: &HookInput, counted: &Counted) -> HookOutput {
        let Some(model) = input.fields.get::<typed::Model>()? else {
            return Ok(None);
        };
        let Some(group) = self.group(&model) else {
            return Ok(None);
        };
        let sent = input.fields.raw(Field::RequestBody)?;
        let borrowers = self.borrowers(counted, &model);
        let asked = sent.filter(|_| !borrowers.is_empty()).map(|sent| Asked {
            body: sent.clone(),
            counted: self.settings.rate_limit_type,
        });
        let facts = Facts {
            borrowing: Some(Fact::Ok(borrowers)),
            ..Facts::default()
        };
        let request = CapacityRequest {
            key: &counted.key,
            call_type: input.call.as_ref().map(|call| call.kind.call_type.as_str()),
            settings: &self.settings,
            stash: &counted.stash,
            fallback: SaturationFallback::default(),
            redis: None,
            facts: &facts,
        };
        match self.engine(asked).pre_call(&request, &model, &group).await {
            PreCall::Allow(published) => {
                counted.admitted.store(true, Ordering::Release);
                let statuses = published.map_or_else(Patch::default, |published| {
                    Patch::default().set(Field::RateLimitStatuses, json!(published))
                });
                let cap = Self::output_cap(counted, sent);
                let patch = if cap.is_empty() {
                    statuses
                } else {
                    statuses.merge(Field::OptionalParams, cap)
                };
                Ok((!patch.ops.is_empty()).then_some(Verdict::Patch(patch)))
            }
            PreCall::Refuse(refusal) => Ok(Some(Verdict::Abort(refusal.into_abort()))),
            PreCall::Raise(error) => Ok(Some(Verdict::Abort(unverifiable(&error)))),
        }
    }

    /// `x-litellm-priority` and `x-litellm-rate-limiter-version`, which the proxy's post-call hook
    /// adds to every complete answer, whatever model it was for.
    fn priority_headers(counted: &Counted) -> Map<String, Value> {
        Engine::post_call_headers(&counted.key)
            .into_iter()
            .map(|(name, value)| (name, Value::from(loaded_str(&value))))
            .collect()
    }

    /// The success log's half: the tokens the call used are added to the group's saturation and
    /// to its priority's counter, and what a share reserved is settled to them, on the post-call
    /// batch the proxy's settlement rides.
    async fn settle(&self, counted: &Counted, used: &UsageRecord, model_group: String) {
        let used = UsageTokens {
            prompt_tokens: tokens(&used.prompt_tokens),
            completion_tokens: tokens(&used.completion_tokens),
            total_tokens: tokens(&used.total_tokens),
            cached_tokens: tokens(&used.cached_tokens),
        };
        let group = LoggedGroup {
            truthy: !model_group.is_empty(),
            text: model_group,
            keyed: true,
            hash_error: None,
        };
        let settlement = Settlement {
            model_group: Some(&group),
            team_id: None,
            team_alias: None,
            key_hash: counted.key.api_key.as_deref(),
            key_alias: None,
            customer_org_id: None,
            customer_org_alias: None,
        };
        let request = RequestRedis::new(tokio::runtime::Handle::current(), POST_CALL_DEADLINE);
        let settled = self
            .engine(None)
            .settle_success(
                &self.settings,
                Some(&counted.stash),
                &settlement,
                counted.priority.as_deref(),
                UsageTokens::settled(Some(&used), self.settings.rate_limit_type),
                Some(&request),
            )
            .await;
        request.flush_post().await;
        if let Err(error) = settled {
            tracing::error!(
                "Error in dynamic rate limiter success event: {}",
                error.python_message()
            );
        }
    }

    /// A failed call is settled by the failure hook before this, which gives back all it held. A
    /// call the route told no usage of settles nothing, and what a share holds for it is given
    /// back when its state is dropped.
    async fn log(&self, input: &HookInput, counted: &Counted) -> HookOutput {
        let succeeded = input.fields.get::<typed::Outcome>()? == Some(Outcome::Success);
        if !succeeded || !counted.admitted.load(Ordering::Acquire) {
            return Ok(None);
        }
        let used = input
            .fields
            .get::<typed::Usage>()?
            .and_then(|usage| usage.response);
        let group = input
            .fields
            .get::<typed::LoggedModelGroup>()?
            .and_then(|group| match group.value {
                Some(Scalar::Text(group)) => Some(group),
                _ => None,
            });
        if let Some((used, group)) = used.zip(group) {
            self.settle(counted, &used, group).await;
        }
        Ok(None)
    }
}

impl Callback for ModelCapacity {
    type State = Counted;

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn handle<'a>(
        &'a self,
        input: &'a HookInput,
        state: Option<&'a Counted>,
    ) -> BoxFuture<'a, HookOutput> {
        Box::pin(async move {
            let Some(counted) = state else {
                return Ok(None);
            };
            match input.hook {
                Hook::PreCall => self.pre_call(input, counted).await,
                Hook::PostCallSuccess => {
                    counted.complete.store(true, Ordering::Release);
                    Ok(None)
                }
                Hook::ResponseHeaders if counted.complete.load(Ordering::Acquire) => {
                    Ok(Some(Verdict::Patch(Patch::default().merge(
                        Field::ResponseHeaders,
                        Self::priority_headers(counted),
                    ))))
                }
                Hook::PostCallFailure => {
                    self.give_back(&counted.stash).await;
                    Ok(None)
                }
                Hook::AttemptEnded => self.log(input, counted).await,
                _ => Ok(None),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use axum::{
        Router,
        body::{Body, to_bytes},
        http::{Request, StatusCode},
        response::Response,
    };
    use litellm_gateway_auth::{
        hash_token,
        keys::{KeyHash, KeyLookup, KeyStatus, LookupFuture},
    };
    use litellm_secret_value::SecretValue;
    use rstest::rstest;
    use tower::ServiceExt;
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::body_partial_json};

    use super::*;
    use crate::{
        KeyLimiter,
        test_support::{self, MASTER_KEY, Served, asking, send_as},
    };

    const RESERVED: &str = "litellm_settings:\n  callbacks: [dynamic_rate_limiter_v3]\n  priority_reservation: {prod: 0.9, dev: 0.1}\n";
    const SPLIT_BY_KEY: &str = "litellm_settings:\n  callbacks: [dynamic_rate_limiter_v3]\n  priority_reservation: {prod: 0.9, dev: 0.1}\n  priority_reservation_settings:\n    priority_class_share_config:\n      split_equally_among_active: [key]\n";

    /// Each key active under the priority its name carries: `sk-prod`, `sk-dev-1`, `sk-dev-2`.
    /// `sk-dev-1` also holds one request a minute of its own.
    struct Prioritized;

    impl KeyLookup for Prioritized {
        fn lookup<'a>(&'a self, hash: &'a KeyHash) -> LookupFuture<'a> {
            Box::pin(async move {
                let named = |token: &str| hash == &KeyHash::from_token(&SecretValue::new(token));
                let priority = ["sk-prod", "sk-dev-1", "sk-dev-2", "sk-limited-dev"]
                    .into_iter()
                    .find(|token| named(token))
                    .map(|token| Arc::from(if token == "sk-prod" { "prod" } else { "dev" }));
                Ok(Some(KeyStatus::Active {
                    expires_at: None,
                    models: Arc::from([]),
                    limits: KeyLimits {
                        rpm: named("sk-limited-dev").then_some(1),
                        ..KeyLimits::default()
                    },
                    priority,
                }))
            })
        }
    }

    const BORROWING: &str = "litellm_settings:\n  callbacks: [dynamic_rate_limiter_v3]\n  priority_reservation_settings:\n    model_limit_borrowing_config:\n      borrow_idle_capacity: [key]\n";
    const BORROWING_TWICE_ITS_LIMIT: &str = "litellm_settings:\n  callbacks: [dynamic_rate_limiter_v3]\n  priority_reservation_settings:\n    model_limit_borrowing_config:\n      borrow_idle_capacity: [key]\n      max_borrow_multiple: 2\n";
    const KEY: &str = "sk-borrower";

    /// Every key active with the same limits.
    struct Holding(KeyLimits);

    impl KeyLookup for Holding {
        fn lookup<'a>(&'a self, _: &'a KeyHash) -> LookupFuture<'a> {
            Box::pin(async move {
                Ok(Some(KeyStatus::Active {
                    expires_at: None,
                    models: Arc::from([]),
                    limits: self.0.clone(),
                    priority: None,
                }))
            })
        }
    }

    fn by_model(model: &str, limit: i64) -> ModelLimits {
        Arc::new([(model.to_owned(), limit)].into_iter().collect())
    }

    /// The gateway for a key that holds `limits`, over the same groups as [`served`].
    async fn holding(limits: KeyLimits, litellm_settings: &str) -> Served {
        test_support::served(Arc::new(Holding(limits)), "", litellm_settings).await
    }

    /// The gateway over the group `claude`, which serves 10 requests and 100000 tokens a minute.
    async fn served(litellm_settings: &str) -> Served {
        test_support::served(Arc::new(Prioritized), "", litellm_settings).await
    }

    async fn call(app: &Router, token: &str, model: &str) -> Response {
        send_as(app, token, asking(model, Some(16))).await
    }

    async fn statuses(app: &Router, tokens: impl IntoIterator<Item = &str>) -> Vec<StatusCode> {
        let mut answered = Vec::new();
        for token in tokens {
            answered.push(call(app, token, "claude").await.status());
        }
        answered
    }

    #[rstest]
    #[case::a_reserved_priority("sk-prod", "prod")]
    #[case::no_priority(MASTER_KEY, "default")]
    #[tokio::test]
    async fn a_complete_answer_names_the_priority_it_was_counted_under(
        #[case] token: &str,
        #[case] priority: &str,
    ) {
        let Served { app, .. } = served(RESERVED).await;

        let answer = call(&app, token, "claude").await;

        assert_eq!(
            (
                answer.status(),
                answer.headers()["x-litellm-priority"].to_str().unwrap(),
                answer.headers()["x-litellm-rate-limiter-version"]
                    .to_str()
                    .unwrap(),
            ),
            (StatusCode::OK, priority, "v3")
        );
    }

    #[tokio::test]
    async fn once_its_group_is_saturated_a_priority_is_held_to_what_it_reserves() {
        let Served { app, .. } = served(RESERVED).await;

        let answered = statuses(
            &app,
            [
                "sk-dev-1", "sk-dev-1", "sk-dev-1", "sk-dev-1", "sk-prod", "sk-dev-1", "sk-prod",
            ],
        )
        .await;

        let ok = StatusCode::OK;
        assert_eq!(
            answered,
            [ok, ok, ok, ok, ok, StatusCode::TOO_MANY_REQUESTS, ok]
        );
    }

    #[rstest]
    #[case::one_pool_for_the_class(RESERVED, StatusCode::TOO_MANY_REQUESTS)]
    #[case::a_slice_for_each_key(SPLIT_BY_KEY, StatusCode::OK)]
    #[tokio::test]
    async fn a_key_new_to_a_saturated_group_is_refused_only_when_its_class_shares_one_pool(
        #[case] config: &str,
        #[case] expected: StatusCode,
    ) {
        let Served { app, .. } = served(config).await;
        statuses(
            &app,
            ["sk-dev-1", "sk-dev-1", "sk-dev-1", "sk-dev-1", "sk-prod"],
        )
        .await;

        assert_eq!(call(&app, "sk-dev-2", "claude").await.status(), expected);
    }

    #[tokio::test]
    async fn a_refusal_is_the_proxys_rate_limit_error_as_the_routes_handler_answers_it() {
        let Served { app, upstream, .. } = served(RESERVED).await;
        statuses(&app, ["sk-dev-1"; 5]).await;

        let refused = call(&app, "sk-dev-1", "claude").await;

        let status = refused.status();
        let body: Value =
            serde_json::from_slice(&to_bytes(refused.into_body(), 4096).await.unwrap()).unwrap();
        assert_eq!(
            (
                status,
                &body["type"],
                &body["error"]["type"],
                upstream.received_requests().await.unwrap().len()
            ),
            (
                StatusCode::TOO_MANY_REQUESTS,
                &json!("error"),
                &json!("rate_limit_error"),
                5
            )
        );
    }

    /// The limiter's headers of a response, in the order they were sent.
    fn limit_headers(response: &Response) -> Vec<(&str, &str)> {
        const SENT: [&str; 5] = [
            "retry-after",
            "rate_limit_type",
            "rate_limited_by",
            "x-litellm-priority",
            "x-litellm-saturation",
        ];
        response
            .headers()
            .iter()
            .filter(|(name, _)| SENT.contains(&name.as_str()))
            .map(|(name, value)| (name.as_str(), value.to_str().unwrap()))
            .collect()
    }

    /// The proxy answers its `ProxyRateLimitError` with the headers in the order the limiter
    /// lists them, which is not the order of their names.
    #[tokio::test]
    async fn an_openai_shaped_route_answers_a_refusal_with_its_headers_in_the_proxys_order() {
        let Served { app, .. } = served(RESERVED).await;
        statuses(&app, ["sk-dev-1"; 5]).await;

        let refused = post_as(&app, "sk-dev-1", "/v1/chat/completions", chat(false)).await;

        assert_eq!(
            (refused.status(), limit_headers(&refused)),
            (
                StatusCode::TOO_MANY_REQUESTS,
                vec![
                    ("retry-after", "60"),
                    ("rate_limit_type", "requests"),
                    ("x-litellm-priority", "dev"),
                    ("x-litellm-saturation", "50.00%"),
                ]
            )
        );
    }

    /// The proxy's failure hook and failure log only give back what a share that borrows held,
    /// so a failed call stays counted as a request of its group and of its priority.
    #[tokio::test]
    async fn a_failed_call_stays_counted_as_a_request_and_adds_no_tokens() {
        let Served {
            app,
            limiter,
            upstream,
            ..
        } = served(RESERVED).await;
        refusing_forty_nine(&upstream).await;

        let failed = send_as(&app, "sk-prod", asking("claude", Some(49))).await;

        assert_eq!(
            (
                failed.status(),
                limiter.counted("requests", "model_saturation_check", "claude"),
                limiter.counted("requests", "priority_model", "claude:prod"),
                limiter.counted("tokens", "model_saturation_check", "claude"),
                limiter.counted("tokens", "priority_model", "claude:prod"),
            ),
            (
                StatusCode::BAD_REQUEST,
                Some(1.0),
                Some(1.0),
                Some(0.0),
                Some(0.0)
            )
        );
    }

    /// 40 input tokens of which 15 were read from cache, and 9 output, which Python charges as 34.
    const RESPONSE: &str = r#"{"id":"response-1","model":"responses-test","output":[],"usage":{"input_tokens":40,"output_tokens":9,"total_tokens":49,"input_tokens_details":{"cached_tokens":15}}}"#;

    /// An OpenAI provider that answers a Responses call, complete or streamed, with [`RESPONSE`].
    async fn responding() -> MockServer {
        let upstream = MockServer::start().await;
        let completed = format!(
            "event: response.completed\ndata: {{\"type\":\"response.completed\",\"response\":{RESPONSE}}}\n\n"
        );
        Mock::given(body_partial_json(json!({"stream": true})))
            .respond_with(ResponseTemplate::new(200).set_body_raw(completed, "text/event-stream"))
            .mount(&upstream)
            .await;
        Mock::given(body_partial_json(json!({"model": "responses-test"})))
            .respond_with(ResponseTemplate::new(200).set_body_raw(RESPONSE, "application/json"))
            .mount(&upstream)
            .await;
        upstream
    }

    async fn post_as(app: &Router, token: &str, path: &str, body: Value) -> Response {
        app.clone()
            .oneshot(
                Request::post(path)
                    .header("authorization", format!("Bearer {token}"))
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    fn chat(stream: bool) -> Value {
        json!({"model": "claude", "max_tokens": 16, "stream": stream,
            "messages": [{"role": "user", "content": "hi"}]})
    }

    fn response(stream: bool) -> Value {
        json!({"model": "responses", "input": "hello", "stream": stream})
    }

    /// The gateway with one more group, `responses`, which serves 10 requests and 100000 tokens a
    /// minute at the OpenAI provider handed back with it.
    async fn with_responses(
        keys: Arc<dyn KeyLookup>,
        litellm_settings: &str,
    ) -> (Served, MockServer) {
        let provider = responding().await;
        let responses = format!(
            "  - model_name: responses\n    litellm_params: {{model: openai/responses-test, api_base: \"{}\", api_key: sk-upstream, rpm: 10, tpm: 100000}}\n",
            provider.uri()
        );
        let served = test_support::serving(keys, "", litellm_settings, &responses).await;
        (served, provider)
    }

    /// The success log settles what the route says the call used, less the prompt tokens read
    /// from cache, whichever route served it and whether it streamed. The chat completions route
    /// serves no stream.
    #[rstest]
    #[case::a_message("/v1/messages", chat(false), "claude", 27.0)]
    #[case::a_streamed_message("/v1/messages", chat(true), "claude", 27.0)]
    #[case::a_chat_completion("/v1/chat/completions", chat(false), "claude", 27.0)]
    #[case::a_response("/v1/responses", response(false), "responses", 34.0)]
    #[case::a_streamed_response("/v1/responses", response(true), "responses", 34.0)]
    #[tokio::test]
    async fn the_tokens_a_call_used_are_added_to_its_group_and_to_its_priority(
        #[case] path: &str,
        #[case] body: Value,
        #[case] group: &str,
        #[case] used: f64,
    ) {
        let (Served { app, limiter, .. }, _provider) =
            with_responses(Arc::new(Prioritized), RESERVED).await;

        let answer = post_as(&app, "sk-prod", path, body).await;
        let status = answer.status();
        to_bytes(answer.into_body(), usize::MAX).await.unwrap();

        let tokens = |descriptor: &str, value: &str| limiter.counted("tokens", descriptor, value);
        let priority = format!("{group}:prod");
        for _ in 0..200 {
            if tokens("priority_model", &priority) == Some(used) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        assert_eq!(
            (
                status,
                tokens("model_saturation_check", group),
                tokens("priority_model", &priority),
                limiter.counted("requests", "model_saturation_check", group)
            ),
            (StatusCode::OK, Some(used), Some(used), Some(1.0))
        );
    }

    #[tokio::test]
    async fn a_model_group_that_sets_no_limit_is_answered_uncounted() {
        let Served { app, limiter, .. } = served(RESERVED).await;

        let answer = call(&app, "sk-prod", "gpt").await;

        assert_eq!(
            (
                answer.status(),
                answer.headers()["x-litellm-priority"].to_str().unwrap(),
                limiter.counted("requests", "model_saturation_check", "gpt")
            ),
            (StatusCode::OK, "prod", None)
        );
    }

    /// The proxy reads an alias's group for its limits and counts the call under the name asked
    /// for.
    #[tokio::test]
    async fn a_call_under_an_alias_is_counted_against_its_groups_limits() {
        let Served { app, limiter, .. } = served(RESERVED).await;

        call(&app, "sk-prod", "sonnet").await;

        assert_eq!(
            limiter.counted("requests", "model_saturation_check", "sonnet"),
            Some(1.0)
        );
    }

    #[tokio::test]
    async fn a_key_that_borrows_is_held_to_its_limit_on_the_group_under_an_alias_of_it() {
        let limits = KeyLimits {
            model_rpm: by_model("claude", 1),
            ..KeyLimits::default()
        };
        let Served { app, .. } = holding(limits, BORROWING_TWICE_ITS_LIMIT).await;

        let first = call(&app, KEY, "sonnet").await.status();
        let second = call(&app, KEY, "sonnet").await.status();
        let third = call(&app, KEY, "sonnet").await.status();

        assert_eq!(
            (first, second, third),
            (
                StatusCode::OK,
                StatusCode::OK,
                StatusCode::TOO_MANY_REQUESTS
            )
        );
    }

    /// Python's authentication refuses the key before any hook, so the group never counts a call
    /// its key was already over the limit for.
    #[tokio::test]
    async fn a_caller_over_its_own_limit_is_refused_before_its_model_group_counts_the_call() {
        let Served { app, limiter, .. } = served(RESERVED).await;

        let answered = statuses(&app, ["sk-limited-dev", "sk-limited-dev"]).await;

        assert_eq!(
            (
                answered,
                limiter.counted("requests", "model_saturation_check", "claude")
            ),
            (
                vec![StatusCode::OK, StatusCode::TOO_MANY_REQUESTS],
                Some(1.0)
            )
        );
    }

    /// The group is saturated from its sixth call on, at half its 10 requests a minute.
    #[rstest]
    #[case::until_its_group_saturates(BORROWING, 7, 6)]
    #[case::up_to_its_borrow_ceiling(BORROWING_TWICE_ITS_LIMIT, 3, 2)]
    #[tokio::test]
    async fn a_key_that_borrows_runs_past_its_own_model_limit_on_idle_capacity(
        #[case] config: &str,
        #[case] calls: usize,
        #[case] answered: usize,
    ) {
        let limits = KeyLimits {
            model_rpm: by_model("claude", 1),
            ..KeyLimits::default()
        };
        let Served { app, upstream, .. } = holding(limits, config).await;

        let last = statuses(&app, vec![KEY; calls]).await.pop();

        assert_eq!(
            (last, upstream.received_requests().await.unwrap().len()),
            (Some(StatusCode::TOO_MANY_REQUESTS), answered)
        );
    }

    /// The tokens the key's borrow ceiling on a group holds.
    struct Held<'a>(&'a KeyLimiter, &'a str);

    fn held_under_the_ceiling(limiter: &KeyLimiter) -> Held<'_> {
        Held(limiter, "claude")
    }

    impl Held<'_> {
        fn now(&self) -> Option<f64> {
            self.0.counted(
                "tokens",
                "model_per_key_max_borrow_multiple",
                &format!("{}:{}", hash_token(KEY), self.1),
            )
        }

        /// What it holds once it reads `expected`, or what it last held.
        async fn once(&self, expected: f64) -> Option<f64> {
            for _ in 0..200 {
                if self.now() == Some(expected) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
            self.now()
        }
    }

    /// A key that may use 30 tokens a minute of `claude`, and 60 while it borrows.
    async fn borrowing_up_to_sixty_tokens() -> Served {
        let limits = KeyLimits {
            model_tpm: by_model("claude", 30),
            ..KeyLimits::default()
        };
        holding(limits, BORROWING_TWICE_ITS_LIMIT).await
    }

    #[tokio::test]
    async fn a_share_that_holds_tokens_reserves_what_the_body_asks_for_and_settles_what_was_used() {
        let Served { app, limiter, .. } = borrowing_up_to_sixty_tokens().await;

        let admitted = send_as(&app, KEY, asking("claude", Some(50)))
            .await
            .status();
        let settled = held_under_the_ceiling(&limiter).once(27.0).await;
        let over = send_as(&app, KEY, asking("claude", Some(50)))
            .await
            .status();
        let fitting = send_as(&app, KEY, asking("claude", Some(20)))
            .await
            .status();

        assert_eq!(
            (admitted, settled, over, fitting),
            (
                StatusCode::OK,
                Some(27.0),
                StatusCode::TOO_MANY_REQUESTS,
                StatusCode::OK
            ),
            "the 51 reserved settle to the 27 used, and a refused call reserves nothing"
        );
    }

    /// The estimate a share reserves for a route is settled to what that route says the call
    /// used, as it is for `/v1/messages`.
    #[tokio::test]
    async fn a_share_settles_what_it_reserved_to_what_a_response_used() {
        let limits = KeyLimits {
            model_tpm: by_model("responses", 1000),
            ..KeyLimits::default()
        };
        let (Served { app, limiter, .. }, _provider) =
            with_responses(Arc::new(Holding(limits)), BORROWING_TWICE_ITS_LIMIT).await;

        let answer = post_as(&app, KEY, "/v1/responses", response(false)).await;

        assert_eq!(
            (
                answer.status(),
                Held(&limiter, "responses").once(34.0).await
            ),
            (StatusCode::OK, Some(34.0))
        );
    }

    /// The provider refuses a call that asks for exactly 49 tokens.
    async fn refusing_forty_nine(upstream: &MockServer) {
        Mock::given(body_partial_json(json!({"max_tokens": 49})))
            .respond_with(
                ResponseTemplate::new(400).set_body_json(json!({"type": "error",
                "error": {"type": "invalid_request_error", "message": "refused upstream"}})),
            )
            .with_priority(1)
            .mount(upstream)
            .await;
    }

    /// The failure hook gives them back before the client is answered, so a retry finds them.
    #[tokio::test]
    async fn a_failed_call_has_given_back_the_tokens_its_share_held_once_it_is_answered() {
        let Served {
            app,
            limiter,
            upstream,
            ..
        } = borrowing_up_to_sixty_tokens().await;
        refusing_forty_nine(&upstream).await;

        let failed = send_as(&app, KEY, asking("claude", Some(49))).await;

        assert_eq!(
            (failed.status(), held_under_the_ceiling(&limiter).now()),
            (StatusCode::BAD_REQUEST, Some(0.0))
        );
    }

    #[tokio::test]
    async fn a_stream_the_client_leaves_gives_back_the_tokens_its_share_held() {
        let Served { app, limiter, .. } = borrowing_up_to_sixty_tokens().await;
        let streamed = json!({"model": "claude", "max_tokens": 50, "stream": true,
            "messages": [{"role": "user", "content": "hi"}]});

        drop(send_as(&app, KEY, streamed).await);

        assert_eq!(held_under_the_ceiling(&limiter).once(0.0).await, Some(0.0));
    }

    /// A share holds tokens before its group saturates only under a borrow ceiling. The route
    /// refuses a body left without `max_tokens`, so nothing reaches the provider.
    #[rstest]
    #[case::a_share_that_holds_tokens(BORROWING_TWICE_ITS_LIMIT, Some(json!(25)))]
    #[case::a_share_that_holds_none(BORROWING, None)]
    #[tokio::test]
    async fn a_body_that_sets_no_output_cap_is_capped_at_a_quarter_of_the_share_that_holds_tokens(
        #[case] config: &str,
        #[case] reached: Option<Value>,
    ) {
        let limits = KeyLimits {
            model_tpm: by_model("claude", 100),
            ..KeyLimits::default()
        };
        let Served { app, upstream, .. } = holding(limits, config).await;

        send_as(&app, KEY, asking("claude", None)).await;

        let received = upstream.received_requests().await.unwrap();
        let provider_saw: Option<Value> = received.last().map(|request| {
            serde_json::from_slice::<Value>(&request.body).unwrap()["max_tokens"].clone()
        });
        assert_eq!(provider_saw, reached);
    }

    /// `gpt` sets no `tpm` or `rpm`, so the dynamic rate limiter does not cover it.
    #[tokio::test]
    async fn a_limit_on_a_group_the_dynamic_rate_limiter_does_not_cover_stays_the_key_limiters() {
        let limits = KeyLimits {
            model_rpm: by_model("gpt", 1),
            ..KeyLimits::default()
        };
        let Served { app, .. } = holding(limits, BORROWING).await;

        let first = call(&app, KEY, "gpt").await.status();
        let second = call(&app, KEY, "gpt").await.status();

        assert_eq!(
            (first, second),
            (StatusCode::OK, StatusCode::TOO_MANY_REQUESTS)
        );
    }

    #[rstest]
    fn the_callback_is_built_only_when_the_config_names_it() {
        let unnamed =
            Config::from_yaml("litellm_settings:\n  priority_reservation: {prod: 0.9}\n").unwrap();

        assert_eq!(
            Settings::read(
                &unnamed,
                parallel_request_limiter::Settings::default(),
                true
            ),
            None
        );
    }
}
