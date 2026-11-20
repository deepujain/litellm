//! The proxy's `parallel_request_limiter_v3` for a key's own `rpm_limit`, `tpm_limit`,
//! `max_parallel_requests` and per-model limits, as a callback of the stage: the pre-call hook
//! counts the request, reserves the tokens its body asks for and takes a parallel slot, or refuses
//! the call. A complete answer gives the slot back before it goes out and carries the
//! `x-ratelimit-*` headers, and the success log charges the tokens the call used to the caller's
//! counters, for every caller, the master key included. The limit check Python's authentication
//! runs before the body asks the same limiter through [`KeyLimiter::before_body`].
//!
//! Accepted difference: a route that tells no usage, every one but `/v1/messages`, charges no
//! tokens.

use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Abort, Activation, Callback, Field, FieldSet, Hook, HookInput, HookOutput, Manifest, Outcome,
    Patch, Scalar, Selection, Subscription, UsageRecord, Verdict, typed,
};
use litellm_callbacks_model_capacity::RATE_LIMIT_ERROR as CAPACITY_RATE_LIMITED;
use litellm_callbacks_parallel_request_limiter::{
    AuthMetadata, CapSettings, Descriptor, DescriptorRequest, Engine, EstimateRequest,
    LimiterState, Lookups, OutputCapFacts, ParallelRequestLimiter, PolicyFacts,
    PostCallFailureFacts, PostCallSuccessFacts, PreCallFacts, PythonServed, RATE_LIMITED,
    RateLimitedModel, RequestFacts, Settings as EngineSettings, SuccessFacts, TargetFacts,
    TokenRateLimitType, UsageObject, aggregate_only, charged_tokens, explicit_output_cap,
    header_map, output_cap, prior_response, tokens_for_request, write_output_cap,
};
use litellm_config::{Config, Environment};
use litellm_gateway_auth::{AuthenticatedRequest, AuthenticationMethod, keys::KeyLimits};
use litellm_gateway_inference::{RESPONSES_CALL_TYPE, RateLimited, call_type};
use litellm_redis_client::RedisClient;
use litellm_redis_store::{Remote, rate_limits::RateLimits};
use litellm_router_core::Router;
use parking_lot::Mutex;
use serde_json::{Map, Value, json};

use super::Registered;

/// The name Python registers the hook under.
pub(crate) const NAME: &str = "parallel_request_limiter";
/// `RedisCache.default_ttl` when `default_redis_ttl` is unset.
pub(crate) const REDIS_DEFAULT_TTL: Duration = Duration::from_secs(60);
/// The `default_in_memory_ttl` of the proxy's `internal_usage_cache`, the limiter's memory tier.
const MEMORY_TTL: Duration = Duration::from_secs(1);
/// The api key and the user Python names a master key caller by, `LITELLM_PROXY_MASTER_KEY_ALIAS`
/// and `LITELLM_PROXY_ADMIN_NAME`.
const MASTER_KEY_ALIAS: &str = "litellm_proxy_master_key";
const PROXY_ADMIN_NAME: &str = "default_user_id";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Settings {
    window_size: i64,
    tpm_reservation_enabled: bool,
    token_type: TokenRateLimitType,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            window_size: 60,
            tpm_reservation_enabled: true,
            token_type: TokenRateLimitType::Total,
        }
    }
}

/// The proxy's `internal_usage_cache` counters, over the coordination Redis when there is one and
/// over this process's memory otherwise. A key's limits and a model group's capacity are counted
/// on the same ones, as they are in the proxy.
pub fn counters(redis: Option<RedisClient>) -> Arc<RateLimits> {
    let remote = redis.map(|client| Remote {
        client,
        default_ttl: REDIS_DEFAULT_TTL,
    });
    Arc::new(RateLimits::new(remote, Some(MEMORY_TTL)))
}

impl Settings {
    pub(crate) fn window_size(self) -> i64 {
        self.window_size
    }

    pub(crate) fn token_type(self) -> TokenRateLimitType {
        self.token_type
    }

    /// The limiter's engine over `counters`, which each limiter builds its own of, as the proxy
    /// does.
    pub(crate) fn engine(self, counters: Arc<RateLimits>) -> Engine {
        Engine::new(
            counters,
            EngineSettings {
                window_size: self.window_size,
                tpm_reservation_enabled: self.tpm_reservation_enabled,
            },
            Arc::new(unix_now),
        )
    }

    /// The window and the reservation as the environment sets them, and
    /// `general_settings.token_rate_limit_type`.
    pub fn read(config: &Config, environment: &Environment) -> Self {
        let defaults = Self::default();
        Self {
            window_size: environment
                .litellm_rate_limit_window_size
                .unwrap_or(defaults.window_size),
            tpm_reservation_enabled: environment
                .litellm_tpm_token_reservation_enabled
                .map_or(defaults.tpm_reservation_enabled, |flag| flag.0),
            token_type: TokenRateLimitType::from_setting(&json!(
                config.general_settings.token_rate_limit_type
            )),
        }
    }
}

/// The tokens a call asks for, input and output, as the proxy estimates them before it runs. `Err`
/// for a body only Python's estimator reads, one that carries media.
fn estimated(
    body: &Value,
    call_type: &str,
    min_tpm_limit: Option<i64>,
    cap: &CapSettings,
) -> Result<i64, PythonServed> {
    let tokens = tokens_for_request(
        EstimateRequest {
            data: body,
            call_type,
            min_configured_tpm_limit: min_tpm_limit,
            configured_output_tokens: None,
            include_media: true,
        },
        cap,
    )?;
    Ok(i64::try_from(tokens).unwrap_or(i64::MAX))
}

/// Whether only Python's estimator tells the tokens the body sent to `path` asks for.
pub(crate) fn python_estimates(body: &mut Map<String, Value>, path: &str) -> bool {
    let mut sent = Value::Object(std::mem::take(body));
    let python = estimated(&sent, call_type(path), None, &CapSettings::default()).is_err();
    if let Value::Object(sent) = sent.take() {
        *body = sent;
    }
    python
}

/// What a call reserves from a token limit, after the output cap the proxy writes into a body
/// that sets none, so one unbounded answer cannot overshoot the smallest limit.
fn reserved(
    body: &mut Value,
    call_type: &str,
    min_tpm_limit: i64,
    cap: &CapSettings,
) -> Result<i64, PythonServed> {
    let explicit = match &*body {
        Value::Object(request) => explicit_output_cap(request, call_type, cap)?,
        _ => None,
    };
    let implicit = output_cap(&OutputCapFacts {
        min_configured_limit: Some(min_tpm_limit),
        responses: call_type == RESPONSES_CALL_TYPE,
        exempt: explicit.is_some(),
        configured_output_tokens: None,
        fraction: cap.fraction,
        default_max_tokens_estimate: cap.default_max_tokens_estimate,
        responses_min_output_tokens: cap.responses_min_output_tokens,
    });
    if let Some(implicit) = implicit {
        write_output_cap(body, call_type, implicit);
    }
    estimated(body, call_type, Some(min_tpm_limit), cap)
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64())
}

/// Who Python says the caller is: its `api_key` and its `user_id`.
pub(crate) struct Caller {
    pub(crate) api_key: String,
    user_id: Option<&'static str>,
    pub(crate) limits: KeyLimits,
}

impl Caller {
    pub(crate) fn of(identity: &AuthenticatedRequest) -> Self {
        let authentication = identity.caller().authentication();
        match authentication.method {
            AuthenticationMethod::MasterKey => Self {
                api_key: MASTER_KEY_ALIAS.to_owned(),
                user_id: Some(PROXY_ADMIN_NAME),
                limits: authentication.limits.clone(),
            },
            AuthenticationMethod::VirtualKey | AuthenticationMethod::External(_) => Self {
                api_key: authentication.credential_id.clone(),
                user_id: None,
                limits: authentication.limits.clone(),
            },
        }
    }

    fn targets(&self) -> TargetFacts {
        TargetFacts {
            user_api_key_hash: Some(self.api_key.clone()),
            user_api_key_user_id: self.user_id.map(str::to_owned),
            ..TargetFacts::default()
        }
    }
}

/// The tokens Python charges a served call: those `token_type` counts, less the prompt tokens read
/// from cache.
fn charged(used: &UsageRecord, token_type: TokenRateLimitType) -> i64 {
    let count = |tokens: &Option<Scalar>| match tokens {
        Some(Scalar::Int(tokens)) => Value::from(i64::try_from(*tokens).unwrap_or(i64::MAX)),
        _ => Value::from(0),
    };
    let usage = UsageObject::Chat {
        members: Map::from_iter([
            ("prompt_tokens".to_owned(), count(&used.prompt_tokens)),
            (
                "completion_tokens".to_owned(),
                count(&used.completion_tokens),
            ),
            ("total_tokens".to_owned(), count(&used.total_tokens)),
            ("cached_tokens".to_owned(), count(&used.cached_tokens)),
        ]),
    };
    match charged_tokens(Some(&usage), token_type) {
        0 => aggregate_only(Some(&usage)),
        charged => charged,
    }
}

/// The limit's ordered headers and words of an abort that is Python's `ProxyRateLimitError`.
pub(crate) fn rate_limited(abort: &Abort) -> Option<RateLimited> {
    let error = abort
        .error
        .as_ref()
        .filter(|error| error.kind == RATE_LIMITED || error.kind == CAPACITY_RATE_LIMITED)?;
    let headers = error
        .fields
        .get("headers")
        .and_then(|headers| serde_json::from_value::<Vec<(String, String)>>(headers.clone()).ok())
        .unwrap_or_else(|| abort.headers.clone().into_iter().collect());
    Some(RateLimited {
        message: abort.message.clone(),
        headers,
    })
}

/// The proxy's `parallel_request_limiter_v3` over [`counters`].
pub struct KeyLimiter {
    manifest: Manifest,
    limiter: ParallelRequestLimiter,
    token_type: TokenRateLimitType,
    /// Tells the model group `model_group_alias` gives the model a body names.
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
                reads: FieldSet::of(&[Field::Model, Field::RequestBody, Field::RateLimitStatuses]),
                writes: FieldSet::of(&[Field::OptionalParams]),
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
                reads: FieldSet::of(&[
                    Field::Outcome,
                    Field::Usage,
                    Field::Model,
                    Field::LoggedModelGroup,
                ]),
                ..Subscription::on(Hook::AttemptEnded)
            },
        ],
        has_request_state: true,
    }
}

/// The members of `capped` that `sent` does not hold the same, which is the output cap the
/// limiter wrote.
fn written(sent: &Map<String, Value>, capped: Map<String, Value>) -> Map<String, Value> {
    capped
        .into_iter()
        .filter(|(name, value)| sent.get(name) != Some(value))
        .collect()
}

impl KeyLimiter {
    pub fn new(settings: Settings, counters: Arc<RateLimits>, router: Router) -> Arc<Self> {
        Arc::new(Self {
            manifest: manifest(),
            limiter: ParallelRequestLimiter::new(settings.engine(counters)),
            token_type: settings.token_type,
            router,
        })
    }

    /// The limiter as the stage registers it, with each request's admission made from its caller.
    pub(crate) fn registered(self: &Arc<Self>) -> Registered {
        let limiter = self.clone();
        Registered::for_caller(self.clone(), move |identity| Admission {
            limiter: limiter.clone(),
            caller: Caller::of(identity),
            state: LimiterState::default(),
            phase: Mutex::new(Phase::Asked),
        })
    }

    /// `requested` with the group `model_group_alias` gives it, so a limit keyed by either is
    /// found.
    fn asked(&self, requested: String) -> RateLimitedModel {
        let registry = self.router.registry();
        let group = registry
            .alias_of(&requested)
            .filter(|group| !group.is_empty())
            .unwrap_or(&requested)
            .to_owned();
        RateLimitedModel { requested, group }
    }

    /// The limits the caller's key holds, as the proxy describes them: its `api_key`, and its
    /// `model_per_key` for the model asked for.
    fn descriptors(&self, caller: &Caller, model: Option<&RateLimitedModel>) -> Vec<Descriptor> {
        let KeyLimits {
            rpm,
            tpm,
            max_parallel_requests,
            model_rpm,
            model_tpm,
        } = &caller.limits;
        let auth = Map::from_iter([
            ("api_key".to_owned(), json!(caller.api_key)),
            ("rpm_limit".to_owned(), json!(rpm)),
            ("tpm_limit".to_owned(), json!(tpm)),
            (
                "max_parallel_requests".to_owned(),
                json!(max_parallel_requests),
            ),
        ]);
        let metadata = AuthMetadata {
            key_model_rpm_limit: json!(**model_rpm),
            key_model_tpm_limit: json!(**model_tpm),
            ..AuthMetadata::default()
        };
        let lookups = Lookups {
            candidates: model
                .map(|model| model.requested.clone())
                .into_iter()
                .collect(),
            groups: model
                .map(|model| (model.requested.clone(), Some(model.group.clone())))
                .into_iter()
                .collect(),
            ..Lookups::default()
        };
        DescriptorRequest {
            auth: &auth,
            rpm_limit_type: None,
            tpm_limit_type: None,
            metadata: &metadata,
            lookups: &lookups,
            request: &RequestFacts::default(),
            requested_model: model.map(|model| model.requested.as_str()),
            window_size: self.limiter.engine().settings().window_size,
        }
        .build()
        .map(|built| built.descriptors)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|descriptor| serde_json::from_value(descriptor).ok())
        .collect()
    }

    /// What the pre-call hook is told of a call: the caller's limits and, when one counts tokens,
    /// what `body` asks for, after the output cap the proxy writes into a body that sets none. A
    /// body nobody here estimates reserves the least.
    fn pre_call_facts(
        &self,
        caller: &Caller,
        call_type: &str,
        model: Option<RateLimitedModel>,
        body: Option<&mut Value>,
    ) -> PreCallFacts {
        let descriptors = self.descriptors(caller, model.as_ref());
        let min_tpm_limit = descriptors
            .iter()
            .filter_map(|descriptor| descriptor.limits().tokens_per_unit)
            .min();
        let reserves = self.limiter.engine().settings().tpm_reservation_enabled;
        let estimated_tokens = match (min_tpm_limit, body) {
            (Some(min_tpm_limit), Some(body)) if reserves => {
                reserved(body, call_type, min_tpm_limit, &self.limiter.cap_settings()).ok()
            }
            _ => None,
        };
        PreCallFacts {
            descriptors,
            requested_model: model.as_ref().map(|model| model.requested.clone()),
            estimated_tokens,
            reserved_model: model,
            ..PreCallFacts::default()
        }
    }

    /// `raise_if_over_limit_before_body`: the refusal of a caller already over a limit that needs
    /// no body. Counters are only read, so the pre-call hook still counts an admitted call.
    pub(crate) async fn before_body(&self, caller: &Caller) -> Option<Abort> {
        self.limiter
            .before_body(
                &self.descriptors(caller, None),
                PolicyFacts::default(),
                None,
            )
            .await
    }

    async fn pre_call(&self, input: &HookInput, admission: &Admission) -> HookOutput {
        let model = input
            .fields
            .get::<typed::Model>()?
            .filter(|model| !model.is_empty())
            .map(|model| self.asked(model));
        let call_type = input
            .call
            .as_ref()
            .map_or("", |call| call.kind.call_type.as_str());
        let sent = match input.fields.raw(Field::RequestBody)? {
            Some(Value::Object(sent)) if admission.caller.limits.counts_tokens() => Some(sent),
            _ => None,
        };
        let mut capped = sent.cloned().map(Value::Object);
        let facts = PreCallFacts {
            prior_response: prior_response(input)?,
            ..self.pre_call_facts(&admission.caller, call_type, model, capped.as_mut())
        };
        if let Some(abort) = self.limiter.pre_call(&admission.state, facts, None).await {
            return Ok(Some(Verdict::Abort(abort)));
        }
        *admission.phase.lock() = Phase::Admitted;
        let cap = match (sent, capped) {
            (Some(sent), Some(Value::Object(capped))) => written(sent, capped),
            _ => Map::new(),
        };
        Ok((!cap.is_empty())
            .then(|| Verdict::Patch(Patch::default().merge(Field::OptionalParams, cap))))
    }

    /// `async_log_success_event`: gives back what the call still holds and charges the tokens it
    /// used to the caller's counters, which a TPM limit reads. A call whose route told no usage
    /// only gives back what it holds.
    async fn log_success(&self, input: &HookInput, admission: &Admission) -> HookOutput {
        let used = input
            .fields
            .get::<typed::Usage>()?
            .and_then(|usage| usage.response);
        let Some(used) = used else {
            self.limiter.release_on_disconnect(&admission.state).await;
            return Ok(None);
        };
        let group = input
            .fields
            .get::<typed::LoggedModelGroup>()?
            .and_then(|group| match group.value {
                Some(Scalar::Text(group)) => Some(group),
                _ => None,
            });
        let served_model = input
            .fields
            .get::<typed::Model>()?
            .zip(group)
            .map(|(requested, group)| RateLimitedModel { requested, group });
        let facts = SuccessFacts {
            total_tokens: charged(&used, self.token_type),
            targets: admission.caller.targets(),
            served_model,
            ..SuccessFacts::default()
        };
        self.limiter
            .settle_success(&admission.state, facts, None)
            .await;
        Ok(None)
    }
}

#[cfg(test)]
impl KeyLimiter {
    /// What this process's memory holds of a counter, for a test to read what was counted.
    pub(crate) fn counted(&self, kind: &str, descriptor: &str, value: &str) -> Option<f64> {
        use litellm_redis_store::{
            Number,
            rate_limits::{Entry, Local, Scope},
        };
        let entry = Entry::new(
            kind,
            Scope {
                descriptor: descriptor.into(),
                value: value.into(),
            },
        )
        .ok()?;
        match self.limiter.engine().store().get_local(&entry).ok()? {
            Some(Local::Number(number)) => Some(Number::as_f64(number)),
            _ => None,
        }
    }
}

/// How far one request got with the limiter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Not admitted, so the limiter holds nothing for it.
    Asked,
    Admitted,
    /// A complete answer gave its slot back, so its limits are known to name in its headers.
    Succeeded,
    /// Nothing is left to give back.
    Settled,
}

/// One request's admission: who it is charged to and what the limiter holds for it. A call that
/// is dropped while it still holds something, a stream the client left, gives it back off the
/// request.
pub struct Admission {
    limiter: Arc<KeyLimiter>,
    caller: Caller,
    state: LimiterState,
    phase: Mutex<Phase>,
}

impl Drop for Admission {
    fn drop(&mut self) {
        let holds = matches!(*self.phase.lock(), Phase::Admitted | Phase::Succeeded);
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        if holds {
            let limiter = self.limiter.clone();
            let state = std::mem::take(&mut self.state);
            runtime.spawn(async move { limiter.limiter.release_on_disconnect(&state).await });
        }
    }
}

impl Callback for KeyLimiter {
    type State = Admission;

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn handle<'a>(
        &'a self,
        input: &'a HookInput,
        state: Option<&'a Admission>,
    ) -> BoxFuture<'a, HookOutput> {
        Box::pin(async move {
            let Some(admission) = state else {
                return Ok(None);
            };
            let phase = *admission.phase.lock();
            match (input.hook, phase) {
                (Hook::PreCall, Phase::Asked) => self.pre_call(input, admission).await,
                (Hook::PostCallSuccess, Phase::Admitted) => {
                    self.limiter
                        .post_call_success(&admission.state, PostCallSuccessFacts::default())
                        .await;
                    *admission.phase.lock() = Phase::Succeeded;
                    Ok(None)
                }
                (Hook::ResponseHeaders, Phase::Succeeded) => {
                    let statuses = admission
                        .state
                        .stash()
                        .rate_limit_response
                        .as_ref()
                        .map(|response| header_map(&response.statuses))
                        .unwrap_or_default();
                    Ok((!statuses.is_empty()).then(|| {
                        Verdict::Patch(Patch::default().merge(Field::ResponseHeaders, statuses))
                    }))
                }
                (Hook::PostCallFailure, Phase::Admitted | Phase::Succeeded) => {
                    self.limiter
                        .post_call_failure(&admission.state, PostCallFailureFacts::default())
                        .await;
                    *admission.phase.lock() = Phase::Settled;
                    Ok(None)
                }
                (Hook::AttemptEnded, Phase::Admitted | Phase::Succeeded)
                    if input.fields.get::<typed::Outcome>()? == Some(Outcome::Success) =>
                {
                    *admission.phase.lock() = Phase::Settled;
                    self.log_success(input, admission).await
                }
                _ => Ok(None),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::future::Future;

    use axum::{
        Router,
        body::to_bytes,
        http::{StatusCode, header},
        response::Response,
    };
    use litellm_gateway_auth::{
        hash_token,
        keys::{KeyHash, KeyLookup, KeyStatus, LookupFuture, ModelLimits},
    };
    use rstest::rstest;

    use super::*;
    use crate::test_support::{self, MASTER_KEY, Served, asking, send_as};

    const KEY: &str = "sk-limited";
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

    /// What `descriptor`'s token counter holds once it reads `expected`, or what it last held.
    async fn settled(
        limiter: &KeyLimiter,
        descriptor: &str,
        value: &str,
        expected: f64,
    ) -> Option<f64> {
        for _ in 0..200 {
            if charged_to(limiter, descriptor, value) == Some(expected) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        charged_to(limiter, descriptor, value)
    }

    fn complete() -> Value {
        asking("claude", Some(16))
    }

    fn streamed() -> Value {
        json!({"model": "claude", "max_tokens": 16, "stream": true,
            "messages": [{"role": "user", "content": "hi"}]})
    }

    fn failing() -> Value {
        asking("broken", Some(16))
    }

    async fn send(app: &Router, body: Value) -> Response {
        send_as(app, KEY, body).await
    }

    fn by_model(limits: &[(&str, i64)]) -> ModelLimits {
        Arc::new(
            limits
                .iter()
                .map(|(model, limit)| ((*model).to_owned(), *limit))
                .collect(),
        )
    }

    async fn served(limits: KeyLimits) -> Served {
        configured(limits, "").await
    }

    /// The gateway for a key holding `limits`, with `general_settings` set besides its master key.
    async fn configured(limits: KeyLimits, general_settings: &str) -> Served {
        test_support::served(Arc::new(Holding(limits)), general_settings, "").await
    }

    fn charged_to(limiter: &KeyLimiter, descriptor: &str, value: &str) -> Option<f64> {
        limiter.counted("tokens", descriptor, value)
    }

    async fn json_body(response: Response) -> Value {
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap()
    }

    fn ratelimit_headers(response: &Response) -> Vec<(String, String)> {
        let mut found: Vec<(String, String)> = response
            .headers()
            .iter()
            .filter(|(name, _)| name.as_str().starts_with("x-ratelimit-api_key"))
            .map(|(name, value)| (name.to_string(), value.to_str().unwrap().to_owned()))
            .collect();
        found.sort();
        found
    }

    fn headers(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    }

    #[tokio::test]
    async fn a_keys_rpm_limit_counts_each_call_and_refuses_the_one_past_it() {
        let Served { app, .. } = served(KeyLimits {
            rpm: Some(2),
            ..KeyLimits::default()
        })
        .await;

        let first = send(&app, complete()).await;
        let second = send(&app, complete()).await;
        let third = send(&app, complete()).await;

        assert_eq!(
            ratelimit_headers(&first),
            headers(&[
                ("x-ratelimit-api_key-limit-requests", "2"),
                ("x-ratelimit-api_key-remaining-requests", "1"),
            ])
        );
        assert_eq!(second.status(), StatusCode::OK);
        assert_eq!(
            ratelimit_headers(&second),
            headers(&[
                ("x-ratelimit-api_key-limit-requests", "2"),
                ("x-ratelimit-api_key-remaining-requests", "0"),
            ])
        );
        assert_eq!(third.status(), StatusCode::TOO_MANY_REQUESTS);
        let refusal: Vec<(&str, &str)> = third
            .headers()
            .iter()
            .filter(|(name, _)| *name != header::CONTENT_TYPE && *name != header::CONTENT_LENGTH)
            .map(|(name, value)| (name.as_str(), value.to_str().unwrap()))
            .collect();
        assert_eq!(
            refusal.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
            [
                "retry-after",
                "rate_limit_type",
                "rate_limited_by",
                "reset_at"
            ]
        );
        assert_eq!(
            &refusal[..3],
            [
                ("retry-after", "60"),
                ("rate_limit_type", "requests"),
                ("rate_limited_by", "api_key"),
            ]
        );
        let body = json_body(third).await;
        assert_eq!(
            (&body["error"]["type"], &body["error"]["litellm_error_type"]),
            (&json!("rate_limit_error"), &json!("throttling_error"))
        );
        let message = body["error"]["message"].as_str().unwrap();
        let expected = format!(
            "Rate limit exceeded for api_key: {}. Limit type: requests. Current limit: 2, Remaining: 0. Limit resets at: ",
            hash_token(KEY)
        );
        assert!(message.starts_with(&expected), "{message}");
    }

    #[tokio::test]
    async fn a_complete_answer_gives_its_parallel_slot_back_before_it_goes_out() {
        let Served { app, .. } = served(KeyLimits {
            max_parallel_requests: Some(1),
            ..KeyLimits::default()
        })
        .await;

        let answered = send(&app, complete()).await;
        let again = send(&app, complete()).await;
        let failed = send(&app, failing()).await;
        let after_the_failure = send(&app, complete()).await;

        assert_eq!(
            (
                answered.status(),
                ratelimit_headers(&answered),
                again.status(),
                failed.status(),
                ratelimit_headers(&failed),
                after_the_failure.status()
            ),
            (
                StatusCode::OK,
                headers(&[
                    ("x-ratelimit-api_key-limit-max_parallel_requests", "1"),
                    ("x-ratelimit-api_key-remaining-max_parallel_requests", "0"),
                ]),
                StatusCode::OK,
                StatusCode::BAD_REQUEST,
                vec![],
                StatusCode::OK
            )
        );
    }

    async fn read_to_its_end(response: Response) {
        to_bytes(response.into_body(), usize::MAX).await.unwrap();
    }

    async fn dropped(response: Response) {
        drop(response);
    }

    async fn served_once_released(app: &Router) -> bool {
        for _ in 0..200 {
            if send(app, complete()).await.status() == StatusCode::OK {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        false
    }

    #[rstest]
    #[case::read(read_to_its_end)]
    #[case::dropped(dropped)]
    #[tokio::test]
    async fn a_streamed_answer_holds_its_parallel_slot_until_its_body_is_done<F>(
        #[case] finish: fn(Response) -> F,
    ) where
        F: Future<Output = ()>,
    {
        let Served { app, .. } = served(KeyLimits {
            max_parallel_requests: Some(1),
            ..KeyLimits::default()
        })
        .await;

        let streaming = send(&app, streamed()).await;
        let refused = send(&app, complete()).await;

        assert_eq!(
            (
                streaming.status(),
                ratelimit_headers(&streaming),
                refused.status(),
                refused
                    .headers()
                    .get("rate_limit_type")
                    .map(|kind| kind.to_str().unwrap().to_owned())
            ),
            (
                StatusCode::OK,
                vec![],
                StatusCode::TOO_MANY_REQUESTS,
                Some("max_parallel_requests".to_owned())
            )
        );
        finish(streaming).await;
        assert!(served_once_released(&app).await);
    }

    #[rstest]
    #[case::the_master_key(MASTER_KEY, KeyLimits { rpm: Some(0), max_parallel_requests: Some(0), ..KeyLimits::default() })]
    #[case::a_key_without_limits(KEY, KeyLimits::default())]
    #[tokio::test]
    async fn a_caller_without_limits_of_its_own_is_never_refused(
        #[case] token: &str,
        #[case] key_limits: KeyLimits,
    ) {
        let Served { app, .. } = served(key_limits).await;

        let first = send_as(&app, token, complete()).await;
        let second = send_as(&app, token, complete()).await;

        assert_eq!(
            (first.status(), second.status(), ratelimit_headers(&second)),
            (StatusCode::OK, StatusCode::OK, vec![])
        );
    }

    #[rstest]
    #[case::complete(complete())]
    #[case::streamed_to_its_end(streamed())]
    #[tokio::test]
    async fn a_served_call_charges_its_tokens_less_the_cached_reads_to_its_key_and_model(
        #[case] body: Value,
    ) {
        let Served { app, limiter, .. } = served(KeyLimits {
            rpm: Some(5),
            ..KeyLimits::default()
        })
        .await;
        let hash = hash_token(KEY);

        read_to_its_end(send(&app, body).await).await;

        assert_eq!(
            (
                settled(&limiter, "api_key", &hash, 27.0).await,
                settled(&limiter, "model_per_key", &format!("{hash}:claude"), 27.0).await
            ),
            (Some(27.0), Some(27.0))
        );
    }

    #[rstest]
    #[case::its_key("api_key", "litellm_proxy_master_key")]
    #[case::its_user("user", "default_user_id")]
    #[case::its_model("model_per_key", "litellm_proxy_master_key:claude")]
    #[tokio::test]
    async fn the_master_key_is_charged_as_the_proxy_names_it(
        #[case] descriptor: &str,
        #[case] value: &str,
    ) {
        let Served { app, limiter, .. } = served(KeyLimits::default()).await;

        send_as(&app, MASTER_KEY, complete()).await;

        assert_eq!(settled(&limiter, descriptor, value, 27.0).await, Some(27.0));
    }

    #[rstest]
    #[case::a_dropped_stream(streamed())]
    #[case::a_failed_call(failing())]
    #[tokio::test]
    async fn a_call_that_tells_no_usage_charges_nothing(#[case] body: Value) {
        let Served { app, limiter, .. } = served(KeyLimits {
            rpm: Some(5),
            max_parallel_requests: Some(1),
            ..KeyLimits::default()
        })
        .await;

        drop(send(&app, body).await);

        assert!(served_once_released(&app).await);
        assert_eq!(
            settled(&limiter, "api_key", &hash_token(KEY), 27.0).await,
            Some(27.0),
            "the call that told no usage was charged too"
        );
    }

    #[rstest]
    #[case::complete(complete(), StatusCode::OK)]
    #[case::streamed(streamed(), StatusCode::OK)]
    #[case::failed(failing(), StatusCode::BAD_REQUEST)]
    #[tokio::test]
    async fn every_answer_of_the_handler_names_the_keys_rpm_limit_and_authentications_refusal_never(
        #[case] body: Value,
        #[case] status: StatusCode,
    ) {
        let Served { app, .. } = served(KeyLimits {
            rpm: Some(1),
            ..KeyLimits::default()
        })
        .await;
        let unlimited = served(KeyLimits::default()).await;

        let admitted = send(&app, body.clone()).await;
        let refused = send(&app, body.clone()).await;
        let without_a_limit = send(&unlimited.app, body).await;

        let named =
            |response: &Response| response.headers().get("x-litellm-key-rpm-limit").cloned();
        assert_eq!(
            (
                admitted.status(),
                named(&admitted),
                refused.status(),
                named(&refused),
                named(&without_a_limit)
            ),
            (
                status,
                Some("1".parse().unwrap()),
                StatusCode::TOO_MANY_REQUESTS,
                None,
                None
            )
        );
    }

    #[tokio::test]
    async fn a_keys_tpm_limit_reserves_what_the_body_asks_for_and_settles_what_was_used() {
        let Served { app, limiter, .. } = served(KeyLimits {
            tpm: Some(60),
            ..KeyLimits::default()
        })
        .await;
        let hash = hash_token(KEY);

        let admitted = send(&app, asking("claude", Some(50))).await;

        assert_eq!(admitted.status(), StatusCode::OK);
        assert_eq!(admitted.headers()["x-litellm-key-tpm-limit"], "60");
        assert_eq!(
            ratelimit_headers(&admitted),
            headers(&[
                ("x-ratelimit-api_key-limit-tokens", "60"),
                ("x-ratelimit-api_key-remaining-tokens", "9"),
            ])
        );
        assert_eq!(
            settled(&limiter, "api_key", &hash, 27.0).await,
            Some(27.0),
            "the 51 reserved were not settled to the 27 used"
        );
        let over = send(&app, asking("claude", Some(50))).await;
        assert_eq!(over.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(over.headers()["x-litellm-key-tpm-limit"], "60");
        assert!(!over.headers().contains_key("retry-after"));
        let error = json_body(over).await["error"].clone();
        assert_eq!(error["type"], "rate_limit_error");
        assert_eq!(error.get("litellm_error_type"), None);
        assert!(
            error["message"]
                .as_str()
                .unwrap()
                .starts_with(&format!("Rate limit exceeded for api_key: {hash}. Limit type: tokens. Current limit: 60, Remaining: 33."))
        );
        assert_eq!(
            send(&app, asking("claude", Some(20))).await.status(),
            StatusCode::OK,
            "the refused call kept its reservation"
        );
    }

    #[tokio::test]
    async fn a_failed_call_gives_its_reserved_tokens_back() {
        let Served { app, .. } = served(KeyLimits {
            tpm: Some(60),
            ..KeyLimits::default()
        })
        .await;

        let failed = send(&app, asking("broken", Some(50))).await;
        let next = send(&app, asking("claude", Some(50))).await;

        assert_eq!(
            (failed.status(), next.status()),
            (StatusCode::BAD_REQUEST, StatusCode::OK)
        );
    }

    #[rstest]
    #[case::none_sent(None, json!(25))]
    #[case::one_sent(Some(50), json!(50))]
    #[tokio::test]
    async fn a_body_that_sets_no_output_cap_is_capped_at_a_quarter_of_the_token_limit(
        #[case] sent: Option<u64>,
        #[case] reached: Value,
    ) {
        let Served { app, upstream, .. } = served(KeyLimits {
            tpm: Some(100),
            rpm: Some(5),
            ..KeyLimits::default()
        })
        .await;

        send(&app, asking("claude", sent)).await;

        let received = upstream.received_requests().await.unwrap();
        let provider_saw: Option<Value> = received.last().map(|request| {
            serde_json::from_slice::<Value>(&request.body).unwrap()["max_tokens"].clone()
        });
        assert_eq!(provider_saw, Some(reached));
    }

    #[tokio::test]
    async fn a_body_only_python_estimates_reserves_the_least() {
        let Served { app, .. } = served(KeyLimits {
            tpm: Some(60),
            ..KeyLimits::default()
        })
        .await;
        let body = json!({"model": "claude", "max_tokens": 1000, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}},
        ]}]});

        assert_eq!(send(&app, body).await.status(), StatusCode::OK);
    }

    #[rstest]
    #[case::requests(
        KeyLimits { model_rpm: by_model(&[("claude", 1)]), ..KeyLimits::default() },
        "requests"
    )]
    #[case::tokens(
        KeyLimits { model_tpm: by_model(&[("claude", 60)]), ..KeyLimits::default() },
        "tokens"
    )]
    #[tokio::test]
    async fn a_keys_model_limit_counts_only_the_model_it_names(
        #[case] limits: KeyLimits,
        #[case] limit_type: &str,
    ) {
        let Served { app, limiter, .. } = served(limits).await;

        let admitted = send(&app, asking("claude", Some(50))).await;
        assert_eq!(admitted.status(), StatusCode::OK);
        settled(
            &limiter,
            "model_per_key",
            &format!("{}:claude", hash_token(KEY)),
            27.0,
        )
        .await;
        let refused = send(&app, asking("claude", Some(50))).await;
        let other = send(&app, asking("gpt", Some(50))).await;

        assert_eq!(refused.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(!refused.headers().contains_key("retry-after"));
        let error = json_body(refused).await["error"].clone();
        assert_eq!(error.get("litellm_error_type"), None);
        assert!(error["message"].as_str().unwrap().starts_with(&format!(
            "Rate limit exceeded for model_per_key: {}:claude. Limit type: {limit_type}.",
            hash_token(KEY)
        )));
        assert_eq!(other.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn a_model_limit_is_counted_under_the_models_group() {
        let Served { app, .. } = served(KeyLimits {
            model_rpm: by_model(&[("claude", 1)]),
            ..KeyLimits::default()
        })
        .await;

        let by_alias = send(&app, asking("sonnet", Some(50))).await;
        let by_group = send(&app, asking("claude", Some(50))).await;

        assert_eq!(
            (by_alias.status(), by_group.status()),
            (StatusCode::OK, StatusCode::TOO_MANY_REQUESTS)
        );
    }

    #[rstest]
    #[case::output("output", 14.0)]
    #[case::input("input", 13.0)]
    #[case::anything_else("tokens", 27.0)]
    #[tokio::test]
    async fn a_served_call_is_charged_the_tokens_the_configured_type_counts(
        #[case] token_rate_limit_type: &str,
        #[case] charged: f64,
    ) {
        let Served { app, limiter, .. } = configured(
            KeyLimits::default(),
            &format!("  token_rate_limit_type: {token_rate_limit_type}\n"),
        )
        .await;

        send(&app, complete()).await;

        assert_eq!(
            settled(&limiter, "api_key", &hash_token(KEY), charged).await,
            Some(charged)
        );
    }

    #[rstest]
    fn the_window_and_the_reservation_come_from_the_environment() {
        let environment = Environment {
            litellm_rate_limit_window_size: Some(30),
            litellm_tpm_token_reservation_enabled: Some(litellm_config::Flag(false)),
            ..Environment::default()
        };

        assert_eq!(
            (
                Settings::read(&Config::default(), &environment),
                Settings::read(&Config::default(), &Environment::default())
            ),
            (
                Settings {
                    window_size: 30,
                    tpm_reservation_enabled: false,
                    token_type: TokenRateLimitType::Total,
                },
                Settings::default()
            )
        );
    }
}
