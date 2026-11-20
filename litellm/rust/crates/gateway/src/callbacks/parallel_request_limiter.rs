//! The proxy's `parallel_request_limiter_v3` for a key's own `rpm_limit`, `tpm_limit`,
//! `max_parallel_requests` and per-model limits, and the default per-model limit a deployment
//! gives a key that holds none, as a callback of the stage: the pre-call hook
//! counts the request, reserves the tokens its body asks for and takes a parallel slot, or refuses
//! the call. A complete answer gives the slot back before it goes out and carries the
//! `x-ratelimit-*` headers, and the success log charges the tokens the call used to the caller's
//! counters, for every caller, the master key included. The limit check Python's authentication
//! runs before the body asks the same limiter through [`KeyLimiter::before_body`].
//!
//! A served call is charged what its route tells it used, which every route tells as Python's
//! litellm `Usage`: the proxy's success log rewrites a Responses answer's usage into one before
//! its limiter reads it, so chat completions, Responses and Messages are all charged the tokens
//! the configured type counts, less the prompt tokens read from cache.
//!
//! Accepted differences: transcription tells no usage yet, so it charges no tokens. A call that
//! fails is told without usage, so a stream that breaks gives its reservation back where the
//! proxy charges what the stream had used.

use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Abort, Activation, Callback, Field, FieldSet, Hook, HookInput, HookOutput, Manifest, Outcome,
    Patch, Scalar, Selection, Subscription, UsageRecord, Verdict, typed,
};
use litellm_callbacks_parallel_request_limiter::{
    AuthMetadata, Candidate, CapSettings, DeploymentLimits, Descriptor, DescriptorRequest, Engine,
    EstimateRequest, LimiterState, Lookups, ModelFacts, OutputCapFacts, ParallelRequestLimiter,
    PolicyFacts, PostCallFailureFacts, PostCallSuccessFacts, PreCallFacts, PythonServed,
    RATE_LIMITED, RateLimitedModel, RequestFacts, Settings as EngineSettings, SuccessFacts,
    TargetFacts, TokenRateLimitType, UsageObject, aggregate_only, charged_tokens,
    explicit_output_cap, header_map, output_cap, tokens_for_request, write_output_cap,
};
use litellm_config::{Config, Environment};
use litellm_gateway_auth::{AuthenticatedRequest, AuthenticationMethod, keys::KeyLimits};
use litellm_gateway_inference::{RESPONSES_CALL_TYPE, RateLimited, call_type};
use litellm_redis_client::RedisClient;
use litellm_redis_store::{Remote, rate_limits::RateLimits};
use litellm_router_core::{
    Router,
    registry::{Deployment, DeploymentFilter},
};
use parking_lot::Mutex;
use serde_json::{Map, Value, json};

use super::Registered;
use crate::latest_aliases::LatestAliases;

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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    window_size: i64,
    tpm_reservation_enabled: bool,
    token_type: TokenRateLimitType,
    /// `NO_MAX_TOKENS_TPM_RESERVATION_FRACTION`: the share of the smallest token limit a body
    /// that sets no output cap is capped at.
    output_cap_fraction: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            window_size: 60,
            tpm_reservation_enabled: true,
            token_type: TokenRateLimitType::Total,
            output_cap_fraction: CapSettings::default().fraction,
        }
    }
}

/// The fractions the proxy takes `NO_MAX_TOKENS_TPM_RESERVATION_FRACTION` from. Any other value
/// leaves its default.
const OUTPUT_CAP_FRACTIONS: std::ops::RangeInclusive<f64> = 0.01..=1.0;

impl Settings {
    /// The window, the reservation and the output cap's fraction as the environment sets them,
    /// and `general_settings.token_rate_limit_type`.
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
            output_cap_fraction: environment
                .no_max_tokens_tpm_reservation_fraction
                .filter(|fraction| OUTPUT_CAP_FRACTIONS.contains(fraction))
                .unwrap_or(defaults.output_cap_fraction),
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
    api_key: String,
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
        .filter(|error| error.kind == RATE_LIMITED)?;
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

/// The proxy's `parallel_request_limiter_v3`, over the coordination Redis when there is one and
/// over this process's memory otherwise, as the proxy's `internal_usage_cache` is.
pub struct KeyLimiter {
    manifest: Manifest,
    limiter: ParallelRequestLimiter,
    token_type: TokenRateLimitType,
    /// Tells the model group `model_group_alias` gives the model a body names.
    router: Router,
    aliases: LatestAliases,
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
    pub fn new(
        settings: Settings,
        redis: Option<RedisClient>,
        router: Router,
        aliases: LatestAliases,
    ) -> Arc<Self> {
        let remote = redis.map(|client| Remote {
            client,
            default_ttl: REDIS_DEFAULT_TTL,
        });
        let engine = Engine::new(
            Arc::new(RateLimits::new(remote, Some(MEMORY_TTL))),
            EngineSettings {
                window_size: settings.window_size,
                tpm_reservation_enabled: settings.tpm_reservation_enabled,
            },
            Arc::new(unix_now),
        );
        Arc::new(Self {
            manifest: manifest(),
            limiter: ParallelRequestLimiter::new(engine).with_cap_settings(CapSettings {
                fraction: settings.output_cap_fraction,
                ..CapSettings::default()
            }),
            token_type: settings.token_type,
            router,
            aliases,
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

    /// `_rate_limited_models`: `model`, and the model it stands for when the cost map serves its
    /// name as an alias, each with its group, so a limit keyed by any of them is found.
    fn candidates(&self, model: &RateLimitedModel) -> Vec<RateLimitedModel> {
        self.aliases
            .expand(&model.requested)
            .into_iter()
            .map(|name| self.asked(name))
            .collect()
    }

    /// What a served call's tokens are settled against for `model`: each candidate's group.
    /// `key_owns` only keeps a team's share of a model from being charged, and a key of a team
    /// is not authenticated here.
    fn settled_as(&self, model: &RateLimitedModel) -> ModelFacts {
        ModelFacts {
            candidates: self
                .candidates(model)
                .into_iter()
                .map(|candidate| Candidate {
                    group: candidate.group,
                    key_owns: false,
                })
                .collect(),
            ptu: None,
        }
    }

    /// `_get_deployment_default_limit`: the least `default_api_key_rpm_limit` and
    /// `default_api_key_tpm_limit` a deployment of `group` sets, the limit of a key that holds
    /// none of its own for a model.
    fn deployment_defaults(&self, group: &str) -> DeploymentLimits {
        let deployments = self.router.registry().deployments(&DeploymentFilter {
            model_name: Some(group),
            ..DeploymentFilter::default()
        });
        let least = |limit: fn(&Deployment) -> Option<u64>| {
            deployments
                .iter()
                .filter_map(|deployment| limit(deployment))
                .min()
                .map(|limit| i64::try_from(limit).unwrap_or(i64::MAX))
        };
        DeploymentLimits {
            rpm: least(|deployment| deployment.params.default_api_key_rpm_limit),
            tpm: least(|deployment| deployment.params.default_api_key_tpm_limit),
        }
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
        let candidates = model.map_or_else(Vec::new, |model| self.candidates(model));
        let lookups = Lookups {
            candidates: candidates
                .iter()
                .map(|candidate| candidate.requested.clone())
                .collect(),
            deployment_limits: candidates
                .iter()
                .map(|candidate| {
                    (
                        candidate.group.clone(),
                        self.deployment_defaults(&candidate.group),
                    )
                })
                .collect(),
            groups: candidates
                .into_iter()
                .map(|candidate| (candidate.requested, Some(candidate.group)))
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

    /// Whether a limit of `identity`'s caller counts the tokens of a call to `model`: one of its
    /// key, or the default one a deployment of the model's group gives it.
    pub(crate) fn counts_tokens(
        &self,
        identity: &AuthenticatedRequest,
        model: Option<&str>,
    ) -> bool {
        let model = model
            .filter(|model| !model.is_empty())
            .map(|model| self.asked(model.to_owned()));
        self.descriptors(&Caller::of(identity), model.as_ref())
            .iter()
            .any(|descriptor| descriptor.limits().tokens_per_unit.is_some())
    }

    /// What the pre-call hook is told of a call: the caller's limits and, when one counts tokens,
    /// what `sent` asks for, after the output cap the proxy writes into a body that sets none,
    /// which is handed back beside it. A body nobody here estimates reserves the least.
    fn pre_call_facts(
        &self,
        caller: &Caller,
        call_type: &str,
        model: Option<RateLimitedModel>,
        sent: Option<&Map<String, Value>>,
    ) -> (PreCallFacts, Map<String, Value>) {
        let descriptors = self.descriptors(caller, model.as_ref());
        let min_tpm_limit = descriptors
            .iter()
            .filter_map(|descriptor| descriptor.limits().tokens_per_unit)
            .min();
        let reserves = self.limiter.engine().settings().tpm_reservation_enabled;
        let (estimated_tokens, cap) = match (min_tpm_limit, sent) {
            (Some(min_tpm_limit), Some(sent)) if reserves => {
                let mut capped = Value::Object(sent.clone());
                let estimated = reserved(
                    &mut capped,
                    call_type,
                    min_tpm_limit,
                    &self.limiter.cap_settings(),
                );
                let cap = match capped {
                    Value::Object(capped) => written(sent, capped),
                    _ => Map::new(),
                };
                (estimated.ok(), cap)
            }
            _ => (None, Map::new()),
        };
        let facts = PreCallFacts {
            descriptors,
            requested_model: model.as_ref().map(|model| model.requested.clone()),
            estimated_tokens,
            reserved_model: model,
            ..PreCallFacts::default()
        };
        (facts, cap)
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
            Some(Value::Object(sent)) => Some(sent),
            _ => None,
        };
        let (facts, cap) = self.pre_call_facts(&admission.caller, call_type, model, sent);
        if let Some(abort) = self.limiter.pre_call(&admission.state, facts, None).await {
            return Ok(Some(Verdict::Abort(abort)));
        }
        *admission.phase.lock() = Phase::Admitted;
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
        let reserved_model = admission.state.stash().reserved_model.clone();
        let facts = SuccessFacts {
            total_tokens: charged(&used, self.token_type),
            targets: admission.caller.targets(),
            models: [&served_model, &reserved_model]
                .into_iter()
                .flatten()
                .map(|model| (model.requested.clone(), self.settled_as(model)))
                .collect(),
            served_model,
            ..SuccessFacts::default()
        };
        self.limiter
            .settle_success(&admission.state, facts, None)
            .await;
        Ok(None)
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
        body::{Body, to_bytes},
        http::{Request, StatusCode, header},
        middleware::from_fn_with_state,
        response::Response,
    };
    use litellm_gateway_auth::{
        authenticate, hash_token,
        keys::{KeyHash, KeyLookup, KeyStatus, LookupFuture, ModelLimits},
    };
    use litellm_model_catalog::{BUNDLED_REGISTRY, Catalog};
    use litellm_redis_store::{
        Number,
        rate_limits::{Entry, Local, Scope},
    };
    use rstest::rstest;
    use tower::ServiceExt;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{body_partial_json, method},
    };

    use super::*;
    use crate::{AppState, middleware};

    const MASTER_KEY: &str = "sk-master";
    const KEY: &str = "sk-limited";
    /// 8 input, 14 output, 100 read from cache and 5 written to it, which Python charges as 27.
    const USAGE: &str = r#"{"input_tokens":8,"output_tokens":14,"cache_read_input_tokens":100,"cache_creation_input_tokens":5}"#;

    struct Holding(KeyLimits);

    impl KeyLookup for Holding {
        fn lookup<'a>(&'a self, _: &'a KeyHash) -> LookupFuture<'a> {
            Box::pin(async move {
                Ok(Some(KeyStatus::Active {
                    expires_at: None,
                    models: Arc::from([]),
                    limits: self.0.clone(),
                }))
            })
        }
    }

    /// The provider every deployment but `broken` calls: it answers a stream when asked for one,
    /// and a complete message otherwise, each with [`USAGE`].
    async fn provider() -> MockServer {
        let upstream = MockServer::start().await;
        let stream = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-test\",\"content\":[],\"usage\":{\"input_tokens\":8,\"cache_read_input_tokens\":100,\"cache_creation_input_tokens\":5}}}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":14}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        );
        Mock::given(method("POST"))
            .and(body_partial_json(json!({"stream": true})))
            .respond_with(ResponseTemplate::new(200).set_body_raw(stream, "text/event-stream"))
            .mount(&upstream)
            .await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                format!(
                    r#"{{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[{{"type":"text","text":"hi"}}],"stop_reason":"end_turn","usage":{USAGE}}}"#
                ),
                "application/json",
            ))
            .mount(&upstream)
            .await;
        upstream
    }

    /// 30 input tokens of which 10 were read from cache, and 7 output, which Python charges as 27.
    const RESPONSES_USAGE: &str = r#"{"input_tokens":30,"output_tokens":7,"total_tokens":37,"input_tokens_details":{"cached_tokens":10}}"#;

    /// The OpenAI provider `responder` calls: it answers a Responses stream when asked for one,
    /// and a complete response otherwise, each with [`RESPONSES_USAGE`].
    async fn responses_provider() -> MockServer {
        let upstream = MockServer::start().await;
        let response = format!(
            r#"{{"id":"resp_1","object":"response","model":"responder-test","status":"completed","output":[],"usage":{RESPONSES_USAGE}}}"#
        );
        Mock::given(method("POST"))
            .and(body_partial_json(json!({"stream": true})))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                format!(
                    "event: response.completed\ndata: {{\"type\":\"response.completed\",\"response\":{response}}}\n\n"
                ),
                "text/event-stream",
            ))
            .mount(&upstream)
            .await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(response, "application/json"))
            .mount(&upstream)
            .await;
        upstream
    }

    /// The inference routes as the gateway mounts them, for a key holding `limits`: the groups
    /// `claude` (also asked for as `sonnet`) and `gpt` answer, `responder` answers the Responses
    /// route, and `broken` is refused upstream.
    struct Served {
        app: Router,
        limiter: Arc<KeyLimiter>,
        upstream: MockServer,
        responding: MockServer,
        _refusing: MockServer,
    }

    impl Served {
        /// What the provider behind `path` was last sent as `param`.
        async fn provider_saw(&self, path: &str, param: &str) -> Option<Value> {
            let provider = match path {
                RESPONSES => &self.responding,
                _ => &self.upstream,
            };
            let received = provider.received_requests().await.unwrap();
            received.last().map(|request| {
                serde_json::from_slice::<Value>(&request.body).unwrap()[param].clone()
            })
        }
    }

    async fn served(limits: KeyLimits) -> Served {
        configured(limits, "").await
    }

    async fn configured(limits: KeyLimits, general_settings: &str) -> Served {
        Setup {
            limits,
            general_settings,
            ..Setup::default()
        }
        .serve()
        .await
    }

    async fn deployed(limits: KeyLimits, groups: &[(&str, &str)]) -> Served {
        Setup {
            limits,
            groups,
            ..Setup::default()
        }
        .serve()
        .await
    }

    /// What [`Served`] is built from. `groups` are further groups, each a name and the
    /// `litellm_params` its deployment sets besides its model.
    #[derive(Default)]
    struct Setup<'a> {
        limits: KeyLimits,
        general_settings: &'a str,
        groups: &'a [(&'a str, &'a str)],
        environment: Environment,
    }

    impl Setup<'_> {
        async fn serve(self) -> Served {
            let Self {
                limits,
                general_settings,
                groups,
                environment,
            } = self;
            let upstream = provider().await;
            let responding = responses_provider().await;
            let refusing = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(
                    ResponseTemplate::new(400).set_body_json(json!({"type": "error",
                    "error": {"type": "invalid_request_error", "message": "refused upstream"}})),
                )
                .mount(&refusing)
                .await;
            let deployment = |name: &str, base: &str| {
                format!(
                    "  - model_name: {name}\n    litellm_params: {{model: anthropic/{name}-test, api_base: \"{base}\", api_key: sk-upstream}}\n"
                )
            };
            let further: String = groups
                .iter()
                .map(|(name, params)| {
                    format!(
                        "  - model_name: {name}\n    litellm_params: {{model: anthropic/further-test, api_base: \"{}\", api_key: sk-upstream{params}}}\n",
                        upstream.uri()
                    )
                })
                .collect();
            let responder = format!(
                "  - model_name: responder\n    litellm_params: {{model: openai/responder-test, api_base: \"{}\", api_key: sk-upstream}}\n",
                responding.uri()
            );
            let config = Config::from_yaml(&format!(
                "model_list:\n{}{}{}{responder}{further}router_settings:\n  model_group_alias: {{sonnet: claude}}\ngeneral_settings:\n  master_key: {MASTER_KEY}\n{general_settings}",
                deployment("claude", &upstream.uri()),
                deployment("gpt", &upstream.uri()),
                deployment("broken", &refusing.uri()),
            ))
            .unwrap();
            let state = AppState {
                keys: Some(Arc::new(Holding(limits))),
                ..AppState::build(&config, &environment).await.unwrap()
            };
            let app = middleware::rate_limited(
                litellm_gateway_inference::router(state.inference.clone()),
                state.key_limiter.clone(),
            )
            .route_layer(from_fn_with_state(state.auth(&config), authenticate));
            Served {
                app,
                limiter: state.key_limiter.clone(),
                upstream,
                responding,
                _refusing: refusing,
            }
        }
    }

    fn charged_to(limiter: &KeyLimiter, descriptor: &str, value: &str) -> Option<f64> {
        let entry = Entry::new(
            "tokens",
            Scope {
                descriptor: descriptor.into(),
                value: value.into(),
            },
        )
        .unwrap();
        match limiter.limiter.engine().store().get_local(&entry).unwrap() {
            Some(Local::Number(number)) => Some(Number::as_f64(number)),
            _ => None,
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

    /// A body asking `model` for at most `max_tokens`, whose prompt is estimated at one token.
    fn asking(model: &str, max_tokens: Option<u64>) -> Value {
        let body = json!({"model": model, "messages": [{"role": "user", "content": "hi"}]});
        match (body, max_tokens) {
            (Value::Object(mut body), Some(max_tokens)) => {
                body.insert("max_tokens".to_owned(), json!(max_tokens));
                Value::Object(body)
            }
            (body, _) => body,
        }
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

    const MESSAGES: &str = "/v1/messages";
    const CHAT: &str = "/v1/chat/completions";
    const RESPONSES: &str = "/v1/responses";

    /// A Responses body asking `responder` for at most `max_output_tokens`.
    fn responding(max_output_tokens: Option<u64>, stream: bool) -> Value {
        let body = json!({"model": "responder", "input": "hi", "stream": stream});
        match (body, max_output_tokens) {
            (Value::Object(mut body), Some(max_output_tokens)) => {
                body.insert("max_output_tokens".to_owned(), json!(max_output_tokens));
                Value::Object(body)
            }
            (body, _) => body,
        }
    }

    async fn post(app: &Router, token: &str, path: &str, body: Value) -> Response {
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

    async fn send_as(app: &Router, token: &str, body: Value) -> Response {
        post(app, token, MESSAGES, body).await
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
    #[case::complete(MESSAGES, complete(), "claude")]
    #[case::streamed_to_its_end(MESSAGES, streamed(), "claude")]
    #[case::a_chat_completion(CHAT, complete(), "claude")]
    #[case::a_response(RESPONSES, responding(None, false), "responder")]
    #[case::a_streamed_response(RESPONSES, responding(None, true), "responder")]
    #[tokio::test]
    async fn a_served_call_charges_its_tokens_less_the_cached_reads_to_its_key_and_model(
        #[case] path: &str,
        #[case] body: Value,
        #[case] group: &str,
    ) {
        let Served { app, limiter, .. } = served(KeyLimits {
            rpm: Some(5),
            ..KeyLimits::default()
        })
        .await;
        let hash = hash_token(KEY);

        let answered = post(&app, KEY, path, body).await;
        let status = answered.status();
        read_to_its_end(answered).await;

        assert_eq!(
            (
                status,
                settled(&limiter, "api_key", &hash, 27.0).await,
                settled(&limiter, "model_per_key", &format!("{hash}:{group}"), 27.0).await
            ),
            (StatusCode::OK, Some(27.0), Some(27.0))
        );
    }

    #[rstest]
    #[case::a_chat_completion(CHAT, asking("claude", Some(50)))]
    #[case::a_response(RESPONSES, responding(Some(50), false))]
    #[case::a_streamed_response(RESPONSES, responding(Some(50), true))]
    #[tokio::test]
    async fn a_token_limit_reserves_what_a_call_of_any_route_asks_for_and_settles_what_it_used(
        #[case] path: &str,
        #[case] body: Value,
    ) {
        let Served { app, limiter, .. } = served(KeyLimits {
            tpm: Some(60),
            ..KeyLimits::default()
        })
        .await;

        let admitted = post(&app, KEY, path, body.clone()).await;
        let status = admitted.status();
        read_to_its_end(admitted).await;
        let used = settled(&limiter, "api_key", &hash_token(KEY), 27.0).await;
        let over = post(&app, KEY, path, body).await;

        assert_eq!(
            (status, used, over.status()),
            (StatusCode::OK, Some(27.0), StatusCode::TOO_MANY_REQUESTS)
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
    #[case::none_sent(MESSAGES, asking("claude", None), "max_tokens", json!(25))]
    #[case::one_sent(MESSAGES, asking("claude", Some(50)), "max_tokens", json!(50))]
    #[case::none_sent_to_chat(CHAT, asking("claude", None), "max_tokens", json!(25))]
    #[case::none_sent_to_responses(
        RESPONSES,
        responding(None, false),
        "max_output_tokens",
        json!(25)
    )]
    #[tokio::test]
    async fn a_body_that_sets_no_output_cap_is_capped_at_a_quarter_of_the_token_limit(
        #[case] path: &str,
        #[case] body: Value,
        #[case] cap: &str,
        #[case] reached: Value,
    ) {
        let served = served(KeyLimits {
            tpm: Some(100),
            rpm: Some(5),
            ..KeyLimits::default()
        })
        .await;

        post(&served.app, KEY, path, body).await;

        assert_eq!(served.provider_saw(path, cap).await, Some(reached));
    }

    #[rstest]
    #[case::set("0.5", json!(50))]
    #[case::beyond_the_whole_limit("1.5", json!(25))]
    #[case::not_a_number("half", json!(25))]
    #[tokio::test]
    async fn the_output_cap_is_the_fraction_of_the_token_limit_the_environment_sets(
        #[case] fraction: &'static str,
        #[case] reached: Value,
    ) {
        let variables = move |name: &str| {
            (name == "NO_MAX_TOKENS_TPM_RESERVATION_FRACTION").then(|| fraction.to_owned())
        };
        let Served { app, upstream, .. } = Setup {
            limits: KeyLimits {
                tpm: Some(100),
                ..KeyLimits::default()
            },
            environment: Environment::read(&variables).unwrap(),
            ..Setup::default()
        }
        .serve()
        .await;

        send(&app, asking("claude", None)).await;

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

    /// The first alias row of the bundled cost map that `picked` takes, with the model it stands
    /// for.
    fn alias_row(picked: fn(&str, &str) -> bool) -> (String, String) {
        let catalog = Catalog::parse(BUNDLED_REGISTRY).unwrap().value;
        catalog
            .aliases()
            .iter()
            .find(|(alias, model)| picked(alias, model))
            .map(|(alias, model)| (alias.clone(), model.clone()))
            .unwrap()
    }

    fn a_bare_alias() -> (String, String) {
        alias_row(|alias, _| !alias.contains('/'))
    }

    fn an_alias_under_its_models_provider() -> (String, String) {
        alias_row(|alias, model| {
            alias
                .split_once('/')
                .is_some_and(|(provider, _)| model.starts_with(&format!("{provider}/")))
        })
    }

    /// An alias asked for through a provider its model's own name does not carry, as a Bedrock
    /// alias of a `global.` model is. Its limit is found under that provider.
    fn an_alias_under_a_provider_hosting_its_model() -> (String, String) {
        let (alias, model) = alias_row(|alias, model| {
            alias
                .split_once('/')
                .is_some_and(|(provider, _)| !model.starts_with(&format!("{provider}/")))
        });
        let limited = format!("{}/{model}", alias.split_once('/').unwrap().0);
        (alias, limited)
    }

    #[rstest]
    #[case::bare(a_bare_alias())]
    #[case::under_its_models_provider(an_alias_under_its_models_provider())]
    #[case::under_a_provider_hosting_its_model(an_alias_under_a_provider_hosting_its_model())]
    #[tokio::test]
    async fn a_limit_on_the_model_a_latest_alias_stands_for_counts_a_call_to_the_alias(
        #[case] (alias, limited): (String, String),
    ) {
        let Served { app, limiter, .. } = deployed(
            KeyLimits {
                model_tpm: by_model(&[(&limited, 60)]),
                ..KeyLimits::default()
            },
            &[(&alias, "")],
        )
        .await;

        let admitted = send(&app, asking(&alias, Some(50))).await;
        let charged = settled(
            &limiter,
            "model_per_key",
            &format!("{}:{limited}", hash_token(KEY)),
            27.0,
        )
        .await;
        let refused = send(&app, asking(&alias, Some(50))).await;

        assert_eq!(
            (admitted.status(), charged, refused.status()),
            (StatusCode::OK, Some(27.0), StatusCode::TOO_MANY_REQUESTS)
        );
    }

    #[rstest]
    #[case::a_key_without_limits(KEY, KeyLimits::default(), &[", default_api_key_rpm_limit: 1"], 1)]
    #[case::the_master_key(MASTER_KEY, KeyLimits::default(), &[", default_api_key_rpm_limit: 1"], 1)]
    #[case::the_least_of_its_groups_deployments(
        KEY,
        KeyLimits::default(),
        &[", default_api_key_rpm_limit: 3", ", default_api_key_rpm_limit: 1", ""],
        1
    )]
    #[case::a_key_with_its_own_limit_for_the_model(
        KEY,
        KeyLimits { model_rpm: by_model(&[("limited", 2)]), ..KeyLimits::default() },
        &[", default_api_key_rpm_limit: 1"],
        2
    )]
    #[tokio::test]
    async fn a_deployments_default_request_limit_counts_a_key_with_no_limit_for_its_model(
        #[case] token: &str,
        #[case] limits: KeyLimits,
        #[case] deployments: &[&str],
        #[case] admitted: usize,
    ) {
        let groups: Vec<(&str, &str)> = deployments
            .iter()
            .map(|params| ("limited", *params))
            .collect();
        let Served { app, .. } = deployed(limits, &groups).await;

        let served = futures_util::future::join_all(
            std::iter::repeat_n((), admitted)
                .map(|()| send_as(&app, token, asking("limited", Some(16)))),
        )
        .await;
        let refused = send_as(&app, token, asking("limited", Some(16))).await;
        let another_group = send_as(&app, token, asking("claude", Some(16))).await;

        assert_eq!(
            served.iter().map(Response::status).collect::<Vec<_>>(),
            vec![StatusCode::OK; admitted]
        );
        assert_eq!(
            (refused.status(), another_group.status()),
            (StatusCode::TOO_MANY_REQUESTS, StatusCode::OK)
        );
        let caller = match token {
            MASTER_KEY => MASTER_KEY_ALIAS.to_owned(),
            _ => hash_token(token),
        };
        let message = json_body(refused).await["error"]["message"].clone();
        assert!(
            message.as_str().unwrap().starts_with(&format!(
                "Rate limit exceeded for model_per_key: {caller}:limited. Limit type: requests. Current limit: {admitted}"
            )),
            "{message}"
        );
    }

    #[tokio::test]
    async fn a_deployments_default_token_limit_reserves_what_the_body_asks_for() {
        let Served { app, upstream, .. } = deployed(
            KeyLimits::default(),
            &[("limited", ", default_api_key_tpm_limit: 60")],
        )
        .await;

        let uncapped = send(&app, asking("limited", None)).await;
        let received = upstream.received_requests().await.unwrap();
        let provider_saw = received.last().map(|request| {
            serde_json::from_slice::<Value>(&request.body).unwrap()["max_tokens"].clone()
        });
        let over = send(&app, asking("limited", Some(50))).await;

        assert_eq!(
            (uncapped.status(), provider_saw, over.status()),
            (
                StatusCode::OK,
                Some(json!(15)),
                StatusCode::TOO_MANY_REQUESTS
            )
        );
    }

    #[rstest]
    #[case::output(MESSAGES, complete(), "output", 14.0)]
    #[case::input(MESSAGES, complete(), "input", 13.0)]
    #[case::anything_else(MESSAGES, complete(), "tokens", 27.0)]
    #[case::a_chat_completions_input(CHAT, complete(), "input", 13.0)]
    #[case::a_responses_output(RESPONSES, responding(None, false), "output", 7.0)]
    #[case::a_responses_input(RESPONSES, responding(None, false), "input", 20.0)]
    #[case::a_streamed_responses_input(RESPONSES, responding(None, true), "input", 20.0)]
    #[tokio::test]
    async fn a_served_call_is_charged_the_tokens_the_configured_type_counts(
        #[case] path: &str,
        #[case] body: Value,
        #[case] token_rate_limit_type: &str,
        #[case] charged: f64,
    ) {
        let Served { app, limiter, .. } = configured(
            KeyLimits::default(),
            &format!("  token_rate_limit_type: {token_rate_limit_type}\n"),
        )
        .await;

        read_to_its_end(post(&app, KEY, path, body).await).await;

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
                    ..Settings::default()
                },
                Settings::default()
            )
        );
    }
}
