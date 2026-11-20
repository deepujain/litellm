use std::sync::Arc;

use axum::{
    Extension, Json,
    body::Bytes,
    extract::State,
    response::{IntoResponse, Response},
};
use litellm_cache_response::CachePolicy;
use litellm_gateway_auth::AuthenticatedRequest;
use litellm_host::call::CallOutput;
use litellm_host_http::Sse;
use litellm_inference_responses::{ResponsesRoute, route::Responses, types::ResponsesCall};
use litellm_router_core::{Attempt, Op, Served};
use serde_json::{Map, Value};

use crate::{
    Ended, Error, Gateway, JsonObject,
    caching::CacheHeaders,
    headers::Routing,
    hooks::{self, CallHooks, Served as ServedCall},
    model_restamp::ReportedModel,
    request,
    routing::{self, Host, RouteAttempt},
    stream_event::{self, Counts},
    stream_outcome::{self, Streamed},
};

/// Python's `call_type` for this route.
const CALL_TYPE: &str = "aresponses";

pub(crate) async fn create<H: Host>(
    State(gateway): State<Arc<Gateway<H>>>,
    identity: AuthenticatedRequest,
    JsonObject(body): JsonObject,
) -> Result<Response, Ended> {
    hooks::around(&gateway, &identity, CALL_TYPE, body, |hooks, body| {
        serve(&gateway, &identity, hooks, body)
    })
    .await
}

/// What a complete answer's `usage` counts.
fn complete_counts(usage: Option<&Value>) -> Option<Counts> {
    serde_json::from_value(usage?.clone()).ok()
}

async fn serve<H: Host>(
    gateway: &Gateway<H>,
    identity: &AuthenticatedRequest,
    hooks: Arc<H::Hooks>,
    body: Map<String, Value>,
) -> Result<Response, Ended> {
    let model = request::model(&body)?.to_owned();
    let (body, cache_options) = crate::caching::prepare(identity, body)?;
    let route = gateway.responses.clone();
    let route = match &gateway.cache {
        Some(cache) => route.with_cache(litellm_cache_response::ScopedCache::new(
            cache.clone(),
            cache_options.scope.clone(),
        )),
        None => route,
    };
    let attempt = ResponsesAttempt {
        gateway,
        route,
        body,
        policy: cache_options.policy,
    };
    let routed = routing::call(
        gateway,
        identity,
        Op::RESPONSES,
        "/responses",
        &model,
        &attempt,
    )
    .await?;
    let routing = Routing::served(&routed);
    let reported = ReportedModel::of(&model, &routed);
    let served = {
        let (model, model_group) = (model.clone(), routed.name.clone());
        move |counts: Option<Counts>| ServedCall {
            model: model.clone(),
            model_group: model_group.clone(),
            usage: counts.map(Counts::responses_usage),
        }
    };
    let (ran, cache) = routed.output.response;
    let complete = match &ran {
        CallOutput::Complete(response) => {
            hooks.post_call_success().await?;
            Some(served(complete_counts(response.extra.get("usage"))))
        }
        CallOutput::Stream { .. } => None,
    };
    let added = hooks.response_headers().await;
    let ran = hooks::told_when_it_ends(
        ran,
        hooks.clone(),
        stream_event::responses_event,
        move |counts| served(Some(counts)),
    );
    let stream = Sse::<Responses, _, _>::new(Json, |error| {
        Bytes::from(Error::from(error).responses_sse_frame())
    });
    let ran = stream_outcome::bytes(reported.complete(ran));
    let response = litellm_host_http::render(stream, ran)?;
    if let Some(served) = complete {
        hooks::log(&hooks, served);
    }
    Ok((Extension(routing), added, cache.apply(response)).into_response())
}

struct ResponsesAttempt<'a, H> {
    gateway: &'a Gateway<H>,
    route: ResponsesRoute,
    body: Map<String, Value>,
    policy: CachePolicy,
}

impl<H: Host> RouteAttempt for ResponsesAttempt<'_, H> {
    type Outcome = (Streamed<Responses>, CacheHeaders);

    async fn run(&self, attempt: &Attempt) -> Result<Served<Self::Outcome>, Error> {
        let deployment = &attempt.deployment;
        let body = routing::attempt_body(&self.body, attempt);
        let call = ResponsesCall {
            model: deployment.params.model.clone(),
            input: body.get("input").cloned().unwrap_or_default(),
            optional_params: body
                .into_iter()
                .filter(|(name, _)| !matches!(name.as_str(), "model" | "input"))
                .collect(),
            api_key: routing::api_key(deployment),
            api_base: deployment.params.api_base.clone(),
            custom_llm_provider: deployment.params.custom_llm_provider.clone(),
            extra_headers: None,
            timeout: routing::timeout(attempt, routing::streamed(&self.body)),
        };
        let machine = self.route.clone().machine(call, self.policy);
        let cache = CacheHeaders::default();
        let ran = litellm_host_http::run(machine, (), cache.clone(), None).await?;
        Ok(match ran {
            CallOutput::Complete(response) => {
                let total_tokens = routing::total_tokens(response.extra.get("usage"));
                Served::new((CallOutput::Complete(response), cache), total_tokens)
            }
            CallOutput::Stream { head, chunks } => {
                let chunks = stream_outcome::reporting_outcome(
                    self.gateway,
                    deployment,
                    chunks,
                    stream_event::responses_event,
                );
                Served {
                    deferred: true,
                    ..Served::new((CallOutput::Stream { head, chunks }, cache), 0)
                }
            }
        })
    }
}
