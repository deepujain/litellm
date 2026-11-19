//! The proxy's callback hooks around one call, at the points `ProxyBaseLLMRequestProcessing`
//! runs them. The [`Host`](crate::Host) the gateway was built with answers them, so a route
//! tells what happened to its call and never learns which callbacks hear it.

use std::{future::Future, sync::Arc};

use axum::{
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use futures_util::stream::BoxStream;
use litellm_framer::sse::SseBlock;
use litellm_gateway_auth::AuthenticatedRequest;
use litellm_host::call::CallOutput;
use litellm_inference::Error as InferenceError;
use litellm_llms_types::formats::chat_completions::ChatCompletionsUsage;
use serde_json::{Map, Value};

use crate::{
    Ended, Gateway, Host,
    stream_event::{Counts, Read},
    stream_outcome::{self, Outcome},
};

/// A call a callback refused, as the proxy's `ProxyException` carries it to a route's handler.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Refusal {
    pub status: StatusCode,
    pub message: String,
    /// The exception's `type`, when it is not the one its status gives.
    pub kind: Option<String>,
    /// The exception's `code`, when it is not its status.
    pub code: Option<String>,
    /// In the order Python sends them.
    pub headers: Vec<(String, String)>,
}

impl Refusal {
    pub fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            ..Self::default()
        }
    }
}

/// A call that was served, as the success log is told it.
#[derive(Clone, Debug, PartialEq)]
pub struct Served {
    /// The model the client asked for.
    pub model: String,
    /// The model group that served it.
    pub model_group: String,
    /// As Python's litellm `Usage` counts it, `None` for an answer that carries no usage.
    pub usage: Option<ChatCompletionsUsage>,
}

/// The hooks of one call, told in the order the proxy runs them:
///
/// - [`pre_call`](Self::pre_call) before the call is routed (`pre_call_hook`)
/// - [`post_call_success`](Self::post_call_success) on a complete answer only, as
///   `post_call_success_hook` never sees a stream
/// - [`response_headers`](Self::response_headers) on every served answer
///   (`post_call_response_headers_hook`)
/// - [`succeeded`](Self::succeeded) once the call is done: a complete answer after it goes out, a
///   stream when it ends (the success log)
/// - [`failed`](Self::failed) for a call that ends on an error, a refusal by a hook included
///   (`post_call_failure_hook`, then the failure log)
///
/// A stream the client drops before its end tells neither `succeeded` nor `failed`, so a callback
/// that holds something for the call gives it back when its state for the request is dropped.
/// Accepted difference: an attempt a retry or a fallback answered for is not logged as a
/// failure, since the route only hears how its call ended.
pub trait CallHooks: Send + Sync + 'static {
    /// Every callback sees the call's `body` in turn and may change it or refuse the call.
    fn pre_call(
        &self,
        body: &mut Map<String, Value>,
    ) -> impl Future<Output = Result<(), Refusal>> + Send;

    fn post_call_success(&self) -> impl Future<Output = Result<(), Refusal>> + Send;

    /// The headers the callbacks add to the answer.
    fn response_headers(&self) -> impl Future<Output = HeaderMap> + Send;

    fn succeeded(&self, served: Served) -> impl Future<Output = ()> + Send;

    fn failed(&self) -> impl Future<Output = ()> + Send;
}

/// Serves one call of the route Python gives `call_type` with the host's hooks around it:
/// `serve` runs on the body the pre-call hook left, and a call that ends on an error, a refusal
/// by that hook included, tells the hooks it failed.
pub(crate) async fn around<H, F, Fut>(
    gateway: &Gateway<H>,
    identity: &AuthenticatedRequest,
    call_type: &'static str,
    mut body: Map<String, Value>,
    serve: F,
) -> Result<Response, Ended>
where
    H: Host,
    F: FnOnce(Arc<H::Hooks>, Map<String, Value>) -> Fut,
    Fut: Future<Output = Result<Response, Ended>>,
{
    let hooks = Arc::new(gateway.host.hooks(identity, call_type));
    let served = match hooks.pre_call(&mut body).await {
        Ok(()) => serve(hooks.clone(), body).await,
        Err(refusal) => Err(refusal.into()),
    };
    if served.is_err() {
        hooks.failed().await;
    }
    served
}

/// Tells the success log off the request, as the proxy logs a complete answer after it went out.
pub(crate) fn log<K: CallHooks>(hooks: &Arc<K>, served: Served) {
    let hooks = hooks.clone();
    tokio::spawn(async move { hooks.succeeded(served).await });
}

/// The answer of a route that never streams, with the hooks told about it and the headers they
/// add.
pub(crate) async fn complete<K: CallHooks>(
    hooks: &Arc<K>,
    served: Served,
    response: Response,
) -> Result<Response, Ended> {
    hooks.post_call_success().await?;
    let added = hooks.response_headers().await;
    log(hooks, served);
    Ok((added, response).into_response())
}

type Blocks = BoxStream<'static, Result<SseBlock, litellm_host_http::Error<InferenceError>>>;

/// `ran`, with `hooks` told how a stream went once it ends: what it `served`, or that it failed.
/// A complete answer is left as it is.
pub(crate) fn told_when_it_ends<R, Head, K: CallHooks>(
    ran: CallOutput<R, Head, SseBlock, litellm_host_http::Error<InferenceError>>,
    hooks: Arc<K>,
    read: Read,
    served: impl Fn(Counts) -> Served + Send + Sync + 'static,
) -> CallOutput<R, Head, SseBlock, litellm_host_http::Error<InferenceError>> {
    let CallOutput::Stream { head, chunks } = ran else {
        return ran;
    };
    let served = Arc::new(served);
    let chunks: Blocks = stream_outcome::watched(
        chunks,
        read,
        stream_outcome::broke,
        move |outcome: Outcome| {
            let (hooks, served) = (hooks.clone(), served.clone());
            async move {
                match outcome {
                    Ok(counts) => hooks.succeeded(served(counts)).await,
                    Err(_) => hooks.failed().await,
                }
            }
        },
    );
    CallOutput::Stream { head, chunks }
}
