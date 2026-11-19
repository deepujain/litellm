use litellm_gateway_auth::AuthenticatedRequest;
use std::{path::Path, sync::Arc};

use axum::{
    Extension, Json,
    extract::State,
    response::{IntoResponse, Response},
};
use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_inference_transcription::{AudioTranscriptionRoute, types::AudioTranscriptionRequest};
use litellm_router_core::{Attempt, Op, Served};
use serde_json::{Map, Value, json};

use crate::{
    Ended, Error, Gateway,
    headers::Routing,
    hooks::{self, Served as ServedCall},
    request::{self, InferenceBody, Upload},
    routing::{self, Host, RouteAttempt},
};

pub(crate) async fn create<H: Host>(
    State(gateway): State<Arc<Gateway<H>>>,
    identity: AuthenticatedRequest,
    body: InferenceBody,
) -> Result<Response, Ended> {
    let InferenceBody { fields, upload } = body;
    hooks::around(&gateway, &identity, CALL_TYPE, fields, |hooks, fields| {
        serve(&gateway, &identity, hooks, fields, upload)
    })
    .await
}

/// Python's `call_type` for this route.
pub(crate) const CALL_TYPE: &str = "atranscription";

async fn serve<H: Host>(
    gateway: &Gateway<H>,
    identity: &AuthenticatedRequest,
    hooks: Arc<H::Hooks>,
    body: Map<String, Value>,
    upload: Option<Upload>,
) -> Result<Response, Ended> {
    let model = request::model(&body)?.to_owned();
    let audio = match upload {
        Some(upload) => {
            let format = upload
                .file_name
                .as_deref()
                .and_then(|name| Path::new(name).extension())
                .and_then(|extension| extension.to_str())
                .map(str::to_ascii_lowercase);
            json!({"data": STANDARD.encode(upload.bytes), "format": format})
        }
        None => body.get("audio").cloned().unwrap_or_default(),
    };
    let attempt = TranscriptionAttempt {
        route: &gateway.audio_transcription,
        body,
        audio,
    };
    let routed = routing::call(
        gateway,
        identity,
        Op::TRANSCRIPTION,
        "/audio/transcriptions",
        &model,
        &attempt,
    )
    .await?;
    let served = ServedCall {
        model,
        model_group: routed.name.clone(),
        usage: None,
    };
    let response = (
        Extension(Routing::served(&routed)),
        Json(routed.output.response),
    )
        .into_response();
    hooks::complete(&hooks, served, response).await
}

struct TranscriptionAttempt<'a> {
    route: &'a AudioTranscriptionRoute,
    body: Map<String, Value>,
    audio: Value,
}

impl RouteAttempt for TranscriptionAttempt<'_> {
    type Outcome = Value;

    async fn run(&self, attempt: &Attempt) -> Result<Served<Value>, Error> {
        let deployment = &attempt.deployment;
        let api_key = routing::api_key(deployment);
        let body = self
            .route
            .execute(AudioTranscriptionRequest {
                model: &deployment.params.model,
                audio: self.audio.clone(),
                api_key: api_key.as_deref(),
                api_base: deployment.params.api_base.as_deref(),
                custom_llm_provider: deployment.params.custom_llm_provider.as_deref(),
                extra_headers: None,
                optional_params: routing::attempt_body(&self.body, attempt)
                    .into_iter()
                    .filter(|(name, _)| !matches!(name.as_str(), "model" | "audio"))
                    .collect(),
                timeout: routing::timeout(attempt, false),
            })
            .await?;
        let total_tokens = routing::total_tokens(body.get("usage"));
        Ok(Served::new(body, total_tokens))
    }
}
