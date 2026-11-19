//! The proxy's call type for the route matched at a path, for whoever has to tell it before the
//! route runs, as the gateway's `admit` does to know which bodies only Python estimates.

use crate::{audio_transcription, chat_completions, messages, ocr, responses};

/// The proxy's call type for a Responses call, whose output cap is its own field.
pub const RESPONSES_CALL_TYPE: &str = responses::CALL_TYPE;

/// The call type the proxy gives the route matched at `path`, which decides what a token estimate
/// reads from the body. A path no inference route serves is `""`.
pub fn call_type(path: &str) -> &'static str {
    if path.ends_with("/messages") {
        messages::CALL_TYPE
    } else if path.ends_with("/chat/completions") {
        chat_completions::CALL_TYPE
    } else if path.ends_with("/responses") {
        responses::CALL_TYPE
    } else if path.ends_with("/ocr") {
        ocr::CALL_TYPE
    } else if path.ends_with("/audio/transcriptions") {
        audio_transcription::CALL_TYPE
    } else {
        ""
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::messages("/v1/messages", "anthropic_messages")]
    #[case::chat("/v1/chat/completions", "acompletion")]
    #[case::a_deployment_path("/openai/deployments/{model}/chat/completions", "acompletion")]
    #[case::responses("/responses", "aresponses")]
    #[case::ocr("/v1/ocr", "aocr")]
    #[case::transcriptions("/v1/audio/transcriptions", "atranscription")]
    #[case::any_other("/v1/embeddings", "")]
    fn each_route_is_estimated_as_the_proxys_call_type(#[case] path: &str, #[case] expected: &str) {
        assert_eq!(call_type(path), expected);
    }
}
