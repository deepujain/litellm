use std::{collections::BTreeMap, future::Future, pin::Pin, sync::Arc, time::SystemTime};

use litellm_secret_value::SecretValue;

use crate::{Error, hash_token};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KeyHash(String);

impl KeyHash {
    pub fn from_token(token: &SecretValue) -> Self {
        Self(hash_token(token.expose()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A limit for each model a key names one for, by the name a caller asks for or its model group.
pub type ModelLimits = Arc<BTreeMap<String, i64>>;

/// The limits a virtual key holds, which the server enforces after authentication.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct KeyLimits {
    pub rpm: Option<i64>,
    pub tpm: Option<i64>,
    pub max_parallel_requests: Option<i64>,
    pub model_rpm: ModelLimits,
    pub model_tpm: ModelLimits,
}

impl KeyLimits {
    /// Whether any limit counts tokens, which a call reserves from its body before it runs.
    pub fn counts_tokens(&self) -> bool {
        self.tpm.is_some() || !self.model_tpm.is_empty()
    }

    /// Whether any limit is told from the request body: its tokens or the model it names.
    pub fn reads_body(&self) -> bool {
        self.counts_tokens() || !self.model_rpm.is_empty()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KeyStatus {
    /// `models` is the key's model list, where none reaches every model.
    Active {
        expires_at: Option<SystemTime>,
        models: Arc<[String]>,
        limits: KeyLimits,
    },
    Revoked,
}

pub type LookupFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Option<KeyStatus>, Error>> + Send + 'a>>;

/// The virtual keys this server authenticates. A key it holds but cannot honor is an error, never
/// a status.
pub trait KeyLookup: Send + Sync {
    fn lookup<'a>(&'a self, hash: &'a KeyHash) -> LookupFuture<'a>;
}
