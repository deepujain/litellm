//! The gateway's answers to router-core's hostcalls, so it routes with no Python behind it, the
//! native counterpart of python-bridge's `PythonRouterHost`. [`NativeBootHost`] answers the
//! deployment build at boot, and [`NativeHost`] one inference call: router-core's hostcalls
//! through [`NativeRouterHost`], and the callback hooks through the [`Callbacks`] it holds.

mod boot;
mod router;

pub use boot::NativeBootHost;
pub use router::NativeRouterHost;

use std::sync::Arc;

use litellm_gateway_auth::AuthenticatedRequest;
use litellm_gateway_inference::routing::{Host, RouteAttempt};
use litellm_router_core::HostFacts;

use crate::{Gateway, callbacks, callbacks::Callbacks};

/// What the native host serves: every route the gateway mounts, and none of the host steps a
/// setting may need.
pub const FACTS: HostFacts = HostFacts {
    capabilities: &[],
    supports: |_, _| true,
};

/// The host the gateway's routes are served through.
#[derive(Default)]
pub struct NativeHost {
    callbacks: Arc<Callbacks>,
}

impl NativeHost {
    pub fn new(callbacks: Arc<Callbacks>) -> Self {
        Self { callbacks }
    }
}

impl Host for NativeHost {
    type RouterHost<'a, A: RouteAttempt + 'a> = NativeRouterHost<'a, A>;
    type Hooks = callbacks::Call;

    fn hooks(&self, _identity: &AuthenticatedRequest, call_type: &'static str) -> callbacks::Call {
        self.callbacks.begin(call_type)
    }

    fn router_host<'a, A: RouteAttempt>(
        &'a self,
        gateway: &'a Gateway,
        identity: &'a AuthenticatedRequest,
        route: &'a A,
    ) -> NativeRouterHost<'a, A> {
        NativeRouterHost {
            gateway,
            identity,
            route,
        }
    }
}
