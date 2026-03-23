import logging
import os

# Try to import posthog, but make it optional
try:
    from posthog import Posthog
    POSTHOG_AVAILABLE = True
except ImportError:
    POSTHOG_AVAILABLE = False
    Posthog = None

MEM0_TELEMETRY = os.environ.get("MEM0_TELEMETRY", "False")  # Disabled by default

if isinstance(MEM0_TELEMETRY, str):
    MEM0_TELEMETRY = MEM0_TELEMETRY.lower() in ("true", "1", "yes")


class AnonymousTelemetry:
    """Minimal telemetry implementation (disabled by default)."""
    
    def __init__(self, vector_store=None):
        self.posthog = None
        self.user_id = None
        
    def capture_event(self, event_name, event_data, user_email):
        pass


# Create global instance
client_telemetry = AnonymousTelemetry()


def capture_event(event_name, instance, additional_data=None):
    """Capture telemetry event (no-op in minimal build)."""
    pass


def capture_client_event(event_name, instance, additional_data=None):
    """Capture client telemetry event (no-op in minimal build)."""
    pass
