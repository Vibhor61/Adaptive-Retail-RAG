from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


_tracer_provider = None


def setup_tracing():
    """
    Initializes OpenTelemetry and exports traces to Phoenix.
    """
    global _tracer_provider
    
    _tracer_provider = TracerProvider()
    trace.set_tracer_provider(_tracer_provider)

    exporter = OTLPSpanExporter(
        endpoint="http://localhost:5107/v1/traces",
        headers={}
    )

    # Batch processor (recommended for performance)
    span_processor = BatchSpanProcessor(exporter)
    _tracer_provider.add_span_processor(span_processor)

    return _tracer_provider


def flush_traces():
    """
    Flushes all pending traces to Phoenix before shutdown.
    """
    global _tracer_provider
    if _tracer_provider:
        _tracer_provider.force_flush(timeout_millis=5000)