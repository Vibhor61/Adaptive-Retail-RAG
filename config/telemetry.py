from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from config.settings import settings

_tracer_provider = None


def setup_tracing():
    global _tracer_provider

    if _tracer_provider is not None:
        return _tracer_provider

    resource = Resource.create(
        {
            "service.name": "retail-rag",
            "service.version": "1.0.0",
            "deployment.environment": "docker"
        }
    )

    _tracer_provider = TracerProvider(resource=resource)

    trace.set_tracer_provider(_tracer_provider)

    exporter = OTLPSpanExporter(
        endpoint=f"http://{settings.phoenix_host}:{settings.phoenix_port}/v1/traces"
    )

    span_processor = BatchSpanProcessor(exporter)
    _tracer_provider.add_span_processor(span_processor)

    return _tracer_provider


def shutdown_tracing():
    global _tracer_provider

    if _tracer_provider:
        _tracer_provider.force_flush()
        _tracer_provider.shutdown()