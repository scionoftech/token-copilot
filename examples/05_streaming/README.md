# Streaming Examples

This directory contains examples of real-time cost event streaming to external systems.

## Overview

Token-copilot supports streaming cost events to various backends for real-time analytics, persistence, and integration with observability stacks.

## Available Streamers

### 1. WebhookStreamer
**File:** `webhook_streaming.py`

Stream events to HTTP webhooks with batching and retry logic.

**Use Cases:**
- Custom analytics services
- Internal monitoring dashboards
- Serverless functions (AWS Lambda, Google Cloud Functions)
- Third-party analytics platforms

**Installation:**
```bash
pip install token-copilot[streaming]
# or
pip install requests
```

**Features:**
- Async/background sending
- Automatic batching
- Retry with exponential backoff
- Custom headers (auth, metadata)

---

### 2. SyslogStreamer
**File:** `syslog_streaming.py`

Stream events to syslog servers using RFC 5424 format.

**Use Cases:**
- Centralized logging (rsyslog, syslog-ng)
- SIEM integration (Splunk, ArcSight)
- Compliance logging
- Traditional enterprise monitoring

**Installation:**
```bash
# No additional dependencies required
pip install token-copilot
```

**Features:**
- TCP and UDP protocols
- RFC 5424 structured data
- JSON format option
- Configurable facility/severity

---

### 3. LogstashStreamer
**File:** `logstash_streaming.py`

Stream events to Logstash for Elasticsearch indexing (ELK stack).

**Use Cases:**
- Elasticsearch/Kibana dashboards
- ELK stack integration
- Log aggregation
- Full-text search over cost data

**Installation:**
```bash
# No additional dependencies required
pip install token-copilot
```

**Logstash Configuration:**
```conf
input {
  tcp {
    port => 5000
    codec => json_lines
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "llm-costs-%{+YYYY.MM.dd}"
  }
}
```

**Features:**
- JSON over TCP
- Custom index names
- Tags and extra fields
- Automatic reconnection

---

### 4. KafkaStreamer
**File:** `kafka_streaming.py`

Stream events to Apache Kafka for event-driven architectures.

**Use Cases:**
- Event-driven microservices
- Stream processing (Kafka Streams, Flink)
- Data lakes and warehouses
- Real-time analytics pipelines

**Installation:**
```bash
pip install token-copilot[streaming]
# or
pip install kafka-python
```

**Features:**
- Partitioning by user/org/model
- Compression (gzip, snappy, lz4, zstd)
- SASL authentication
- Delivery guarantees

---

### 5. OpenTelemetryStreamer
**File:** `opentelemetry_streaming.py`

Export events as OpenTelemetry spans (OTLP protocol).

**Use Cases:**
- Modern observability platforms (Honeycomb, Datadog, New Relic)
- Distributed tracing
- Service mesh integration
- Cloud-native monitoring

**Installation:**
```bash
pip install token-copilot[streaming]
# or
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
```

**Supported Platforms:**
- Jaeger
- Zipkin
- Honeycomb
- Datadog
- New Relic
- Grafana Tempo
- AWS X-Ray
- Google Cloud Trace

**Features:**
- HTTP and gRPC protocols
- Service metadata
- Custom attributes
- Automatic span batching

---

## Quick Start

### Basic Usage

```python
from token_copilot import TokenCoPilotCallback
from token_copilot.streaming import WebhookStreamer

# Create streamer
streamer = WebhookStreamer(url="https://your-service.com/events")

# Create callback with streamer
callback = TokenCoPilotCallback(
    budget_limit=100.00,
    streamer=streamer
)

# Use with LangChain
llm = ChatOpenAI(callbacks=[callback])
```

### Multiple Streamers

You can stream to multiple backends simultaneously:

```python
class MultiStreamer:
    def __init__(self, streamers):
        self.streamers = streamers
        self.enabled = True

    def send_if_enabled(self, event):
        for streamer in self.streamers:
            streamer.send_if_enabled(event)

    def close(self):
        for streamer in self.streamers:
            streamer.close()

# Stream to webhook AND Kafka
multi = MultiStreamer([
    WebhookStreamer(url="https://..."),
    KafkaStreamer(bootstrap_servers=["kafka:9092"], topic="costs")
])

callback = TokenCoPilotCallback(streamer=multi)
```

---

## Event Format

All streamers send events in the following JSON format:

```json
{
  "timestamp": "2025-12-01T10:30:45.123Z",
  "event_type": "llm_cost",
  "model": "gpt-4o-mini",
  "input_tokens": 1000,
  "output_tokens": 500,
  "total_tokens": 1500,
  "cost": 0.0245,
  "user_id": "user_123",
  "org_id": "org_456",
  "session_id": "session_789",
  "feature": "chat",
  "endpoint": "/api/chat",
  "environment": "production",
  "metadata": {
    "custom_field": "value"
  }
}
```

---

## Configuration

### Environment Variables

You can configure streamers via environment variables:

```bash
# Webhook
export WEBHOOK_URL="https://your-service.com/events"
export WEBHOOK_AUTH_TOKEN="your-token"

# Kafka
export KAFKA_BOOTSTRAP_SERVERS="kafka:9092"
export KAFKA_TOPIC="llm-costs"

# Syslog
export SYSLOG_HOST="logs.company.com"
export SYSLOG_PORT="514"
```

### Production Best Practices

1. **Use async mode** for webhooks to avoid blocking
2. **Enable batching** to reduce network overhead
3. **Set appropriate retry policies** for reliability
4. **Monitor streamer health** and failures
5. **Use compression** for Kafka in high-volume scenarios
6. **Configure proper authentication** (API keys, SASL, SSL)

---

## Running Examples

Each example can be run independently:

```bash
# Install dependencies
pip install token-copilot[streaming,examples]

# Set your OpenAI API key
export OPENAI_API_KEY="your-key"

# Run an example
python webhook_streaming.py
python kafka_streaming.py
python opentelemetry_streaming.py
```

---

## Troubleshooting

### Webhook fails
- Check URL is accessible
- Verify authentication headers
- Check for firewall/network issues
- Enable debug logging

### Kafka connection errors
- Verify bootstrap servers are reachable
- Check SASL credentials if using auth
- Ensure topic exists or auto-creation is enabled

### Syslog not receiving
- Verify syslog server is listening on port
- Check protocol (TCP vs UDP)
- Ensure firewall allows traffic

### OpenTelemetry spans not appearing
- Verify endpoint URL and port
- Check authentication headers
- Ensure collector/platform is running
- Wait for batch flush (up to 30 seconds)

---

## More Information

- [Main Documentation](../../README.md)
- [API Reference](../../docs/)
- [GitHub Issues](https://github.com/scionoftech/token-copilot/issues)
