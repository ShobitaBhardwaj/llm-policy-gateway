# LLM Safety Middleware

LLM Safety Middleware is a policy-driven service that sits between an application and any large language model provider. It inspects requests before they reach the model, validates responses before they reach the end user, and applies configurable safety, compliance, and formatting rules defined in YAML.

## Problem Statement

Applications that call LLMs directly often run into the same set of risks:

- users can paste secrets, payment card numbers, passwords, or private keys by mistake
- malicious prompts can try to override instructions, extract hidden context, or jailbreak the model
- model responses can be off-topic, toxic, non-compliant, or structurally invalid
- policy changes often require engineering work even when the rule itself is simple

This project solves that by adding a middleware layer that:

- blocks unsafe input before it reaches the provider
- validates model output before it reaches the client
- lets non-engineers update allowed and disallowed behavior through YAML rules
- retries once with a repair prompt when the first response fails validation
- returns a safe fallback when a compliant response cannot be produced

## Advantages

- provider-agnostic design: the same guardrail layer can sit in front of multiple LLM providers
- policy-driven behavior: rules can be changed without rewriting application logic
- safer operations: sensitive content is blocked and log previews are redacted
- better UX: structured output can be enforced with schema checks and automatic retries
- easier governance: business rules like "always cite sources" or "never discuss competitors" live in one place
- testable architecture: the guardrail engine is separated from the HTTP layer and can be unit tested directly

## Core Capabilities

- input guardrails for:
  - prompt injection
  - jailbreak attempts
  - PII and secret leakage, including payment card numbers
- output guardrails for:
  - valid JSON and required fields
  - toxic or abusive responses
  - off-topic responses
  - citation requirements
  - business policy violations
- configurable YAML policy profiles
- retry-with-repair flow for retryable output failures
- safe fallback response when validation still fails
- streaming endpoint that validates first, then emits compliant chunks

## Project Structure

- `llm_guard/`
  - core models, detectors, policy engine, schema validator, service layer, and API wiring
- `llm_guard/providers/`
  - provider adapter interface
  - built-in `echo` provider for local testing
  - OpenAI-compatible provider adapter
- `config/policies/default.yaml`
  - default rules and profile configuration
- `tests/`
  - unit tests for schema validation, policy behavior, and retry/fallback flow
- `Dockerfile`
  - containerized runtime

## Requirements

- Python `3.9+`
- `pip`
- optional: Docker, if you want to run the service in a container
- optional: an OpenAI-compatible upstream endpoint if you want to call a real model provider

## Installation

### Local Python Installation

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project:

```bash
pip install --upgrade pip
pip install -e .
```

Installed dependencies:

- `fastapi`
- `uvicorn`
- `pydantic`
- `PyYAML`
- `httpx`

### Docker Installation

Build the image:

```bash
docker build -t llm-safety-middleware .
```

Run the container:

```bash
docker run --rm -p 8000:8000 llm-safety-middleware
```

## Setup and Configuration

The service reads configuration from environment variables.

### Environment Variables

- `POLICY_PATH`
  - path to the YAML policy file
  - default: `config/policies/default.yaml`
- `OPENAI_COMPAT_BASE_URL`
  - base URL for an OpenAI-compatible upstream service
  - example: `https://your-provider.example.com/v1`
- `OPENAI_COMPAT_API_KEY`
  - bearer token for the OpenAI-compatible provider
- `STREAM_CHUNK_SIZE`
  - chunk size used by the validated streaming endpoint
  - default: `120`

Example:

```bash
export POLICY_PATH=config/policies/default.yaml
export OPENAI_COMPAT_BASE_URL=https://your-provider.example.com/v1
export OPENAI_COMPAT_API_KEY=your-secret-token
export STREAM_CHUNK_SIZE=120
```

### Policy File

The default policy file is located at `config/policies/default.yaml`.

It demonstrates:

- allowed providers
- input size and output size limits
- blocking prompt injection and PII
- blocking competitor discussion
- requiring citations
- blocking medical advice
- retry configuration
- safe fallback messaging

Profiles are selected per request using `policy_profile`.

## Running the Service

Start the API server locally:

```bash
uvicorn llm_guard.app:app --host 0.0.0.0 --port 8000
```

Available endpoints:

- `GET /healthz`
  - basic process health
- `GET /readyz`
  - verifies that the policy loads and providers are available
- `POST /v1/chat/completions`
  - synchronous validated completion endpoint
- `POST /v1/chat/completions/stream`
  - validated streaming endpoint using server-sent events

## Usage

### Request Shape

The main request fields are:

- `provider`
  - provider adapter name such as `echo` or `openai_compatible`
- `model`
  - upstream model identifier
- `messages`
  - chat messages in `{role, content}` format
- `response_schema`
  - optional JSON schema used to enforce output structure
- `policy_profile`
  - policy profile to apply, default is `default`
- `parameters`
  - optional provider-specific parameters
- `metadata`
  - optional request metadata, also used by the mock `echo` provider for local testing

### Local Example with the Mock Provider

This example uses the built-in `echo` provider, which returns the value from `metadata.mock_response`.

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "echo",
    "model": "mock-model",
    "policy_profile": "default",
    "messages": [
      {
        "role": "user",
        "content": "Summarize our refund policy as JSON and cite sources."
      }
    ],
    "response_schema": {
      "type": "object",
      "required": ["answer", "sources"],
      "properties": {
        "answer": { "type": "string" },
        "sources": {
          "type": "array",
          "items": { "type": "string" },
          "minItems": 1
        }
      },
      "additionalProperties": false
    },
    "metadata": {
      "mock_response": "{\"answer\":\"Refunds are processed within 7 business days.\",\"sources\":[\"https://example.com/policy\"]}"
    }
  }'
```

Example successful response:

```json
{
  "request_id": "805e863f-cfbe-422f-b7ce-89d422c7f16f",
  "provider": "echo",
  "model": "mock-model",
  "content": "{\"answer\":\"Refunds are processed within 7 business days.\",\"sources\":[\"https://example.com/policy\"]}",
  "structured_output": {
    "answer": "Refunds are processed within 7 business days.",
    "sources": [
      "https://example.com/policy"
    ]
  },
  "finish_reason": "stop",
  "guardrails": {
    "status": "passed",
    "policy_profile": "default",
    "policy_version": "2026-07-04",
    "retry_count": 0,
    "input_violations": [],
    "output_violations": []
  }
}
```

### Streaming Example

```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions/stream \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "echo",
    "model": "mock-model",
    "messages": [
      {
        "role": "user",
        "content": "Return a policy summary."
      }
    ],
    "metadata": {
      "mock_response": "This is a validated response."
    }
  }'
```

The streaming endpoint emits:

- a `meta` event
- one or more `chunk` events
- a final `done` event with the validated response payload

## How Validation Works

### Input Guardrails

Before the request reaches the provider, the middleware checks for:

- oversized prompts
- prompt injection patterns
- jailbreak attempts
- PII and credential leakage
- input-side policy rules from YAML

If input validation fails, the request is blocked with `GUARDRAIL_INPUT_BLOCKED`.

### Output Guardrails

After the provider responds, the middleware checks for:

- policy rule violations
- oversized output
- toxicity
- output PII leakage
- JSON parsing and schema conformance
- citation requirements
- topic relevance

If the response fails a retryable validation rule, the middleware sends one repair retry to the provider with instructions to correct the output. If the response still fails, or the failure is non-retryable, the middleware returns a safe fallback or an output-unavailable error.

## Policy Customization

Policies are defined in YAML and intended to be editable by non-engineers.

Example ideas supported by the current engine:

- `block`
- `redact`
- `rewrite`
- `require_citation`
- `require_topic`
- `require_json`
- `fallback`

Example business rules:

- "Never discuss competitors"
- "Always cite sources"
- "Block all medical advice"

## Testing

Run the unit test suite:

```bash
python3 -m unittest discover -s tests -v
```

Run a compile check without writing outside the workspace:

```bash
PYTHONPYCACHEPREFIX=work/pycache python3 -m compileall llm_guard tests
```

## Current Notes

- the API layer depends on `fastapi` and `uvicorn`, so install project dependencies before starting the server
- the built-in `echo` provider is useful for local validation and automated tests
- the OpenAI-compatible adapter is optional and enabled only when `OPENAI_COMPAT_BASE_URL` is configured
- the current JSON schema validator is intentionally lightweight and implemented in-project rather than relying on an external schema library
- the streaming endpoint validates the full response before emitting chunks, which favors safety over minimal latency

## When This Middleware Is Useful

This service is a strong fit when you need:

- a governed LLM gateway for internal tools
- output contracts for downstream automation
- centralized safety rules across multiple apps
- a separation between business policy and application code
- safer rollout of LLM features in regulated or customer-facing workflows
