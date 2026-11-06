# Changelog

Release notes are listed newest first. Changes under Unreleased have not yet
been published in a tagged release.

## Unreleased

### Added

- `tau-server` HTTP server with OpenAI-compatible chat completion endpoints.
- Rust inference runtime with CPU and CUDA execution for supported local models.
- Dense Qwen3.5 / Qwen3.8 and Qwen3.5 / Qwen3.6 MoE text-model serving.
- Configurable MoE prefill and decode CUDA projection settings.
- Server and runtime tests, including their required JSON and numerical fixtures.
