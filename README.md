# Tau.rs

`Tau.rs` (pronounced `Taurus`) is a dense + MoE inference engine written
in `Rust` from ground up with custom `CUDA` kernels. It is influenced by peer
inference engines in `C++` and
 `Python`. `Tau.rs`' design is most influenced by `llama.cpp` (in `C++`); its
optimization routines are most influenced by `SGLang` (in `Python`); and its
development used tool chain is `vLLM` (in `Python`).

`Tau.rs` aims to deliver comparable inference performance on latest dense and
MoE models with much less complexity compared to `SGLang` and `vLLM`, hence allowing
lean development and more flexibility in adaptation. It also aims to achieve
better reliability and security in production largely attributed to the `Rust`
programming language.

`tau-server` provides HTTP and OpenAI-compatible chat completion endpoints for
supported local safetensors models, including `Qwen3.5` / `Qwen3.8` dense and
`Qwen3.5` / `Qwen3.6` MoE text models.

See [CHANGELOG.md](CHANGELOG.md) for release notes.

## Build

Use Rust with edition 2024 support, a C++ compiler, and the NVIDIA CUDA toolkit.
Inference binaries include CPU and CUDA support; the toolkit is required even when
serving on CPU. The binaries need the linked CUDA runtime and cuBLAS libraries.
The complete default test suite requires a usable NVIDIA GPU.

```sh
cargo build --release --locked -p tau-server
```

The executable is `target/release/tau-server`. When cross-building for another
NVIDIA architecture, set both `CUDA_COMPUTE_CAP` and `TAU_CUDA_ARCH` to the target
compute capability accepted by the CUDA toolchain (for example, `90` for H200).
Build and validate on the intended hardware.

## Run

```sh
./target/release/tau-server \
  --model /path/to/model \
  --device gpu \
  --dtype bf16 \
  --listen 0.0.0.0:18081

curl http://127.0.0.1:18081/v1/models
# Replace MODEL_ID below with the identifier returned by /v1/models.
curl http://127.0.0.1:18081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MODEL_ID","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

Use the model identifier returned by `/v1/models` in chat requests. To serve on
CPU, use `--device cpu --dtype f32`. Run `tau-server --help` for all options.
Without `--model`, the server provides a deterministic loopback engine for API
smoke checks; this does not run a language model.

## Test

```sh
cargo fmt --all -- --check
cargo test --workspace --locked -- --test-threads=1
```

Unit tests, their compile-time JSON references and small numerical fixtures are
included. Model-backed tests and microbenchmarks marked `#[ignore]` need separate
model artifacts and are excluded by the default command. Model weights are not
bundled. Tests expose
`TAU_*_TEST_MODEL` environment overrides in their source for supplying local
artifacts; run specific ignored tests only after preparing their required
checkpoint or generated fixture. This source snapshot does not include the
development fixture-generation harness.

## License

Apache-2.0. See [LICENSE](LICENSE).
