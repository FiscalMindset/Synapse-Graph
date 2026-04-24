# Synapse-Graph Backend

Step 1 provides the local inference substrate for Synapse-Graph:

- Ollama is used as the preferred local generation engine when `http://127.0.0.1:11434` is reachable.
- A parallel Hugging Face / PyTorch runner loads an instrumented causal LM and registers forward hooks on every transformer attention module.
- Each generation step captures the last-query attention slice for every layer, summarizes the hottest heads, and emits a step-wise activation path that the FastAPI and OpenMetadata layers can ingest next.

## Recommended Runtime

Use Python `3.11` or `3.12`. PyTorch and Transformers support is still uneven on `3.14`, so the project metadata intentionally pins below `3.13`.

## Quick Wiring Example

```python
import asyncio

from app.inference import GenerationRequest, NeuralInferenceEngine


async def main() -> None:
    engine = NeuralInferenceEngine()
    await engine.startup()

    response = await engine.generate(
        GenerationRequest(
            prompt="Explain why attention heads matter for interpretability.",
            max_new_tokens=96,
        )
    )

    print(response.text)
    print(response.trace.steps[0].high_activation_path)

    await engine.shutdown()


asyncio.run(main())
```
