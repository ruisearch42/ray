from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="qwen-0.5b",
        model_source="Qwen/Qwen2.5-0.5B-Instruct",
    ),
    deployment_config=dict(
        autoscaling_config=dict(
            min_replicas=1,
            max_replicas=2,
        )
    ),
    # Pass the desired accelerator type (e.g. A10G, L4, etc.)
    accelerator_type="L4",
    # You can customize the engine arguments (e.g. vLLM engine kwargs)
    engine_kwargs=dict(
        data_parallel_size=2,
    ),
)

app = build_openai_app({"llm_configs": [llm_config]})
serve.run(app, blocking=True)
