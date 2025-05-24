import os
from typing import Any, Callable, Dict, Optional

from ray import serve

from ray.llm._internal.serve.configs.constants import (
    DEFAULT_HEALTH_CHECK_PERIOD_S,
    DEFAULT_HEALTH_CHECK_TIMEOUT_S,
)

class MockCoreEngineActorManager:
    def __init__(
        self,
        local_engine_count: int,
        start_index: int,
        local_start_index: int,
        vllm_config: VllmConfig,
        addresses,
        executor_class: type[Executor],
        log_stats: bool,
    ):
        pass

class MockDPCoordinator:
    def __init__(self, parallel_config: ParallelConfig):
        pass


class MockAPIServerProcessManager:
    def __init__(
        self,
        target_server_fn: Callable,
        listen_address: str,
        sock: Any,
        args: argparse.Namespace,
        num_servers: int,
        input_addresses: list[str],
        output_addresses: list[str],
        stats_update_address: Optional[str] = None,
    ):
        pass


class DPManager:
    def __init__(
            self,
            llm_config: LLMConfig,
            data_parallel_size: int,
            data_parallel_size_local: int,
            data_parallel_address: str,
            ):

        engine_args = self._get_dp_config(llm_config,
                            data_parallel_size,
                            data_parallel_size_local,
                            data_parallel_address,
                            )
    
        vllm_config = engine_args.create_engine_config()
        parallel_config = vllm_config.parallel_config


        self.dp_engine_manager = MockCoreEngineActorManager(
            local_engine_count=data_parallel_size_local,
            start_index=0,
            local_start_index=0,
            vllm_config=vllm_config,
            addresses=[],
            executor_class=Executor,
            log_stats=False,
        )

    def _get_dp_args(self,
                     llm_config: LLMConfig,
                    data_parallel_size: int,
                    data_parallel_size_local: int,
                    data_parallel_address: str,
                    ) -> "AsyncEngineArgs":
        engine_config = llm_config.get_engine_config()

        # This `model` is the local path on disk, or the hf model id.
        # If it is the hf_model_id, vLLM automatically downloads the correct model from HF.
        # We want this to be the local path on the disk when we already downloaded the
        # model artifacts from a remote storage during node initialization,
        # so vLLM will not require HF token for it and try to download it again.
        model = engine_config.actual_hf_model_id
        if isinstance(llm_config.model_loading_config.model_source, str):
            model = llm_config.model_loading_config.model_source

        return vllm.engine.arg_utils.AsyncEngineArgs(
            **{
                "model": model,
                "data_parallel_backend": "ray",
                "data_parallel_size": data_parallel_size,
                "data_parallel_size_local": data_parallel_size_local,
                "data_parallel_address": data_parallel_address,
                "disable_log_stats": False,
                **engine_config.get_initialization_kwargs(),
            }
        )

    @classmethod
    def as_deployment(
        cls, deployment_options: Dict[str, Any] = None
    ) -> serve.Deployment:
        """Convert the LLMServer to a Ray Serve deployment.

        Args:
            deployment_options: A dictionary of deployment options.

        Returns:
            A Ray Serve deployment.
        """
        deployment_options = deployment_options or {}
        return DPDeployment.options(**deployment_options)

@serve.deployment(
    # TODO make this configurable
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 1,
    },
    health_check_period_s=DEFAULT_HEALTH_CHECK_PERIOD_S,
    health_check_timeout_s=DEFAULT_HEALTH_CHECK_TIMEOUT_S,
)
class DPDeployment(DPManager):
    ...
