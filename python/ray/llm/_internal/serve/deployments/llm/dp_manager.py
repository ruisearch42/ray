from typing import Any, Dict
from ray.llm._internal.serve.deployments.llm.vllm.vllm_engine import (
    _get_vllm_engine_config,
)
from ray.llm._internal.serve.configs.server_models import LLMConfig
from ray import serve
from ray.llm._internal.serve.configs.constants import (
    DEFAULT_HEALTH_CHECK_PERIOD_S,
    DEFAULT_HEALTH_CHECK_TIMEOUT_S,
)
import ray
from vllm.v1.utils import CoreEngineActorManager
from vllm.v1.executor.abstract import Executor


class DPManager:
    def __init__(self, llm_config: LLMConfig):
        self.llm_config = llm_config
        # Create engine config on a task with access to GPU,
        # as GPU capability may be queried.
        ref = (
            ray.remote(
                num_cpus=0,
                num_gpus=1,
                accelerator_type=self.llm_config.accelerator_type,
            )(_get_vllm_engine_config)
            # .options(
            #     runtime_env=node_initialization.runtime_env,
            #     scheduling_strategy=PlacementGroupSchedulingStrategy(
            #         placement_group=node_initialization.placement_group,
            #     ),
            # )
            .remote(self.llm_config)
        )
        engine_args, engine_config = ray.get(ref)

        # NEXT: determine addresses
        addresses = None
        self.actor_manager = CoreEngineActorManager(
            vllm_config=engine_config,
            addresses=addresses,
            executor_class=Executor.get_class(engine_config),
            log_stats=not engine_args.disable_log_stats,
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
        return DPManagerDeployment.options(**deployment_options)


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
class DPManagerDeployment(DPManager):
    ...
