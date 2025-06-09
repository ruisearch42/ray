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
from vllm.v1.utils import EngineCoreActorManager
from vllm.v1.executor.abstract import Executor


class DPManager:
    def __init__(self, llm_config: LLMConfig):
        self.llm_config = llm_config
        engine_args, engine_config = _get_vllm_engine_config(self.llm_config)
        # NEXT: determine addresses
        addresses = None
        self.actor_manager = EngineCoreActorManager(
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
