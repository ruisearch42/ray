import os
import argparse
from typing import Any, Callable, Dict, Optional

from ray import serve
from vllm.v1.executor.abstract import Executor
from vllm import AsyncEngineArgs
from vllm.config import ParallelConfig, VllmConfig

from ray.llm._internal.serve.configs.server_models import (
    LLMConfig,
)
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
            api_server_count: int,
            ):

        # TODO: should api_server_count and MockAPIServerProcessManager
        # be part of LLMServer deployment? 
        engine_args = self._get_dp_config(llm_config,
                            data_parallel_size,
                            data_parallel_size_local,
                            data_parallel_address,
                            )
    
        vllm_config = engine_args.create_engine_config()
        parallel_config = vllm_config.parallel_config
        local_only = data_parallel_size == data_parallel_size_local

        # Set up input and output addresses.
        input_addresses = [
            get_engine_client_zmq_addr(local_only, data_parallel_address)
            for _ in range(api_server_count)
        ]
        output_addresses = [
            get_engine_client_zmq_addr(local_only, data_parallel_address)
            for _ in range(api_server_count)
        ]

        addresses: dict[str, Any] = {
            "input_addresses": input_addresses,
            "output_addresses": output_addresses,
        }

        self.dp_engine_manager = MockCoreEngineActorManager(
            local_engine_count=data_parallel_size_local,
            start_index=0,
            local_start_index=0,
            vllm_config=vllm_config,
            addresses=addresses,
            executor_class=Executor.get_class(vllm_config),
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


def get_tcp_uri(ip: str, port: int) -> str:
    # Brackets are not permitted in ipv4 addresses,
    # see https://github.com/python/cpython/issues/103848
    return f"tcp://[{ip}]:{port}" if ":" in ip else f"tcp://{ip}:{port}"


def get_open_zmq_ipc_path() -> str:
    base_rpc_path = envs.VLLM_RPC_BASE_PATH
    return f"ipc://{base_rpc_path}/{uuid4()}"

def get_engine_client_zmq_addr(local_only: bool,
                               host: str,
                               port: int = 0) -> str:
    return get_open_zmq_ipc_path() if local_only else (get_tcp_uri(
        host, port or get_open_port()))

def get_open_port() -> int:
    """
    Get an open port for the vLLM process to listen on.
    An edge case to handle, is when we run data parallel,
    we need to avoid ports that are potentially used by
    the data parallel master process.
    Right now we reserve 10 ports for the data parallel master
    process. Currently it uses 2 ports.
    """
    if "VLLM_DP_MASTER_PORT" in os.environ:
        dp_master_port = envs.VLLM_DP_MASTER_PORT
        reserved_port_range = range(dp_master_port, dp_master_port + 10)
        while True:
            candidate_port = _get_open_port()
            if candidate_port not in reserved_port_range:
                return candidate_port
    return _get_open_port()

def _get_open_port() -> int:
    port = envs.VLLM_PORT
    if port is not None:
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                port += 1  # Increment port number if already in use
                logger.info("Port %d is already in use, trying port %d",
                            port - 1, port)
    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]