import os
import argparse
from typing import Any, Callable, Dict, Optional

from ray import serve
import ray
from ray._private.state import available_resources_per_node
from ray.util.scheduling_strategies import (
    PlacementGroupSchedulingStrategy)
from ray.util.state import list_nodes
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
        placement_groups=None,
        local_dp_ranks: Optional[list[int]] = None,
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
        # TODO: move other params to LLMConfig

        self.data_parallel_address = data_parallel_address
        # TODO: should api_server_count and MockAPIServerProcessManager
        # be part of LLMServer deployment? 
        engine_args = self._get_dp_config(llm_config,
                            data_parallel_size,
                            data_parallel_size_local,
                            data_parallel_address,
                            )
    
        self.vllm_config = engine_args.create_engine_config()
        parallel_config = self.vllm_config.parallel_config
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

        coordinator = MockDPCoordinator(parallel_config)
        addresses.update(coordinator.get_engine_socket_addresses())
        stats_update_address = coordinator.get_stats_publish_address()
        placement_groups, local_dp_ranks = self._decide_placement()

        self.dp_engine_manager = MockCoreEngineActorManager(
            local_engine_count=data_parallel_size_local,
            start_index=0,
            local_start_index=0,
            vllm_config=self.vllm_config,
            addresses=addresses,
            executor_class=Executor.get_class(self.vllm_config),
            log_stats=False,
            placement_groups=placement_groups,
            local_dp_ranks=local_dp_ranks,
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

        return AsyncEngineArgs(
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

    def _decide_placement(self):
        # TODO: is head_node_ip the same as data_parallel_address?
        head_node_ip = self.data_parallel_address
        vllm_config = self.vllm_config
        local_engine_count = vllm_config.parallel_config.data_parallel_size_local
        dp_size = vllm_config.parallel_config.data_parallel_size
        world_size = vllm_config.parallel_config.world_size

        nodes = list_nodes()
        nodes = sorted(list_nodes(),
                        key=lambda node: node.node_ip != head_node_ip)
        assert nodes[0].node_ip == head_node_ip, (
            "The first node must be the head node")
        assert len(nodes) == 1 or nodes[1].node_ip != head_node_ip, (
            "There can only be one head node")

        available_resources = available_resources_per_node()
        world_size = vllm_config.parallel_config.world_size
        placement_groups = []
        local_dp_ranks = []

        for node in nodes:
            node_ip = node.node_ip
            node_resources = available_resources[node.node_id]
            # For now, each DP rank can only be assigned to one node
            # TODO(rui): support allocating a single DP rank
            # to multiple nodes
            available_engine_count = node_resources["GPU"] // world_size
            if node_ip == head_node_ip:
                assert available_engine_count >= local_engine_count, (
                    "Not enough resources to allocate DP ranks "
                    f"on DP master node {node_ip}")
                for i in range(local_engine_count):
                    bundles = [{
                        "GPU": 1.0,
                        "node:" + head_node_ip: 0.001
                    }] * world_size + [{
                        "CPU": 1.0
                    }]
                    pg = ray.util.placement_group(
                        name=f"dp_rank_{len(placement_groups)}",
                        strategy="STRICT_PACK",
                        bundles=bundles,
                    )
                    placement_groups.append(pg)
                    local_dp_ranks.append(i)
            else:
                for i in range(available_engine_count):
                    if len(placement_groups) == dp_size:
                        break
                    bundles = [{"GPU": 1.0}] * world_size + [{"CPU": 1.0}]
                    pg = ray.util.placement_group(
                        name=f"dp_rank_{len(placement_groups)}",
                        strategy="STRICT_PACK",
                        bundles=bundles,
                    )
                    placement_groups.append(pg)
                    local_dp_ranks.append(i)
        return placement_groups, local_dp_ranks

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