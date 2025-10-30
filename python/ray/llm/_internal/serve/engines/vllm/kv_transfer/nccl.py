from ray.llm._internal.serve.engines.vllm.kv_transfer.base import BaseConnectorBackend
import logging


logger = logging.getLogger(__name__)
class P2pNcclConnectorBackend(BaseConnectorBackend):
    def setup(self) -> None:
        # from vllm import envs as vllm_envs, utils as vllm_utils

        base_port = self.kv_transfer_config["kv_port"]
        port = int(base_port) + self._compute_port_offset()
        logger.info(f"{base_port=}, {port=}")

        # ip = vllm_utils.get_ip()
        # zmq_address = f"{ip}:{port}"
        self.kv_transfer_config["kv_port"] = str(port)
