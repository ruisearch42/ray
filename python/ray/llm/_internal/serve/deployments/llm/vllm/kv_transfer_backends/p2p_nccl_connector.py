import os

from ray.llm._internal.serve.deployments.llm.vllm.kv_transfer_backends.base import (
    BaseConnectorBackend,
)


class P2pNcclConnectorBackend(BaseConnectorBackend):

    def setup(self) -> None:
        pass
