from ray.llm._internal.serve.engines.vllm.kv_transfer.base import BaseConnectorBackend

class P2pNcclConnectorBackend(BaseConnectorBackend):

    def setup(self) -> None:
        pass