import re
from typing import List, Optional
from ray.python.ray.serve._private.request_router.common import PendingRequest
from ray.python.ray.serve._private.request_router.replica_wrapper import RunningReplica
from ray.serve._private.request_router.request_router import FIFOMixin, LocalityMixin, MultiplexMixin, RequestRouter

import ray
import random
import logging
logger = logging.getLogger("ray")


class PDNCCLRequestRouter(FIFOMixin, LocalityMixin, MultiplexMixin, RequestRouter):

    async def choose_replicas(
        self, 
        candidate_replicas: List[RunningReplica], 
        pending_request: Optional[PendingRequest] = None
    ) -> List[List[RunningReplica]]:

        request_id = pending_request.metadata.request_id
        print(f"PDNCCLRequestRouter routing request {request_id}")

        decode_ip, decode_port = self.parse_request_id(request_id, is_prefill=False)
        prefill_ip, prefill_port = self.parse_request_id(request_id, is_prefill=True)

        decode_replica_id = self.parse_replica_id(request_id, is_prefill=False)
        prefill_replica_id = self.parse_replica_id(request_id, is_prefill=True)

        for replica in candidate_replicas:
            if replica.replica_id.unique_id == decode_replica_id:
                return [replica]
            if replica.replica_id.unique_id == prefill_replica_id:
                return [replica]

    @staticmethod
    def parse_request_id(request_id: str, is_prefill=True) -> tuple[str, int]:
        # TODO: import from vllm
        # Regular expression to match the string hostname and integer port
        if is_prefill:
            pattern = r"___decode_addr_(.*):(\d+)"
        else:
            pattern = r"___prefill_addr_(.*):(\d+)___"

        # Use re.search to find the pattern in the request_id
        match = re.search(pattern, request_id)
        if match:
            # Extract the ranks
            ip = match.group(1)
            port = int(match.group(2))

            return ip, port
        raise ValueError(
            f"Request id {request_id} does not contain hostname and port")
    
    @staticmethod
    def parse_replica_id(request_id: str, is_prefill=True) -> str:
        if is_prefill:
            pattern = r"___prefill_replica_id_(.*)___decode_replica_id"
        else:
            pattern = r"___decode_replica_id_(.*)___uuid"
        match = re.search(pattern, request_id)
        if match:
            return match.group(1)
        raise ValueError(f"Request id {request_id} does not contain replica id")