import logging
import numpy as np
import math
import pandas as pd

from typing import Dict, Any, Optional, List

from ray.data import Dataset

from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch, convert_ma_batch_to_sample_batch
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.utils.numpy import convert_to_numpy

from ray.rllib.offline.estimators.off_policy_estimator import OffPolicyEstimator
from ray.rllib.offline.estimators.fqe_torch_model import FQETorchModel
from ray.rllib.offline.offline_evaluator import OfflineEvaluator
from ray.rllib.offline.offline_evaluation_utils import (
    compute_is_weights,
    compute_q_and_v_values,
)

logger = logging.getLogger()


@DeveloperAPI
class DoublyRobust(OffPolicyEstimator):
    r"""The Doubly Robust estimator.

    Let s_t, a_t, and r_t be the state, action, and reward at timestep t.

    This method trains a Q-model for the evaluation policy \pi_e on behavior
    data generated by \pi_b. Currently, RLlib implements this using
    Fitted-Q Evaluation (FQE). You can also implement your own model
    and pass it in as `q_model_config = {"type": your_model_class, **your_kwargs}`.

    For behavior policy \pi_b and evaluation policy \pi_e, define the
    cumulative importance ratio at timestep t as:
    p_t = \sum_{t'=0}^t (\pi_e(a_{t'} | s_{t'}) / \pi_b(a_{t'} | s_{t'})).

    Consider an episode with length T. Let V_T = 0.
    For all t in {0, T - 1}, use the following recursive update:
    V_t^DR = (\sum_{a \in A} \pi_e(a | s_t) Q(s_t, a))
        + p_t * (r_t + \gamma * V_{t+1}^DR - Q(s_t, a_t))

    This estimator computes the expected return for \pi_e for an episode as:
    V^{\pi_e}(s_0) = V_0^DR
    and returns the mean and standard deviation over episodes.

    For more information refer to https://arxiv.org/pdf/1911.06854.pdf"""

    @override(OffPolicyEstimator)
    def __init__(
        self,
        policy: Policy,
        gamma: float,
        epsilon_greedy: float = 0.0,
        normalize_weights: bool = True,
        q_model_config: Optional[Dict] = None,
    ):
        """Initializes a Doubly Robust OPE Estimator.

        Args:
            policy: Policy to evaluate.
            gamma: Discount factor of the environment.
            epsilon_greedy: The probability by which we act acording to a fully random
                policy during deployment. With 1-epsilon_greedy we act
                according the target policy.
            normalize_weights: If True, the inverse propensity scores are normalized to
                their sum across the entire dataset. The effect of this is similar to
                weighted importance sampling compared to standard importance sampling.
            q_model_config: Arguments to specify the Q-model. Must specify
                a `type` key pointing to the Q-model class.
                This Q-model is trained in the train() method and is used
                to compute the state-value and Q-value estimates
                for the DoublyRobust estimator.
                It must implement `train`, `estimate_q`, and `estimate_v`.
                TODO (Rohan138): Unify this with RLModule API.
        """

        super().__init__(policy, gamma, epsilon_greedy)
        q_model_config = q_model_config or {}
        q_model_config["gamma"] = gamma

        self._model_cls = q_model_config.pop("type", FQETorchModel)
        self._model_configs = q_model_config
        self._normalize_weights = normalize_weights

        self.model = self._model_cls(
            policy=policy,
            **q_model_config,
        )
        assert hasattr(
            self.model, "estimate_v"
        ), "self.model must implement `estimate_v`!"
        assert hasattr(
            self.model, "estimate_q"
        ), "self.model must implement `estimate_q`!"

    @override(OffPolicyEstimator)
    def estimate_on_single_episode(self, episode: SampleBatch) -> Dict[str, Any]:
        estimates_per_epsiode = {}

        rewards, old_prob = episode["rewards"], episode["action_prob"]
        new_prob = self.compute_action_probs(episode)

        weight = new_prob / old_prob

        v_behavior = 0.0
        v_target = 0.0
        q_values = self.model.estimate_q(episode)
        q_values = convert_to_numpy(q_values)
        v_values = self.model.estimate_v(episode)
        v_values = convert_to_numpy(v_values)
        assert q_values.shape == v_values.shape == (episode.count,)

        for t in reversed(range(episode.count)):
            v_behavior = rewards[t] + self.gamma * v_behavior
            v_target = v_values[t] + weight[t] * (
                rewards[t] + self.gamma * v_target - q_values[t]
            )
        v_target = v_target.item()

        estimates_per_epsiode["v_behavior"] = v_behavior
        estimates_per_epsiode["v_target"] = v_target

        return estimates_per_epsiode

    @override(OffPolicyEstimator)
    def estimate_on_single_step_samples(
        self, batch: SampleBatch
    ) -> Dict[str, List[float]]:
        estimates_per_epsiode = {}

        rewards, old_prob = batch["rewards"], batch["action_prob"]
        new_prob = self.compute_action_probs(batch)

        q_values = self.model.estimate_q(batch)
        q_values = convert_to_numpy(q_values)
        v_values = self.model.estimate_v(batch)
        v_values = convert_to_numpy(v_values)

        v_behavior = rewards

        weight = new_prob / old_prob
        v_target = v_values + weight * (rewards - q_values)

        estimates_per_epsiode["v_behavior"] = v_behavior
        estimates_per_epsiode["v_target"] = v_target

        return estimates_per_epsiode

    @override(OffPolicyEstimator)
    def train(self, batch: SampleBatchType) -> Dict[str, Any]:
        """Trains self.model on the given batch.

        Args:
        batch: A SampleBatch or MultiAgentbatch to train on

        Returns:
            A dict with key "loss" and value as the mean training loss.
        """
        batch = convert_ma_batch_to_sample_batch(batch)
        losses = self.model.train(batch)
        return {"loss": np.mean(losses)}

    @override(OfflineEvaluator)
    def estimate_on_dataset(
        self, dataset: Dataset, *, n_parallelism: int = ...
    ) -> Dict[str, Any]:
        """Estimates the policy value using the Doubly Robust estimator.

        The doubly robust estimator uses normalization of importance sampling weights
        (aka. propensity ratios) to the average of the importance weights across the
        entire dataset. This is done to reduce the variance of the estimate (similar to
        weighted importance sampling). You can disable this by setting
        `normalize_weights=False` in the constructor.

        Note: This estimate works for only discrete action spaces for now.

        Args:
            dataset: Dataset to compute the estimate on. Each record in dataset should
                include the following columns: `obs`, `actions`, `action_prob` and
                `rewards`. The `obs` on each row shoud be a vector of D dimensions.
            n_parallelism: Number of parallelism to use for the computation.

        Returns:
            A dict with the following keys:
                v_target: The estimated value of the target policy.
                v_behavior: The estimated value of the behavior policy.
                v_gain: The estimated gain of the target policy over the behavior
                    policy.
                v_std: The standard deviation of the estimated value of the target.
        """

        # step 1: compute the weights and weighted rewards
        batch_size = max(dataset.count() // n_parallelism, 1)
        updated_ds = dataset.map_batches(
            compute_is_weights,
            batch_size=batch_size,
            batch_format="pandas",
            fn_kwargs={
                "policy_state": self.policy.get_state(),
                "estimator_class": self.__class__,
            },
        )

        # step 2: compute q_values and v_values
        batch_size = max(updated_ds.count() // n_parallelism, 1)
        updated_ds = updated_ds.map_batches(
            compute_q_and_v_values,
            batch_size=batch_size,
            batch_format="pandas",
            fn_kwargs={
                "model_class": self.model.__class__,
                "model_state": self.model.get_state(),
            },
        )

        # step 3: compute the v_target
        def compute_v_target(batch: pd.DataFrame, normalizer: float = 1.0):
            weights = batch["weights"] / normalizer
            batch["v_target"] = batch["v_values"] + weights * (
                batch["rewards"] - batch["q_values"]
            )
            batch["v_behavior"] = batch["rewards"]
            return batch

        normalizer = updated_ds.mean("weights") if self._normalize_weights else 1.0
        updated_ds = updated_ds.map_batches(
            compute_v_target,
            batch_size=batch_size,
            batch_format="pandas",
            fn_kwargs={"normalizer": normalizer},
        )

        v_behavior = updated_ds.mean("v_behavior")
        v_target = updated_ds.mean("v_target")
        v_gain_mean = v_target / v_behavior
        v_gain_ste = (
            updated_ds.std("v_target")
            / normalizer
            / v_behavior
            / math.sqrt(dataset.count())
        )

        return {
            "v_behavior": v_behavior,
            "v_target": v_target,
            "v_gain_mean": v_gain_mean,
            "v_gain_ste": v_gain_ste,
        }
