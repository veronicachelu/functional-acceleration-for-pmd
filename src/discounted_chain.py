import numpy as np
import importlib
from src.utils import sample, random_dist
from scipy import linalg
from itertools import product
from src.mrp import MRP
from src import runner
import src.mdp
src.mdp = importlib.reload(src.mdp)
from src.mdp import MDP

runner = importlib.reload(runner)

class DiscountingChain(MDP):
    """Discounting Chain environment."""

    def __init__(self, mapping_seed: Optional[int] = None):
        """Builds the Discounting Chain environment.

        Args:
          mapping_seed: Optional integer, specifies which reward is bonus.
        """
        super().__init__()
        self._episode_len = 100
        self._reward_timestep = [1, 3, 10, 30, 100]
        self._n_actions = len(self._reward_timestep)
        if mapping_seed is None:
            mapping_seed = np.random.randint(0, self._n_actions)
        else:
            mapping_seed = mapping_seed % self._n_actions

        self._rewards = np.ones(self._n_actions)
        self._rewards[mapping_seed] += 0.1

        self._timestep = 0
        self._context = -1

        self.bsuite_num_episodes = sweep.NUM_EPISODES

    def _get_observation(self):
        obs = np.zeros(shape=(1, 2), dtype=np.float32)
        obs[0, 0] = self._context
        obs[0, 1] = self._timestep / self._episode_len
        return obs

    def _reset(self) -> dm_env.TimeStep:
        self._timestep = 0
        self._context = -1
        observation = self._get_observation()
        return dm_env.restart(observation)

    def _step(self, action: int) -> dm_env.TimeStep:
        if self._timestep == 0:
            self._context = action

        self._timestep += 1
        if self._timestep == self._reward_timestep[self._context]:
            reward = self._rewards[self._context]
        else:
            reward = 0.

        observation = self._get_observation()
        if self._timestep == self._episode_len:
            return dm_env.termination(reward=reward, observation=observation)
        return dm_env.transition(reward=reward, observation=observation)


def random_discounted_chain(seed, n_states, n_actions, gamma)