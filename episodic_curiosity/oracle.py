# coding=utf-8
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Computes some oracle reward based on the actual agent position."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import pickle

class OracleExplorationReward(object):
  """Class that computes the ideal exploration bonus."""

  def __init__(self, reward_grid_size=30.0, cell_reward_normalizer=900.0):
    """Creates a new oracle to compute the exploration reward.

    Args:
      reward_grid_size: Size of a cell that contains a unique reward.
      cell_reward_normalizer: Denominator for computation of a cell reward
    """
    self._reward_grid_size = reward_grid_size

    # Make the total sum of exploration reward that can be collected
    # independent of the grid size.
    # Here, we assume that the position is laying on a 2D manifold,
    # hence the multiplication by the area of a 2D cell.
    self._cell_reward = float(reward_grid_size * reward_grid_size)

    # Somewhat normalize the exploration reward so that it is neither
    # too big or too small.
    self._cell_reward /= cell_reward_normalizer

    self.reset()

  def reset(self):
    self._collected_positions = set()

  def update_position(self, agent_position):
    """Set the new state (i.e. the position).

    Args:
      agent_position: x,y,z position of the agent.

    Returns:
      The exploration bonus for having visited this position.
    """
    x, y, z = agent_position
    quantized_x = int(x / self._reward_grid_size)
    quantized_y = int(y / self._reward_grid_size)
    quantized_z = int(z / self._reward_grid_size)
    position_id = (quantized_x, quantized_y, quantized_z)
    if position_id in self._collected_positions:
      # No reward if the position has already been explored.
      return 0.0
    else:
      self._collected_positions.add(position_id)
      return self._cell_reward

class OracleExplorationRewardForMR(object):
  """Class that computes the ideal exploration bonus."""

  def __init__(self, reward_grid_size=2, cell_reward_normalizer=4.):
    """Creates a new oracle to compute the exploration reward.

    Args:
      reward_grid_size: Size of a cell that contains a unique reward.
      cell_reward_normalizer: Denominator for computation of a cell reward
    """
    self._reward_grid_size = reward_grid_size


    '''
    # Make the total sum of exploration reward that can be collected
    # independent of the grid size.
    # Here, we assume that the position is laying on a 2D manifold,
    # hence the multiplication by the area of a 2D cell.
    self._cell_reward = float(reward_grid_size * reward_grid_size)

    # Somewhat normalize the exploration reward so that it is neither
    # too big or too small.
    self._cell_reward /= cell_reward_normalizer
    '''
    self._cell_reward = 1.0
    self.reset()

  def reset(self):
    self._collected_positions = {}

  def update_position(self, agent_position):
    """Set the new state (i.e. the position).

    Args:
      agent_position: x,y,z position of the agent.

    Returns:
      The exploration bonus for having visited this position.
    """
    x, y, room_id, nkeys, room_level = agent_position
    #print(agent_position)
    quantized_x = int(x / self._reward_grid_size)
    quantized_y = int(y / self._reward_grid_size)

    position_id = (quantized_x, quantized_y, room_id, nkeys, room_level)

    count = self._collected_positions.get(position_id, 0)

    self._collected_positions[position_id] = count +1

    rewards = 1. / (np.sqrt(count+1))

    return rewards

class OracleExplorationRewardForAllEpisodes(object):
  """Class that computes the ideal exploration bonus."""

  def __init__(self,coef=0.5 , reward_grid_size=1, cell_reward_normalizer=4.):
    """Creates a new oracle to compute the exploration reward.

    Args:
      reward_grid_size: Size of a cell that contains a unique reward.
      cell_reward_normalizer: Denominator for computation of a cell reward
    """
    self._reward_grid_size = reward_grid_size

    self._coef = coef

    self._exclude_room = []

    self.reset()

  def copy(self,others):
    self._reward_grid_size = others._reward_grid_size
    self._coef = others._coef
    self._collected_positions_reader = others._collected_positions_reader.copy()
    self._collected_positions_writer = others._collected_positions_writer.copy()


  def reset(self):
    self._collected_positions_reader = {}
    self._collected_positions_writer = {}


  def _get_key(self,agent_position):
    x, y, room_id, nkeys, room_level = agent_position
    #print(agent_position)
    quantized_x = int(x / self._reward_grid_size)
    quantized_y = int(y / self._reward_grid_size)

    position_id = (quantized_x, quantized_y, room_id, room_level)

    return position_id

  def get_count(self,agent_position, is_reader=True):

    position_id = self._get_key(agent_position)

    if is_reader:
      count = self._collected_positions_reader.get(position_id, 0)
    else:
      count = self._collected_positions_writer.get(position_id, 0)

    return count

  def get_reward(self, agent_position):

    """Set the new state (i.e. the position).

    Args:
      agent_position: x,y,z position of the agent.

    Returns:
      The exploration bonus for having visited this position.
    """

    count = self.get_count(agent_position, is_reader=True)


    rewards = self._coef * 1. / (np.sqrt(count+1))

    x, y, room_id, nkeys, room_level = agent_position
    if room_id in self._exclude_room:
      rewards = 0.


    return rewards

  def reset_count_by_position(self,agent_position):


    position_id = self._get_key(agent_position)

    self._collected_positions_writer[position_id] = 0

  def update_position(self, agent_position):
    """Set the new state (i.e. the position).

    Args:
      agent_position: x,y,z position of the agent.

    Returns:
      The exploration bonus for having visited this position.
    """

    position_id = self._get_key(agent_position)

    #get reward
    rewards = self.get_reward(agent_position)

    #update position
    count = self._collected_positions_writer.get(position_id, 0)
    self._collected_positions_writer[position_id] = count +1

    return rewards

  def sync(self):
    self._collected_positions_reader = self._collected_positions_writer.copy()

  def save(self,path):
    f = open(path,'wb')
    pickle.dump(self,f)
    f.close()

  def load(self,path):
    f = open(path,'rb')

    t = pickle.load(f)

    f.close()
    self.copy(t)

class OracleRewardForAllEpisodes(object):
  """Class that computes the ideal exploration bonus."""

  def __init__(self,coef=1 , reward_grid_size=1, cell_reward_normalizer=4.):
    """Creates a new oracle to compute the exploration reward.

    Args:
      reward_grid_size: Size of a cell that contains a unique reward.
      cell_reward_normalizer: Denominator for computation of a cell reward
    """
    self._reward_grid_size = reward_grid_size

    self._coef = coef

    self.reset()

  def copy(self,others):
    self._reward_grid_size = others._reward_grid_size
    self._coef = others._coef
    self._collected_positions_reader = others._collected_positions_reader.copy()
    self._collected_positions_writer = others._collected_positions_writer.copy()


  def reset(self):
    self._collected_positions_reader = {}
    self._collected_positions_writer = {}



  def get_count(self,key, is_reader=True):


    if is_reader:
      count = self._collected_positions_reader.get(key, 0)
    else:
      count = self._collected_positions_writer.get(key, 0)

    return count

  def get_reward(self, key):

    """Set the new state (i.e. the position).

    Args:
      key

    Returns:
      The exploration bonus for having visited this position.
    """

    count = self.get_count(key, is_reader=True)


    rewards = self._coef * 1. / (np.sqrt(count+1))

    return rewards


  def update_key(self, key):
    """Set the new state (i.e. the position).

    Args:
      key

    Returns:
      The exploration bonus for having visited this position.
    """


    #get reward
    rewards = self.get_reward(key)

    #update position
    count = self._collected_positions_writer.get(key, 0)
    self._collected_positions_writer[key] = count +1

    return rewards

  def sync(self):
    self._collected_positions_reader = self._collected_positions_writer.copy()

  def save(self,path):
    f = open(path,'wb')
    pickle.dump(self,f)
    f.close()

  def load(self,path):
    f = open(path,'rb')

    t = pickle.load(f)

    f.close()
    self.copy(t)