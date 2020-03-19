from abc import ABC, abstractmethod
from multiprocessing import Process, Pipe
from baselines import logger
from utils import tile_images
from episodic_curiosity import oracle
from episodic_curiosity import episodic_memory

class AlreadySteppingError(Exception):
    """
    Raised when an asynchronous step is running while
    step_async() is called again.
    """
    def __init__(self):
        msg = 'already running an async step'
        Exception.__init__(self, msg)

class NotSteppingError(Exception):
    """
    Raised when an asynchronous step is not running but
    step_wait() is called.
    """
    def __init__(self):
        msg = 'not running an async step'
        Exception.__init__(self, msg)

class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        logger.warn('Render not defined for %s'%self)

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

class VecEnvWrapper(VecEnv):
    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        VecEnv.__init__(self,
            num_envs=venv.num_envs,
            observation_space=observation_space or venv.observation_space,
            action_space=action_space or venv.action_space)

    def step_async(self, actions):
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close(self):
        return self.venv.close()

    def render(self):
        self.venv.render()

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

import numpy as np
from gym import spaces




class VecFrameStack(VecEnvWrapper):
    """
    Vectorized environment base class
    """
    def __init__(self, venv, nstack):
        self.venv = venv
        self.nstack = nstack
        wos = venv.observation_space # wrapped ob space
        low = np.repeat(wos.low, self.nstack, axis=-1)
        high = np.repeat(wos.high, self.nstack, axis=-1)
        self.stackedobs = np.zeros((venv.num_envs,)+low.shape, low.dtype)
        observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs

        ram_states = self.venv.clone_full_state()

        monitor_rews = self.venv.get_cur_monitor_rewards()

        ec_rewards = np.zeros_like(rews)

        return self.stackedobs, rews, ec_rewards, news, infos, ram_states, monitor_rews

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs

    def close(self):
        self.venv.close()
    def restore_full_state_by_idx(self, state, env_idx):
        obs, rewards, dones, infos = self.venv.restore_full_state_by_idx(state, env_idx)
        return obs
    
    def set_cur_monitor_rewards_by_idx(self, rews, env_idx):
        self.venv.set_cur_monitor_rewards_by_idx(rews,env_idx)




def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'render':
            remote.send(env.render(mode='rgb_array'))
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd =='clone_full_state':
            state = env.unwrapped.clone_full_state()
            remote.send(state)
        elif cmd=='restore_full_state':
            env.unwrapped.restore_full_state(data)
            ob, reward, done, info = env.step(0)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'get_cur_monitor_rewards':
            rews = env.get_cur_rewards()
            remote.send(rews)
        elif cmd == 'set_cur_monitor_rewards':
            env.set_cur_rewards(data)
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.viewer = None
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def clone_full_state(self):
        for remote in self.remotes:
            remote.send(('clone_full_state', None))
        return np.stack([remote.recv() for remote in self.remotes])
    def restore_full_state_by_idx(self,state, env_idx):
        remote = self.remotes[env_idx]
        remote.send(('restore_full_state', state))
        return remote.recv()

    def get_cur_monitor_rewards(self):
        for remote in self.remotes:
            remote.send(('get_cur_monitor_rewards', None))
        monitor_rewards = [remote.recv() for remote in self.remotes]
        return monitor_rewards

    def set_cur_monitor_rewards_by_idx(self, rews, env_idx):
        remote = self.remotes[env_idx]
        remote.send(('set_cur_monitor_rewards', rews))


    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode='human'):
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        bigimg = tile_images(imgs)
        if mode == 'human':
            #import cv2
            #cv2.imshow('vecenv', bigimg[:,:,::-1])
            #cv2.waitKey(1)
            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()

            #print(imgs)
            self.viewer.imshow(bigimg[:, :, ::-1])
            #print(bigimg[:, :, ::-1].shape)
            return bigimg[:, :, ::-1]
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError


class MovingAverage(object):
  """Computes the moving average of a variable."""

  def __init__(self, capacity):
    self._capacity = capacity
    self._history = np.array([0.0] * capacity)
    self._size = 0

  def add(self, value):
    index = self._size % self._capacity
    self._history[index] = value
    self._size += 1

  def mean(self):
    if not self._size:
      return None
    if self._size < self._capacity:
      return np.mean(self._history[0:self._size])
    return np.mean(self._history)

class CuriosityEnvWrapperFrameStack(VecEnvWrapper):
  """Environment wrapper that adds additional curiosity reward."""

  def __init__(self,
               vec_env,
               vec_episodic_memory,
               observation_embedding_fn,
               exploration_reward = 'episodic_curiosity',
               scale_task_reward = 1.0,
               scale_surrogate_reward = 0.0,
               append_ec_reward_as_channel = False,
               bonus_reward_additive_term = 0,
               exploration_reward_min_step = 0,
               similarity_threshold = 0.5,
               nstack = 4,
               only_train_r = False
               ):
    if exploration_reward == 'episodic_curiosity':
      if len(vec_episodic_memory) != vec_env.num_envs:
        raise ValueError('Each env must have a unique episodic memory.')


    '''
    # Note: post-processing of the observation might change the [0, 255]
    # range of the observation...
    if self._should_postprocess_observation(vec_env.observation_space.shape):
      observation_space_shape = target_image_shape[:]
      if append_ec_reward_as_channel:
        observation_space_shape[-1] += 1
      observation_space = gym.spaces.Box(
          low=0, high=255, shape=observation_space_shape, dtype=np.float)
    else:
      observation_space = vec_env.observation_space
      assert not append_ec_reward_as_channel, (
          'append_ec_reward_as_channel not compatible with non-image-like obs.')
    '''
    self.nstack = nstack
    wos = vec_env.observation_space # wrapped ob space
    low = np.repeat(wos.low, self.nstack, axis=-1)
    high = np.repeat(wos.high, self.nstack, axis=-1)
    self.stackedobs = np.zeros((vec_env.num_envs,)+low.shape, low.dtype)
    observation_space = spaces.Box(low=low, high=high, dtype=vec_env.observation_space.dtype)
    VecEnvWrapper.__init__(self, vec_env, observation_space=observation_space)


    self._bonus_reward_additive_term = bonus_reward_additive_term
    self._vec_episodic_memory = vec_episodic_memory
    self._observation_embedding_fn = observation_embedding_fn
    #self._target_image_shape = target_image_shape
    self._append_ec_reward_as_channel = append_ec_reward_as_channel

    self._exploration_reward = exploration_reward
    self._scale_task_reward = scale_task_reward
    self._scale_surrogate_reward = scale_surrogate_reward
    self._exploration_reward_min_step = exploration_reward_min_step

    '''
    # Oracle reward.
    self._oracles = [oracle.OracleExplorationReward()
                     for _ in range(self.venv.num_envs)]
    '''
    self._oracles = [oracle.OracleExplorationRewardForMR()
                     for _ in range(self.venv.num_envs)]
    # Cumulative task reward over an episode.
    self._episode_task_reward = [0.0] * self.venv.num_envs
    self._episode_bonus_reward = [0.0] * self.venv.num_envs

    
    self._skip_threshold = 1
    self._skip_count = [self._skip_threshold] * self.venv.num_envs

    # Stats on the task and exploration reward.
    self._stats_task_reward = MovingAverage(capacity=100)
    self._stats_bonus_reward = MovingAverage(capacity=100)

    # Total number of steps so far per environment.
    self._step_count = 0

    self._similarity_threshold = similarity_threshold

    self._only_train_r = only_train_r

    # Observers are notified each time a new time step is generated by the
    # environment.
    # Observers implement a function "on_new_observation".
    self._observers = []

  def add_observer(self, observer):
    self._observers.append(observer)


  def _compute_curiosity_reward(self, observations, infos, dones):
    # Computes the surrogate reward.
    # This extra reward is set to 0 when the episode is finished.
    if infos[0].get('frame') is not None:
      frames = np.array([info['frame'] for info in infos])
    else:
      frames = observations
    embedded_observations = self._observation_embedding_fn(frames)
    '''
    similarity_to_memory = [
        episodic_memory.similarity_to_memory(embedded_observations[k],
                                             self._vec_episodic_memory[k])
        for k in range(self.venv.num_envs)
    ]
    '''

    similarity_to_memory = [0.] * self.venv.num_envs

    # Updates the episodic memory of every environment.
    for k in range(self.venv.num_envs):



      if self._skip_count[k] < self._skip_threshold:

        #skip the frame, don't clac episodic rew to speed up training
        similarity_to_memory[k] = 1.0
        self._skip_count[k] = self._skip_count[k] + 1

      else:
        similarity_to_memory[k] = episodic_memory.similarity_to_memory(embedded_observations[k],
                                                self._vec_episodic_memory[k])
        self._skip_count[k] = 0

      # If we've reached the end of the episode, resets the memory
      # and always adds the first state of the new episode to the memory.
      if dones[k]:
        self._skip_count[k] = self._skip_threshold
        self._vec_episodic_memory[k].reset(k==0)
        self._vec_episodic_memory[k].add(embedded_observations[k], infos[k])
        continue

      # Only add the new state to the episodic memory if it is dissimilar
      # enough.
      if similarity_to_memory[k] < self._similarity_threshold:
        self._vec_episodic_memory[k].add(embedded_observations[k], infos[k])
    # Augment the reward with the exploration reward.

    bonus_rewards = [ 1 - s for (s, d) in zip(similarity_to_memory, dones) ]

    #bonus_rewards = [ 1.0 if s <0.5 else 0.0 for (s, d) in zip(similarity_to_memory, dones) ]
    '''
    bonus_rewards = [
        0.0 if d else 0.5 - s + self._bonus_reward_additive_term
        for (s, d) in zip(similarity_to_memory, dones)
    ]
    '''
    bonus_rewards = np.array(bonus_rewards)
    return bonus_rewards

  def _compute_oracle_reward(self, infos, dones):
    bonus_rewards = [
        self._oracles[k].update_position(infos[k]['position'])
        for k in range(self.venv.num_envs)]
    bonus_rewards = np.array(bonus_rewards)

    for k in range(self.venv.num_envs):
      if dones[k]:
        self._oracles[k].reset()

    return bonus_rewards

  def step_wait(self):
    """Overrides VecEnvWrapper.step_wait."""
    #print("here0")
    observations, rewards, dones, infos = self.venv.step_wait()
    #print("here1")
    ram_states = self.venv.clone_full_state()

    monitor_rews = self.venv.get_cur_monitor_rewards()

    #print("here2")
    for observer in self._observers:
      observer.on_new_observation(observations, rewards, dones, infos)

    self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
    for (i, new) in enumerate(dones):
        if new:
            self.stackedobs[i] = 0
    self.stackedobs[..., -observations.shape[-1]:] = observations

    self._step_count += 1

    if (self._step_count % 1000) == 0:
      logger.info('step={} task_reward={} bonus_reward={} scale_bonus={}'.format(
          self._step_count,
          self._stats_task_reward.mean(),
          self._stats_bonus_reward.mean(),
          self._scale_surrogate_reward))

    for i in range(self.venv.num_envs):
      infos[i]['task_reward'] = rewards[i]
      infos[i]['task_observation'] = observations[i]

    # Exploration bonus.

    if self._only_train_r:
        bonus_rewards = np.zeros(self.venv.num_envs)
    else:
        if self._exploration_reward == 'episodic_curiosity':
          bonus_rewards = self._compute_curiosity_reward(observations, infos, dones)
        #elif self._exploration_reward == 'oracle':
        #  bonus_rewards = self._compute_oracle_reward(infos, dones)
        elif self._exploration_reward == 'none':
          bonus_rewards = np.zeros(self.venv.num_envs)
          bonus_rewards = 1.0 + bonus_rewards
        else:
          raise ValueError('Unknown exploration reward: {}'.format(
              self._exploration_reward))

    # Combined rewards.
    scale_surrogate_reward = self._scale_surrogate_reward
    if self._step_count < self._exploration_reward_min_step:
      # This can be used for online training during the first N steps,
      # the R network is totally random and the surrogate reward has no
      # meaning.
      #scale_surrogate_reward = 0.0
      ec_rewards = 1.0
    else:
      ec_rewards = bonus_rewards

    # Update the statistics.
    for i in range(self.venv.num_envs):
      self._episode_task_reward[i] += rewards[i]
      self._episode_bonus_reward[i] += bonus_rewards[i]
      if dones[i]:
        self._stats_task_reward.add(self._episode_task_reward[i])
        self._stats_bonus_reward.add(self._episode_bonus_reward[i])
        self._episode_task_reward[i] = 0.0
        self._episode_bonus_reward[i] = 0.0

    # Post-processing on the observation. Note that the reward could be used
    # as an input to the agent. For simplicity we add it as a separate channel.
    '''
    postprocessed_observations = self._postprocess_observation(observations,
                                                               reward_for_input)
    '''
    return self.stackedobs, rewards, ec_rewards, dones, infos, ram_states, monitor_rews

  def get_episodic_memory(self, k):
    """Returns the episodic memory for the k-th environment."""
    return self._vec_episodic_memory[k]

  def reset(self):
    """Overrides VecEnvWrapper.reset."""
    obs = self.venv.reset()
    self.stackedobs[...] = 0
    self.stackedobs[..., -obs.shape[-1]:] = obs
    # Clears the episodic memory of every environment.
    if self._vec_episodic_memory is not None:
      for i in range(self.venv.num_envs):
        self._vec_episodic_memory[i].reset(i==0)

        self._skip_count[i] = self._skip_threshold
      #for memory in self._vec_episodic_memory:
      #  memory.reset()

    return self.stackedobs

  def restore_full_state_by_idx(self, state, env_idx):
    obs, rewards, dones, infos = self.venv.restore_full_state_by_idx(state, env_idx)
    return obs

  def set_cur_monitor_rewards_by_idx(self, rews, env_idx):
    self.venv.set_cur_monitor_rewards_by_idx(rews,env_idx)

  def close(self):
    self.venv.close()
