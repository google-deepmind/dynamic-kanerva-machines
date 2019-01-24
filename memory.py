"""Memory module for Kanerva Machines.

  Functions of the module always take inputs with shape:
  [seq_length, batch_size, ...]

  Examples:

    # Initialisation
    memory = KanervaMemory(code_size=100, memory_size=32)
    prior_memory = memory.get_prior_state(batch_size)

    # Update memory posterior
    posterior_memory, _, _, _ = memory.update_state(z_episode, prior_memory)

    # Read from the memory using cues z_q
    read_z, dkl_w = memory.read_with_z(z_q, posterior_memory)

    # Compute the KL-divergence between posterior and prior memory
    dkl_M = memory.get_dkl_total(posterior_memory)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

MemoryState = collections.namedtuple(
    'MemoryState',
    # Mean of memory slots, [batch_size, memory_size, word_size]
    # Covariance of memory slots, [batch_size, memory_size, memory_size]
    ('M_mean', 'M_cov'))

EPSILON = 1e-6


# disable lint warnings for cleaner algebraic expressions
# pylint: disable=invalid-name
class KanervaMemory(snt.AbstractModule):
  """A memory-based generative model."""

  def __init__(self,
               code_size,
               memory_size,
               num_opt_iters=1,
               w_prior_stddev=1.0,
               obs_noise_stddev=1.0,
               sample_w=False,
               sample_M=False,
               name='KanervaMemory'):
    """Initialise the memory module.

    Args:
      code_size: Integer specifying the size of each encoded input.
      memory_size: Integer specifying the total number of rows in the memory.
      num_opt_iters: Integer specifying the number of optimisation iterations.
      w_prior_stddev: Float specifying the standard deviation of w's prior.
      obs_noise_stddev: Float specifying the standard deviation of the
        observational noise.
      sample_w: Boolean specifying whether to sample w or simply take its mean.
      sample_M: Boolean specifying whether to sample M or simply take its mean.
      name: String specfying the name of this module.
    """
    super(KanervaMemory, self).__init__(name=name)
    self._memory_size = memory_size
    self._code_size = code_size
    self._num_opt_iters = num_opt_iters
    self._sample_w = sample_w
    self._sample_M = sample_M
    self._w_prior_stddev = tf.constant(w_prior_stddev)

    with self._enter_variable_scope():
      log_w_stddev = snt.TrainableVariable(
          [], name='w_stddev',
          initializers={'w': tf.constant_initializer(np.log(0.3))})()
      if obs_noise_stddev > 0.0:
        self._obs_noise_stddev = tf.constant(obs_noise_stddev)
      else:
        log_obs_stddev = snt.TrainableVariable(
            [], name='obs_stdddev',
            initializers={'w': tf.constant_initializer(np.log(1.0))})()
        self._obs_noise_stddev = tf.exp(log_obs_stddev)
    self._w_stddev = tf.exp(log_w_stddev)
    self._w_prior_dist = tfp.distributions.MultivariateNormalDiag(
        loc=tf.zeros([self._memory_size]),
        scale_identity_multiplier=self._w_prior_stddev)

  def _build(self):
    raise ValueError('`_build()` should not be called for this module since'
                     'it takes no inputs and all of its variables are'
                     'constructed in `__init__`')

  def _get_w_dist(self, mu_w):
    return tfp.distributions.MultivariateNormalDiag(
        loc=mu_w, scale_identity_multiplier=self._w_stddev)

  def sample_prior_w(self, seq_length, batch_size):
    """Sample w from its prior.

    Args:
      seq_length: length of sequence
      batch_size: batch size of samples
    Returns:
      w: [batch_size, memory_size]
    """
    return self._w_prior_dist.sample([seq_length, batch_size])

  def read_with_z(self, z, memory_state):
    """Query from memory (specified by memory_state) using embedding z.

    Args:
      z: Tensor with dimensions [episode_length, batch_size, code_size]
        containing an embedded input.
      memory_state: Instance of `MemoryState`.

    Returns:
      A tuple of tensors containing the mean of read embedding and the
        KL-divergence between the w used in reading and its prior.
    """
    M = self.sample_M(memory_state)
    w_mean = self._solve_w_mean(z, M)
    w_samples = self.sample_w(w_mean)
    dkl_w = self.get_dkl_w(w_mean)
    z_mean = self.get_w_to_z_mean(w_samples, M)
    return z_mean, dkl_w

  def wrap_z_dist(self, z_mean):
    """Wrap the mean of z as an observation (Gaussian) distribution."""
    return tfp.distributions.MultivariateNormalDiag(
        loc=z_mean, scale_identity_multiplier=self._obs_noise_stddev)

  def sample_w(self, w_mean):
    """Sample w from its posterior distribution."""
    if self._sample_w:
      return self._get_w_dist(w_mean).sample()
    else:
      return w_mean

  def sample_M(self, memory_state):
    """Sample the memory from its distribution specified by memory_state."""
    if self._sample_M:
      noise_dist = tfp.distributions.MultivariateNormalFullCovariance(
          covariance_matrix=memory_state.M_cov)
      # C, B, M
      noise = tf.transpose(noise_dist.sample(self._code_size),
                           [1, 2, 0])
      return memory_state.M_mean + noise
    else:
      return memory_state.M_mean

  def get_w_to_z_mean(self, w_p, R):
    """Return the mean of z by reading from memory using weights w_p."""
    return tf.einsum('sbm,bmc->sbc', w_p, R)  # Rw

  def _read_cov(self, w_samples, memory_state):
    episode_size, batch_size = w_samples.get_shape().as_list()[:2]
    _, U = memory_state  # cov: [B, M, M]
    wU = tf.einsum('sbm,bmn->sbn', w_samples, U)
    wUw = tf.einsum('sbm,sbm->sb', wU, w_samples)
    wUw.get_shape().assert_is_compatible_with([episode_size, batch_size])
    return wU, wUw

  def get_dkl_total(self, memory_state):
    """Compute the KL-divergence between a memory distribution and its prior."""
    R, U = memory_state
    B, K, _ = R.get_shape().as_list()
    U.get_shape().assert_is_compatible_with([B, K, K])
    R_prior, U_prior = self.get_prior_state(B)
    p_diag = tf.matrix_diag_part(U_prior)
    q_diag = tf.matrix_diag_part(U)  # B, K
    t1 = self._code_size * tf.reduce_sum(q_diag / p_diag, -1)
    t2 = tf.reduce_sum((R - R_prior)**2 / tf.expand_dims(
        p_diag, -1), [-2, -1])
    t3 = -self._code_size * self._memory_size
    t4 = self._code_size * tf.reduce_sum(tf.log(p_diag) - tf.log(q_diag), -1)
    return t1 + t2 + t3 + t4

  def _get_dkl_update(self, memory_state, w_samples, new_z_mean, new_z_var):
    """Compute memory_kl after updating prior_state."""
    B, K, C = memory_state.M_mean.get_shape().as_list()
    S = w_samples.get_shape().as_list()[0]

    # check shapes
    w_samples.get_shape().assert_is_compatible_with([S, B, K])
    new_z_mean.get_shape().assert_is_compatible_with([S, B, C])

    delta = new_z_mean - self.get_w_to_z_mean(w_samples, memory_state.M_mean)
    _, wUw = self._read_cov(w_samples, memory_state)
    var_z = wUw + new_z_var + self._obs_noise_stddev**2
    beta = wUw / var_z

    dkl_M = -0.5 * (self._code_size * beta
                    - tf.reduce_sum(tf.expand_dims(beta / var_z, -1)
                                    * delta**2, -1)
                    + self._code_size * tf.log(1 - beta))
    dkl_M.get_shape().assert_is_compatible_with([S, B])
    return dkl_M

  @snt.reuse_variables
  def _get_prior_params(self):
    log_var = snt.TrainableVariable(
        [], name='prior_var_scale',
        initializers={'w': tf.constant_initializer(
            np.log(1.0))})()
    self._prior_var = tf.ones([self._memory_size]) * tf.exp(log_var) + EPSILON
    prior_cov = tf.matrix_diag(self._prior_var)
    prior_mean = snt.TrainableVariable(
        [self._memory_size, self._code_size],
        name='prior_mean',
        initializers={'w': tf.truncated_normal_initializer(
            mean=0.0, stddev=1.0)})()
    return prior_mean, prior_cov

  @property
  def prior_avg_var(self):
    """return the average of prior memory variance."""
    return tf.reduce_mean(self._prior_var)

  def _solve_w_mean(self, new_z_mean, M):
    """Minimise the conditional KL-divergence between z wrt w."""
    w_matrix = tf.matmul(M, M, transpose_b=True)
    w_rhs = tf.einsum('bmc,sbc->bms', M, new_z_mean)
    w_mean = tf.matrix_solve_ls(
        matrix=w_matrix, rhs=w_rhs,
        l2_regularizer=self._obs_noise_stddev**2 / self._w_prior_stddev**2)
    w_mean = tf.einsum('bms->sbm', w_mean)
    return w_mean

  def get_prior_state(self, batch_size):
    """Return the prior distribution of memory as a MemoryState."""
    prior_mean, prior_cov = self._get_prior_params()
    batch_prior_mean = tf.stack([prior_mean] * batch_size)
    batch_prior_cov = tf.stack([prior_cov] * batch_size)
    return MemoryState(M_mean=batch_prior_mean,
                       M_cov=batch_prior_cov)

  def update_state(self, z, memory_state):
    """Update the memory state using Bayes' rule.

    Args:
      z: A tensor with dimensions [episode_length, batch_size, code_size]
        containing a sequence of embeddings to write into memory.
      memory_state: A `MemoryState` namedtuple containing the memory state to
        be written to.

    Returns:
      A tuple containing the following elements:
      final_memory: A `MemoryState` namedtuple containing the new memory state
        after the update.
      w_mean_episode: The mean of w for the written episode.
      dkl_w_episode: The KL-divergence of w for the written episode.
      dkl_M_episode: The KL-divergence between the memory states before and
        after the update.
    """

    episode_size, batch_size = z.get_shape().as_list()[:2]
    w_array = tf.TensorArray(dtype=tf.float32, size=episode_size,
                             element_shape=[1, batch_size, self._memory_size])
    dkl_w_array = tf.TensorArray(dtype=tf.float32, size=episode_size,
                                 element_shape=[1, batch_size])
    dkl_M_array = tf.TensorArray(dtype=tf.float32, size=episode_size,
                                 element_shape=[1, batch_size])
    init_var = (0, memory_state, w_array, dkl_w_array, dkl_M_array)
    cond = lambda i, m, d_2, d_3, d_4: i < episode_size
    def loop_body(i, old_memory, w_array, dkl_w_array, dkl_M_array):
      """Update memory step-by-step."""
      z_step = tf.expand_dims(z[i], 0)
      new_memory = old_memory
      for _ in xrange(self._num_opt_iters):
        w_step_mean = self._solve_w_mean(z_step, self.sample_M(new_memory))
        w_step_sample = self.sample_w(w_step_mean)
        new_memory = self._update_memory(old_memory,
                                         w_step_mean,
                                         z_step, 0)
      dkl_w_step = self.get_dkl_w(w_step_mean)
      dkl_M_step = self._get_dkl_update(old_memory,
                                        w_step_sample,
                                        z_step, 0)
      return (i+1,
              new_memory,
              w_array.write(i, w_step_sample),
              dkl_w_array.write(i, dkl_w_step),
              dkl_M_array.write(i, dkl_M_step))

    _, final_memory, w_mean, dkl_w, dkl_M = tf.while_loop(
        cond, loop_body, init_var)
    w_mean_episode = w_mean.concat()
    dkl_w_episode = dkl_w.concat()
    dkl_M_episode = dkl_M.concat()
    dkl_M_episode.get_shape().assert_is_compatible_with(
        [episode_size, batch_size])

    return final_memory, w_mean_episode, dkl_w_episode, dkl_M_episode

  def _update_memory(self, old_memory, w_samples, new_z_mean, new_z_var):
    """Setting new_z_var=0 for sample based update."""
    old_mean, old_cov = old_memory
    wR = self.get_w_to_z_mean(w_samples, old_memory.M_mean)
    wU, wUw = self._read_cov(w_samples, old_memory)
    sigma_z = wUw + new_z_var + self._obs_noise_stddev**2  # [S, B]
    delta = new_z_mean - wR  # [S, B, C]
    c_z = wU / tf.expand_dims(sigma_z, -1)  # [S, B, M]
    posterior_mean = old_mean + tf.einsum('sbm,sbc->bmc', c_z, delta)
    posterior_cov = old_cov - tf.einsum('sbm,sbn->bmn', c_z, wU)
    # Clip diagonal elements for numerical stability
    posterior_cov = tf.matrix_set_diag(
        posterior_cov,
        tf.clip_by_value(tf.matrix_diag_part(posterior_cov), EPSILON, 1e10))
    new_memory = MemoryState(M_mean=posterior_mean, M_cov=posterior_cov)
    return new_memory

  def get_dkl_w(self, w_mean):
    """Return the KL-divergence between posterior and prior weights w."""
    posterior_dist = self._get_w_dist(w_mean)
    dkl_w = posterior_dist.kl_divergence(self._w_prior_dist)
    dkl_w.get_shape().assert_is_compatible_with(
        w_mean.get_shape().as_list()[:-1])
    return dkl_w
