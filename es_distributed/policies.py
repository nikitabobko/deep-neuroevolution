import logging
import pickle
import time

import h5py
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from . import tf_util as U

logger = logging.getLogger(__name__)


class Policy:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.scope = self._initialize(*args, **kwargs)
        self.all_variables = tf.get_collection(tf.GraphKeys.VARIABLES, self.scope.name)

        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope.name)
        self.num_params = sum(int(np.prod(v.get_shape().as_list())) for v in self.trainable_variables)
        self._setfromflat = U.SetFromFlat(self.trainable_variables)
        self._getflat = U.GetFlat(self.trainable_variables)

        logger.info('Trainable variables ({} parameters)'.format(self.num_params))
        for v in self.trainable_variables:
            shp = v.get_shape().as_list()
            logger.info('- {} shape:{} size:{}'.format(v.name, shp, np.prod(shp)))
        logger.info('All variables')
        for v in self.all_variables:
            shp = v.get_shape().as_list()
            logger.info('- {} shape:{} size:{}'.format(v.name, shp, np.prod(shp)))

        placeholders = [tf.placeholder(v.value().dtype, v.get_shape().as_list()) for v in self.all_variables]
        self.set_all_vars = U.function(
            inputs=placeholders,
            outputs=[],
            updates=[tf.group(*[v.assign(p) for v, p in zip(self.all_variables, placeholders)])]
        )

    def reinitialize(self):
        for v in self.trainable_variables:
            v.reinitialize.eval()

    def _initialize(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, filename):
        assert filename.endswith('.h5')
        with h5py.File(filename, 'w', libver='latest') as f:
            for v in self.all_variables:
                f[v.name] = v.eval()
            # TODO: it would be nice to avoid pickle, but it's convenient to pass Python objects to _initialize
            # (like Gym spaces or numpy arrays)
            f.attrs['name'] = type(self).__name__
            f.attrs['args_and_kwargs'] = np.void(pickle.dumps((self.args, self.kwargs), protocol=-1))

    @classmethod
    def Load(cls, filename, extra_kwargs=None):
        with h5py.File(filename, 'r') as f:
            args, kwargs = pickle.loads(f.attrs['args_and_kwargs'].tostring())
            if extra_kwargs:
                kwargs.update(extra_kwargs)
            policy = cls(*args, **kwargs)
            policy.set_all_vars(*[f[v.name][...] for v in policy.all_variables])
        return policy

    # === Rollouts/training ===

    def rollout(self, env, *, render=False, timestep_limit=None, save_obs=False, random_stream=None):
        """
        If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
        Otherwise, no action noise will be added.
        """
        raise NotImplementedError

    def act(self, ob, random_stream=None):
        raise NotImplementedError

    def set_trainable_flat(self, x):
        self._setfromflat(x)

    def get_trainable_flat(self):
        return self._getflat()

    @property
    def needs_ob_stat(self):
        raise NotImplementedError

    def set_ob_stat(self, ob_mean, ob_std):
        raise NotImplementedError


def bins(x, dim, num_bins, name):
    scores = U.dense(x, dim * num_bins, name, U.normc_initializer(0.01))
    scores_nab = tf.reshape(scores, [-1, dim, num_bins])
    return tf.argmax(scores_nab, 2)  # 0 ... num_bins-1


class ESAtariPolicy(Policy):
    def _initialize(self, ob_space, ac_space):
        self.ob_space_shape = ob_space.shape
        self.ac_space = ac_space
        self.num_actions = ac_space.n

        with tf.variable_scope(type(self).__name__) as scope:
            o = tf.placeholder(tf.float32, [None] + list(self.ob_space_shape))
            is_ref_ph = tf.placeholder(tf.bool, shape=[])

            a = self._make_net(o, is_ref_ph)
            self._act = U.function([o, is_ref_ph], a)
        return scope

    def _make_net(self, o, is_ref):
        x = o
        x = layers.convolution2d(x, num_outputs=16, kernel_size=8, stride=4, activation_fn=None, scope='conv1')
        x = layers.batch_norm(x, scale=True, is_training=is_ref, decay=0., updates_collections=None,
                              activation_fn=tf.nn.relu, epsilon=1e-3)
        x = layers.convolution2d(x, num_outputs=32, kernel_size=4, stride=2, activation_fn=None, scope='conv2')
        x = layers.batch_norm(x, scale=True, is_training=is_ref, decay=0., updates_collections=None,
                              activation_fn=tf.nn.relu, epsilon=1e-3)

        x = layers.flatten(x)
        x = layers.fully_connected(x, num_outputs=256, activation_fn=None, scope='fc')
        x = layers.batch_norm(x, scale=True, is_training=is_ref, decay=0., updates_collections=None,
                              activation_fn=tf.nn.relu, epsilon=1e-3)
        a = layers.fully_connected(x, num_outputs=self.num_actions, activation_fn=None, scope='out')
        return tf.argmax(a, 1)

    def set_ref_batch(self, ref_batch):
        self.ref_list = []
        self.ref_list.append(ref_batch)
        self.ref_list.append(True)

    @property
    def needs_ob_stat(self):
        return False

    @property
    def needs_ref_batch(self):
        return True

    def initialize_from(self, filename):
        """
        Initializes weights from another policy, which must have the same architecture (variable names),
        but the weight arrays can be smaller than the current policy.
        """
        with h5py.File(filename, 'r') as f:
            f_var_names = []
            f.visititems(lambda name, obj: f_var_names.append(name) if isinstance(obj, h5py.Dataset) else None)
            assert set(v.name for v in self.all_variables) == set(f_var_names), 'Variable names do not match'

            init_vals = []
            for v in self.all_variables:
                shp = v.get_shape().as_list()
                f_shp = f[v.name].shape
                assert len(shp) == len(f_shp) and all(a >= b for a, b in zip(shp, f_shp)), \
                    'This policy must have more weights than the policy to load'
                init_val = v.eval()
                # ob_mean and ob_std are initialized with nan, so set them manually
                if 'ob_mean' in v.name:
                    init_val[:] = 0
                    init_mean = init_val
                elif 'ob_std' in v.name:
                    init_val[:] = 0.001
                    init_std = init_val
                # Fill in subarray from the loaded policy
                init_val[tuple([np.s_[:s] for s in f_shp])] = f[v.name]
                init_vals.append(init_val)
            self.set_all_vars(*init_vals)

    def act(self, train_vars, random_stream=None):
        return self._act(*train_vars)

    def rollout(self, env, *, render=False, timestep_limit=None, save_obs=False, random_stream=None, worker_stats=None,
                policy_seed=None):
        """
        If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
        Otherwise, no action noise will be added.
        """
        env_timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

        timestep_limit = env_timestep_limit if timestep_limit is None else min(timestep_limit, env_timestep_limit)
        novelty_vector = []
        t = 0

        if save_obs:
            obs = []

        if policy_seed:
            env.seed(policy_seed)
            np.random.seed(policy_seed)
            if random_stream:
                random_stream.seed(policy_seed)

        ob = env.reset()
        # self.act(self.ref_list, random_stream=random_stream)  # passing ref batch through network

        global_rew = 0

        for _ in range(timestep_limit):
            start_time = time.time()
            ac = self.act([ob[None], False], random_stream=random_stream)[0]

            if worker_stats:
                worker_stats.time_comp_act += time.time() - start_time

            start_time = time.time()
            ob, rew, done, info = env.step(ac)
            ram = env.unwrapped._get_ram()  # extracts RAM state information

            if save_obs:
                obs.append(ob)
            if worker_stats:
                worker_stats.time_comp_step += time.time() - start_time

            global_rew += rew
            novelty_vector.append(ram)

            t += 1
            if render:
                env.render()
            if done:
                break

        if save_obs:
            return global_rew, t, np.array(obs), np.array(novelty_vector)
        return global_rew, t, np.array(novelty_vector)
