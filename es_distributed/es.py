import logging
import time
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .dist import MasterClient, WorkerClient, CoolWorkerClient, CoolMasterClient

logger = logging.getLogger(__name__)

Config = namedtuple('Config', [
    'l2coeff', 'noise_stdev', 'episodes_per_batch', 'timesteps_per_batch',
    'calc_obstat_prob', 'eval_prob', 'snapshot_freq',
    'return_proc_mode', 'episode_cutoff_mode'
])
Task = namedtuple('Task', ['params', 'timestep_limit'])
Result = namedtuple('Result', ['worker_id', 'task_id', 'rew_sum'])


# def task(task_id, params, timestep_limit):
#     return {
#         'task_id': task_id,
#         'params': params,
#         'timestep_limit': timestep_limit
#     }


class RunningStat(object):
    def __init__(self, shape, eps):
        self.sum = np.zeros(shape, dtype=np.float32)
        self.sumsq = np.full(shape, eps, dtype=np.float32)
        self.count = eps

    def increment(self, s, ssq, c):
        self.sum += s
        self.sumsq += ssq
        self.count += c

    @property
    def mean(self):
        return self.sum / self.count

    @property
    def std(self):
        return np.sqrt(np.maximum(self.sumsq / self.count - np.square(self.mean), 1e-2))

    def set_from_init(self, init_mean, init_std, init_count):
        self.sum[:] = init_mean * init_count
        self.sumsq[:] = (np.square(init_mean) + np.square(init_std)) * init_count
        self.count = init_count


class SharedNoiseTable(object):
    def __init__(self):
        import ctypes, multiprocessing
        seed = 123
        count = 250000000  # 1 gigabyte of 32-bit numbers. Will actually sample 2 gigabytes below.
        logger.info('Sampling {} random numbers with seed {}'.format(count, seed))
        self._shared_mem = multiprocessing.Array(ctypes.c_float, count)
        self.noise = np.ctypeslib.as_array(self._shared_mem.get_obj())
        assert self.noise.dtype == np.float32
        self.noise[:] = np.random.RandomState(seed).randn(count)  # 64-bit to 32-bit conversion here
        logger.info('Sampled {} bytes'.format(self.noise.size * 4))

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, stream, dim):
        return stream.randint(0, len(self.noise) - dim + 1)


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def make_session(single_threaded):
    import tensorflow as tf
    if not single_threaded:
        return tf.InteractiveSession()
    return tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1))


def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)


def get_ref_batch(env, batch_size=32):
    ref_batch = []
    ob = env.reset()
    while len(ref_batch) < batch_size:
        ob, rew, done, info = env.step(env.action_space.sample())
        ref_batch.append(ob)
        if done:
            ob = env.reset()
    return ref_batch


def batched_weighted_sum(weights, vecs, batch_size):
    total = 0.
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size), itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(np.asarray(batch_weights, dtype=np.float32), np.asarray(batch_vecs, dtype=np.float32))
        num_items_summed += len(batch_weights)
    return total, num_items_summed


def setup(exp, single_threaded):
    import gym
    gym.undo_logger_setup()
    from . import policies, tf_util

    config = Config(**exp['config'])
    print('-------pidr 0')
    env = gym.make(exp['env_id'])
    print('-------pidr 0.1')
    if exp['policy']['type'] == "ESAtariPolicy":
        from .atari_wrappers import wrap_deepmind
        env = wrap_deepmind(env)
    sess = make_session(single_threaded=single_threaded)
    print('-------pidr 1')
    policy = getattr(policies, exp['policy']['type'])(env.observation_space, env.action_space, **exp['policy']['args'])
    tf_util.initialize()
    return config, env, sess, policy


class HackFloat(float):
    def __init__(self, *args):
        super().__init__()
        self.dtype = tf.float32


master = None
tslimit = None


def differential_evolution_one_step_objective_function(*population):
    global master
    global tslimit
    assert len(population) == master.num_workers

    master.declare_tasks(list(map(lambda x: Task(tf.get_default_session().run(x), tslimit), population)))

    return list(map(lambda x: -x, master.pop_results()))


def run_master(log_dir, exp, num_workers, sockets):
    # exp is parsed JSON
    logger.info('run_master: {}'.format(locals()))
    from . import tabular_logger as tlogger
    logger.info('Tabular logging to {}'.format(log_dir))
    tlogger.start(log_dir)
    config, env, sess, policy = setup(exp, single_threaded=False)
    theta = policy.get_trainable_flat()

    # global ref_batch
    # ref_batch = get_ref_batch(env, batch_size=128)
    # policy.set_ref_batch(ref_batch)

    global tslimit
    tslimit = config.episode_cutoff_mode

    tstart = time.time()

    population = [theta + np.random.sample(theta.size) for _ in range(num_workers)]

    global master
    master = CoolMasterClient(num_workers, sockets)

    generation = 0

    import pydevd_pycharm
    pydevd_pycharm.settrace('localhost', port=5005, stdoutToServer=True, stderrToServer=True)
    print('------master')  # todo

    while True:
        step_tstart = time.time()

        tlogger.log('********** Iteration {} **********'.format(generation))

        master.declare_tasks(list(map(lambda x: Task(x, tslimit), population)))
        print('send tasks')

        print('Waiting result')
        results = master.pop_results()

        crossover_prob = HackFloat(0.9)

        rews_inverted = list(map(lambda x: -x, results))
        step = tfp.optimizer.differential_evolution_one_step(differential_evolution_one_step_objective_function,
                                                             list(map(lambda x: tf.constant(x, dtype=tf.float32),
                                                                      population)),
                                                             population_values=rews_inverted,
                                                             crossover_prob=crossover_prob)

        # noinspection PyTypeChecker
        theta = population[np.argmin(rews_inverted)]
        # updating policy
        policy.set_trainable_flat(theta)

        step_tend = time.time()

        tlogger.record_tabular("Rew", 5)  # todo
        tlogger.record_tabular("TimeElapsedThisIter", step_tend - step_tstart)
        tlogger.record_tabular("TimeElapsed", step_tend - tstart)
        tlogger.dump_tabular()

        if config.snapshot_freq != 0 and generation % config.snapshot_freq == 0:
            import os.path as osp
            filename = 'snapshot_iter{:05d}.h5'.format(generation)
            assert not osp.exists(filename)
            policy.save(filename)
            tlogger.log('Saved snapshot {}'.format(filename))

        generation += 1


def rollout_and_update_ob_stat(policy, env, timestep_limit, rs, task_ob_stat, calc_obstat_prob):
    if policy.needs_ob_stat and calc_obstat_prob != 0 and rs.rand() < calc_obstat_prob:
        rollout_rews, rollout_len, obs, rollout_nov = policy.rollout(
            env, timestep_limit=timestep_limit, save_obs=True, random_stream=rs)
        task_ob_stat.increment(obs.sum(axis=0), np.square(obs).sum(axis=0), len(obs))
    else:
        rollout_rews, rollout_len, rollout_nov = policy.rollout(env, timestep_limit=timestep_limit, random_stream=rs)
    return rollout_rews, rollout_len, rollout_nov


def run_worker(exp, socket):
    logger.info('run_worker: {}'.format(locals()))

    worker = CoolWorkerClient(socket)
    # after creating worker we are subscribed and must notify master about it todo remove comment
    # write_pipe.write("0")

    config, env, sess, policy = setup(exp, single_threaded=True)
    # rs = np.random.RandomState()

    assert policy.needs_ob_stat == (config.calc_obstat_prob != 0)

    while True:
        task_data = worker.get_current_task()
        # policy.set_ref_batch(task_data.ref_batch)

        # Evaluation: noiseless weights and noiseless actions
        policy.set_trainable_flat(task_data.params)
        rew_sum, eval_length, _ = policy.rollout(env, timestep_limit=task_data.timestep_limit)
        worker.push_result(rew_sum)
