import logging
import os
import pickle
import time
import json
import numpy as np
from collections import deque
from pprint import pformat

import redis

logger = logging.getLogger(__name__)

EXP_KEY = 'es:exp'
TASK_CHANNEL = 'es:task_channel'
RESULTS_KEY = 'es:results'
ARCHIVE_KEY = 'es:archive'


###########################################


def get_task_id_key(worker_id):
    return 'es:task_id' + str(worker_id)


def get_task_data_key(worker_id):
    return 'es:task_data' + str(worker_id)


def get_result_key(worker_id):
    return 'es:results' + str(worker_id)


###########################################

def serialize(x):
    # return json.dumps(x)
    return pickle.dumps(x, protocol=-1)


def deserialize(x):
    # ret = json.loads(x)
    # ret['params'] = np.array(ret['params'])
    # return ret
    return pickle.loads(x)


def retry_connect(redis_cfg, tries=300, base_delay=4.):
    for i in range(tries):
        try:
            r = redis.StrictRedis(**redis_cfg)
            r.ping()
            return r
        except redis.ConnectionError as e:
            if i == tries - 1:
                raise
            else:
                delay = base_delay * (1 + (os.getpid() % 10) / 9)
                logger.warning('Could not connect to {}. Retrying after {:.2f} sec ({}/{}). Error: {}'.format(
                    redis_cfg, delay, i + 2, tries, e))
                time.sleep(delay)


def retry_get(pipe, key, tries=300, base_delay=4.):
    for i in range(tries):
        # Try to (m)get
        if isinstance(key, (list, tuple)):
            vals = pipe.mget(key)
            if all(v is not None for v in vals):
                return vals
        else:
            val = pipe.get(key)
            if val is not None:
                return val
        # Sleep and retry if any key wasn't available
        if i != tries - 1:
            delay = base_delay * (1 + (os.getpid() % 10) / 9)
            logger.warning('{} not set. Retrying after {:.2f} sec ({}/{})'.format(key, delay, i + 2, tries))
            time.sleep(delay)
    raise RuntimeError('{} not set'.format(key))


class MasterClient:
    def __init__(self, master_redis_cfg):
        self.task_counter = 0
        self.master_redis = retry_connect(master_redis_cfg)
        logger.info('[master] Connected to Redis: {}'.format(self.master_redis))

    def declare_json(self, exp):
        self.master_redis.set(EXP_KEY, serialize(exp))
        self.master_redis.flushdb()
        print("------flush json")
        logger.info('[master] Declared experiment {}'.format(pformat(exp)))

    def declare_tasks(self, tasks_data):
        task_id = self.task_counter
        self.task_counter += 1

        worker_id = 0
        for task_data in tasks_data:
            serialized_task_data = serialize(task_data)
            self.master_redis.set(get_task_id_key(worker_id), task_id)
            self.master_redis.set(get_task_data_key(worker_id), serialized_task_data)

            # (self.master_redis.pipeline()
            #  .mset({get_task_id_key(worker_id): task_id, get_task_data_key(worker_id): serialized_task_data})
            #  .publish(TASK_CHANNEL, serialize((task_id, serialized_task_data)))
            #  .execute())  # TODO: can we avoid transferring task data twice and serializing so much?
            logger.debug('[master] Declared task {}'.format(task_id))

            worker_id += 1
        self.master_redis.flushdb()
        return task_id

    def pop_result(self, num_workers, expected_task_id):
        results = []
        for worker_id in range(num_workers):
            while True:
                task_id, result = deserialize(self.master_redis.blpop(RESULTS_KEY)[1])
                if task_id == expected_task_id:
                    break
            results.append(result)
            logger.debug('[master] Popped a result for task {}'.format(task_id))
        return results

    def flush_results(self):
        return max(self.master_redis.pipeline().llen(RESULTS_KEY).ltrim(RESULTS_KEY, -1, -1).execute()[0] - 1, 0)

    def add_to_novelty_archive(self, novelty_vector):
        self.master_redis.rpush(ARCHIVE_KEY, serialize(novelty_vector))
        logger.info('[master] Added novelty vector to archive')

    def get_archive(self):
        archive = self.master_redis.lrange(ARCHIVE_KEY, 0, -1)
        return [deserialize(novelty_vector) for novelty_vector in archive]


# class RelayClient:
#     """
#     Receives and stores task broadcasts from the master
#     Batches and pushes results from workers to the master
#     """
#
#     def __init__(self, master_redis_cfg, relay_redis_cfg):
#         self.master_redis = retry_connect(master_redis_cfg)
#         logger.info('[relay] Connected to master: {}'.format(self.master_redis))
#         self.local_redis = retry_connect(relay_redis_cfg)
#         logger.info('[relay] Connected to relay: {}'.format(self.local_redis))
#         self.results_published = 0
#
#     def run(self, num_workers):
#         # Initialization: read exp and latest task from master
#         self.local_redis.set(EXP_KEY, retry_get(self.master_redis, EXP_KEY))
#         for worker_id in range(num_workers):
#             self._declare_task_local(num_workers, *retry_get(self.master_redis, (get_task_id_key(worker_id), get_task_data_key(worker_id))))
#
#         # Start subscribing to tasks
#         p = self.master_redis.pubsub(ignore_subscribe_messages=True)
#         p.subscribe(**{TASK_CHANNEL: lambda msg: self._declare_task_local(*deserialize(msg['data']))})
#         p.run_in_thread(sleep_time=0.001)
#
#         # Loop on RESULTS_KEY and push to master
#         batch_sizes, last_print_time = deque(maxlen=20), time.time()  # for logging
#         while True:
#             results = []
#             start_time = curr_time = time.time()
#             while curr_time - start_time < 0.001:
#                 results.append(self.local_redis.blpop(RESULTS_KEY)[1])
#                 curr_time = time.time()
#             self.results_published += len(results)
#             self.master_redis.rpush(RESULTS_KEY, *results)
#             # Log
#             batch_sizes.append(len(results))
#             if curr_time - last_print_time > 5.0:
#                 logger.info('[relay] Average batch size {:.3f} ({} total)'.format(sum(batch_sizes) / len(batch_sizes), self.results_published))
#                 last_print_time = curr_time
#
#     def flush_results(self):
#         number_flushed = max(self.local_redis.pipeline().llen(RESULTS_KEY).ltrim(RESULTS_KEY, -1, -1).execute()[0] -1, 0)
#         number_flushed_master = max(self.master_redis.pipeline().llen(RESULTS_KEY).ltrim(RESULTS_KEY, -1, -1).execute()[0] -1, 0)
#         logger.warning('[relay] Flushed {} results from worker redis and {} from master'
#             .format(number_flushed, number_flushed_master))
#
#     def _declare_task_local(self, worker_id, task_id, task_data):
#         logger.info('[relay] Received task {}'.format(task_id))
#         self.results_published = 0
#         self.local_redis.mset({get_task_id_key(worker_id): task_id, get_task_data_key(worker_id): task_data})
#         self.flush_results()


class WorkerClient:
    def __init__(self, relay_redis_cfg, master_redis_cfg):
        self.local_redis = retry_connect(relay_redis_cfg)
        logger.info('[worker] Connected to relay: {}'.format(self.local_redis))
        self.master_redis = retry_connect(master_redis_cfg)
        logger.warning('[worker] Connected to master: {}'.format(self.master_redis))

        self.cached_task_id, self.cached_task_data = None, None

    def get_experiment(self):
        # Grab experiment info
        exp = deserialize(retry_get(self.local_redis, EXP_KEY))
        logger.info('[worker] Experiment: {}'.format(exp))
        return exp

    def get_archive(self):
        archive = self.master_redis.lrange(ARCHIVE_KEY, 0, -1)
        return [deserialize(novelty_vector) for novelty_vector in archive]

    def get_current_task(self, worker_id):
        with self.local_redis.pipeline() as pipe:
            while True:
                task_id_key = get_task_id_key(worker_id)
                task_data_key = get_task_data_key(worker_id)
                try:
                    pipe.watch(task_id_key)
                    task_id = int(retry_get(pipe, task_id_key))
                    if task_id == self.cached_task_id:
                        logger.debug('[worker] Returning cached task {}'.format(task_id))
                        break
                    pipe.multi()
                    self.cached_task_data = deserialize(pipe.get(task_data_key))
                    self.cached_task_id = task_id
                    logger.info('[worker] Getting new task {}. Cached task was {}'.format(task_id, self.cached_task_id))
                    # self.cached_task_id, self.cached_task_data = task_id, deserialize(pipe.execute()[0])
                    break
                except redis.WatchError:
                    continue
        return self.cached_task_id, self.cached_task_data

    def push_result(self, task_id, result):
        self.local_redis.rpush(RESULTS_KEY, serialize((task_id, result)))
        logger.debug('[worker] Pushed result for task {}'.format(task_id))


class CoolWorkerClient:
    def __init__(self, socket):
        self.socket = socket
        # self.worker_id = worker_id
        # self.r = redis.Redis()
        # self.p = self.r.pubsub()
        # self.p.subscribe(get_task_data_key(worker_id))

    def get_current_task(self):
        print('Waiting for task')
        # try:
        size = self.socket.recv(32)
        size = size.decode('ascii')
        print("sosi hui kursavaya = ", len(size), "   ",  size)
        size = int(size)
        print('Got task!')
        parts = size // 4096
        data = b''
        recv_len = 0
        for _ in range(parts):
            chunk = self.socket.recv(4096)
            if len(chunk) != 4096:
                print("Ebis ono vse konem!", len(chunk))
            recv_len += len(chunk)
            data += chunk
        if size % 4096 != 0:
            chunk = self.socket.recv(4096)[0:size % 4096]
            if len(chunk) != 4096:
                print("Ebis ono vse konem!", size % 4096, "  ", len(chunk))
            recv_len += len(chunk)
            data += chunk

        if len(data) != size:
            print("WTF!?")

        task_data = deserialize(data)
        # except BaseException:
        #     print("Fuck!")
        #     pass
        return task_data

    def push_result(self, result):
        data = str(result)
        data = data.encode('ascii')
        data = data.zfill(32)
        sent = self.socket.sendall(data)
        if sent != 32:
            print("Blyaaaaaaaaaaaaaa!", sent, 32)
        # self.r.publish(get_result_key(self.worker_id), serialize(result))


class CoolMasterClient:
    def __init__(self, num_workers, sockets):
        # self.r = redis.Redis()
        # self.p = self.r.pubsub()
        self.num_workers = num_workers
        self.sockets = sockets
        # for worker_id in range(num_workers):
        #     self.p.subscribe(get_result_key(worker_id))

    def declare_tasks(self, tasks_data):
        worker_id = 0
        for task_data in tasks_data:
            socket = self.sockets[worker_id]

            serialized_task_data = serialize(task_data)
            size = len(serialized_task_data)

            if size == 4036432:
                print("send " + str(4036432))

            size = str(size).encode('ascii')
            size = size.zfill(32)
            sent = socket.sendall(size)
            if sent != 32:
                print("Sukaaaaaaaaaaaaaaaaaaaa", 32, sent)

            size = len(serialized_task_data)

            parts = size // 4096

            for i in range(parts):
                sent = socket.sendall(serialized_task_data[4096*i:4096*(i + 1)])
                if sent != 4096:
                    print("fjslfhslfs nenavizhu!", sent, 4096)

            if size % 4096 != 0:
                sent = socket.sendall(serialized_task_data[4096*parts:] + b'0'*(4096 - size % 4096))
                if sent != size % 4096:
                    print("Ebal v rot govno python govno", sent, size % 4096)

            # socket.send(serialized_task_data)
            print('send to ' + str(worker_id))

            worker_id += 1
        # worker_id = 0
        # for task_data in tasks_data:
        #     serialized_task_data = serialize(task_data)
        #     self.r.publish(get_task_data_key(worker_id), serialized_task_data)
        #     worker_id += 1

    def pop_results(self):
        results = []

        for worker_id in range(self.num_workers):
            data = self.sockets[worker_id].recv(32)
            data = data.decode('ascii')
            data = float(data)
            result = data
            results.append(result)

        return results

        # results = [0 for _ in range(self.num_workers)]
        # indices = set()
        #
        # for message in self.p.listen():
        #     data = message['data']
        #     if not isinstance(data, bytes):
        #         continue
        #
        #     result = deserialize(data)
        #     results[result.worker_id] = result
        #     indices.add(result.worker_id)
        #
        #     if len(indices) == self.num_workers:
        #         break
        #
        # for result in results:
        #     if result == 0:
        #         i = 0
        # return results
