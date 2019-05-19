import logging
import pickle

logger = logging.getLogger(__name__)

EXP_KEY = 'es:exp'
TASK_CHANNEL = 'es:task_channel'
RESULTS_KEY = 'es:results'
ARCHIVE_KEY = 'es:archive'


def get_task_id_key(worker_id):
    return 'es:task_id' + str(worker_id)


def get_task_data_key(worker_id):
    return 'es:task_data' + str(worker_id)


def get_result_key(worker_id):
    return 'es:results' + str(worker_id)


def serialize(x):
    # return json.dumps(x)
    return pickle.dumps(x, protocol=-1)


def deserialize(x):
    # ret = json.loads(x)
    # ret['params'] = np.array(ret['params'])
    # return ret
    return pickle.loads(x)


def recv_from_socket(socket, size):
    data = b''
    while size > 0:
        chunk = socket.recv(min(size, 4096))
        size -= len(chunk)
        data += chunk
    return data


class CoolWorkerClient:
    def __init__(self, socket):
        self.socket = socket

    def get_current_task(self):
        size = recv_from_socket(self.socket, 32)
        size = size.decode('ascii')
        size = int(size)
        parts = size // 4096
        data = b''
        recv_len = 0
        for _ in range(parts):
            chunk = recv_from_socket(self.socket, 4096)
            recv_len += len(chunk)
            data += chunk
        if size % 4096 != 0:
            chunk = recv_from_socket(self.socket, 4096)[0:size % 4096]
            recv_len += len(chunk)
            data += chunk

        assert len(data) == size

        task_data = deserialize(data)
        return task_data

    def push_result(self, result):
        data = str(result)
        data = data.encode('ascii')
        data = data.zfill(32)
        self.socket.sendall(data)


class CoolMasterClient:
    def __init__(self, num_workers, sockets):
        self.num_workers = num_workers
        self.sockets = sockets

    def declare_tasks(self, tasks_data):
        worker_id = 0
        for task_data in tasks_data:
            socket = self.sockets[worker_id]

            serialized_task_data = serialize(task_data)
            size = len(serialized_task_data)

            size = str(size).encode('ascii')
            size = size.zfill(32)
            socket.sendall(size)

            size = len(serialized_task_data)

            parts = size // 4096

            for i in range(parts):
                socket.sendall(serialized_task_data[4096 * i:4096 * (i + 1)])

            if size % 4096 != 0:
                socket.sendall(serialized_task_data[4096 * parts:] + b'0' * (4096 - size % 4096))

            worker_id += 1

    def pop_results(self):
        results = []

        for worker_id in range(self.num_workers):
            data = recv_from_socket(self.sockets[worker_id], 32)
            data = data.decode('ascii')
            data = float(data)
            result = data
            results.append(result)

        return results
