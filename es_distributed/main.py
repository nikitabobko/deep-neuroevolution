import errno
import json
import logging
import os
import sys
import socket
import click

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

@click.group()
def cli():
    logging.basicConfig(
        format='[%(asctime)s pid=%(process)d] %(message)s',
        level=logging.INFO,
        stream=sys.stderr)

def import_algo(name):
    if name == 'es':
        from . import es as algo
    elif name == 'ns-es' or name == "nsr-es":
        from . import nses as algo
    elif name == 'ga':
        from . import ga as algo
    elif name == 'rs':
        from . import rs as algo
    else:
        raise NotImplementedError()
    return algo

@cli.command()
@click.option('--algo')
@click.option('--exp_file')
@click.option('--log_dir')
@click.option('--num_workers', type=int, default=0)
def master(algo, exp_file, log_dir, num_workers):
    with open(exp_file, 'r') as f:
        exp = json.loads(f.read())
    log_dir = os.path.expanduser(log_dir) if log_dir else '/tmp/es_master_{}'.format(os.getpid())
    mkdir_p(log_dir)
    algo = import_algo(algo)

    logging.info('Spawning {} workers'.format(num_workers))

    bind_ip = '127.0.0.1'
    bind_port = 9999

    write_pipes = []
    for worker_id in range(num_workers):
        r, w = os.pipe()

        if os.fork() == 0:
            # child
            os.close(w)
            # r = os.fdopen(r)
            os.read(r, 1)
            # r.read()

            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print("trying to connect")
            client.connect((bind_ip, bind_port))
            print("connected")

            # if worker_id == 0:
            #     import pydevd_pycharm
            #     pydevd_pycharm.settrace('localhost', port=7007, stdoutToServer=True, stderrToServer=True)
            #     print('----slave 0')
            # elif worker_id == 1:
            #     import pydevd_pycharm
            #     pydevd_pycharm.settrace('localhost', port=7008, stdoutToServer=True, stderrToServer=True)
            # elif worker_id == 2:
            #     import pydevd_pycharm
            #     pydevd_pycharm.settrace('localhost', port=7009, stdoutToServer=True, stderrToServer=True)
            # elif worker_id == 3:
            #     import pydevd_pycharm
            #     pydevd_pycharm.settrace('localhost', port=7010, stdoutToServer=True, stderrToServer=True)
            # else:
            #     assert False

            algo.run_worker(exp, client)
            return
        else:
            # parent
            os.close(r)
            # w = os.fdopen(w, 'w')
            write_pipes.append(w)

    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((bind_ip, bind_port))
        server.listen(num_workers)  # max backlog of connections

        for w in write_pipes:
            fdopen = os.fdopen(w, 'w')
            fdopen.write("0")
            fdopen.close()

        print("here")
        client_socks = []
        for worker_id in range(num_workers):
            client_sock, _ = server.accept()
            client_socks.append(client_sock)

        algo.run_master(log_dir, exp, num_workers, client_socks)
    except KeyboardInterrupt:
        print("Server is closed")
        server.close()
        sys.exit()
    os.wait()


# @cli.command()
# @click.option('--algo')
# @click.option('--num_workers', type=int, default=0)
# @click.option('--exp_file')
# def workers(algo, num_workers, exp_file):
#     with open(exp_file, 'r') as f:
#         exp = json.loads(f.read())
#
#     # Start the workers
#     algo = import_algo(algo)
#     num_workers = num_workers if num_workers else os.cpu_count()



if __name__ == '__main__':
    cli()
