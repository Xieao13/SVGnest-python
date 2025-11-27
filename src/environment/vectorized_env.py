import multiprocessing as mp
import numpy as np
from typing import List, Callable

def worker(remote: mp.connection.Connection, parent_remote: mp.connection.Connection, env_fn: Callable):
    """
    在独立的进程中运行环境。
    """
    parent_remote.close()
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                next_state, reward, done, info = env.step(data)
                if done:
                    # 如果环境结束，自动重置并返回新的初始状态
                    next_state = env.reset()
                remote.send((next_state, reward, done, info))
            elif cmd == 'reset':
                state = env.reset()
                remote.send(state)
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
    except KeyboardInterrupt:
        print("SubprocVecEnv worker: KeyboardInterrupt")
    finally:
        env.close() if hasattr(env, 'close') else None

class SubprocVecEnv:
    """
    一个简单的、基于多进程的并行环境包裹器。
    """
    def __init__(self, env_fns: List[Callable]):
        self.num_envs = len(env_fns)
        self.waiting = False
        self.closed = False
        
        # 使用'fork'上下文以获得更好的性能，但在某些系统上可能需要'spawn'
        ctx = mp.get_context('fork')

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, env_fn)
            process = ctx.Process(target=worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

    def step(self, actions):
        """同时在所有环境中执行一步。"""
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        if self.waiting:
            raise RuntimeError("Already waiting for a step to complete")
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        next_states, rewards, dones, infos = zip(*results)
        return np.stack(next_states), np.stack(rewards), np.stack(dones), infos

    def reset(self):
        """重置所有环境。"""
        for remote in self.remotes:
            remote.send(('reset', None))
        states = [remote.recv() for remote in self.remotes]
        return np.stack(states)

    def close(self):
        """清理所有进程。"""
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True