"""Training and sampling plumbing shared by the learned baselines."""
import contextlib
import os
import random
import time

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name):
    if name.startswith('cuda') and not torch.cuda.is_available():
        print('CUDA is not available, running on CPU', flush=True)
        return torch.device('cpu')
    return torch.device(name)


def autocast(device, enabled=True):
    """bf16 autocast on GPU; a no-op on CPU."""
    if enabled and device.type == 'cuda':
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


class Logger:
    def __init__(self, path):
        self.path = path

    def __call__(self, msg):
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
        print(line, flush=True)
        with open(self.path, 'a') as f:
            f.write(line + '\n')


def save_checkpoint(path, state):
    """Write to a temp file first, so a job killed mid-save keeps the old checkpoint."""
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path):
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location='cpu', weights_only=False)


class TrainClock:
    """Wall-clock budget. A run is split over SLURM jobs (segments): each segment
    stops after segment_hours, the run stops after total_hours over all segments."""

    def __init__(self, total_hours, segment_hours, hours_before=0.0):
        self.start = time.time()
        self.total_hours = total_hours
        self.segment_hours = segment_hours
        self.hours_before = hours_before

    def segment(self):
        return (time.time() - self.start) / 3600

    def total(self):
        return self.hours_before + self.segment()

    def run_over(self):
        return self.total() >= self.total_hours

    def segment_over(self):
        return self.segment() >= self.segment_hours


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state):
        self.shadow = {k: v.to(self.shadow[k].device) for k, v in state.items()}


def shard(indices, shard_id=None, num_shards=None):
    """This job's share of indices; defaults to the SLURM array task."""
    if shard_id is None:
        shard_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    if num_shards is None:
        count = os.environ.get('SLURM_ARRAY_TASK_COUNT')
        if count is None and 'SLURM_ARRAY_TASK_MAX' in os.environ:
            count = int(os.environ['SLURM_ARRAY_TASK_MAX']) + 1
        num_shards = int(count or 1)
    return list(indices)[shard_id::num_shards], shard_id, num_shards


def parse_ints(text):
    return tuple(int(v) for v in str(text).split(',') if v.strip())


def evenly_spaced(n_total, n_pick):
    if n_total == 0 or n_pick <= 0:
        return []
    return sorted(set(np.linspace(0, n_total - 1, min(n_pick, n_total)).round().astype(int).tolist()))


class TrainRecord:
    """Everything about a training run, kept up to date in the method's folder:

    train_info.json   setting, data sizes, model, optimizer, loss function,
                      stopping rule, one entry per SLURM job segment (job id,
                      start, end, hours), totals, stop reason
    loss.csv          one row per logged loss (iteration or epoch, hours, ...)

    A resumed run reads its existing record and adds a segment.
    """

    def __init__(self, out, info, columns):
        import json
        self.json_path = os.path.join(out, 'train_info.json')
        self.csv_path = os.path.join(out, 'loss.csv')
        self.columns = columns
        self.info = {}
        if os.path.exists(self.json_path):
            with open(self.json_path) as f:
                self.info = json.load(f)
        self.info.update(info)
        self.info.setdefault('started', time.strftime('%Y-%m-%d %H:%M:%S'))
        self.info.setdefault('segments', []).append(
            {'job': os.environ.get('SLURM_JOB_ID'), 'node': os.environ.get('SLURMD_NODENAME'),
             'start': time.strftime('%Y-%m-%d %H:%M:%S'), 'end': None, 'hours': 0.0})
        self.info.pop('finished', None)
        self.info['stop_reason'] = 'running'
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w') as f:
                f.write(','.join(columns) + '\n')
        self.write()

    def write(self):
        import json
        tmp = self.json_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.info, f, indent=1)
        os.replace(tmp, self.json_path)

    def loss(self, **row):
        with open(self.csv_path, 'a') as f:
            f.write(','.join(f'{row[c]:.6g}' if isinstance(row[c], float) else str(row[c]) for c in self.columns) + '\n')

    def update(self, segment_hours, **fields):
        """Progress: called at every checkpoint and at the end."""
        self.info.update(fields)
        self.info['segments'][-1]['hours'] = round(segment_hours, 4)
        self.info['segments'][-1]['end'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self.info['total_hours'] = round(sum(s['hours'] for s in self.info['segments']), 4)
        self.write()

    def finish(self, reason, segment_hours, **fields):
        self.update(segment_hours, **fields)
        self.info['stop_reason'] = reason
        self.info['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self.write()
