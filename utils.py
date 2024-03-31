import os
import sys
import glob
import torch
import shutil
import logging
import datetime
import socket
import wandb
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.logger import LoggerHook, TextLoggerHook
from mmcv.runner.dist_utils import master_only
from torch.utils.tensorboard import SummaryWriter


def init_logging(filename=None, debug=False):
    logging.root = logging.RootLogger('DEBUG' if debug else 'INFO')
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] - %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


def backup_code(work_dir, verbose=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for pattern in ['*.py', 'configs/*.py', 'models/*.py', 'loaders/*.py', 'loaders/pipelines/*.py']:
        for file in glob.glob(pattern):
            src = os.path.join(base_dir, file)
            dst = os.path.join(work_dir, 'backup', os.path.dirname(file))

            if verbose:
                logging.info('Copying %s -> %s' % (os.path.relpath(src), os.path.relpath(dst)))
            
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(src, dst)


@HOOKS.register_module()
class MyTextLoggerHook(TextLoggerHook):
    def _log_info(self, log_dict, runner):
        # print exp name for users to distinguish experiments
        # at every ``interval_exp_name`` iterations and the end of each epoch
        if runner.meta is not None and 'exp_name' in runner.meta:
            if (self.every_n_iters(runner, self.interval_exp_name)) or (
                    self.by_epoch and self.end_of_epoch(runner)):
                exp_info = f'Exp name: {runner.meta["exp_name"]}'
                runner.logger.info(exp_info)

        # by epoch: Epoch [4][100/1000]
        # by iter:  Iter [100/100000]
        if self.by_epoch:
            log_str = f'Epoch [{log_dict["epoch"]}/{runner.max_epochs}]' \
                        f'[{log_dict["iter"]}/{len(runner.data_loader)}] '
        else:
            log_str = f'Iter [{log_dict["iter"]}/{runner.max_iters}] '

        log_str += 'loss: %.2f, ' % log_dict['loss']

        if 'time' in log_dict.keys():
            # MOD: skip the first iteration since it's not accurate
            if runner.iter == self.start_iter:
                time_sec_avg = log_dict['time']
            else:
                self.time_sec_tot += (log_dict['time'] * self.interval)
                time_sec_avg = self.time_sec_tot / (runner.iter - self.start_iter)

            eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            log_str += f'eta: {eta_str}, '
            log_str += f'time: {log_dict["time"]:.2f}s, ' \
                        f'data: {log_dict["data_time"] * 1000:.0f}ms, '
            # statistic memory
            if torch.cuda.is_available():
                log_str += f'mem: {log_dict["memory"]}M'

        runner.logger.info(log_str)

    def log(self, runner):
        if 'eval_iter_num' in runner.log_buffer.output:
            # this doesn't modify runner.iter and is regardless of by_epoch
            cur_iter = runner.log_buffer.output.pop('eval_iter_num')
        else:
            cur_iter = self.get_iter(runner, inner_iter=True)

        log_dict = {
            'mode': self.get_mode(runner),
            'epoch': self.get_epoch(runner),
            'iter': cur_iter
        }

        # only record lr of the first param group
        cur_lr = runner.current_lr()
        if isinstance(cur_lr, list):
            log_dict['lr'] = cur_lr[0]
        else:
            assert isinstance(cur_lr, dict)
            log_dict['lr'] = {}
            for k, lr_ in cur_lr.items():
                assert isinstance(lr_, list)
                log_dict['lr'].update({k: lr_[0]})

        if 'time' in runner.log_buffer.output:
            # statistic memory
            if torch.cuda.is_available():
                log_dict['memory'] = self._get_max_memory(runner)

        log_dict = dict(log_dict, **runner.log_buffer.output)

        # MOD: disable writing to files
        # self._dump_log(log_dict, runner)
        self._log_info(log_dict, runner)

        return log_dict

    def after_train_epoch(self, runner):
        if 'eval_iter_num' in runner.log_buffer.output:
            runner.log_buffer.output.pop('eval_iter_num')

        if runner.log_buffer.ready:
            metrics = self.get_loggable_tags(runner)
            runner.logger.info('--- Evaluation Results ---')
            runner.logger.info('RayIoU: %.4f' % metrics['val/RayIoU'])


@HOOKS.register_module()
class MyTensorboardLoggerHook(LoggerHook):
    def __init__(self, log_dir=None, interval=10, ignore_last=True, reset_flag=False, by_epoch=True):
        super(MyTensorboardLoggerHook, self).__init__(
            interval, ignore_last, reset_flag, by_epoch)
        self.log_dir = log_dir

    @master_only
    def before_run(self, runner):
        super(MyTensorboardLoggerHook, self).before_run(runner)
        if self.log_dir is None:
            self.log_dir = runner.work_dir
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)

        for key, value in tags.items():
            # MOD: merge into the 'train' group
            if key == 'learning_rate':
                key = 'train/learning_rate'

            # MOD: skip momentum
            ignore = False
            if key == 'momentum':
                ignore = True

            # MOD: skip intermediate losses
            for i in range(5):
                if key[:13] == 'train/d%d.loss' % i:
                    ignore = True

            if self.get_mode(runner) == 'train' and key[:5] != 'train':
                ignore = True

            if self.get_mode(runner) != 'train' and key[:3] != 'val':
                ignore = True

            if ignore:
                continue

            if key[:5] == 'train':
                self.writer.add_scalar(key, value, self.get_iter(runner))
            elif key[:3] == 'val':
                self.writer.add_scalar(key, value, self.get_epoch(runner))

    @master_only
    def after_run(self, runner):
        self.writer.close()


# modified from mmcv.runner.hooks.logger.wandb
@HOOKS.register_module()
class MyWandbLoggerHook(LoggerHook):
    """Class to log metrics with wandb.

    It requires `wandb`_ to be installed.


    Args:
        log_dir (str): directory for saving logs
            Default None.
        project_name (str): name for your project (mainly used to specify saving path on wandb server)
            Default None.
        team_name (str): name for your team (mainly used to specify saving path on wandb server)
            Default None.
        experiment_name (str): name for your run, if not specified, use the last part of log_dir
            Default None.
        interval (int): Logging interval (every k iterations).
            Default 10.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`.
            Default: True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default: False.
        commit (bool): Save the metrics dict to the wandb server and increment
            the step. If false ``wandb.log`` just updates the current metrics
            dict with the row argument and metrics won't be saved until
            ``wandb.log`` is called with ``commit=True``.
            Default: True.
        by_epoch (bool): Whether EpochBasedRunner is used.
            Default: True.
        with_step (bool): If True, the step will be logged from
            ``self.get_iters``. Otherwise, step will not be logged.
            Default: True.
        out_suffix (str or tuple[str], optional): Those filenames ending with
            ``out_suffix`` will be uploaded to wandb.
            Default: ('.log.json', '.log', '.py').
            `New in version 1.4.3.`

    .. _wandb:
        https://docs.wandb.ai
    """
    def __init__(self, log_dir=None, project_name=None, team_name=None, experiment_name=None, 
                 interval=10, ignore_last=True, reset_flag=False, by_epoch=True, commit=True, 
                 with_step=True, out_suffix = ('.log.json', '.log', '.py')):
        
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.import_wandb()
        self.commit = commit
        self.with_step = with_step
        self.out_suffix = out_suffix
        
        self.log_dir = log_dir
        self.project_name = project_name
        self.team_name = team_name
        self.experiment_name = experiment_name
        if commit:
            os.system('wandb online')
        else:
            os.system('wandb offline')
            
    def import_wandb(self) -> None:
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')
        self.wandb = wandb
        
    @master_only
    def before_run(self, runner) -> None:
        super().before_run(runner)
        if self.log_dir is None:
            self.log_dir = runner.work_dir
        if self.experiment_name is None:
            self.experiment_name = os.path.basename(self.log_dir)
        init_kwargs = dict(
            project=self.project_name,
            entity=self.team_name,
            notes=socket.gethostname(),
            name=self.experiment_name,
            dir=self.log_dir,
            reinit=True
        )
            
        if self.wandb is None:
            self.import_wandb()
        if init_kwargs:
            self.wandb.init(**init_kwargs)  # type: ignore
        else:
            self.wandb.init()  # type: ignore
    
    @master_only
    def log(self, runner) -> None:
        tags = self.get_loggable_tags(runner)
        mode = self.get_mode(runner)
        if not tags:
            return
        if 'learning_rate' in tags.keys():
            tags['train/learning_rate'] = tags['learning_rate']
            del tags['learning_rate']
        if 'momentum' in tags.keys():
            del tags['momentum']
        tags = {k: v for k, v in tags.items() if k.startswith(mode)}
        
        if self.with_step:
            self.wandb.log(
                tags, step=self.get_iter(runner), commit=self.commit)
        else:
            tags['global_step'] = self.get_iter(runner)
            self.wandb.log(tags, commit=self.commit)

    @master_only
    def after_run(self, runner) -> None:
        self.wandb.join()
