# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

import json
import logging
import os
import os.path as osp
from typing import Dict, Optional, Union
import torch
from detectron2.config import CfgNode
from detectron2.utils.events import CommonMetricPrinter as _CommonMetricPrinter
from detectron2.utils.events import EventWriter, get_event_storage


class WandbWriter(EventWriter):
    """Write all scalars to a wandb tool.
    based on https://github.com/facebookresearch/detectron2/pull/3716
    """

    def __init__(
        self,
        max_iter: int,
        run_name: str,
        output_dir: str,
        project: str = "ODISE",
        config: Union[Dict, CfgNode] = {},  # noqa: B006
        resume: bool = False,
        window_size: int = 20,
        **kwargs,
    ):
        """
        Args:
            project (str): W&B Project name
            config Union[Dict, CfgNode]: the project level configuration object
            window_size (int): the scalars will be median-smoothed by this window size
            kwargs: other arguments passed to `wandb.init(...)`
        """
        logger = logging.getLogger(__name__)
        # this could avoid hanging when process failed before Trainer.train()
        os.environ.setdefault("WANDB_START_METHOD", "thread")
        logger.info(f"Setting WANDB_START_METHOD to '{os.environ['WANDB_START_METHOD']}'")

        import wandb

        self._window_size = window_size
        self._run = (
            wandb.init(
                project=project,
                name=run_name,
                config=config,
                dir=output_dir,
                resume=resume,
                **kwargs,
            )
            if not wandb.run
            else wandb.run
        )
        self._run._label(repo="vision")
        self._max_iter = max_iter

        # manually write "wandb-resume.json" file
        # it is automatically created by wandb.init() only resume=True
        # so we manually create it when resume=False
        # such that we could resume a run even if we passed resume=False previously
        resume_file = osp.join(output_dir, "wandb/wandb-resume.json")
        if not resume:
            logger.warning("Manually create wandb-resume.json file")
            with open(resume_file, "w") as f:
                json.dump({"run_id": self._run.id}, f)

    def write(self):
        storage = get_event_storage()

        log_dict = {}
        for k, (v, _) in storage.latest_with_smoothing_hint(self._window_size).items():
            log_dict[k] = v

        log_dict["progress"] = storage.iter / self._max_iter

        self._run.log(log_dict)

    def close(self):
        try:
            storage = get_event_storage()
            iteration = storage.iter
            if iteration >= self._max_iter - 1:
                # finish the run after reaching max_iter
                # finish() will automatically remove the "wandb-resume.json" file
                self._run.finish()
            else:
                # mark the run haven't finish yet
                self._run.finish(1)
        except AssertionError:
            # no trainer/event_storage yet
            # mark the run haven't finish yet
            self._run.finish(1)


class CommonMetricPrinter(_CommonMetricPrinter):
    """
    Print **common** metrics to the terminal, including
    iteration time, ETA, memory, all losses, and the learning rate.
    It also applies smoothing using a window of 20 elements.

    It's meant to print common metrics in common ways.
    To print something in more customized ways, please implement a similar printer by yourself.
    """

    def __init__(self, max_iter: Optional[int] = None, window_size: int = 20, run_name: str = ""):
        super().__init__(max_iter=max_iter, window_size=window_size)
        self.run_name = run_name

    def write(self):
        storage = get_event_storage()
        iteration = storage.iter
        if iteration == self._max_iter:
            # This hook only reports training progress (loss, ETA, etc) but not other data,
            # therefore do not write anything after training succeeds, even if this method
            # is called.
            return

        try:
            data_time = storage.history("data_time").avg(20)
        except KeyError:
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            data_time = None
        try:
            iter_time = storage.history("time").global_avg()
        except KeyError:
            iter_time = None
        try:
            lr = "{:.5g}".format(storage.history("lr").latest())
        except KeyError:
            lr = "N/A"

        eta_string = self._get_eta(storage)

        storage_hide_keys = ["eta_seconds", "data_time", "time", "lr"]

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
        self.logger.info(
            "{run_name}  {eta}iter: {iter}{max_iter}  {losses}  {time}{data_time}lr: {lr}  {memory}".format(  # noqa: E501,B950
                run_name=self.run_name,
                eta=f"eta: {eta_string}  " if eta_string else "",
                iter=iteration,
                max_iter=f"/{self._max_iter}" if self._max_iter else "",
                # NOTE: Jiarui makes losses include all the storage.histories() except for
                # non-smoothing metrics. This hack will make writter log all the metrics in
                # the storage, but excluding metrics from EvalHook since smoothing_hint=False.
                losses="  ".join(
                    [
                        "{}: {:.4g}".format(k, v.median(self._window_size))
                        for k, v in storage.histories().items()
                        if k not in storage_hide_keys and storage.smoothing_hints()[k]
                    ]
                ),
                time="time: {:.4f}  ".format(iter_time) if iter_time is not None else "",
                data_time="data_time: {:.4f}  ".format(data_time) if data_time is not None else "",
                lr=lr,
                memory="max_mem: {:.0f}M".format(max_mem_mb) if max_mem_mb is not None else "",
            )
        )


class WriterStack:
    def __init__(self, logger, writers=None):
        self.logger = logger
        self.writers = writers

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.logger.error("Error occurred in the writer", exc_info=(exc_type, exc_val, exc_tb))
            self.logger.error("Closing all writers")
            if self.writers is not None:
                for writer in self.writers:
                    writer.close()
            self.logger.error("All writers closed")
