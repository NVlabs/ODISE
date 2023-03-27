# ------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# To view a copy of this license, visit
# https://github.com/facebookresearch/detectron2/blob/main/LICENSE
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

import itertools
import logging
import numpy as np
import time
from typing import Iterable, Mapping, Union
import detectron2.utils.comm as comm
import torch
from detectron2.engine import SimpleTrainer as _SimpleTrainer
from detectron2.utils.events import get_event_storage
from torch._six import inf
from torch.nn.parallel import DataParallel, DistributedDataParallel

from odise.utils.parameter_count import parameter_count_table

logger = logging.getLogger(__name__)

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def get_grad_norm(parameters: _tensor_or_tensors, norm_type: float = 2.0) -> torch.Tensor:
    r"""get gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
            norm_type,
        )

    return total_norm


class SimpleTrainer(_SimpleTrainer):
    def __init__(self, model, data_loader, optimizer, *, grad_clip=None):
        super().__init__(model, data_loader, optimizer)
        self.grad_clip = grad_clip
        logger.info(f"Trainer: {self.__class__.__name__}")
        logger.info(f"grad_clip: {grad_clip}")
        logger.info("All parameters: \n" + parameter_count_table(model))

        # print trainable parameters
        logger.info("Trainable parameters: \n" + parameter_count_table(model, trainable_only=True))

    def raise_loss_nan(self, losses):
        losses = losses.detach().cpu()
        loss_nan = (~torch.isfinite(losses)).any()
        all_loss_nan = comm.all_gather(loss_nan)
        all_loss_nan = [l.item() for l in all_loss_nan]
        if any(all_loss_nan):
            raise FloatingPointError(
                f"Loss became infinite or NaN for rank: {np.where(all_loss_nan)[0].tolist()} "
                f"at iteration={self.storage.iter}!\n"
            )

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        # NOTE: added "runner_meta" if input is dict, by Jiarui
        if isinstance(data, dict):
            data["runner_meta"] = dict()
            data["runner_meta"]["iter"] = self.iter
            data["runner_meta"]["max_iter"] = self.max_iter
        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        # NOTE: added "grad_norm" by Jiarui
        if self.grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        else:
            grad_norm = get_grad_norm(self.model.parameters())
        self.storage.put_scalar("grad_norm", grad_norm)

        # disable to see if it is necessary
        # NaN check before write metric, exit all process
        # self.raise_loss_nan(losses)

        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

    # Almost the same as parent, except for log all `metric_dict` instead of
    # len(metric_dict) > 1 in orginal detectron2, maybe a bug
    def _write_metrics(
        self,
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ) -> None:
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
            prefix (str): prefix for logging keys
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={storage.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict):
                storage.put_scalars(**metrics_dict)


class NativeScalerWithGradNormCount:
    """
    Reference: https://github.com/microsoft/Swin-Transformer/blob/afeb877fba1139dfbc186276983af2abb02c2196/main.py#L194
    """  # noqa

    state_dict_key = "amp_scaler"

    def __init__(self):
        from torch.cuda.amp import GradScaler

        self._scaler = GradScaler()

    def __call__(
        self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm(parameters)

            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_optimizer_parameters(optimizer):
    return itertools.chain(*[x["params"] for x in optimizer.param_groups])


class AMPTrainer(SimpleTrainer):
    """
    Like :class:`SimpleTrainer`, but uses PyTorch's native automatic mixed precision
    in the training loop.
    """

    def __init__(self, model, data_loader, optimizer, grad_clip=None):
        """
        Args:
            model, data_loader, optimizer: same as in :class:`SimpleTrainer`.
            grad_scaler: torch GradScaler to automatically scale gradients.
        """
        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        # NOTE: Jiarui relaxed this check, because it is not necessary for AMP
        elif isinstance(model, DataParallel):
            assert not len(model.device_ids) > 1, unsupported
        # assert not isinstance(model, DataParallel), unsupported

        super().__init__(model, data_loader, optimizer, grad_clip=grad_clip)

        self.grad_scaler = NativeScalerWithGradNormCount()

    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        # detectron2 use trivial collate function, so data is just a list of dict
        # we don't add meta info for them
        # for webdataset, data is a dict of tensors
        if isinstance(data, dict):
            # NOTE: added "runner_meta" by Jiarui
            data["runner_meta"] = dict()
            data["runner_meta"]["iter"] = self.iter
            data["runner_meta"]["max_iter"] = self.max_iter
        with autocast():
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        # instead of using self.optimizer.zero_grad(),
        # we zero out the gradients for all parameters in the model
        self.model.zero_grad()

        # NOTE: the following operations are wrapped
        # in __call__() of NativeScalerWithGradNormCount():
        # self.grad_scaler.scale(losses).backward()
        # self.grad_scaler.step(self.optimizer)
        # self.grad_scaler.update()
        grad_norm = self.grad_scaler(
            losses,
            self.optimizer,
            clip_grad=self.grad_clip,
            # instead of using self.model.parameters(), we use parameters in the optimizer
            # so we don't clip the parameters that are not in the optimizer
            parameters=get_optimizer_parameters(self.optimizer),
            update_grad=True,
        )
        self.storage.put_scalar("grad_norm", grad_norm)
        self.storage.put_scalar(
            "clipped_grad_norm", get_grad_norm(get_optimizer_parameters(self.optimizer))
        )

        loss_scale_value = self.grad_scaler.state_dict()["scale"]
        self.storage.put_scalar("loss_scale", loss_scale_value)

        self._write_metrics(loss_dict, data_time)

    def state_dict(self):
        ret = super().state_dict()
        ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.grad_scaler.load_state_dict(state_dict["grad_scaler"])
