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
import json
import os
from collections import OrderedDict
from detectron2.evaluation import COCOEvaluator as _COCOEvaluator
from detectron2.evaluation import COCOPanopticEvaluator as _COCOPanopticEvaluator
from detectron2.evaluation import SemSegEvaluator as _SemSegEvaluator
from detectron2.evaluation.coco_evaluation import _evaluate_predictions_on_coco
from detectron2.utils.file_io import PathManager
from tabulate import tabulate


class COCOEvaluator(_COCOEvaluator):
    def __init__(self, *, dataset_name, **kwargs):
        super().__init__(dataset_name=dataset_name, **kwargs)
        self.dataset_name = dataset_name

    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """

        results = super().evaluate(img_ids)
        prefix_results = OrderedDict()
        for k, v in results.items():
            prefix_results[f"{self.dataset_name}/{k}"] = v

        return prefix_results


class COCOPanopticEvaluator(_COCOPanopticEvaluator):
    def __init__(self, *, dataset_name, **kwargs):
        super().__init__(dataset_name=dataset_name, **kwargs)
        self.dataset_name = dataset_name

    def evaluate(self):
        results = super().evaluate()
        if results is None:
            return
        prefix_results = OrderedDict()
        for k, v in results.items():
            prefix_results[f"{self.dataset_name}/{k}"] = v

        return prefix_results


class SemSegEvaluator(_SemSegEvaluator):
    def __init__(self, *, dataset_name, prefix="", **kwargs):
        super().__init__(dataset_name=dataset_name, **kwargs)
        self.dataset_name = dataset_name
        if len(prefix) and not prefix.endswith("_"):
            prefix += "_"
        self.prefix = prefix

    def evaluate(self):
        results = super().evaluate()
        if results is None:
            return
        results_per_category = []
        for name in self._class_names:
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            results_per_category.append((str(name), float(results["sem_seg"][f"IoU-{name}"])))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "IoU"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category IoU: \n" + table)

        prefix_results = OrderedDict()
        for k, v in results.items():
            prefix_results[f"{self.dataset_name}/{self.prefix}{k}"] = v

        return prefix_results


# Copied from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/evaluation/instance_evaluation.py  # noqa
# modified from COCOEvaluator for instance segmentation
class InstanceSegEvaluator(COCOEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.
    The metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means
    the metric cannot be computed (e.g. due to no predictions made).

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            # all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            # num_classes = len(all_contiguous_ids)
            # assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                # assert category_id < num_classes, (
                #     f"A prediction has class={category_id}, "
                #     f"but the dataset only has {num_classes} classes and "
                #     f"predicted class id should be in [0, {num_classes - 1}]."
                # )
                assert category_id in reverse_id_mapping, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has class ids in {dataset_id_to_contiguous_id}."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                    use_fast_impl=self._use_fast_impl,
                    img_ids=img_ids,
                    max_dets_per_image=self._max_dets_per_image,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res
