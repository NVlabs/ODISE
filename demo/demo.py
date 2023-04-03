#!/usr/bin/env python
#
# ------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
# To view a copy of this license, visit
# https://github.com/facebookresearch/Mask2Former/blob/main/LICENSE
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

import argparse
import glob
import itertools
import numpy as np
import os
import tempfile
import time
import warnings
from contextlib import ExitStack
import cv2
import nltk
import torch
import tqdm
from detectron2.config import LazyConfig, instantiate
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.engine import create_ddp_model
from detectron2.evaluation import inference_context
from detectron2.utils.env import seed_all_rng
from detectron2.utils.logger import setup_logger
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer, random_color
from mask2former.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES
from torch import nn

from odise.checkpoint import ODISECheckpointer
from odise.config import instantiate_odise
from odise.data import get_openseg_labels
from odise.engine.defaults import get_model_from_module

nltk.download("popular", quiet=True)
nltk.download("universal_tagset", quiet=True)

# constants
WINDOW_NAME = "ODISE demo"

COCO_THING_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("coco_panoptic", True))
    if COCO_CATEGORIES[idx]["isthing"] == 1
]
COCO_THING_COLORS = [c["color"] for c in COCO_CATEGORIES if c["isthing"] == 1]
COCO_STUFF_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("coco_panoptic", True))
    if COCO_CATEGORIES[idx]["isthing"] == 0
]
COCO_STUFF_COLORS = [c["color"] for c in COCO_CATEGORIES if c["isthing"] == 0]

ADE_THING_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("ade20k_150", True))
    if ADE20K_150_CATEGORIES[idx]["isthing"] == 1
]
ADE_THING_COLORS = [c["color"] for c in ADE20K_150_CATEGORIES if c["isthing"] == 1]
ADE_STUFF_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("ade20k_150", True))
    if ADE20K_150_CATEGORIES[idx]["isthing"] == 0
]
ADE_STUFF_COLORS = [c["color"] for c in ADE20K_150_CATEGORIES if c["isthing"] == 0]

LVIS_CLASSES = get_openseg_labels("lvis_1203", True)
# use beautiful coco colors
LVIS_COLORS = list(
    itertools.islice(itertools.cycle([c["color"] for c in COCO_CATEGORIES]), len(LVIS_CLASSES))
)


def get_nouns(caption, with_preposition):
    if with_preposition:
        # Taken from Su Nam Kim Paper...
        grammar = r"""
            NBAR:
                {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

            NP:
                {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
                {<NBAR>} # If pattern is not found, just a single NBAR is ok
        """
    else:
        # Taken from Su Nam Kim Paper...
        grammar = r"""
            NBAR:
                {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

            NP:
                {<NBAR>} # If pattern is not found, just a single NBAR is ok
        """
    tokenized = nltk.word_tokenize(caption)
    chunker = nltk.RegexpParser(grammar)

    chunked = chunker.parse(nltk.pos_tag(tokenized))
    continuous_chunk = []
    current_chunk = []

    for subtree in chunked:
        if isinstance(subtree, nltk.Tree):
            current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    if current_chunk:
        named_entity = " ".join(current_chunk)
        if named_entity not in continuous_chunk:
            continuous_chunk.append(named_entity)

    return continuous_chunk


class VisualizationDemo(object):
    def __init__(self, model, metadata, aug, instance_mode=ColorMode.IMAGE):
        """
        Args:
            model (nn.Module):
            metadata (MetadataCatalog): image metadata.
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.model = model
        self.metadata = metadata
        self.aug = aug
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

    def predict(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        height, width = original_image.shape[:2]
        aug_input = T.AugInput(original_image, sem_seg=None)
        self.aug(aug_input)
        image = aug_input.image
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        predictions = self.model([inputs])[0]
        return predictions

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predict(image)
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        for frame in frame_gen:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield process_predictions(frame, self.predict(frame))


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ODISE demo.")
    parser.add_argument(
        "--config-file",
        default="configs/Panoptic/odise_label_coco_50e.py",
        type=str,
        help="path to config file",
    )
    parser.add_argument(
        "--init-from",
        type=str,
        help="init from the given checkpoint",
        default="odise://Panoptic/odise_label_coco_50e",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocab",
        help="extra vocabulary, in format 'a1,a2;b1,b2',"
        "where a1,a2 are synonyms vocabularies for the first class"
        "first word will be displayed as the class name",
    )
    parser.add_argument(
        "--label",
        help="label set to use, could be multiple options from 'COCO', 'ADE' and 'LVIS'.",
        choices=["COCO", "ADE", "LVIS", ""],
        nargs="+",
        default=["COCO", "ADE", "LVIS"],
    )
    parser.add_argument("--caption", help="caption contains nouns (noun phrases) to be segmented")
    parser.add_argument(
        "opts",
        help="Modify config options at the end of the command, " "in 'path.key=value' format",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    setup_logger()
    logger = setup_logger(name="odise")

    cfg = LazyConfig.load(args.config_file)
    cfg.model.overlap_threshold = 0
    cfg.model.clip_head.alpha = 0.35
    cfg.model.clip_head.beta = 0.65
    seed_all_rng(42)

    dataset_cfg = cfg.dataloader.test
    wrapper_cfg = cfg.dataloader.wrapper

    extra_classes = []

    if args.vocab:
        for words in args.vocab.split(";"):
            extra_classes.append([word.strip() for word in words.split(",")])

    if args.caption:
        caption_words = []
        caption_words.extend(get_nouns(args.caption, True))
        caption_words.extend(get_nouns(args.caption, False))
        for word in list(set(caption_words)):
            extra_classes.append([word.strip()])

    logger.info(f"extra classes: {extra_classes}")
    extra_colors = [random_color(rgb=True, maximum=1) for _ in range(len(extra_classes))]

    # demo_thing_classes = extra_classes + COCO_THING_CLASSES + ADE_THING_CLASSES + LVIS_CLASSES
    # demo_stuff_classes = COCO_STUFF_CLASSES + ADE_STUFF_CLASSES
    # demo_thing_colors = extra_colors + COCO_THING_COLORS + ADE_THING_COLORS + LVIS_COLORS
    # demo_stuff_colors = COCO_STUFF_COLORS + ADE_STUFF_COLORS

    demo_thing_classes = extra_classes
    demo_stuff_classes = []
    demo_thing_colors = extra_colors
    demo_stuff_colors = []

    if "COCO" in args.label:
        demo_thing_classes += COCO_THING_CLASSES
        demo_stuff_classes += COCO_STUFF_CLASSES
        demo_thing_colors += COCO_THING_COLORS
        demo_stuff_colors = COCO_STUFF_COLORS
    if "ADE" in args.label:
        demo_thing_classes += ADE_THING_CLASSES
        demo_stuff_classes += ADE_STUFF_CLASSES
        demo_thing_colors += ADE_THING_COLORS
        demo_stuff_colors += ADE_STUFF_COLORS
    if "LVIS" in args.label:
        demo_thing_classes += LVIS_CLASSES
        demo_thing_colors += LVIS_COLORS

    demo_metadata = MetadataCatalog.get("odise_demo_metadata")
    demo_metadata.thing_classes = [c[0] for c in demo_thing_classes]
    demo_metadata.stuff_classes = [
        *demo_metadata.thing_classes,
        *[c[0] for c in demo_stuff_classes],
    ]
    demo_metadata.thing_colors = demo_thing_colors
    demo_metadata.stuff_colors = demo_thing_colors + demo_stuff_colors
    demo_metadata.stuff_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.stuff_classes))
    }
    demo_metadata.thing_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.thing_classes))
    }

    wrapper_cfg.labels = demo_thing_classes + demo_stuff_classes
    wrapper_cfg.metadata = demo_metadata

    aug = instantiate(dataset_cfg.mapper).augmentations

    model = instantiate_odise(cfg.model)
    model.to(cfg.train.device)
    ODISECheckpointer(model).load(args.init_from)
    # look for the last wrapper
    while "model" in wrapper_cfg:
        wrapper_cfg = wrapper_cfg.model
    wrapper_cfg.model = get_model_from_module(model)

    inference_model = create_ddp_model(instantiate(cfg.dataloader.wrapper))
    with ExitStack() as stack:
        if isinstance(inference_model, nn.Module):
            stack.enter_context(inference_context(inference_model))
        stack.enter_context(torch.no_grad())
        demo = VisualizationDemo(inference_model, demo_metadata, aug)

        if args.input:
            if len(args.input) == 1:
                args.input = glob.glob(os.path.expanduser(args.input[0]))
                assert args.input, "The input path(s) was not found"
            for path in tqdm.tqdm(args.input, disable=not args.output):
                # use PIL, to be consistent with evaluation
                img = utils.read_image(path, format="RGB")
                start_time = time.time()
                predictions, visualized_output = demo.run_on_image(img)
                logger.info(
                    "{}: {} in {:.2f}s".format(
                        path,
                        "detected {} instances".format(len(predictions["instances"]))
                        if "instances" in predictions
                        else "finished",
                        time.time() - start_time,
                    )
                )

                if args.output:
                    if os.path.isdir(args.output):
                        assert os.path.isdir(args.output), args.output
                        out_filename = os.path.join(args.output, os.path.basename(path))
                    else:
                        assert len(args.input) == 1, "Please specify a directory with args.output"
                        out_filename = args.output
                    visualized_output.save(out_filename)
                else:
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                    if cv2.waitKey(0) == 27:
                        break  # esc to quit
        elif args.webcam:
            assert args.input is None, "Cannot have both --input and --webcam!"
            assert args.output is None, "output not yet supported with --webcam!"
            cam = cv2.VideoCapture(0)
            for vis in tqdm.tqdm(demo.run_on_video(cam)):
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, vis)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
            cam.release()
            cv2.destroyAllWindows()
        elif args.video_input:
            video = cv2.VideoCapture(args.video_input)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames_per_second = video.get(cv2.CAP_PROP_FPS)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            basename = os.path.basename(args.video_input)
            codec, file_ext = (
                ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
            )
            if codec == ".mp4v":
                warnings.warn("x264 codec not available, switching to mp4v")
            if args.output:
                if os.path.isdir(args.output):
                    output_fname = os.path.join(args.output, basename)
                    output_fname = os.path.splitext(output_fname)[0] + file_ext
                else:
                    output_fname = args.output
                assert not os.path.isfile(output_fname), output_fname
                output_file = cv2.VideoWriter(
                    filename=output_fname,
                    # some installation of opencv may not support x264 (due to its license),
                    # you can try other format (e.g. MPEG)
                    fourcc=cv2.VideoWriter_fourcc(*codec),
                    fps=float(frames_per_second),
                    frameSize=(width, height),
                    isColor=True,
                )
            assert os.path.isfile(args.video_input)
            for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
                if args.output:
                    output_file.write(vis_frame)
                else:
                    cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                    cv2.imshow(basename, vis_frame)
                    if cv2.waitKey(1) == 27:
                        break  # esc to quit
            video.release()
            if args.output:
                output_file.release()
            else:
                cv2.destroyAllWindows()
