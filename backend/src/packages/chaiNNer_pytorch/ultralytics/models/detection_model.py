from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from api import Iterator, IteratorOutputInfo, NodeContext
from nodes.properties.inputs import ImageInput, PthFileInput, SliderInput
from nodes.properties.outputs import (
    ImageOutput,
    NumberOutput,
    TextOutput,
)

from .. import ultralytics


@ultralytics.register(
    schema_id="chainner:pytorch:ultralytics:load_model",
    name="Detection Model",
    description=[
        (
            "Load PyTorch state dict (.pth), TorchScript (.pt), or Checkpoint (.ckpt) files into an"
            " auto-detected supported model architecture."
        ),
        (
            "- For Super-Resolution, we support most variations of the RRDB"
            " architecture (ESRGAN, Real-ESRGAN, RealSR, BSRGAN, SPSR), Real-ESRGAN's"
            " SRVGG architecture, Swift-SRGAN, SwinIR, Swin2SR, HAT, Omni-SR, SRFormer, and DAT."
        ),
        (
            "- For Face-Restoration, we support GFPGAN (1.2, 1.3, 1.4), RestoreFormer,"
            " and CodeFormer."
        ),
        "- For Inpainting, we support LaMa and MAT.",
        (
            "Links to the official models can be found in [chaiNNer's"
            " README](https://github.com/chaiNNer-org/chaiNNer#pytorch), and"
            " community-trained models on [OpenModelDB](https://openmodeldb.info/)."
        ),
    ],
    icon="PyTorch",
    inputs=[
        ImageInput().with_id(0),
        PthFileInput(primary_input=True).with_id(1),
        SliderInput(
            label="Conf Threshold",
            minimum=0,
            maximum=1,
            default=0.25,
            slider_step=0.01,
            precision=2,
        ),
        SliderInput(
            label="IoU Threshold",
            minimum=0,
            maximum=1,
            default=0.25,
            slider_step=0.01,
            precision=2,
        ),
    ],
    outputs=[
        TextOutput("Class Name"),
        NumberOutput("Top"),
        NumberOutput("Left"),
        NumberOutput("Bottom"),
        NumberOutput("Right"),
        NumberOutput("Index"),
        ImageOutput(
            "Annotated Image",
        ),
    ],
    node_context=True,
    iterator_outputs=IteratorOutputInfo(outputs=[0, 1, 2, 3, 4, 5]),
    kind="newIterator",
    see_also=[
        "chainner:pytorch:load_models",
    ],
)
def detection_model_node(
    context: NodeContext,
    img: np.ndarray,
    path: Path,
    conf: float,
    iou: float,
) -> tuple[Iterator[tuple[str, int, int, int, int, int]], np.ndarray]:
    def get_model_results(result: dict, index: int):
        cls_name = result["cls_name"]
        top, left, bottom, right = result["xyxy"]
        return cls_name, int(top), int(left), int(bottom), int(right), index

    assert os.path.exists(path), f"Model file at location {path} does not exist"

    assert os.path.isfile(path), f"Path {path} is not a file"

    model = YOLO(model=path, task="detect")
    img = (img * 255).astype(np.uint8)
    result = model(
        img,
        conf=conf,
        iou=iou,
        verbose=False,
    )[0]
    ann_img = result.plot()
    model_result = [
        {"cls_name": model.names[int(cls_id)], "xyxy": list(xyxy.numpy().astype(int))}
        for cls_id, xyxy in zip(result.boxes.cls, result.boxes.xyxy)
    ]

    return Iterator.from_list(model_result, get_model_results, True), ann_img
