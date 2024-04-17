from __future__ import annotations

import numpy as np

from api import KeyInfo
from nodes.properties.inputs import ImageInput, NumberInput
from nodes.properties.outputs import ImageOutput

from .. import ultralytics


@ultralytics.register(
    schema_id="chainner:image:crop",
    name="Crop",
    description="Crop an image.",
    icon="MdCrop",
    inputs=[
        ImageInput(),
        NumberInput("Top", unit="px").with_id(1),
        NumberInput("Left", unit="px").with_id(2),
        NumberInput("Bottom", unit="px").with_id(3),
        NumberInput("Right", unit="px").with_id(4),
    ],
    outputs=[ImageOutput()],
    key_info=KeyInfo.enum(1),
)
def crop_node(
    img: np.ndarray,
    top: int,
    left: int,
    bottom: int,
    right: int,
) -> np.ndarray:
    img = (img * 255).astype(np.uint8)
    crop = img[left:right, top:bottom]
    return crop
