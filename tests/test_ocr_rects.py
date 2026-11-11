"""`BaseOcrModel.get_ocr_rects` against its original, slower formulation."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, find_objects, label

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Size
from docling.datamodel.pipeline_options import RapidOcrOptions
from docling.models.base_ocr_model import BaseOcrModel


class _OcrModel(BaseOcrModel):
    def __call__(self, conv_res, page_batch):
        yield from page_batch

    @classmethod
    def get_options_type(cls):
        return RapidOcrOptions


def _model(**options) -> _OcrModel:
    return _OcrModel(
        enabled=True,
        artifacts_path=None,
        options=RapidOcrOptions(**options),
        accelerator_options=AcceleratorOptions(device="cpu", num_threads=1),
    )


def _page(size: Size, rects: list[BoundingBox]):
    backend = SimpleNamespace(get_bitmap_rects=lambda: iter(rects))
    return SimpleNamespace(size=size, _backend=backend)


def _reference_ocr_rects(model, page) -> list[BoundingBox]:
    """The algorithm as first written: one 20x20 dilation, on every page."""
    size = page.size
    image = Image.new("1", (round(size.width), round(size.height)))
    draw = ImageDraw.Draw(image)
    for rect in page._backend.get_bitmap_rects():
        x0, y0, x1, y1 = (round(v) for v in rect.as_tuple())
        draw.rectangle([(x0, y0), (x1, y1)], fill=1)
    np_image = binary_dilation(np.array(image) > 0, structure=np.ones((20, 20)))
    labeled_image, _ = label(np_image > 0)
    boxes = [
        BoundingBox(
            l=slc[1].start,
            t=slc[0].start,
            r=slc[1].stop - 1,
            b=slc[0].stop - 1,
            coord_origin=CoordOrigin.TOPLEFT,
        )
        for slc in find_objects(labeled_image)
    ]
    coverage = np.sum(np_image > 0) / (size.width * size.height)

    full_page = [
        BoundingBox(
            l=0, t=0, r=size.width, b=size.height, coord_origin=CoordOrigin.TOPLEFT
        )
    ]
    if model.options.force_full_page_ocr or coverage > max(
        0.75, model.options.bitmap_area_threshold
    ):
        return full_page
    elif coverage > model.options.bitmap_area_threshold:
        return boxes
    return []


def _random_rects(rng: random.Random, size: Size, count: int) -> list[BoundingBox]:
    rects = []
    for _ in range(count):
        left = rng.uniform(-8, size.width)
        top = rng.uniform(-8, size.height)
        width = rng.uniform(0, size.width / rng.choice([1, 2, 6, 25]))
        height = rng.uniform(0, size.height / rng.choice([1, 2, 6, 25]))
        rects.append(
            BoundingBox(
                l=left,
                t=top,
                r=left + width,
                b=top + height,
                coord_origin=CoordOrigin.TOPLEFT,
            )
        )
    return rects


@pytest.mark.parametrize("bitmap_area_threshold", [0.05, 0.0, 0.5])
def test_ocr_rects_match_the_reference(bitmap_area_threshold: float):
    rng = random.Random(20261112)
    model = _model(bitmap_area_threshold=bitmap_area_threshold)
    kinds = set()
    for _ in range(120):
        width, height = rng.choice([(595.3, 841.9), (960, 540), (612, 792), (41, 23)])
        size = Size(width=width, height=height)
        rects = _random_rects(rng, size, rng.choice([0, 1, 1, 2, 4, 12]))
        page = _page(size, rects)

        expected = _reference_ocr_rects(model, page)
        actual = model.get_ocr_rects(page)

        assert actual == expected
        kinds.add(
            "none"
            if not expected
            else "full"
            if expected[0].r == size.width and expected[0].b == size.height
            else "parts"
        )
    assert kinds == {"none", "full", "parts"}  # every branch was exercised


def test_forced_full_page_ocr_takes_the_whole_page_without_bitmaps():
    size = Size(width=595, height=842)

    rects = _model(force_full_page_ocr=True).get_ocr_rects(_page(size, []))

    assert rects == [
        BoundingBox(l=0, t=0, r=595, b=842, coord_origin=CoordOrigin.TOPLEFT)
    ]


def test_a_page_without_bitmaps_has_no_ocr_rects():
    assert _model().get_ocr_rects(_page(Size(width=595, height=842), [])) == []
