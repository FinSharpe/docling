import sys
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import RapidOcrOptions
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel


@pytest.mark.parametrize(
    ("backend", "det_name", "cls_name", "rec_name"),
    [
        (
            "onnxruntime",
            "ch_PP-OCRv4_det_mobile.onnx",
            "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
            "ch_PP-OCRv4_rec_mobile.onnx",
        ),
        (
            "torch",
            "ch_PP-OCRv4_det_mobile.pth",
            "ch_ptocr_mobile_v2.0_cls_mobile.pth",
            "ch_PP-OCRv4_rec_mobile.pth",
        ),
    ],
)
def test_rapidocr_default_models_use_3_8_mobile_assets(
    backend: str,
    det_name: str,
    cls_name: str,
    rec_name: str,
):
    model_paths = RapidOcrModel._default_models[backend]

    assert "/v3.8.0/" in model_paths["det_model_path"]["url"]
    assert model_paths["det_model_path"]["path"].endswith(det_name)
    assert model_paths["cls_model_path"]["path"].endswith(cls_name)
    assert model_paths["rec_model_path"]["path"].endswith(rec_name)
    assert model_paths["rec_keys_path"]["path"].endswith(
        "paddle/PP-OCRv4/rec/ch_PP-OCRv4_rec_mobile/ppocr_keys_v1.txt"
    )
    assert model_paths["font_path"]["path"] == "resources/fonts/FZYTK.TTF"

    for detail in model_paths.values():
        assert "_infer" not in detail["path"]
        assert "_infer" not in detail["url"]


@pytest.mark.parametrize(
    ("backend", "det_name", "cls_name", "rec_name"),
    [
        (
            "onnxruntime",
            "ch_PP-OCRv4_det_mobile.onnx",
            "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
            "ch_PP-OCRv4_rec_mobile.onnx",
        ),
        (
            "torch",
            "ch_PP-OCRv4_det_mobile.pth",
            "ch_ptocr_mobile_v2.0_cls_mobile.pth",
            "ch_PP-OCRv4_rec_mobile.pth",
        ),
    ],
)
def test_rapidocr_model_initialization_uses_mobile_default_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: str,
    det_name: str,
    cls_name: str,
    rec_name: str,
):
    captured: dict[str, object] = {}

    class FakeEngineType(str, Enum):
        ONNXRUNTIME = "onnxruntime"
        OPENVINO = "openvino"
        PADDLE = "paddle"
        TORCH = "torch"

    class FakeRapidOCR:
        def __init__(self, params):
            captured["params"] = params

    monkeypatch.setitem(
        sys.modules,
        "rapidocr",
        SimpleNamespace(EngineType=FakeEngineType, RapidOCR=FakeRapidOCR),
    )

    model_root = tmp_path / RapidOcrModel._model_repo_folder
    for detail in RapidOcrModel._default_models[backend].values():
        file_path = model_root / detail["path"]
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"")

    RapidOcrModel(
        enabled=True,
        artifacts_path=tmp_path,
        options=RapidOcrOptions(backend=backend),
        accelerator_options=AcceleratorOptions(device="cpu", num_threads=1),
    )

    params = captured["params"]
    assert Path(params["Det.model_path"]).name == det_name
    assert Path(params["Cls.model_path"]).name == cls_name
    assert Path(params["Rec.model_path"]).name == rec_name
    assert Path(params["Rec.rec_keys_path"]).name == "ppocr_keys_v1.txt"
    assert Path(params["Global.font_path"]).name == "FZYTK.TTF"


def _text_lines_image():
    """A page-like image with text lines of different lengths and sizes."""
    import numpy
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1500, 1100), "white")
    draw = ImageDraw.Draw(image)
    lines = [
        ("Revenue from operations 2,463.57", 44),
        ("Total", 60),
        ("Profit before exceptional items and tax", 36),
        ("31st March 2026", 52),
        ("Other comprehensive income for the year, net of tax", 30),
        ("Earnings per share", 48),
        ("19,864", 56),
        ("Notes to the consolidated financial statements", 40),
        ("Cash and cash equivalents at the end of the period", 34),
        ("EBITDA", 64),
    ]
    y = 30
    for text, size in lines:
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:  # Pillow < 10.1 has no sized default font
            pytest.skip("needs a scalable default font")
        draw.text((40 + 7 * (y % 9), y), text, fill="black", font=font)
        y += size + 44
    return numpy.array(image)


def test_rapidocr_skipping_covered_boxes_keeps_the_other_lines_unchanged():
    """Recognizing only the boxes without programmatic text must give those
    boxes exactly what recognizing every box gives them."""
    pytest.importorskip("rapidocr")
    pytest.importorskip("onnxruntime")
    from docling_core.types.doc import BoundingBox, CoordOrigin
    from docling_core.types.doc.page import BoundingRectangle, TextCell

    model = RapidOcrModel(
        enabled=True,
        artifacts_path=None,
        options=RapidOcrOptions(lang=["english"]),
        accelerator_options=AcceleratorOptions(device="cpu", num_threads=2),
    )
    if not model._skip_covered_enabled():
        pytest.skip("this RapidOCR version lacks the staged API")

    im = _text_lines_image()
    ocr_rect = BoundingBox(
        l=0,
        t=0,
        r=im.shape[1] / model.scale,
        b=im.shape[0] / model.scale,
        coord_origin=CoordOrigin.TOPLEFT,
    )

    all_lines = model._read_lines(im, ocr_rect, None)
    assert all_lines is not None and len(all_lines) >= 8

    def cell(box) -> TextCell:
        return TextCell(
            index=0,
            text="x",
            orig="x",
            from_ocr=False,
            rect=model._cell_rect(box, ocr_rect),
        )

    # Every subset of covered lines, including none and all of them: the
    # recognizer's batches are laid out over all boxes, so which boxes are
    # dropped must not change what the others read.
    for covered_ids in (
        set(),
        {0},
        {1, 2, 5},
        set(range(0, len(all_lines), 2)),
        set(range(1, len(all_lines))),
        set(range(len(all_lines))),
    ):
        idx = model._build_cell_index([cell(all_lines[i][0]) for i in covered_ids])
        expected = [
            line
            for line in all_lines
            if not model._rect_overlaps_index(idx, model._cell_rect(line[0], ocr_rect))
        ]
        if not covered_ids:
            assert expected == all_lines

        lines = model._read_uncovered_lines(im, ocr_rect, idx)

        assert lines is not None
        assert [(box, text) for box, text, _ in lines] == [
            (box, text) for box, text, _ in expected
        ]
        assert [score for *_, score in lines] == pytest.approx(
            [score for *_, score in expected], abs=1e-5
        )


def test_rapidocr_reads_every_box_when_skipping_is_off_or_ocr_is_forced():
    for options in (
        RapidOcrOptions(skip_recognition_on_programmatic_text=False),
        RapidOcrOptions(force_full_page_ocr=True),
    ):
        model = RapidOcrModel(
            enabled=False,
            artifacts_path=None,
            options=options,
            accelerator_options=AcceleratorOptions(device="cpu", num_threads=1),
        )
        assert model._skip_covered_enabled() is False
