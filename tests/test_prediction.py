"""Tests for prediction module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import sahi.prediction
import sahi.utils.cv
from sahi.prediction import PredictionResult, PredictionScore
from sahi.utils.cv import read_image_as_pil

# Deliberately non-square, so a swapped width and height is visible rather than a no-op.
_PIXELS = np.random.default_rng(5).integers(0, 256, (30, 40, 3), dtype=np.uint8)

# One entry per container the sizing has to agree with a full decode on.
SIZING_FIXTURES = {
    "rgb.jpg": lambda: Image.fromarray(_PIXELS),
    "gray.jpg": lambda: Image.fromarray(_PIXELS[..., 0]),
    "rgb.png": lambda: Image.fromarray(_PIXELS),
    "rgba.png": lambda: Image.fromarray(np.dstack([_PIXELS, _PIXELS[..., :1]])),
    "palette.png": lambda: Image.fromarray(_PIXELS).convert("P", palette=Image.Palette.ADAPTIVE),
    "rgb.bmp": lambda: Image.fromarray(_PIXELS),
    "rgb.webp": lambda: Image.fromarray(_PIXELS),
    "rgb.tif": lambda: Image.fromarray(_PIXELS),
}


def assert_size_parity(source: object) -> None:
    """The size recorded without decoding must be the size a full decode reports."""
    expected = read_image_as_pil(source).size  # type: ignore[arg-type,union-attr]
    result = PredictionResult(object_prediction_list=[], image=source)  # type: ignore[arg-type]

    assert (result.image_width, result.image_height) == expected


class TestPrediction:
    """Test cases for prediction functionality."""

    def test_prediction_score(self) -> None:
        """Test PredictionScore value and comparison operations."""
        prediction_score = PredictionScore(np.array(0.6))
        assert isinstance(prediction_score.value, float)
        assert prediction_score.is_greater_than_threshold(0.5)
        assert not prediction_score.is_greater_than_threshold(0.7)


class TestPredictionResultImage:
    """Test the source image is decoded only when its pixels are asked for.

    Sliced prediction streams a gigapixel scan a band at a time so the whole image is
    never resident, and decoding it here to read `.size` would undo that.
    """

    @pytest.fixture
    def image_path(self, tmp_path: Path) -> str:
        path = str(tmp_path / "scan.png")
        Image.fromarray(np.full((30, 40, 3), 77, dtype=np.uint8)).save(path)
        return path

    def test_dimensions_do_not_decode_the_image(self, image_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test width and height come from the header, with no decode."""
        calls: list[object] = []
        real = sahi.prediction.read_image_as_pil
        monkeypatch.setattr(
            sahi.prediction,
            "read_image_as_pil",
            lambda image, *args, **kwargs: (calls.append(image), real(image, *args, **kwargs))[1],
        )

        result = PredictionResult(object_prediction_list=[], image=image_path)

        assert (result.image_width, result.image_height) == (40, 30)
        assert calls == [], "constructing a PredictionResult must not decode the image"

    def test_image_still_returns_the_source_pixels(self, image_path: str) -> None:
        """Test `.image` returns what an eager decode would have returned."""
        result = PredictionResult(object_prediction_list=[], image=image_path)

        np.testing.assert_array_equal(np.asarray(result.image), np.full((30, 40, 3), 77, dtype=np.uint8))

    def test_image_is_decoded_only_once(self, image_path: str) -> None:
        """Test the decoded image is cached across repeated access."""
        result = PredictionResult(object_prediction_list=[], image=image_path)

        assert result.image is result.image

    def test_missing_file_still_fails_at_construction(self, tmp_path: Path) -> None:
        """Test a bad path still raises at construction rather than on first access."""
        with pytest.raises(FileNotFoundError):
            PredictionResult(object_prediction_list=[], image=str(tmp_path / "nope.png"))

    def test_url_source_is_fetched_exactly_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test a remote image is downloaded once.

        There is no header to peek at, so sizing it costs a full download and that
        download is kept rather than repeated on the first `.image` access.
        """
        calls: list[object] = []
        fetched = Image.new("RGB", (40, 30))

        def fake_fetch(image: object, *args: object, **kwargs: object) -> Image.Image:
            calls.append(image)
            return fetched

        monkeypatch.setattr(sahi.prediction, "read_image_as_pil", fake_fetch)
        monkeypatch.setattr(sahi.utils.cv, "read_image_as_pil", fake_fetch)

        result = PredictionResult(object_prediction_list=[], image="https://example.com/scan.jpg")

        assert (result.image_width, result.image_height) == (40, 30)
        assert result.image is fetched
        assert len(calls) == 1, f"expected one fetch, saw {len(calls)}"

    @pytest.mark.parametrize("as_array", [True, False])
    def test_accepts_in_memory_images(self, as_array: bool) -> None:
        """Test arrays and Pillow images, which are already decoded, keep working."""
        pixels = np.full((30, 40, 3), 12, dtype=np.uint8)
        image = pixels if as_array else Image.fromarray(pixels)

        result = PredictionResult(object_prediction_list=[], image=image)

        assert (result.image_width, result.image_height) == (40, 30)
        np.testing.assert_array_equal(np.asarray(result.image), pixels)

    def test_accepts_a_channels_first_array(self) -> None:
        """Test a CHW array is sized as the transposed image it decodes to, not as stored."""
        assert_size_parity(np.transpose(_PIXELS, (2, 0, 1)))

    def test_accepts_a_path_object(self, image_path: str) -> None:
        """Test a Path sizes the same as the equivalent str, which sizing splits on."""
        assert_size_parity(Path(image_path))

    @pytest.mark.parametrize("name", list(SIZING_FIXTURES))
    def test_reported_size_matches_a_full_decode(self, tmp_path: Path, name: str) -> None:
        """Test every container reports the size its own decode would have reported."""
        path = tmp_path / name
        SIZING_FIXTURES[name]().save(path)

        assert_size_parity(str(path))

    @pytest.mark.parametrize("suffix", [".jpg", ".tif"])
    @pytest.mark.parametrize("orientation", range(1, 9))
    def test_reported_size_matches_a_full_decode_per_orientation(
        self, tmp_path: Path, suffix: str, orientation: int
    ) -> None:
        """Test a quarter-turned image reports the turned size, as an eager decode did.

        Orientations 5-8 swap width and height, and a TIFF turns inside Pillow's plugin
        on its own terms, so the header read has to reach the same answer both ways.
        """
        path = tmp_path / f"rotated{suffix}"
        image = Image.fromarray(_PIXELS)
        exif = image.getexif()
        exif[0x0112] = orientation
        image.save(path, exif=exif)

        assert_size_parity(str(path))

    def test_image_can_be_replaced(self, image_path: str) -> None:
        """Test `.image` is still assignable, as it was before it became lazy."""
        result = PredictionResult(object_prediction_list=[], image=image_path)
        replacement = Image.fromarray(np.full((10, 20, 3), 5, dtype=np.uint8))

        result.image = replacement

        assert result.image is replacement
