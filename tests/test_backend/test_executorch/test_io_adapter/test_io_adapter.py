import numpy as np
import pytest
import torch

from src.backend.io_adapter_base import IOAdapter
from src.backend.executorch.io_adapter.io_adapter import ExecuTorchIOAdapter


@pytest.fixture
def adapter():
    return ExecuTorchIOAdapter(
        "executorch",
        config={"input_shape": (1, 3, 4, 6)},
    )


class TestExecuTorchIOAdapterRegistration:
    def test_registered_in_factory(self):
        assert IOAdapter._METHODS["executorch"] is ExecuTorchIOAdapter


class TestExecuTorchIOAdapterInit:
    def test_default_config(self):
        adapter = ExecuTorchIOAdapter("executorch")

        assert adapter._config == {}
        assert adapter._input_shape is None

    def test_custom_input_shape(self):
        adapter = ExecuTorchIOAdapter(
            "executorch",
            config={"input_shape": (1, 3, 240, 320)},
        )

        assert adapter._input_shape == (1, 3, 240, 320)


class TestExecuTorchIOAdapterPreprocess:
    def test_converts_bgr_to_normalized_rgb(self):
        adapter = ExecuTorchIOAdapter("executorch")
        image = np.array([[[10, 20, 30]]], dtype=np.uint8)

        result = adapter.preprocess({"image": image})

        assert result["image"].shape == (1, 3, 1, 1)
        torch.testing.assert_close(
            result["image"][0, :, 0, 0],
            torch.tensor([30.0, 20.0, 10.0]) / 255.0,
        )

    def test_resizes_to_configured_shape(self, adapter):
        image = np.zeros((2, 3, 3), dtype=np.uint8)

        result = adapter.preprocess({"image": image})

        assert result["image"].shape == (1, 3, 4, 6)

    def test_expands_grayscale_image(self, adapter):
        image = np.full((2, 3), 127, dtype=np.uint8)

        result = adapter.preprocess({"image": image})

        assert result["image"].shape == (1, 3, 4, 6)
        torch.testing.assert_close(result["image"][0, 0], result["image"][0, 1])
        torch.testing.assert_close(result["image"][0, 1], result["image"][0, 2])

    def test_preserves_extra_inputs(self, adapter):
        result = adapter.preprocess({
            "image": np.zeros((2, 3, 3), dtype=np.uint8),
            "extra": 123,
        })

        assert result["extra"] == 123

    @pytest.mark.parametrize("inputs", [None, {}, {"value": 1}])
    def test_rejects_missing_image(self, adapter, inputs):
        with pytest.raises(ValueError, match="containing 'image'"):
            adapter.preprocess(inputs)

    def test_rejects_non_numpy_image(self, adapter):
        with pytest.raises(TypeError, match="OpenCV numpy image"):
            adapter.preprocess({"image": torch.zeros(3, 4, 5)})

    def test_rejects_unsupported_image_shape(self, adapter):
        with pytest.raises(ValueError, match="Unsupported image shape"):
            adapter.preprocess({"image": np.zeros((2, 3, 4), dtype=np.uint8)})


class TestExecuTorchIOAdapterPostprocess:
    def test_preserves_dictionary(self, adapter):
        outputs = {"value": torch.tensor(1)}

        assert adapter.postprocess(outputs) is outputs

    def test_converts_list_to_tuple(self, adapter):
        outputs = [torch.tensor(1), torch.tensor(2)]

        result = adapter.postprocess(outputs)

        assert isinstance(result, tuple)
        assert result[0] is outputs[0]
        assert result[1] is outputs[1]

    def test_preserves_single_output(self, adapter):
        output = torch.tensor(1)

        assert adapter.postprocess(output) is output
