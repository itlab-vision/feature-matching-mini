import pytest
import torch

from src.backend.executorch.utils import nms_topk, prepare_image_tensor, sample_descriptors


class TestNmsTopK:
    def test_returns_strongest_local_maximum(self):
        scores = torch.tensor([
            [0.0, 0.1, 0.0],
            [0.2, 0.9, 0.3],
            [0.0, 0.4, 0.0],
        ])

        keypoints, values = nms_topk(scores, num_keypoints=1, radius=1)

        torch.testing.assert_close(keypoints, torch.tensor([[1.0, 1.0]]))
        torch.testing.assert_close(values, torch.tensor([0.9]))

    def test_returns_xy_coordinates(self):
        scores = torch.zeros(3, 4)
        scores[2, 3] = 1.0

        keypoints, _ = nms_topk(scores, num_keypoints=1, radius=1)

        torch.testing.assert_close(keypoints, torch.tensor([[3.0, 2.0]]))

    @pytest.mark.parametrize("shape", [(1,), (1, 2, 3)])
    def test_rejects_non_spatial_scores(self, shape):
        with pytest.raises(ValueError, match="Expected an HxW score map"):
            nms_topk(torch.zeros(shape), num_keypoints=1)

    @pytest.mark.parametrize("num_keypoints", [0, 10])
    def test_rejects_invalid_number_of_keypoints(self, num_keypoints):
        with pytest.raises(ValueError, match="num_keypoints must be"):
            nms_topk(torch.zeros(3, 3), num_keypoints=num_keypoints)


class TestSampleDescriptors:
    def test_samples_and_normalizes_descriptors(self):
        dense = torch.tensor([[[[3.0, 0.0], [0.0, 3.0]], [[4.0, 5.0], [5.0, 4.0]]]])
        keypoints = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

        descriptors = sample_descriptors(dense, keypoints, image_height=2, image_width=2)

        assert descriptors.shape == (2, 2)
        torch.testing.assert_close(descriptors.norm(dim=1), torch.ones(2))
        torch.testing.assert_close(descriptors[0], torch.tensor([0.6, 0.8]))

    def test_rejects_invalid_dense_shape(self):
        with pytest.raises(ValueError, match="Expected descriptors shaped"):
            sample_descriptors(
                torch.zeros(2, 4, 3, 3),
                torch.zeros(1, 2),
                image_height=3,
                image_width=3,
            )


class TestPrepareImageTensor:
    def test_adds_batch_dimension(self):
        result = prepare_image_tensor(torch.ones(3, 4, 5))

        assert result.shape == (1, 3, 4, 5)

    def test_converts_dtype_and_device(self):
        result = prepare_image_tensor(torch.ones(3, 4, 5, dtype=torch.float64))

        assert result.dtype == torch.float32
        assert result.device.type == "cpu"

    def test_resizes_to_configured_shape(self):
        result = prepare_image_tensor(torch.ones(3, 4, 5), (1, 3, 8, 10))

        assert result.shape == (1, 3, 8, 10)

    def test_rejects_non_tensor(self):
        with pytest.raises(TypeError, match="Expected image to be a torch.Tensor"):
            prepare_image_tensor("image")

    @pytest.mark.parametrize("image", [torch.zeros(3, 4), torch.zeros(2, 3, 4, 5)])
    def test_rejects_invalid_image_shape(self, image):
        with pytest.raises(ValueError, match="Expected CHW"):
            prepare_image_tensor(image)

    @pytest.mark.parametrize("input_shape", [(3, 4, 5), (2, 3, 4, 5)])
    def test_rejects_invalid_input_shape(self, input_shape):
        with pytest.raises(ValueError, match="input_shape must have"):
            prepare_image_tensor(torch.zeros(3, 4, 5), input_shape)

    def test_rejects_channel_mismatch(self):
        with pytest.raises(ValueError, match="Model expects 1 channels"):
            prepare_image_tensor(torch.zeros(3, 4, 5), (1, 1, 4, 5))
