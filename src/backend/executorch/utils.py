import torch
import torch.nn.functional as F


def nms_topk(scores, num_keypoints, radius=4):
    if scores.ndim != 2:
        raise ValueError(f"Expected an HxW score map, got {tuple(scores.shape)}")

    height, width = scores.shape
    if not 0 < num_keypoints <= height * width:
        raise ValueError(f"num_keypoints must be in [1, {height * width}], got {num_keypoints}")

    local_max = F.max_pool2d(scores[None, None], kernel_size=radius * 2 + 1,
                             stride=1, padding=radius)[0, 0]
    filtered = scores.masked_fill(scores != local_max, float("-inf"))
    values, indices = torch.topk(filtered.reshape(-1), num_keypoints)
    keypoints = torch.stack((indices % width, torch.div(indices, width, rounding_mode="floor")), dim=1)

    return keypoints.to(torch.float32), values


def sample_descriptors(dense, keypoints, image_height, image_width):
    if dense.ndim != 4 or dense.shape[0] != 1:
        raise ValueError(f"Expected descriptors shaped (1, C, H, W), got {tuple(dense.shape)}")

    x = keypoints[:, 0] / max(image_width - 1, 1) * 2 - 1
    y = keypoints[:, 1] / max(image_height - 1, 1) * 2 - 1
    grid = torch.stack((x, y), dim=-1).reshape(1, 1, -1, 2)

    descriptors = F.grid_sample(dense, grid, align_corners=True)[0, :, 0].transpose(0, 1)
    return F.normalize(descriptors, p=2, dim=1)


def prepare_image_tensor(image, input_shape=None):
    if not isinstance(image, torch.Tensor):
        raise TypeError("Expected image to be a torch.Tensor")

    if image.ndim == 3:
        image = image.unsqueeze(0)

    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"Expected CHW or [1, C, H, W] image, got {tuple(image.shape)}")

    image = image.to(dtype=torch.float32).cpu()

    if input_shape is not None:
        if len(input_shape) != 4 or input_shape[0] != 1:
            raise ValueError("input_shape must have the form [1, C, H, W]")

        _, channels, height, width = input_shape
        if image.shape[1] != channels:
            raise ValueError(f"Model expects {channels} channels, got {image.shape[1]}")

        if image.shape[-2:] != (height, width):
            image = F.interpolate(image, size=(height, width), mode="bilinear", align_corners=False)

    return image
