import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))

from thirdparty.lightglue.lightglue import ALIKED, DISK, LightGlue, SuperPoint  # noqa: E402
from thirdparty.lightglue.lightglue.lightglue import filter_matches  # noqa: E402


logger = logging.getLogger("LightGlueConversion")


class SafeInstanceNorm(torch.nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, image):
        mean = image.mean(dim=(2, 3), keepdim=True)
        centered = image - mean
        variance = (centered * centered).mean(dim=(2, 3), keepdim=True)
        return centered / torch.sqrt(variance + self.eps)


def replace_instance_norm_for_export(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.InstanceNorm2d):
            setattr(module, name, SafeInstanceNorm(child.eps))
        else:
            replace_instance_norm_for_export(child)
    return module


class DiskDenseWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = replace_instance_norm_for_export(model.model)

    def forward(self, image):
        heatmaps, descriptors = self.model.heatmap_and_dense_descriptors(image)
        return heatmaps.contiguous(), descriptors.contiguous()


class SuperPointDenseWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        if image.shape[1] == 3:
            image = image[:, 0:1] * 0.2989 + image[:, 1:2] * 0.5870 + image[:, 2:3] * 0.1140

        model = self.model
        x = model.relu(model.conv1a(image))
        x = model.pool(model.relu(model.conv1b(x)))
        x = model.relu(model.conv2a(x))
        x = model.pool(model.relu(model.conv2b(x)))
        x = model.relu(model.conv3a(x))
        x = model.pool(model.relu(model.conv3b(x)))
        x = model.relu(model.conv4a(x))
        x = model.relu(model.conv4b(x))

        score_logits = model.convPb(model.relu(model.convPa(x)))
        descriptors = model.convDb(model.relu(model.convDa(x)))
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
        return score_logits, torch.softmax(score_logits, dim=1), descriptors


def deform_conv2d_export(input, weight, offset, mask=None, bias=None, stride=(1, 1), padding=(0, 0),
                         dilation=(1, 1), groups=1, offset_groups=1, use_mask=False):
    if groups != 1 or offset_groups != 1:
        raise ValueError("The export implementation expects groups=1 and offset_groups=1")
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation
    kernel_h, kernel_w = weight.shape[-2:]
    input_padded = torch.nn.functional.pad(input, (pad_w, pad_w, pad_h, pad_h))
    output_h = (input.shape[-2] + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    output_w = (input.shape[-1] + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    base_x = torch.arange(output_w, device=input.device, dtype=input.dtype) * stride_w
    base_y = torch.arange(output_h, device=input.device, dtype=input.dtype) * stride_h

    samples = []
    for kernel_y in range(kernel_h):
        for kernel_x in range(kernel_w):
            kernel_index = kernel_y * kernel_w + kernel_x
            offset_y = offset[:, 2 * kernel_index]
            offset_x = offset[:, 2 * kernel_index + 1]
            x = base_x[None, None, :] + kernel_x * dilation_w + offset_x
            y = base_y[None, :, None] + kernel_y * dilation_h + offset_y
            x = (x + 0.5) * 2 / input_padded.shape[-1] - 1
            y = (y + 0.5) * 2 / input_padded.shape[-2] - 1
            grid = torch.stack((x, y), dim=-1)
            sample = torch.nn.functional.grid_sample(
                input_padded, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )
            samples.append(sample)

    samples = torch.stack(samples, dim=1)
    if use_mask:
        samples = samples * mask.unsqueeze(2)
    weight = weight.flatten(2)
    output = torch.einsum("bkchw,ock->bohw", samples, weight)
    if bias is not None:
        output = output + bias.reshape(1, -1, 1, 1)
    return output


class AlikedFixedWrapper(torch.nn.Module):
    def __init__(self, model, normalize_output=False):
        super().__init__()
        self.model = model
        self.normalize_output = normalize_output

    def forward(self, image):
        outputs = self.model({"image": image})
        keypoints = outputs["keypoints"]
        if self.normalize_output:
            height, width = image.shape[-2:]
            size = torch.tensor([width - 1, height - 1], device=image.device, dtype=image.dtype)
            keypoints = keypoints / size * 2 - 1
        return (keypoints.contiguous(), outputs["descriptors"].contiguous(),
                outputs["keypoint_scores"].contiguous())


class LightGlueFixedWrapper(torch.nn.Module):
    def __init__(self, model, add_scale_ori):
        super().__init__()
        self.model = model
        self.add_scale_ori = add_scale_ori
        for layer in self.model.transformers:
            layer.self_attn.inner_attn.has_sdp = False

    def _assignment_scores(self, descriptors0, descriptors1):
        assignment = self.model.log_assignment[-1]
        projected0 = assignment.final_proj(descriptors0)
        projected1 = assignment.final_proj(descriptors1)
        dimension = projected0.shape[-1]
        projected0 = projected0 / dimension**0.25
        projected1 = projected1 / dimension**0.25
        similarity = torch.matmul(projected0, projected1.permute(0, 2, 1))
        matchability0 = assignment.matchability(descriptors0)
        matchability1 = assignment.matchability(descriptors1)

        certainties = torch.nn.functional.logsigmoid(matchability0) + \
            torch.nn.functional.logsigmoid(matchability1).permute(0, 2, 1)
        scores0 = torch.nn.functional.log_softmax(similarity, 2)
        scores1 = torch.nn.functional.log_softmax(similarity.permute(0, 2, 1).contiguous(), 2).permute(0, 2, 1)
        main_scores = scores0 + scores1 + certainties
        dustbin0 = torch.nn.functional.logsigmoid(-matchability0.squeeze(-1)).unsqueeze(-1)
        dustbin1 = torch.nn.functional.logsigmoid(-matchability1.squeeze(-1)).unsqueeze(1)
        top = torch.cat((main_scores, dustbin0), dim=2)
        bottom = torch.cat((dustbin1, similarity.new_zeros(similarity.shape[0], 1, 1)), dim=2)
        return torch.cat((top, bottom), dim=1)

    def forward(self, keypoints0, keypoints1, descriptors0, descriptors1,
                scales0=None, oris0=None, scales1=None, oris1=None):
        if self.add_scale_ori:
            keypoints0 = torch.cat((keypoints0, scales0.unsqueeze(-1), oris0.unsqueeze(-1)), dim=-1)
            keypoints1 = torch.cat((keypoints1, scales1.unsqueeze(-1), oris1.unsqueeze(-1)), dim=-1)

        descriptors0 = self.model.input_proj(descriptors0.contiguous())
        descriptors1 = self.model.input_proj(descriptors1.contiguous())
        encoding0 = self.model.posenc(keypoints0)
        encoding1 = self.model.posenc(keypoints1)

        for layer in self.model.transformers:
            descriptors0, descriptors1 = layer(descriptors0, descriptors1, encoding0, encoding1)

        scores = self._assignment_scores(descriptors0, descriptors1)
        return filter_matches(scores, self.model.conf.filter_threshold)


class LightGluePairwiseWrapper(torch.nn.Module):
    def __init__(self, model, add_scale_ori):
        super().__init__()
        self.model = LightGlueFixedWrapper(model, add_scale_ori)

    def forward(self, keypoints0, keypoints1, descriptors0, descriptors1,
                scales0=None, oris0=None, scales1=None, oris1=None):
        matches0, _, scores0, _ = self.model(
            keypoints0, keypoints1, descriptors0, descriptors1, scales0, oris0, scales1, oris1
        )
        valid = matches0[0] > -1
        query_indices = torch.where(valid)[0]
        train_indices = matches0[0][valid]
        return torch.stack((query_indices, train_indices), dim=1), scores0[0][valid]


def load_extractor(model_name, num_keypoints, normalize_output=False):
    if model_name == "disk_lightglue":
        return DiskDenseWrapper(DISK(max_num_keypoints=num_keypoints)), "disk_lightglue"
    if model_name == "superpoint_lightglue":
        return SuperPointDenseWrapper(SuperPoint(max_num_keypoints=num_keypoints)), "superpoint_lightglue"
    if model_name == "aliked_lightglue":
        model = ALIKED(model_name="aliked-n16", max_num_keypoints=num_keypoints, detection_threshold=-1)
        return AlikedFixedWrapper(model, normalize_output), "aliked_lightglue"
    raise ValueError(f"Unknown extractor '{model_name}'")


def load_lightglue(features, num_keypoints, pairwise):
    model = LightGlue(features=features, flash=False, depth_confidence=-1, width_confidence=-1).eval()
    wrapper_class = LightGluePairwiseWrapper if pairwise else LightGlueFixedWrapper
    wrapper = wrapper_class(model, features in ("sift", "doghardnet"))
    return wrapper, f"lightglue_{features}_k{num_keypoints}"


def build_lightglue_inputs(features, num_keypoints):
    descriptor_dim = 256 if features == "superpoint" else 128
    inputs = [
        torch.randn(1, num_keypoints, 2),
        torch.randn(1, num_keypoints, 2),
        torch.randn(1, num_keypoints, descriptor_dim),
        torch.randn(1, num_keypoints, descriptor_dim),
    ]
    names = ["kpts0", "kpts1", "desc0", "desc1"]
    if features in ("sift", "doghardnet"):
        inputs.extend((torch.ones(1, num_keypoints), torch.zeros(1, num_keypoints),
                       torch.ones(1, num_keypoints), torch.zeros(1, num_keypoints)))
        names.extend(("scales0", "oris0", "scales1", "oris1"))
    return tuple(inputs), names


def load_model(model_name, input_shape, num_keypoints, matcher_features, pairwise, normalize_output=False):
    if model_name == "lightglue":
        return load_lightglue(matcher_features, num_keypoints, pairwise)
    return load_extractor(model_name, num_keypoints, normalize_output)


def export_onnx(model, output_stem, model_name, inputs, input_names, output_dir):
    output_names = {
        "disk_lightglue": ["heatmap", "descriptors"],
        "superpoint_lightglue": ["score_logits", "probabilities", "descriptors"],
        "aliked_lightglue": ["keypoints", "descriptors", "scores"],
        "lightglue": ["matches0", "mscores0"],
    }[model_name]
    dynamic_axes = None
    if model_name == "lightglue":
        dynamic_axes = {"matches0": {0: "num_matches"}, "mscores0": {0: "num_matches"}}
        for name in input_names:
            axis = 0 if name.endswith("0") else 1
            dynamic_axes[name] = {1: f"num_keypoints{axis}"}

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_stem}.onnx"
    torch.onnx.export(
        model,
        inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=True,
    )
    logger.info("Saved ONNX model: %s", output_path.resolve())
    return output_path


def export_executorch(model, output_stem, inputs, output_dir, partitioner_name):
    from executorch.exir import to_edge_transform_and_lower

    partitioners = None
    if partitioner_name == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        partitioners = [XnnpackPartitioner(per_op_mode=True)]

    exported_program = torch.export.export(model, inputs, strict=True)
    edge_program = to_edge_transform_and_lower(exported_program, partitioner=partitioners)
    executorch_program = edge_program.to_executorch()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_stem}_{partitioner_name}.pte"
    with output_path.open("wb") as output_file:
        executorch_program.write_to_file(output_file)
    logger.info("Saved ExecuTorch model: %s", output_path.resolve())
    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True,
                        choices=("disk_lightglue", "superpoint_lightglue", "aliked_lightglue", "lightglue"))
    parser.add_argument("--format", required=True, choices=("onnx", "executorch"))
    parser.add_argument("--input-shape", required=True, type=int, nargs=4, metavar=("N", "C", "H", "W"))
    parser.add_argument("--num-keypoints", type=int, default=512)
    parser.add_argument("--matcher-features", default="superpoint",
                        choices=("superpoint", "disk", "aliked", "sift", "doghardnet"))
    parser.add_argument("--partitioner", default="none", choices=("none", "xnnpack"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="[ %(levelname)s ] %(message)s")
    args = parse_args()
    pairwise = args.model_name == "lightglue" and args.format == "onnx"
    if args.model_name == "aliked_lightglue":
        import torchvision
        torchvision.ops.deform_conv2d = deform_conv2d_export
    model, output_stem = load_model(
        args.model_name, args.input_shape, args.num_keypoints, args.matcher_features, pairwise,
        normalize_output=args.format == "onnx"
    )
    model = model.cpu().eval()

    if args.model_name == "lightglue":
        inputs, input_names = build_lightglue_inputs(args.matcher_features, args.num_keypoints)
    else:
        inputs = (torch.randn(tuple(args.input_shape), dtype=torch.float32),)
        input_names = ["image"]

    if args.format == "onnx":
        export_onnx(model, output_stem, args.model_name, inputs, input_names, args.output_dir)
    else:
        export_executorch(model, output_stem, inputs, args.output_dir, args.partitioner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
