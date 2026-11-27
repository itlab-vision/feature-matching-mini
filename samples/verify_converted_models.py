import argparse
import logging
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from executorch.runtime import Runtime

sys.path.append(str(Path(__file__).parent.parent))  # noqa: E402

from models.convert_lightglue_benchmark import load_extractor  # noqa: E402
from src.backend.lightglue_utils import keypoints_to_numpy, normalize_keypoints  # noqa: E402
from src.detectors import Detector  # noqa: E402
from src.descriptors import Descriptor  # noqa: E402
import src.dnn_extractors  # noqa: E402, F401
import src.backend.torch  # noqa: E402, F401
import src.backend.io_adapter  # noqa: E402, F401
from thirdparty.lightglue.lightglue import LightGlue  # noqa: E402


MODELS_DIR = Path("models")
METHOD_SHAPES = {
    "disk": (1, 3, 1024, 1024),
    "superpoint": (1, 3, 480, 640),
    "aliked": (1, 3, 640, 640),
}
OUTPUT_NAMES = {
    "disk": ["heatmap", "descriptors"],
    "superpoint": ["score_logits", "probabilities", "descriptors"],
    "aliked": ["keypoints", "descriptors", "scores"],
}


def tensor_values(outputs):
    return [output.detach().cpu().numpy() if isinstance(output, torch.Tensor) else np.asarray(output)
            for output in outputs]


def descriptor_cosine(reference, converted):
    reference = np.asarray(reference, dtype=np.float32).reshape(-1, reference.shape[-1])
    converted = np.asarray(converted, dtype=np.float32).reshape(-1, converted.shape[-1])
    reference = reference / np.maximum(np.linalg.norm(reference, axis=-1, keepdims=True), 1e-12)
    converted = converted / np.maximum(np.linalg.norm(converted, axis=-1, keepdims=True), 1e-12)
    return float(np.sum(reference * converted, axis=-1).mean())


def compare_extractor(method):
    input_shape = METHOD_SHAPES[method]
    torch.manual_seed(0)
    image = torch.rand(input_shape, dtype=torch.float32)
    model, _ = load_extractor(f"{method}_lightglue", 512)
    model = model.eval()
    with torch.no_grad():
        reference = tensor_values(model(image))
    if method == "aliked":
        size = np.asarray([input_shape[-1] - 1, input_shape[-2] - 1], dtype=np.float32)
        reference[0] = reference[0] / size * 2 - 1

    rows = []
    for backend in ("onnx", "opencv", "executorch"):
        if backend == "onnx":
            session = ort.InferenceSession(str(MODELS_DIR / f"{method}_lightglue.onnx"),
                                           providers=["CPUExecutionProvider"])
            converted = session.run(OUTPUT_NAMES[method], {"image": image.numpy()})
        elif backend == "opencv":
            net = cv.dnn.readNet(str(MODELS_DIR / f"{method}_lightglue.onnx"))
            net.setInput(image.numpy(), name="image")
            converted = net.forward(OUTPUT_NAMES[method])
        else:
            method_handle = Runtime.get().load_program(
                str(MODELS_DIR / f"{method}_lightglue_xnnpack.pte")
            ).load_method("forward")
            converted = tensor_values(method_handle.execute((image,)))
        converted = tensor_values(converted)
        if method == "aliked" and backend == "executorch":
            size = np.asarray([input_shape[-1] - 1, input_shape[-2] - 1], dtype=np.float32)
            converted[0] = converted[0] / size * 2 - 1
        differences = [np.abs(np.asarray(a) - np.asarray(b)) for a, b in zip(reference, converted)]
        descriptor_index = 2 if method == "superpoint" else 1
        row = {
            "model_part": "extractor",
            "method": method,
            "backend": backend,
            "tensor_mean_abs": float(np.mean([diff.mean() for diff in differences])),
            "tensor_max_abs": float(max(diff.max() for diff in differences)),
            "descriptor_cosine": descriptor_cosine(reference[descriptor_index], converted[descriptor_index]),
        }
        if method in ("disk", "aliked"):
            score_index = 0 if method == "disk" else 2
            reference_scores = np.asarray(reference[score_index]).reshape(-1)
            converted_scores = np.asarray(converted[score_index]).reshape(-1)
            reference_top = set(np.argsort(-reference_scores, kind="stable")[:512])
            converted_top = set(np.argsort(-converted_scores, kind="stable")[:512])
            row["score_top512_overlap"] = len(reference_top & converted_top) / 512
        rows.append(row)
    return rows


def extract_torch_features(method, image):
    name = f"{method}_lightglue"
    config = {"backend": "torch", "device": "cpu", "nfeatures": 512}
    if method in ("sift", "doghardnet"):
        config["threshold"] = 0.0
    logger = logging.getLogger("ConvertedModelVerification")
    detector = Detector.create(name, logger, config)
    descriptor = Descriptor.create(name, logger, config)
    features = descriptor.compute(image, detector.detect(image))
    detector.close()
    descriptor.close()
    return features


def torch_matcher_inputs(features):
    count = len(features["keypoints"])
    data = {
        "keypoints": torch.from_numpy(keypoints_to_numpy(features["keypoints"])).float()[None],
        "descriptors": torch.from_numpy(np.asarray(features["descriptors"], dtype=np.float32))[None],
        "keypoint_scores": torch.from_numpy(np.asarray(features["scores"], dtype=np.float32)).reshape(1, count),
        "image_size": torch.tensor([[features["width"], features["height"]]], dtype=torch.float32),
    }
    scales = features.get("scales")
    oris = features.get("oris")
    data["scales"] = torch.ones(1, count) if scales is None else torch.as_tensor(scales).float().reshape(1, count)
    data["oris"] = torch.zeros(1, count) if oris is None else torch.as_tensor(oris).float().reshape(1, count)
    return data


def matcher_inputs(features0, features1, method, backend):
    keypoints0 = normalize_keypoints(features0["keypoints"], features0["height"], features0["width"])
    keypoints1 = normalize_keypoints(features1["keypoints"], features1["height"], features1["width"])
    inputs = {
        "kpts0": keypoints0[None].astype(np.float32),
        "kpts1": keypoints1[None].astype(np.float32),
        "desc0": np.asarray(features0["descriptors"], dtype=np.float32)[None],
        "desc1": np.asarray(features1["descriptors"], dtype=np.float32)[None],
    }
    if method in ("sift", "doghardnet"):
        for index, features in enumerate((features0, features1)):
            count = len(features["keypoints"])
            scales = features.get("scales")
            oris = features.get("oris")
            inputs[f"scales{index}"] = np.ones((1, count), np.float32) if scales is None else \
                np.asarray(scales, dtype=np.float32).reshape(1, count)
            inputs[f"oris{index}"] = np.zeros((1, count), np.float32) if oris is None else \
                np.asarray(oris, dtype=np.float32).reshape(1, count)
    return inputs


def pairs_and_scores(matches, scores):
    return {(int(pair[0]), int(pair[1])): float(score) for pair, score in zip(matches, scores)}


def compare_matcher(method, image0, image1):
    features0 = extract_torch_features(method, image0)
    features1 = extract_torch_features(method, image1)
    model = LightGlue(features=method, flash=False, depth_confidence=-1, width_confidence=-1).eval()
    with torch.no_grad():
        output = model({"image0": torch_matcher_inputs(features0), "image1": torch_matcher_inputs(features1)})
    reference = pairs_and_scores(output["matches"][0].cpu().numpy(), output["scores"][0].cpu().numpy())

    rows = []
    for backend in ("onnx", "opencv", "executorch"):
        inputs = matcher_inputs(features0, features1, method, backend)
        if backend == "onnx":
            model_path = MODELS_DIR / f"lightglue_{method}_k512.onnx"
            session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            matches, scores = session.run(["matches0", "mscores0"], inputs)
            converted = pairs_and_scores(matches, scores)
        elif backend == "opencv":
            model_name = f"lightglue_{method}_opencv.onnx"
            net = cv.dnn.readNet(str(MODELS_DIR / model_name))
            for name, value in inputs.items():
                net.setInput(value, name=name)
            matches, scores = net.forward(["matches0", "mscores0"])
            converted = pairs_and_scores(matches, scores)
        else:
            model_path = MODELS_DIR / f"lightglue_{method}_k512_xnnpack.pte"
            method_handle = Runtime.get().load_program(str(model_path)).load_method("forward")
            outputs = method_handle.execute(tuple(torch.from_numpy(value) for value in inputs.values()))
            matches0, _, scores0, _ = outputs
            valid = matches0[0] >= 0
            query = torch.arange(matches0.shape[1])[valid]
            converted = pairs_and_scores(torch.stack((query, matches0[0][valid]), dim=1).numpy(),
                                         scores0[0][valid].numpy())
        common = set(reference) & set(converted)
        union = set(reference) | set(converted)
        rows.append({
            "model_part": "matcher",
            "method": method,
            "backend": backend,
            "reference_matches": len(reference),
            "converted_matches": len(converted),
            "match_iou": len(common) / len(union) if union else 1.0,
            "match_recall": len(common) / len(reference) if reference else 1.0,
            "score_mae": float(np.mean([abs(reference[pair] - converted[pair]) for pair in common]))
            if common else 0.0,
        })
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i1", "--image1", type=Path, default=Path("test_data/20251126_131827.jpg"))
    parser.add_argument("-i2", "--image2", type=Path, default=Path("test_data/20251126_131833.jpg"))
    parser.add_argument("-o", "--output", type=Path, default=Path("results_6_11/model_consistency.csv"))
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    return parser.parse_args()


def main():
    global MODELS_DIR
    args = parse_args()
    MODELS_DIR = args.models_dir
    rows = []
    for method in ("disk", "superpoint", "aliked"):
        rows.extend(compare_extractor(method))
    image0 = cv.imread(str(args.image1))
    image1 = cv.imread(str(args.image2))
    for method in ("sift", "disk", "aliked", "doghardnet", "superpoint"):
        rows.extend(compare_matcher(method, image0, image1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.8f")
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
