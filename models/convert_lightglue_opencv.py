import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

sys.path.append(str(Path(__file__).parent.parent))

from thirdparty.lightglue.lightglue import LightGlue  # noqa: E402


def parameter_key(model, tensor, state):
    if tensor.name in state:
        return tensor.name
    consumers = [node for node in model.graph.node if tensor.name in node.input and node.op_type == "MatMul"]
    if not consumers:
        raise ValueError(f"No MatMul consumer for {tensor.name}")
    parts = consumers[0].name.strip("/").split("/")
    if parts[-1] == "MatMul":
        parts.pop()
    key = ".".join(parts) + ".weight"
    key = key.replace(".ffn.ffn.", ".ffn.")
    if key not in state and key != "input_proj.weight":
        raise ValueError(f"Cannot map {tensor.name} to {key}")
    return key


def replace_weights(model, features):
    lightglue = LightGlue(features=features, flash=False, depth_confidence=-1, width_confidence=-1)
    state = {key: value.detach().cpu().numpy() for key, value in lightglue.state_dict().items()}
    initializers = []
    for tensor in model.graph.initializer:
        key = "input_proj.bias" if tensor.name == "input_proj.bias" else parameter_key(model, tensor, state)
        if features == "superpoint" and key.startswith("input_proj."):
            value = np.zeros(256, dtype=np.float32) if key == "input_proj.bias" else np.eye(256, dtype=np.float32)
        else:
            value = state[key]
        if tensor.name.startswith("onnx::MatMul_"):
            converted = value.T
        elif tuple(value.shape) == tuple(tensor.dims):
            converted = value
        elif tuple(value.T.shape) == tuple(tensor.dims):
            converted = value.T
        else:
            raise ValueError(f"Shape mismatch for {key}: {value.shape} vs {tensor.dims}")
        initializers.append(numpy_helper.from_array(np.ascontiguousarray(converted), tensor.name))
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(initializers)


def set_descriptor_dimension(model, dimension):
    for graph_input in model.graph.input:
        if graph_input.name in ("desc0", "desc1"):
            graph_input.type.tensor_type.shape.dim[2].Clear()
            graph_input.type.tensor_type.shape.dim[2].dim_value = dimension


def add_scale_orientation(model):
    axes = numpy_helper.from_array(np.asarray([2], dtype=np.int64), "scale_ori_unsqueeze_axes")
    model.graph.initializer.extend([axes])
    for index in (0, 1):
        for name in ("scales", "oris"):
            graph_input = helper.make_tensor_value_info(
                f"{name}{index}", onnx.TensorProto.FLOAT, [1, f"num_keypoints{index}"]
            )
            model.graph.input.extend([graph_input])

    nodes = []
    for node in model.graph.node:
        if node.op_type == "MatMul" and node.name in ("/posenc/Wr/MatMul", "/posenc/Wr_1/MatMul"):
            index = 0 if node.name == "/posenc/Wr/MatMul" else 1
            inputs = []
            for name in ("scales", "oris"):
                output = f"{name}{index}_3d"
                nodes.append(helper.make_node("Unsqueeze", [f"{name}{index}", axes.name], [output],
                                              name=f"/{name}{index}/Unsqueeze"))
                inputs.append(output)
            concatenated = f"kpts{index}_scale_ori"
            nodes.append(helper.make_node("Concat", [node.input[0], *inputs], [concatenated],
                                          name=f"/posenc{index}/ConcatScaleOri", axis=2))
            node.input[0] = concatenated
        nodes.append(node)
    model.graph.ClearField("node")
    model.graph.node.extend(nodes)


def make_empty_matches_safe(model):
    nodes = {node.name: node for node in model.graph.node}
    query_indices = nodes["/Unsqueeze"].output[0]
    train_indices = nodes["/Where_1"].output[0]
    scores = nodes["/Where"].output[0]
    axes = numpy_helper.from_array(np.asarray([2], dtype=np.int64), "empty_safe_unsqueeze_axes")
    model.graph.initializer.append(axes)

    for node in model.graph.node:
        for index, output in enumerate(node.output):
            if output in ("matches0", "mscores0"):
                node.output[index] = f"packed_{output}"

    model.graph.node.extend([
        helper.make_node("Unsqueeze", [query_indices, axes.name], ["empty_safe_query_indices"],
                         name="/EmptySafe/UnsqueezeQuery"),
        helper.make_node("Unsqueeze", [train_indices, axes.name], ["empty_safe_train_indices"],
                         name="/EmptySafe/UnsqueezeTrain"),
        helper.make_node("Concat", ["empty_safe_query_indices", "empty_safe_train_indices"], ["matches0"],
                         name="/EmptySafe/Concat", axis=2),
        helper.make_node("Identity", [scores], ["mscores0"], name="/EmptySafe/Scores"),
    ])

    output_shapes = {"matches0": [1, "num_keypoints0", 2], "mscores0": [1, "num_keypoints0"]}
    for graph_output in model.graph.output:
        dimensions = output_shapes[graph_output.name]
        graph_output.type.tensor_type.shape.ClearField("dim")
        for dimension in dimensions:
            dim = graph_output.type.tensor_type.shape.dim.add()
            if isinstance(dimension, str):
                dim.dim_param = dimension
            else:
                dim.dim_value = dimension

    producers = {output: index for index, node in enumerate(model.graph.node) for output in node.output}
    pending = [graph_output.name for graph_output in model.graph.output]
    required_indices = set()
    while pending:
        output = pending.pop()
        index = producers.get(output)
        if index is None or index in required_indices:
            continue
        required_indices.add(index)
        pending.extend(model.graph.node[index].input)
    required_nodes = [node for index, node in enumerate(model.graph.node) if index in required_indices]
    model.graph.ClearField("node")
    model.graph.node.extend(required_nodes)


def convert(features, output_dir):
    model = onnx.load("models/disk_lightglue_2outputs.onnx")
    replace_weights(model, features)
    set_descriptor_dimension(model, 256 if features == "superpoint" else 128)
    if features in ("sift", "doghardnet"):
        add_scale_orientation(model)
    make_empty_matches_safe(model)
    onnx.checker.check_model(model)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"lightglue_{features}_opencv.onnx"
    onnx.save(model, str(output_path), save_as_external_data=False)
    print(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, choices=("superpoint", "disk", "aliked", "sift", "doghardnet"))
    parser.add_argument("--output-dir", type=Path, default=Path("results_6_11/models"))
    args = parser.parse_args()
    convert(args.features, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
