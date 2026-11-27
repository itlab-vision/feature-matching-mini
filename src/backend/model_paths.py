ONNX_DETECTOR_MODELS = {
    'disk': 'disk_1024.onnx',
    'aliked': 'aliked_lightglue.onnx',
    'disk_lightglue': 'disk_1024.onnx',
    'aliked_lightglue': 'aliked_lightglue.onnx',
    'superpoint_lightglue': 'superpoint_lightglue.onnx',
}
ONNX_DESCRIPTOR_MODELS = ONNX_DETECTOR_MODELS


EXECUTORCH_DETECTOR_MODELS = {
    'd2net': 'd2net_xnnpack_1x3x480x640.pte',
    'disk_lightglue': 'disk_lightglue_xnnpack_1x3x480x640.pte',
    'superpoint_lightglue': 'superpoint_lightglue_xnnpack_1x3x480x640.pte',
    'xfeat': 'xfeat_xnnpack_1x3x480x640.pte',
}
EXECUTORCH_DESCRIPTOR_MODELS = {
    'd2net': 'd2net_xnnpack_1x3x480x640.pte',
    'disk_lightglue': 'disk_lightglue_xnnpack_1x3x480x640.pte',
    'superpoint_lightglue': 'superpoint_lightglue_xnnpack_1x3x480x640.pte',
    'xfeat': 'xfeat_xnnpack_1x3x480x640.pte',
    'hardnet': 'hardnet_xnnpack_1x1x32x32.pte',
    'tfeat': 'tfeat_xnnpack_1x1x32x32.pte',
}
EXECUTORCH_MATCHER_MODELS = {
    'disk_lightglue': 'lightglue_disk_k256_xnnpack_1x3x480x640.pte',
    'superpoint_lightglue': 'lightglue_superpoint_k256_xnnpack_1x3x480x640.pte',
}
