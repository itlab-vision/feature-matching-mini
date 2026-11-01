from src.backend.model_wrapper_base import ModelWrapper


ONNX_DETECTOR_MODELS = {
    'disk': 'disk_1024.onnx',
    'aliked': 'aliked-n32-top2k-640.onnx',
}
ONNX_DESCRIPTOR_MODELS = {
    'disk': 'disk_1024.onnx',
    'aliked': 'aliked-n32-top2k-640.onnx',
}
ONNX_MATCHER_MODELS = {
    'disk': 'disk_lightglue_2outputs.onnx',
    'aliked': 'lightglue_for_aliked.onnx',
}


@ModelWrapper.register("onnx")
class OnnxModelLoader(ModelWrapper):
    def __init__(self, model_name, model_path=None, config=None, logger=None):
        if config is None:
            config = {}

        super().__init__(model_name=model_name, model_path=model_path, config=config, logger=logger)

    def load(self):
        pass
