from src.backend.model_wrapper_base import ModelWrapper


@ModelWrapper.register("opencv")
class OpenCVModelLoader(ModelWrapper):
    def __init__(self, model_name, model_path=None, config=None, logger=None):
        if config is None:
            config = {}

        super().__init__(model_name=model_name, model_path=model_path, config=config, logger=logger)

    def load(self):
        pass
