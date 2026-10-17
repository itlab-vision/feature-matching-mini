from src.backend.io_adapter_base import IOAdapter


@IOAdapter.register("opencv")
class OpenCVIOAdapter(IOAdapter):
    def __init__(self, model_name, model_wrapper, logger=None, config=None):
        if config is None:
            config = {}
        self.scoreThreshold = config.pop('score_threshold', 0.1)

        super().__init__(model_name, model_wrapper, logger, config)
