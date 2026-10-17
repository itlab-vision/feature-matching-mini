from src.backend.io_adapter_base import IOAdapter
from src.backend.torch.utils import get_device


@IOAdapter.register("torch")
class TorchIOAdapter(IOAdapter):
    def __init__(self, model_name, logger=None, config=None):
        if config is None:
            config = {}

        super().__init__(model_name, logger, config)

        device = config.get('device', None)
        if device is None:
            self._device = get_device()
        else:
            self._device = device
