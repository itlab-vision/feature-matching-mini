from src.backend.io_adapter.patch_descriptor import PatchDescriptorIOAdapter
from src.backend.io_adapter_base import IOAdapter


@IOAdapter.register('tfeat')
class TFeatIOAdapter(PatchDescriptorIOAdapter):
    pass
