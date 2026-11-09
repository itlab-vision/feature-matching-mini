from src.backend.io_adapter.loftr import LoFTRIOAdapter
from src.backend.io_adapter_base import IOAdapter


@IOAdapter.register("roma")
class RoMaIOAdapter(LoFTRIOAdapter):
    pass
