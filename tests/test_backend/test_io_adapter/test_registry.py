import numpy as np
import pytest

from src.backend.io_adapter import D2NetIOAdapter, XFeatIOAdapter
from src.backend.io_adapter_base import IOAdapter


@pytest.mark.parametrize('name, adapter_class', [
    ('d2net', D2NetIOAdapter),
    ('xfeat', XFeatIOAdapter),
])
def test_executorch_uses_shared_numpy_adapter(name, adapter_class):
    adapter = IOAdapter.create('executorch', name)
    assert isinstance(adapter, adapter_class)
    inputs = adapter.preprocess({'image': np.zeros((4, 6, 3), dtype=np.uint8)})
    assert isinstance(inputs['image'], np.ndarray)
