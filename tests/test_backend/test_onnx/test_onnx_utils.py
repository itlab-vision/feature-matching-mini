import pytest
import onnxruntime as ort
from unittest.mock import MagicMock
from logging import Logger

from src.backend.onnx.onnx_utils import OnnxSessionConfiguration


@pytest.fixture
def mock_logger():
    return MagicMock(spec=Logger)


class TestOnnxSessionConfiguration:
    def test_set_session_options_defaults(self, mock_logger):
        config = {}
        session_config = OnnxSessionConfiguration(mock_logger, config)
        sess_options = session_config.set_session_options()

        assert sess_options.intra_op_num_threads == 0
        assert sess_options.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
        assert sess_options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    def test_set_session_options_parallel(self, mock_logger):
        config = {
            'intra_op_num_threads': 4,
            'inter_op_num_threads': 2,
            'execution_mode': 'ORT_PARALLEL'
        }
        session_config = OnnxSessionConfiguration(mock_logger, config)
        sess_options = session_config.set_session_options()

        assert sess_options.intra_op_num_threads == 4
        assert sess_options.inter_op_num_threads == 2
        assert sess_options.execution_mode == ort.ExecutionMode.ORT_PARALLEL

    def test_set_session_options_unknown_execution_mode_fallback(self, mock_logger):
        config = {'execution_mode': 'UNKNOWN_MODE'}
        session_config = OnnxSessionConfiguration(mock_logger, config)
        sess_options = session_config.set_session_options()

        assert sess_options.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
        mock_logger.warning.assert_called_once()

    def test_set_session_options_inter_threads_warning(self, mock_logger):
        config = {
            'execution_mode': 'ORT_SEQUENTIAL',
            'inter_op_num_threads': 2
        }
        session_config = OnnxSessionConfiguration(mock_logger, config)
        session_config.set_session_options()

        mock_logger.warning.assert_called_once_with(
            "Parameter num_inter_threads is only set in ORT_PARALLEL mode"
        )

    def test_parse_provider_options_empty(self, mock_logger):
        session_config = OnnxSessionConfiguration(mock_logger, {})
        result = session_config._parse_provider_options([])
        assert result == {}

    def test_parse_provider_options_types_conversion(self, mock_logger):
        option_strings = [
            "CUDAExecutionProvider.device_id=1",
            "CUDAExecutionProvider.do_copy_in_default_stream=false",
            "CUDAExecutionProvider.use_arena=true",
            "CustomProvider.nested.param=123"
        ]
        session_config = OnnxSessionConfiguration(mock_logger, {})
        parsed = session_config._parse_provider_options(option_strings)

        assert parsed['CUDAExecutionProvider']['device_id'] == 1
        assert parsed['CUDAExecutionProvider']['do_copy_in_default_stream'] is False
        assert parsed['CUDAExecutionProvider']['use_arena'] is True
        assert parsed['CustomProvider']['nested']['param'] == 123

    def test_parse_provider_options_invalid_format(self, mock_logger):
        option_strings = ["InvalidOptionWithoutEqual", "CUDAExecutionProvider.device_id=0"]
        session_config = OnnxSessionConfiguration(mock_logger, {})
        parsed = session_config._parse_provider_options(option_strings)

        assert 'InvalidOptionWithoutEqual' not in parsed
        assert parsed['CUDAExecutionProvider']['device_id'] == 0

    def test_get_providers_with_options_cpu(self, mock_logger):
        config = {'device': 'cpu'}
        session_config = OnnxSessionConfiguration(mock_logger, config)
        providers = session_config.get_providers_with_options()

        assert len(providers) == 1
        assert providers[0][0] == 'CPUExecutionProvider'

    def test_get_providers_with_options_cuda(self, mock_logger):
        config = {
            'device': 'cuda',
            'provider_option': ['CUDAExecutionProvider.device_id=1']
        }
        session_config = OnnxSessionConfiguration(mock_logger, config)
        providers = session_config.get_providers_with_options()

        assert len(providers) == 2
        assert providers[0][0] == 'CUDAExecutionProvider'
        assert providers[0][1]['device_id'] == 1
        assert providers[1][0] == 'CPUExecutionProvider'

    def test_get_providers_with_options_tensorrt(self, mock_logger):
        config = {'device': 'tensorrt'}
        session_config = OnnxSessionConfiguration(mock_logger, config)
        providers = session_config.get_providers_with_options()

        assert len(providers) == 3
        assert providers[0][0] == 'TensorrtExecutionProvider'
        assert providers[1][0] == 'CUDAExecutionProvider'
        assert providers[2][0] == 'CPUExecutionProvider'

    def test_get_providers_with_options_custom_device(self, mock_logger):
        config = {'device': 'CustomExecutionProvider'}
        session_config = OnnxSessionConfiguration(mock_logger, config)
        providers = session_config.get_providers_with_options()

        assert len(providers) == 2
        assert providers[0][0] == 'CustomExecutionProvider'
        assert providers[1][0] == 'CPUExecutionProvider'
