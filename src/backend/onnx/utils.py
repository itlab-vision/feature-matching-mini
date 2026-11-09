import onnxruntime as ort


class OnnxSessionConfiguration:
    _ORT_EXECUTION_MODE = {
        'ORT_SEQUENTIAL': ort.ExecutionMode.ORT_SEQUENTIAL,
        'ORT_PARALLEL': ort.ExecutionMode.ORT_PARALLEL,
    }

    def __init__(self, logger, config):
        self._logger = logger
        self._config = config

    def set_session_options(self):
        sess_options = ort.SessionOptions()

        number_threads = self._config.get('intra_op_num_threads', 0)
        execution_mode_str = self._config.get('execution_mode', 'ORT_SEQUENTIAL').upper()
        num_inter_threads = self._config.get('inter_op_num_threads', 0)

        sess_options.intra_op_num_threads = number_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if execution_mode_str in OnnxSessionConfiguration._ORT_EXECUTION_MODE:
            sess_options.execution_mode = OnnxSessionConfiguration._ORT_EXECUTION_MODE[execution_mode_str]
        else:
            self._logger.warning(f"Unknown execution mode '{execution_mode_str}', falling back to ORT_SEQUENTIAL")
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        if execution_mode_str == 'ORT_PARALLEL':
            sess_options.inter_op_num_threads = num_inter_threads
        elif num_inter_threads > 0:
            self._logger.warning("Parameter num_inter_threads is only set in ORT_PARALLEL mode")

        return sess_options

    def _parse_provider_options(self, option_strings):
        nested_dict = {}
        if not option_strings:
            return nested_dict

        for item in option_strings:
            if '=' not in item:
                continue
            key, val = item.split('=', 1)

            if val.isdigit():
                val = int(val)
            elif val.lower() == 'true':
                val = True
            elif val.lower() == 'false':
                val = False

            parts = key.split('.')
            current = nested_dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = val

        return nested_dict

    def get_providers_with_options(self):
        device = self._config.get('device', 'cpu')

        default_providers_options = {
            'CUDAExecutionProvider': {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested',
                'cudnn_conv_algo_search': 'DEFAULT',
                'do_copy_in_default_stream': True,
                'cudnn_conv_use_max_workspace': '1',
                'cudnn_conv1d_pad_to_nc1d': '1',
            },
            'TensorrtExecutionProvider': {
                'device_id': 0,
                'trt_fp16_enable': False,
                'trt_max_workspace_size': 4294967296,
            },
            'CoreMLExecutionProvider': {},
            'CPUExecutionProvider': {},
        }

        config_providers_options = self._config.get('provider_option', [])
        providers_options = default_providers_options.copy()
        config_providers_options = self._parse_provider_options(config_providers_options)
        for prov, opts in config_providers_options.items():
            if prov in providers_options:
                providers_options[prov].update(opts)
            else:
                providers_options[prov] = opts

        if device.lower() == 'cuda':
            return [
                ('CUDAExecutionProvider', providers_options.get('CUDAExecutionProvider', {})),
                ('CPUExecutionProvider', providers_options.get('CPUExecutionProvider', {}))
            ]
        elif device.lower() == 'tensorrt':
            return [
                ('TensorrtExecutionProvider', providers_options.get('TensorrtExecutionProvider', {})),
                ('CUDAExecutionProvider', providers_options.get('CUDAExecutionProvider', {})),
                ('CPUExecutionProvider', providers_options.get('CPUExecutionProvider', {}))
            ]
        elif device.lower() == 'mps':
            return [
                ('CoreMLExecutionProvider', providers_options.get('CoreMLExecutionProvider', {})),
                ('CPUExecutionProvider', providers_options.get('CPUExecutionProvider', {}))
            ]
        elif device.lower() == 'cpu':
            return [
                ('CPUExecutionProvider', providers_options.get('CPUExecutionProvider', {}))
            ]
        else:
            return [
                (device, providers_options.get(device, {})),
                ('CPUExecutionProvider', providers_options.get('CPUExecutionProvider', {}))
            ]
