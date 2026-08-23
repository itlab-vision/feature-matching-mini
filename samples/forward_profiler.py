import time
import cv2 as cv
import torch
import numpy as np
import logging
import pandas as pd
from pathlib import Path
from torch.profiler import profile, record_function, ProfilerActivity
from executorch.runtime import Runtime


logging.basicConfig(level=logging.INFO, format='[ %(levelname)s ] %(message)s')
logger = logging.getLogger("ForwardProfiler")


class ForwardProfiler:
    def __init__(self, input_layers, input_shape, warmup, iterations):
        self.input_layers = input_layers
        self.input_shape_list = self._split_shape_list(input_shape)
        logger.info(f"Shape: {self.input_shape_list}")
        self.warmup = warmup
        self.iterations = iterations

    def _split_shape_list(self, shape):
        groups = shape.split(';')
        res_list = []
        for group in groups:
            tokens = group.strip().split()
            if tokens:
                res_list.append(tuple(int(x) for x in tokens))

        return res_list

    def _save_to_csv(self, save_path, model_path, data_list):
        save_path.mkdir(parents=True, exist_ok=True)
        model_stem = Path(model_path).stem
        new_save_path = str(save_path / f"{model_stem}.csv")

        df = pd.DataFrame(data_list)
        df.to_csv(new_save_path, index=False, float_format='%.4f')
        logger.info(f"Results saved to {new_save_path}")

    def _convert_pytorch_table_to_csv(self, save_path, model_name, data_list):
        sum_self_cpu = 0

        for item in data_list:
            sum_self_cpu += getattr(item, 'self_cpu_time_total', 0)

        data_rows = []
        for item in data_list:
            cpu_total = getattr(item, 'cpu_time_total', 0)
            self_cpu = getattr(item, 'self_cpu_time_total', 0)
            cuda_total = getattr(item, 'device_time_total', 0)
            self_cuda = getattr(item, 'self_device_time_total', 0)

            count = getattr(item, 'count', 0)
            cpu_mem = getattr(item, 'cpu_memory_usage', 0)
            self_cpu_mem = getattr(item, 'self_cpu_memory_usage', 0)
            cuda_mem = getattr(item, 'device_memory_usage', 0)
            self_cuda_mem = getattr(item, 'self_device_memory_usage', 0)

            cpu_avg = cpu_total / count if count > 0 else 0
            cuda_avg = cuda_total / count if count > 0 else 0

            self_cpu_pct = (self_cpu / sum_self_cpu) * 100 if sum_self_cpu > 0 else 0
            cpu_total_pct = (cpu_total / sum_self_cpu) * 100 if sum_self_cpu > 0 else 0

            data_rows.append({
                "Name": item.key,
                "Self CPU %": round(self_cpu_pct, 4),
                "Self CPU (ms)": round(self_cpu / 1000, 4),
                "CPU Total %": round(cpu_total_pct, 4),
                "CPU total (ms)": round(cpu_total / 1000, 4),
                "CPU time avg (ms)": round(cpu_avg / 1000, 4),

                "Self CUDA (ms)": round(self_cuda / 1000, 4),
                "CUDA total (ms)": round(cuda_total / 1000, 4),
                "CUDA time avg (ms)": round(cuda_avg / 1000, 4),

                "CPU Mem (Bytes)": cpu_mem,
                "Self CPU Mem (Bytes)": self_cpu_mem,
                "CUDA Mem (Bytes)": cuda_mem,
                "Self CUDA Mem (Bytes)": self_cuda_mem,

                "# of Calls": count
            })
        self._save_to_csv(save_path, model_name, data_rows)

    def profile_opencv_dnn(self, model_path, save_path):
        net = cv.dnn.readNet(model_path)

        if len(self.input_layers) != len(self.input_shape_list):
            raise ValueError(
                f"Mismatch: {len(self.input_layers)} input names provided, "
                f"but {len(self.input_shape_list)} shapes parsed from --shape argument."
            )

        for i, name in enumerate(self.input_layers):
            shape = self.input_shape_list[i]
            data = np.random.randn(*shape).astype(np.float32)
            net.setInput(data, name=name)

        for _ in range(self.warmup):
            net.forward()

        freq = cv.getTickFrequency()
        total_network_ticks = 0
        layer_names = net.getLayerNames()
        accumulated_layer_timings = np.zeros(len(layer_names))
        for _ in range(self.iterations):
            net.forward()
            ticks, layer_timings = net.getPerfProfile()
            total_network_ticks += ticks
            accumulated_layer_timings += np.array(layer_timings)

        total_time_ms = (total_network_ticks / self.iterations / freq) * 1000
        layers_profile_result = []
        for i, ticks in enumerate(accumulated_layer_timings):
            layer_time_ms = (ticks / self.iterations / freq) * 1000
            layer_name = net.getLayer(layer_names[i]).name
            layer_type = net.getLayer(layer_names[i]).type
            layers_profile_result.append({
                "name": layer_name,
                "type": layer_type,
                "time_ms": layer_time_ms
            })

        significant_layers = [
            layer for layer in layers_profile_result
            if layer['time_ms'] > 0.1
        ]
        self._save_to_csv(save_path, model_path, significant_layers)

        return {
            "total_time_ms": total_time_ms,
            "layers_profile": significant_layers
        }

    def profile_executorch(self, model_path, save_path, debug_buffer_size):
        runtime = Runtime.get()
        program = runtime.load_program(
            Path(model_path),
            enable_etdump=True,
            debug_buffer_size=debug_buffer_size
        )
        method = program.load_method("forward")

        inputs = []
        for shape in self.input_shape_list:
            t = torch.randn(*shape, dtype=torch.float32)
            inputs.append(t)

        for _ in range(self.warmup):
            method.execute(inputs)

        start = time.perf_counter()
        for _ in range(self.iterations):
            method.execute(inputs)
        end = time.perf_counter()
        total_time_ms = ((end - start) / self.iterations) * 1000

        save_path.mkdir(parents=True, exist_ok=True)
        model_stem = Path(model_path).stem
        etdump_file = str(save_path / f"{model_stem}_etdump_output.etdp")
        debug_file = str(save_path / f"{model_stem}_debug_output.bin")
        program.write_etdump_result_to_file(etdump_file, debug_file)

        return {
            "total_time_ms": total_time_ms,
            "etdump_file": etdump_file,
            "debug_file": debug_file
        }

    def profile_pytorch(self, model, model_name, device, save_path, use_dict):
        model.eval()
        model.to(device)

        inputs = []
        for shape in self.input_shape_list:
            t = torch.randn(*shape, dtype=torch.float32, device=device)
            inputs.append(t)

        input_data = None

        if use_dict:
            nested_dict = {}
            for name, tensor in zip(self.input_layers, inputs):
                if '.' in name:
                    parent_key, child_key = name.split('.', 1)
                    if parent_key not in nested_dict:
                        nested_dict[parent_key] = {}
                    nested_dict[parent_key][child_key] = tensor
                else:
                    nested_dict[name] = tensor

            input_data = nested_dict

        with torch.no_grad():
            for _ in range(self.warmup):
                if use_dict:
                    model(input_data)
                else:
                    model(*inputs)

        activities = [ProfilerActivity.CPU]
        if device == "cuda" and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        start_time = time.perf_counter()
        with profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=False
        ) as prof:
            with record_function("model_forward"):
                for _ in range(self.iterations):
                    with torch.no_grad():
                        if use_dict:
                            model(input_data)
                        else:
                            model(*inputs)

                if device == "cuda":
                    torch.cuda.synchronize()
        end_time = time.perf_counter()
        avg_forward_ms = ((end_time - start_time) * 1000) / self.iterations

        averaged_items = prof.key_averages()
        res_table = averaged_items.table(row_limit=-1)

        self._convert_pytorch_table_to_csv(save_path, model_name, averaged_items)

        save_path.mkdir(parents=True, exist_ok=True)
        new_save_path = str(save_path / f"{model_name}_trace.json")
        prof.export_chrome_trace(new_save_path)
        return {
            "total_time_ms": avg_forward_ms,
            "trace_table": res_table,
            "trace_file": str(new_save_path)
        }
