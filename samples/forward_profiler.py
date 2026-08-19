import time
import cv2 as cv
import torch
import numpy as np
from pathlib import Path
from torch.profiler import profile, record_function, ProfilerActivity
from executorch.runtime import Runtime


class ForwardProfiler:
    def __init__(self, input_shape, warmup, iterations):
        self.input_shape = input_shape
        self.warmup = warmup
        self.iterations = iterations

    def profile_opencv_dnn(self, model_path):
        net = cv.dnn.readNet(model_path)

        blob = cv.dnn.blobFromImage(
            np.random.randint(0, 256, self.input_shape, dtype=np.uint8),
            scalefactor=1.0 / 255.0,
            size=(self.input_shape[1], self.input_shape[0]),
            swapRB=True
        )
        net.setInput(blob)

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
        return {
            "total_time_ms": total_time_ms,
            "layers_profile": layers_profile_result
        }

    def profile_executorch(self, model_path, save_path, debug_buffer_size):
        runtime = Runtime.get()
        program = runtime.load_program(
            Path(model_path),
            enable_etdump=True,
            debug_buffer_size=debug_buffer_size
        )
        method = program.load_method("forward")
        tmp_input = torch.randn(self.input_shape)

        for _ in range(self.warmup):
            method.execute([tmp_input])

        start = time.perf_counter()
        for _ in range(self.iterations):
            method.execute([tmp_input])
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

    def profile_pytorch(self, model, model_name, device, save_path):
        model.eval()
        model.to(device)
        tmp_input = torch.randn(self.input_shape, device=device)

        with torch.no_grad():
            for _ in range(self.warmup):
                model(tmp_input)

        activities = [ProfilerActivity.CPU]
        if device == "cuda" and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        with profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=False
        ) as prof:
            with record_function("model_forward"):
                for _ in range(self.iterations):
                    with torch.no_grad():
                        model(tmp_input)

                if device == "cuda":
                    torch.cuda.synchronize()

        avg = prof.key_averages().total_average()
        if device == "cuda" and torch.cuda.is_available():
            total_time_us = avg.cuda_time_total
        else:
            total_time_us = avg.cpu_time_total

        avg_forward_ms = (total_time_us / 1e3) / self.iterations

        averaged_items = prof.key_averages()
        for item in averaged_items:
            item.cpu_time_total /= self.iterations
            item.cuda_time_total /= self.iterations
        res_table = averaged_items.table(row_limit=-1)

        save_path.mkdir(parents=True, exist_ok=True)
        new_save_path = str(save_path / f"{model_name}_trace.json")
        prof.export_chrome_trace(new_save_path)
        return {
            "total_time_ms": avg_forward_ms,
            "trace_table": res_table,
            "trace_file": str(new_save_path)
        }
