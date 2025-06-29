import json
import numpy as np
import tensorflow as tf

def build_full_model_trace(interpreter, model_json, sample_input):
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], sample_input.astype(input_details[0]['dtype']))
    interpreter.invoke()

    tensors = model_json["subgraphs"][0]["tensors"]
    operators = model_json["subgraphs"][0]["operators"]
    opcodes = model_json["operator_codes"]

    full_trace = []

    for op_idx, op in enumerate(operators):
        entry = {}
        opcode_index = op["opcode_index"]
        op_type = opcodes[opcode_index]["builtin_code"]
        input_idxs = op.get("inputs", [])
        output_idxs = op.get("outputs", [])

        entry["op_index"] = op_idx
        entry["op_type"] = op_type
        entry["inputs"] = []
        entry["outputs"] = []

        for i in input_idxs:
            if i == -1:
                continue
            tensor_info = tensors[i]
            try:
                value = interpreter.get_tensor(i)
                sample_vals = value
            except:
                sample_vals = "Unavailable"

            q = tensor_info.get("quantization", {})
            scales = q.get("scale", [])
            zero_points = q.get("zero_point", [])

            entry["inputs"].append({
                "name": tensor_info.get("name", f"tensor_{i}"),
                "index": i,
                "dtype": tensor_info.get("type", "UNKNOWN"),
                "shape": tensor_info.get("shape", []),
                "quantization": {
                    "scale": scales,
                    "zero_point": zero_points
                },
                "values": sample_vals
            })

        for o in output_idxs:
            tensor_info = tensors[o]
            try:
                value = interpreter.get_tensor(o)
                sample_vals = value
            except:
                sample_vals = "Unavailable"

            q = tensor_info.get("quantization", {})
            scales = q.get("scale", [])
            zero_points = q.get("zero_point", [])

            entry["outputs"].append({
                "name": tensor_info.get("name", f"tensor_{o}"),
                "index": o,
                "dtype": tensor_info.get("type", "UNKNOWN"),
                "shape": tensor_info.get("shape", []),
                "quantization": {
                    "scale": scales,
                    "zero_point": zero_points
                },
                "values": sample_vals
            })

        full_trace.append(entry)

    return full_trace

# Example usage (unchanged from your snippet)
interpreter = tf.lite.Interpreter(model_path="model_int8.tflite", experimental_preserve_all_tensors=True)
x_q = np.round(x_sample / input_scale + input_zp).astype(np.int8)
x_q = np.clip(x_q, -128, 127)
x_q = x_q.reshape(1, 70, 567, 1)

with open("model_int8.json") as f:
    model_json = json.load(f)

full_trace = build_full_model_trace(interpreter=interpreter, model_json=model_json, sample_input=x_q)
