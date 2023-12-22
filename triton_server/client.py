from functools import lru_cache

import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import np_to_triton_dtype


@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


def call_triton_catboost(inputs):
    triton_client = get_client()

    inputs_tensor = InferInput(
        name="INPUTS", shape=inputs.shape, datatype=np_to_triton_dtype(inputs.dtype)
    )
    inputs_tensor.set_data_from_numpy(inputs, binary_data=True)

    infer_output = InferRequestedOutput("OUTPUTS", binary_data=True)
    query_response = triton_client.infer(
        "catboost", [inputs_tensor], outputs=[infer_output]
    )
    outputs = query_response.as_numpy("OUTPUTS")
    return outputs


def main():
    inputs = np.array(
        [
            [
                1.734,
                1.313,
                1.695,
                1.263,
                1.936,
                1.159,
                1.857,
                1.889,
                1.572,
                1.886,
                1.133,
                1.149,
                1.329,
                1.894,
                1.561,
                1.143,
                1.469,
                1.449,
                1.77,
            ],
            [
                10.94,
                10.19,
                10.33,
                10.7,
                10.01,
                10.28,
                10.66,
                10.61,
                10.05,
                10.43,
                10.04,
                10.92,
                10.22,
                10.016,
                10.78,
                10.445,
                10.71,
                10.195,
                10.43,
            ],
            [
                3.885,
                3.01,
                3.332,
                3.812,
                3.93,
                3.898,
                3.291,
                3.625,
                3.336,
                3.8,
                3.674,
                3.879,
                3.592,
                3.78,
                3.264,
                3.104,
                3.045,
                3.846,
                3.72,
            ],
        ],
        dtype=np.float16,
    )

    outputs = call_triton_catboost(inputs)
    real_outputs = [121947.86687316, 626841.67834731, 101892.98425913]
    assert np.allclose(outputs, real_outputs)


if __name__ == "__main__":
    main()
