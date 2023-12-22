import numpy as np
import triton_python_backend_utils as pb_utils
from catboost import CatBoostRegressor


class TritonPythonModel:
    def initialize(self, args):
        self.model = CatBoostRegressor()
        self.model.load_model("/assets/model.cbm")

    def infer(self, numpy_inputs):
        # inputs has categorical features
        categ_idx = self.model.get_cat_feature_indices()

        inputs = []
        for sample in numpy_inputs.tolist():
            for i in categ_idx:
                sample[i] = np.int64(sample[i])

            inputs.append(sample)
        return self.model.predict(inputs)

    def execute(self, requests):
        responses = []
        for request in requests:
            inputs = pb_utils.get_input_tensor_by_name(request, "INPUTS").as_numpy()

            outputs = self.infer(inputs)

            outputs_tensor = pb_utils.Tensor("OUTPUTS", outputs)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[outputs_tensor]
            )
            responses.append(inference_response)
        return responses
