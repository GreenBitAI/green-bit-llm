from vllm import LLM
from .base import BaseInferenceBackend

class VLLMInferenceBackend(BaseInferenceBackend):
    def __init__(self, model_path, **kwargs):
        self.model = LLM(model_path, **kwargs)
        
    def do_generate(self, prompt, params):
        outputs = self.model.generate(prompt, params)
        return outputs

    def generate(self, prompt, params=None):
        if isinstance(prompt, str):
            prompt = [prompt]
        outputs = self.do_generate(prompt, params)
        for i,output in enumerate(outputs):
            print("Prompt:",prompt[i])
            print("Generated text:",output.outputs[0].text)