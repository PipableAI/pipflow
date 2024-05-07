import json
import requests
from pip_flow.models.device import Device
from transformers import AutoModelForCausalLM, AutoTokenizer


INFERENCE_URL = "https://playground.pipable.ai/infer"


class PipFlow:
    def __init__(
        self,
        model_key: str,
        device: str,
        url: str = INFERENCE_URL,
    ):
        self.device = Device(device)
        self.model_key = model_key
        self.model = None
        self.tokenizer = None
        self.url = url
        if self.device == Device.CLOUD:
            self.url = url
        else:
            self._load_model()

    def generate(
        self, prompt: str, max_new_tokens: int = 500, eos_token: str = "doc"
    ) -> str:
        if self.device == Device.CLOUD:
            payload = {
                "model_name": self.model_key,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
            }
            response = requests.request(
                method="POST", url=self.url, data=payload, timeout=120
            )
            if response.status_code == 200:
                response = json.loads(response.text)["response"]
            else:
                raise Exception(f"Error generating response using {self.url}.")
        elif self.device == Device.CUDA:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device.value)
            print(inputs.input_ids.shape[-1])
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        response = response.split(f"<{eos_token}>")[1].split(f"</{eos_token}>")[0]
        return response

    def _load_model(self):
        if self.model is None or self.tokenizer is None:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_key).to(
                self.device.value
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_key)
