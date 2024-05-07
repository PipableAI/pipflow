import inspect
import json
import sys
from typing import List
import requests
from pip_flow.models.device import Device
from transformers import AutoModelForCausalLM, AutoTokenizer

from pip_flow.models.function import Function
from pip_flow.models.plan import Plan


INFERENCE_URL = "https://playground.pipable.ai/infer"


class modified_dict(dict):
    def __missing__(self, key):
        return f"{{{key}}}"


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
        self.functions: List[Function] = []
        self.prompt_templates = {}
        self.live_prompt = None
        self.last_question = None
        self.last_plan = None

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

    def _add_function(self, signature: str, docs: str, name: str, full_name: str):
        """
        A function to add a new function to the planner functions list if it is not already present.

        Args:
            signature (str): The signature of the function.
            docs (str): Documentation of the function.
            name (str): The name of the function.
            full_name (str): The full name of the function along with module.
        Returns:
            None
        """
        try:
            func = Function(
                signature=signature, docs=docs, name=name, full_name=full_name
            )
            if func not in self.functions:
                self.functions.append(func)
        except Exception as e:
            raise Exception(f"Unable to add function {name} with error {e}.")

    def generate_docs(self, function: callable, max_new_tokens=500) -> str:
        prompt = f"""
<function_code>
{inspect.getsource(function)}
</function_code>
<question>
Document the function above giving the function description , parameter name and description , dtypes , possible param values, default param value and return type.
</question>
<doc>
"""
        try:
            response = self.generate(prompt, max_new_tokens, eos_token="doc")
            return response
        except Exception as e:
            raise ValueError(f"Unable to generate the code with error: {e}") from e

    def register_callables(self, functions: List[callable], generate_docs=False):
        """
        Registers a list of callable functions with the planner.

        Args:
            functions (List[callable]): A list of callable functions to be registered.
            generate_docs = False (bool): Whether to generate documentation for the functions using the LLM. Defaults to False.
        Raises:
            Exception: If there is an error while registering a function. The exception message will include the name of the function and the error message.

        Returns:
            None
        """
        for function in functions:
            try:
                signature = str(inspect.signature(function))
                docs = function.__doc__
                if generate_docs:
                    try:
                        docs = self.generate_docs(function)
                    except Exception as e:
                        print(
                            f"Unable to generate docs for function using model. Error :{e}"
                        )
                name = function.__name__
                full_name = function.__module__ + "." + function.__qualname__
                self._add_function(signature, docs, name, full_name)
            except Exception as e:
                print(f"Unable to register function {function} with error {e}.")

    def add_plan_template(self, key: str, base_prompt: str):
        self.prompt_templates[key] = base_prompt

    def make_live_prompt(self, key: str, config: dict):
        base_prompt = self.prompt_templates[key]
        if "question" in config:
            self.last_question = config["question"]
        live_prompt = base_prompt.format_map(modified_dict(**config))
        self.live_prompt = live_prompt
        return live_prompt

    def generate_plan(self, max_new_tokens: int = 900) -> Plan:
        try:

            live_prompt = self.live_prompt
            response = self.generate(live_prompt, max_new_tokens, "json")
            response = response.replace("None", "null")
        except Exception as e:
            raise ValueError(f"Unable to generate the plan with error: {e}") from e
        try:
            plan = Plan.model_validate_json(response)
        except Exception as e:
            raise ValueError(
                f"Unable to parse the response: {response} with error: {e}"
            ) from e
        self.last_plan = plan
        return plan

    def plan_to_code(self, max_new_tokens=600) -> str:
        plan = self.last_plan
        names = []
        for tasks in plan.tasks:
            names.append(tasks.function_name)

        full_names = [x.full_name for x in self.functions if x.name in names]
        prompt = f"""
<json>
{str(plan)}
</json>
<instructions>
- Use exact values of parameters in the function calls as given in the tasks of the plan.
- Make sure that constants and the values of the parameters in the task are used in the code.
- Also use imports wherever necessary.
- Add proper comments above each code line.
- Assign the values to the variables you are using.
</instructions>
<question>
Given the above plan, Just return a small python code that executes the plan using just these exact function calls provided in the plan.
The question to resolve:
{self.last_question}
Functions to use:
{full_names}
</question>
<response>
"""
        try:
            response = self.generate(prompt, max_new_tokens, eos_token="response")
            if "ipykernel" in sys.modules:
                from IPython.display import display, Markdown

                display(Markdown(response))
            else:
                return response
        except Exception as e:
            raise ValueError(f"Unable to generate the code with error: {e}") from e
