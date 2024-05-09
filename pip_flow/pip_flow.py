import inspect
import json
import sys
import warnings
from typing import List

import pandas as pd
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
from IPython.core.display import display, HTML

from pip_flow.models.device import Device
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
        self.last_question = None
        self.last_plan = None
        self.latest_config = {}
        self.last_base_prompt = None
        self.train_data = pd.DataFrame(columns=["input", "output"])
        self._base_setup()
        if self.device == Device.CLOUD:
            self.url = url
        else:
            self._load_model()

    def _base_setup(self):
        default_key = "default"
        self.prompt_templates[
            default_key
        ] = """
<functions>
{func_info}
</functions>
<json_structure>
{{
  "tasks": [
    {{
      "task_id": 1,
      "function_name": "function name",
      "parameters": [
        {{
        "name":"name of this parameter according to annotations.",
        "value":"value to be passed for this parameter",
        "dtype":"type annotation of the variable",
        "description": "An explanation of why this value should be utilized."
        }},
        {{
        "name":"self",
        "value":"variable name to be passed for this parameter self.",
        "dtype":"type annotation of the self parameter",
        "description": "An explanation of why the cariable should be used for this self parameter."
        }}
      ],
      "outputs": ["variable_1"],
      "description": "some description"
    }},
    {{
      "task_id": 2,
      "function_name": "function name",
      "parameters": [
        {{
        "name":"self",
        "value":"variable name to be passed for this parameter self.",
        "dtype":"type annotation of the self parameter",
        "description": "An explanation of why the cariable should be used for this self parameter."
        }},
        {{,
        "name":"name of this parameter according to annotations.",
        "value":"value to be passed for this parameter",
        "dtype":"type annotation of the variable",
        "description": "An explanation of why this value should be utilized."
        }}
      ],
      "outputs": ["variable_2"],
      "description": "some description"
    }}
  ]
}}
</json_structure>
<instructions>
- use self as the param\eter name when passing the object variable to some method.
- Use names of functions from the list {function_list} while making plans
- name outputs as variable_1 , variable_2 , variable_3 , variable_4 and more variables in chronological order.
- give attention to the type annotation of the parameter given while filling values.
{instructions}
</instructions>
<question>
Given the above functions,
- Do not give the parameters in json which have null values and default values of the function, only give the sequencial function calls with parameters to execute the below question:
{question}
</question>
"""
        self.last_base_prompt = self.prompt_templates[default_key]

    def save_training_data(self, filepath: str = "train.csv"):
        try:
            self.train_data.to_csv(filepath)
            print(f"saved training data to {filepath}")
        except Exception as e:
            print(f"error saving training data as {e}")

    def save_templates(self, filepath: str = "templates.json"):
        try:
            data = self.prompt_templates
            with open(filepath, "w+") as json_file:
                json.dump(data, json_file, indent=4)
        except Exception as e:
            print(f"Couldn't save template with error {e}")

    def load_templates(self, filepath: str = "templates.json"):
        try:
            with open(filepath, "r") as json_file:
                loaded_data = json.load(json_file)
            self.prompt_templates = loaded_data
        except Exception as e:
            print(f"Couldn load template with error {e}")

    def _update_config(self):
        self.latest_config = {
            "func_info": str(
                [
                    f"""--name:{function.name}\n--annotations:{function.signature}\n--doc:{function.docs}\n\n"""
                    for function in self.functions
                ]
            ),
            "function_list" : str([f.name for f in self.functions]),
            "instructions": "",
        }

    def generate(
        self, prompt: str, max_new_tokens: int = 500, eos_token: str = "doc"
    ) -> str:
        prompt += f"<{eos_token}>\n"
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
        df = self.train_data
        df.loc[len(df)] = {"input": prompt, "output": response}
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
"""
        try:
            response = self.generate(prompt, max_new_tokens, eos_token="doc")
            return response
        except Exception as e:
            raise ValueError(f"Unable to generate the code with error: {e}") from e

    def register_callables(self, functions: List[callable], generate_docs=False):
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
        self._update_config()

    def add_plan_template(self, key: str, base_prompt: str):
        self.prompt_templates[key] = base_prompt

    def make_live_prompt(self, key: str = "default", config: dict | None = None):
        try:
            self.last_base_prompt = self.prompt_templates[key]
            if config is not None:
                self.latest_config.update(config)
            return "Updated config and base prompt template successfully fetched."
        except Exception as e:
            raise KeyError(e)

    def generate_plan(self, question: str, max_new_tokens: int = 900) -> Plan:
        try:

            base_prompt = self.last_base_prompt
            config = self.latest_config
            config["question"] = question
            live_prompt = base_prompt.format_map(modified_dict(**config))
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

    def visualise_plan(self, plan: Plan | None = None):
        warnings.warn("Visualisation is not supported in VS code yet.")
        if plan is None:
            plan = self.last_plan
        if "ipykernel" in sys.modules:
            from IPython.display import HTML, IFrame, display
            from pyvis.network import Network

            CALL_NODE_SIZE = 20
            CALL_NODE_COLOR = "red"
            PARAM_NODE_SIZE = 10
            PARAM_NODE_COLOR = "blue"
            FINAL_NODE_SIZE = 10
            FINAL_NODE_COLOR = "green"

            net = Network(
                notebook=True, cdn_resources="remote", height="100px", width="100%"
            )

            for task in plan.tasks:
                net.add_node(
                    task.task_id,
                    label=task.function_name,
                    size=CALL_NODE_SIZE,
                    color=CALL_NODE_COLOR,
                    physics=False,
                )

            params = []
            for task in plan.tasks:
                for param in task.parameters:
                    params.append((task.task_id, param))

            outputs = []
            for task in plan.tasks:
                for output in task.outputs:
                    outputs.append((task.task_id, output))

            for output in outputs:
                for param in params:
                    if param[1].value == output[1]:
                        net.add_edge(
                            output[0],
                            param[0],
                            label=param[1].name,
                            arrows="to",
                            physics=False,
                        )

            for param in params:
                if param[1].value not in [output[1] for output in outputs]:
                    net.add_node(
                        param[1].value,
                        label=param[1].name,
                        size=PARAM_NODE_SIZE,
                        color=PARAM_NODE_COLOR,
                        physics=False,
                    )
                    net.add_edge(param[1].value, param[0], arrows="to", physics=False)

            for task in plan.tasks:
                if task.task_id == len(plan.tasks):
                    for output in task.outputs:
                        net.add_node(
                            output,
                            label=output,
                            size=FINAL_NODE_SIZE,
                            color=FINAL_NODE_COLOR,
                        )
                        net.add_edge(task.task_id, output, arrows="to")

            file_name = "network.html"
            net.show(file_name)

            # display(HTML(file_name))
            with open(file_name, "r") as file:
                html_content = file.read()
            # print(html_content)
            # import IPython

            # iframe = f"<iframe srcdoc={html_content} width=700 height=350></iframe>"
            # display(html_content)
            display(HTML(filename="network.html"))

            # Display the HTML content
            # HTML(html_content)

            # display(HTML(filename="network.html"))
            # IFrame(src="network.html", width=900, height=600)
            # legend_html = """
            # <div style='margin-top: 20px; padding: 10px; border: 1px solid black; width: 300px;'>
            #     <h3>Legend:</h3>
            #     <ul>
            #         <li><span style='color: red;'>●</span> Methods or Functions</li>
            #         <li><span style='color: blue;'>●</span> Parameters</li>
            #         <li><span style='color: green;'>●</span> Final Output</li>
            #     </ul>
            # </div>
            # """

            # display(HTML(legend_html))
        else:
            print("This method is only available on Interactive Notebooks")

    def plan_to_code(
        self, plan: Plan | None = None, question: str = None, max_new_tokens=600
    ) -> str:
        if plan is None:
            plan = self.last_plan
        if question is None:
            question = self.last_question
        names = []
        for tasks in plan.tasks:
            names.append(tasks.function_name)

        full_names = [x.full_name for x in self.functions if x.name in names]
        prompt = f"""
<json>
{str(plan)}
</json>
<instructions>
- Use try except to produce executable code.
- Use exact values of parameters in the function calls as given in the tasks of the plan.
- Make sure that constants and the values of the parameters in the task are used in the code.
- Also use imports wherever necessary.
- Add proper comments above each code line.
- Assign the values to the variables you are using.
</instructions>
<question>
Given the above plan, Just return a small python code that executes the plan using just these exact function calls provided in the plan.
The question to resolve:
{question}
Functions to use:
{full_names}
</question>
"""
        try:
            response = self.generate(prompt, max_new_tokens, eos_token="response")
            if "ipykernel" in sys.modules:
                from IPython.display import Markdown, display

                display(Markdown(response))
            else:
                return response
        except Exception as e:
            raise ValueError(f"Unable to generate the code with error: {e}") from e
