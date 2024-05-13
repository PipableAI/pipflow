import inspect
import json
import sys
import warnings
from typing import Callable, List

import pandas as pd
import requests
from IPython.display import HTML, Code, Markdown, display
from transformers import AutoModelForCausalLM, AutoTokenizer

from pip_flow.models.device import Device
from pip_flow.models.function import Function
from pip_flow.models.plan import Plan

INFERENCE_URL = "https://playground.pipable.ai/infer"


class PipFlow:
    def __init__(
        self,
        model_key: str = "PipableAI/pip-code-bandit",
        device: str = "cloud",
        url: str = INFERENCE_URL,
    ):
        """
        Initializes the PipFlow object with the provided model key, device, and URL.
        Sets up the device, model key, tokenizer, URL, and initializes other attributes like functions, prompt_templates, last_question, last_plan, latest_config, last_base_prompt, train_data.
        """
        self.device = Device(device)
        self.model_key = model_key
        self.model = None
        self.tokenizer = None
        self.url = url
        self.functions: List[Function] = []
        self.prompt_templates = {}
        self.last_plan_question = None
        self.last_plan = None
        self.latest_configs = {}
        self.latest_base_prompt = None
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
- use self as the parameter name when passing the object variable to some method.
- Use only the requeried functions from the list {function_list} while making plans.
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
        self.latest_base_prompt = self.prompt_templates[default_key]

    def save_training_data(self, filepath: str = "train.csv"):
        """
        Saves the training data to a CSV file.

        Parameters:
            filepath (str): The path to the CSV file. Defaults to "train.csv".

        Returns:
            None

        Raises:
            Exception: If there is an error saving the training data.
        """
        try:
            self.train_data.to_csv(filepath)
            print(f"saved training data to {filepath}")
        except Exception as e:
            print(f"error saving training data as {e}")

    def save_templates(self, filepath: str = "templates.json"):
        """
        Saves the prompt templates to a JSON file.

        Parameters:
            filepath (str): The path to the JSON file. Defaults to "templates.json".

        Returns:
            None

        Raises:
            Exception: If there is an error saving the templates.
        """
        try:
            data = self.prompt_templates
            with open(filepath, "w+") as json_file:
                json.dump(data, json_file, indent=4)
        except Exception as e:
            print(f"Couldn't save template with error {e}")

    def load_templates(self, filepath: str = "templates.json"):
        """
        Load prompt templates from a JSON file.

        Args:
            filepath (str, optional): The path to the JSON file. Defaults to "templates.json".

        Raises:
            Exception: If there is an error loading the templates.

        Returns:
            None
        """
        try:
            with open(filepath, "r") as json_file:
                loaded_data = json.load(json_file)
            self.prompt_templates = loaded_data
        except Exception as e:
            print(f"Couldn load template with error {e}")

    def _update_config(self):
        self.latest_configs = {
            "func_info": str(
                [
                    f"""--name:{function.name}\n--annotations:{function.signature}\n--doc:{function.docs}\n\n"""
                    for function in self.functions
                ]
            ),
            "function_list": str([f.full_name for f in self.functions]),
            "instructions": "",
        }

    def generate(
        self, prompt: str, max_new_tokens: int = 500, eos_token: str = "response"
    ) -> str:
        """
        Generates a response based on the given prompt using a language model.

        Args:
            prompt (str): The input prompt for generating the response.
            max_new_tokens (int, optional): The maximum number of new tokens to generate in the response. Defaults to 500.
            eos_token (str, optional): The end of sentence token. Defaults to "response".

        Returns:
            str: The generated response.

        Raises:
            Exception: If there is an error generating the response using the specified URL.

        Note:
            - If the device is set to CLOUD, the function sends a POST request to the specified URL with the prompt and max_new_tokens as payload.
            - If the device is set to CUDA, the function uses the tokenizer and model to generate the response.
            - The function appends the prompt with the end of sentence token and splits the response to remove the token.
            - The function adds the prompt and response to the train_data DataFrame.
        """
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

    def generate_docs(
        self, function: Callable = None, code: str = None, max_new_tokens=500
    ) -> str:
        if function is None and code is None:
            raise ValueError("Provide either function or code.")
        if code is None:
            code = inspect.getsource(function)
        prompt = f"""
<function_code>
{code}
</function_code>
<question>
Document the function above giving the function description , parameter name and description , dtypes , possible param values, default param value and return type.
</question>
"""
        try:
            response = self.generate(prompt, max_new_tokens, eos_token="doc")
            if "ipykernel" in sys.modules:
                display(Markdown(data=response))
            return response
        except Exception as e:
            raise ValueError(f"Unable to generate the code with error: {e}") from e

    def add_callables(self, functions: List[Callable], generate_docs=False):
        """
        Registers a list of callable functions with the planner. If `generate_docs` is set to True,
        the function will attempt to generate documentation for each function using the `generate_docs`
        method. If the generation of docs fails, it will print an error message. Each function is added
        to the planner's list of functions if it is not already present. If a function cannot be registered,
        an error message is printed. Finally, the configuration is updated.

        :param functions: A list of callable functions to be registered.
        :type functions: List[Callable]
        :param generate_docs: A flag indicating whether to generate documentation for each function.
        :type generate_docs: bool, optional
        :return: None
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
        self._update_config()

    def add_plan_template(self, key: str, base_prompt: str):
        """
        A function to add a plan template based on the provided key and base prompt.

        :param key: A string representing the key for the plan template.
        :param base_prompt: A string representing the base prompt for the plan template.
        :return: None
        """
        self.prompt_templates[key] = base_prompt

    def make_live_prompt(self, key: str = "default", config: dict | None = None):
        """
        A function to make a live prompt based on the provided key and configuration.

        :param key: A string representing the key for the live prompt.
        :param config: A dictionary representing the configuration for the live prompt.
        :return: A message indicating the successful update of the config and base prompt template.
        """
        try:
            self.latest_base_prompt = self.prompt_templates[key]
            if config is not None:
                self.latest_configs.update(config)
            return "Updated config and base prompt template successfully."
        except Exception as e:
            raise KeyError(e)

    def generate_plan(self, question: str, max_new_tokens: int = 900) -> Plan:
        """
        Generates a plan based on the given question and maximum number of new tokens.

        Args:
            question (str): The question for which the plan needs to be generated.
            max_new_tokens (int, optional): The maximum number of new tokens allowed in the generated plan. Defaults to 900.

        Returns:
            Plan: The generated plan.

        Raises:
            ValueError: If the plan generation fails or if the response cannot be parsed.
        """
        try:
            base_prompt = self.latest_base_prompt
            config = self.latest_configs
            config["question"] = question
            live_prompt = base_prompt.format_map(modified_dict(**config))
            response = self.generate(live_prompt, max_new_tokens, "json")
        except Exception as e:
            raise ValueError(f"Unable to generate the plan with error: {e}") from e
        try:
            plan = Plan.model_validate_json(response)
            self.last_plan = plan
            self.last_plan_question = question
            print(plan)
        except Exception as e:
            raise ValueError(
                f"Unable to parse the response: {response} with error: {e}"
            ) from e
        return plan

    def visualise_plan(self, plan: Plan | None = None):
        """
        Visualizes a given plan using a pyvis Network.

        Args:
            plan (Plan | None, optional): The plan to visualize. If None, the last plan generated will be used. Defaults to None.

        Returns:
            None

        Raises:
            Warning: If visualization is not supported in the current environment.
            ValueError: If the plan generation fails or if the response cannot be parsed.
            KeyError: If the specified key is not found in the prompt templates.

        Notes:
            - This method will display the Plan on Interactive Notebooks.
            - Visualization is not supported in VS code yet.
            - The nodes in the network represent the tasks in the plan, with the task ID as the node label.
            - The edges in the network represent the dependencies between tasks, with the parameter name as the edge label.
            - The size and color of the nodes are determined by their type: call nodes, parameter nodes, and final nodes.
            - The network is displayed in a separate HTML file named "network.html".
        """
        warnings.warn("Visualisation is not supported in VS code yet.")
        if plan is None:
            plan = self.last_plan
        if "ipykernel" in sys.modules:
            from pyvis.network import Network

            CALL_NODE_SIZE = 20
            CALL_NODE_COLOR = "#9dd2d8"
            PARAM_NODE_SIZE = 10
            PARAM_NODE_COLOR = "#e2b9db"
            FINAL_NODE_SIZE = 10
            FINAL_NODE_COLOR = "#9dc3e2"

            net = Network(
                notebook=True,
                cdn_resources="remote",
                bgcolor="#222222",
                font_color="white",
                directed=True,
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

            display(HTML(filename="network.html"))
        else:
            print("This method is only available on Interactive Notebooks")

    def plan_to_code(
        self, plan: Plan | None = None, question: str = None, max_new_tokens=600
    ) -> str:
        """
        Generates executable code based on a given plan and question.

        Args:
            plan (Plan | None, optional): The plan to generate code for. If None, the last plan generated will be used. Defaults to None.
            question (str, optional): The question to resolve. If None, the last plan question will be used. Defaults to None.
            max_new_tokens (int, optional): The maximum number of new tokens allowed in the generated code. Defaults to 600.

        Returns:
            str: The generated executable code that executes the plan using the exact function calls provided in the plan.

        Raises:
            ValueError: If unable to generate the code with the given plan and question.
        """
        if plan is None:
            plan = self.last_plan
        if question is None:
            question = self.last_plan_question
        names = []
        for tasks in plan.tasks:
            names.append(tasks.function_name)

        prompt = f"""
<json>
{str(plan)}
</json>
<instructions>
- Use try except to produce executable code.
- Make sure that constants and the values of the parameters in the task are used in the code.
- Also use imports wherever necessary.
- Add proper comments above each code line.
- Assign the values to the variables you are using.
</instructions>
<question>
Functions to use:
- Use only the required functions from the list {str([x.full_name for x in self.functions])} while writing code.
Given the above plan and functions to use, Just return a small python code that executes the plan using just these exact function calls provided in the plan.
The question to resolve:
{question}
</question>
"""
        try:
            response = self.generate(prompt, max_new_tokens, eos_token="response")
            response = response.replace("```python", "")
            response = response.replace("```", "")
            if "ipykernel" in sys.modules:
                display(Code(data=response, language="css"))
            else:
                return response
        except Exception as e:
            raise ValueError(f"Unable to generate the code with error: {e}") from e

    def generate_function_call(
        self,
        question: str,
        function: Callable = None,
        docstring: str = None,
        code: str = None,
        max_new_tokens: int = 500,
    ) -> str:
        """
        Generates a function call in Python language based on a given question, and either the docstring of the function or a undocuemneted code.

        Args:
            docstring (str): The documentation string template for the function.
            question (str): The question prompting the function call generation.
            code (str, optional): The code of the function. This can be used when the docstring is not present.
        Returns:
            str: The Python function call generated based on the question and the provided docstring template.
        """
        try:
            if function is not None:
                docstring = inspect.getdoc(function)
            elif code is not None:
                print("Generating docstring for the code..")
                docstring = self.generate_docs(code=code, max_new_tokens=max_new_tokens)
            elif docstring is not None:
                pass
            else:
                raise ValueError("Provide function, docstring or code.")
            prompt = f"""
Give a function call in python langugae for the following question:
<doc>
{docstring}
</doc>
<instruction>
1. Strictly use named parameters mentioned in the doc to generate function calls.
2. Only return the response as python parsable string version of function call.
3. mention the 'self' parameter if required.
</instruction>
<question>
{question}
</question>"""
            result = self.generate(
                prompt=prompt, max_new_tokens=200, eos_token="function_call"
            )
            if "ipykernel" in sys.modules:
                display(Code(data=result, language="css"))
            return result
        except Exception as e:
            raise RuntimeError(f"An error occurred: {e}")

    def generate_sql(
        self,
        schema: str,
        question: str,
        instructions: str = None,
        examples: str = None,
        max_new_tokens: int = 400,
    ) -> str:
        """
        Generate SQL queries based on the provided schema and question.

        Args:
            schema (str): The schema for the SQL query.
            question (str): The question related to the SQL query.
            instructions (str, optional): Additional instructions for generating the SQL query. Defaults to None.
            examples (str, optional): An examples for generating the SQL query. Defaults to None.

        Returns:
            str: The generated SQL query.

        Raises:
            ValueError: If unable to generate the SQL query using the model.

        """
        try:
            prompt = "Generate simple SQL queries from the schema mentioned for the following questions."

            if instructions:
                prompt += f"\n<instructions>{instructions}</instructions>"

            if examples:
                prompt += f"\n<example>{examples}</example>"

            prompt += f"""
            <schema>{schema}</schema>
            <question>{question}</question>
            """
            resposne = self.generate(prompt, max_new_tokens, "sql")
            resposne = resposne.replace("<p>", "").replace("</p>", "")

            if "ipykernel" in sys.modules:
                display(Code(data=resposne, language="css"))
            return resposne

        except Exception as e:
            message = f"Unable to generate the SQL query using model with error: {e}"
            raise ValueError(message) from e

    def parse_data_to_json(
        self, data: str, question: str, max_new_tokens: int = 300
    ) -> str:
        """
        Parses the configuration data to json format and generates a json response based on the given data and question.

        Args:
            data (str): The configuration data to parse.
            question (str): The question related to the configuration data.
            eos_token (str): The end of sentence token.

        Returns:
            str: The generated response based on the parsed data and question.

        Raises:
            ValueError: If unable to parse the data with the specified error.
        """
        try:
            prompt = f"""
            <file>{data}</file>
            <question>{question}</question>
            """
            response = self.generate(
                prompt, max_new_tokens=max_new_tokens, eos_token="json"
            )
            return response
        except Exception as e:
            raise ValueError(f"Unable to parse the data with error: {e}") from e


class modified_dict(dict):
    def __missing__(self, key):
        """
        Return a string representation of the missing key in the dictionary.

        Parameters:
            key (Any): The key that is missing from the dictionary.

        Returns:
            str: The string representation of the missing key in the format "{{key}}".
        """
        return f"{{{key}}}"
