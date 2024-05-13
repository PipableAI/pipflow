# PipFlow

PipFlow is a Python package designed to facilitate pipeline planning and execution using natural language prompts and a pre-trained language model.

## Introduction

PipFlow simplifies the process of generating and executing pipelines by leveraging language models to understand natural language prompts and convert them into executable code. With PipFlow, you can create complex pipelines using simple, human-readable instructions.

## Installation

You can install PipFlow using pip:

```
pip3 install git+https://github.com/PipableAI/pipflow.git
```

## Getting Started

To get started with PipFlow, follow these steps:

1. Initialize a PipFlow object with your preferred model key and device.
2. Add callable functions to the PipFlow object using the `add_callables` method.
3. Generate plans and visualize them using the `generate_plan` and `visualize_plan` methods.
4. Get code for the generated plans using the `plan_to_code` method.

## Example Usage

For detailed use of library visit [Pip Flow Examples](https://colab.research.google.com/drive/10av3SxFf0Psx_IkmZbcUhiVznStV5pVS?usp=sharing).

```python
from pipflow import PipFlow

# Initialize PipFlow object
pip_flow = PipFlow()

# define func1, func2 and func3 of your choice
# ex: from pandas import read_csv as func1
# Add callable functions
pip_flow.add_callables([func1, func2, func3])

# Generate plan
plan = pip_flow.generate_plan("How to process data efficiently?")

# Visualize plan
pip_flow.visualize_plan(plan)

# Generate executable code
code = pip_flow.plan_to_code(plan)

# Execute code
exec(code)
```

## API Reference

### PipFlow Class


#### \_\_init\_\_

```python
def __init__(
    self,
    model_key: str = "PipableAI/pip-code-bandit",
    device: str = "cloud",
    url: str = INFERENCE_URL,
)
```

Initializes the PipFlow object with the provided model key, device, and URL.

**Parameters:**
- `model_key` (str): The key identifying the language model. Default is "PipableAI/pip-code-bandit".
- `device` (str): The device to use for inference, either "cloud" or "cuda". Default is "cloud".
- `url` (str): The URL for inference if the device is set to "cloud". Default is the value of `INFERENCE_URL`.

---

### Utility Methods

---

#### save_training_data

```python
def save_training_data(filepath: str = "train.csv")
```

Saves the training data to a CSV file.

**Parameters:**
- `filepath` (str): The path to the CSV file. Defaults to "train.csv".

**Returns:**
- None

**Raises:**
- Exception: If there is an error saving the training data.

---

#### save_templates

```python
def save_templates(filepath: str = "templates.json")
```

Saves the prompt templates to a JSON file.

**Parameters:**
- `filepath` (str): The path to the JSON file. Defaults to "templates.json".

**Returns:**
- None

**Raises:**
- Exception: If there is an error saving the templates.

---

#### load_templates

```python
def load_templates(filepath: str = "templates.json")
```

Load prompt templates from a JSON file.

**Parameters:**
- `filepath` (str, optional): The path to the JSON file. Defaults to "templates.json".

**Raises:**
- Exception: If there is an error loading the templates.

**Returns:**
- None

---

#### generate

```python
def generate(
    self, prompt: str, max_new_tokens: int = 500, eos_token: str = "response"
) -> str
```

Generates a response based on the given prompt using a language model.

**Parameters:**
- `prompt` (str): The input prompt for generating the response.
- `max_new_tokens` (int, optional): The maximum number of new tokens to generate in the response. Defaults to 500.
- `eos_token` (str, optional): The end of sentence token. Defaults to "response".

**Returns:**
- str: The generated response.

**Raises:**
- Exception: If there is an error generating the response using the specified URL.

**Note:**
- If the device is set to CLOUD, the function sends a POST request to the specified URL with the prompt and max_new_tokens as payload.
- If the device is set to CUDA, the function uses the tokenizer and model to generate the response.
- The function appends the prompt with the end of sentence token and splits the response to remove the token.
- The function adds the prompt and response to the train_data DataFrame.

---

#### generate_docs

```python
def generate_docs(
    self, function: Callable = None, code: str = None, max_new_tokens=500
) -> str
```

Generates documentation for a given function or code using a language model.

**Parameters:**
- `function` (Callable): The function for which documentation needs to be generated.
- `code` (str): The code snippet for which documentation needs to be generated.
- `max_new_tokens` (int, optional): The maximum number of new tokens to generate in the documentation. Defaults to 500.

**Returns:**
- str: The generated documentation.

**Raises:**
- ValueError: If unable to generate documentation for the given function or code.

---

#### add_callables

```python
def add_callables(self, functions: List[Callable], generate_docs=False)
```

Registers a list of callable functions with the planner.

**Parameters:**
- `functions` (List[Callable]): A list of callable functions to be registered.
- `generate_docs` (bool, optional): A flag indicating whether to generate documentation for each function. Defaults to False.

**Returns:**
- None

---

#### add_plan_template

```python
def add_plan_template(self, key: str, base_prompt: str)
```

Adds a plan template based on the provided key and base prompt.

**Parameters:**
- `key` (str): A string representing the key for the plan template.
- `base_prompt` (str): A string representing the base prompt for the plan template.

**Returns:**
- None

---

#### freeze_template

```python
def freeze_template(self, key: str = "default", config: dict | None = None)
```

Update the prompt based on the provided key and configuration of variables.

**Parameters:**
- `key` (str, optional): The key for the prompt. Defaults to "default".
- `config` (dict | None, optional): The configuration of the variables for the prompt. Defaults to None.

**Returns:**
- str: A message indicating the successful update of the config and base prompt template.

**Raises:**
- KeyError: If the specified key is not found in the prompt templates.

---

#### generate_plan

```python
def generate_plan(self, question: str, max_new_tokens: int = 900) -> Plan
```

Generates a plan based on the given question and maximum number of new tokens.

**Parameters:**
- `question` (str): The question for which the plan needs to be generated.
- `max_new_tokens` (int, optional): The maximum number of new tokens allowed in the generated plan. Defaults to 900.

**Returns:**
- Plan: The generated plan.

**Raises:**
- ValueError: If the plan generation fails or if the response cannot be parsed.

---

#### visualise_plan

```python
def visualise_plan(self, plan: Plan | None = None)
```

Visualizes a given plan using a pyvis Network.

**Parameters:**
- `plan` (Plan | None, optional): The plan to visualize. If None, the last plan generated will be used. Defaults to None.

**Returns:**
- None

**Raises:**
- Warning: If visualization is not supported in the current environment.
- ValueError: If the plan generation fails or if the response cannot be parsed.
- KeyError: If the specified key is not found in the prompt templates.

**Notes:**
- This method will display the Plan on Interactive Notebooks.
- Visualization is not supported in VS code yet.
- The nodes in the network represent the tasks in the plan, with the task ID as the node label.
- The edges in the network represent the dependencies between tasks, with the parameter name as the edge label.
- The size and color of the nodes are determined by their type: call nodes, parameter nodes, and final nodes.
- The network is displayed in a separate HTML file named "network.html".

---

#### plan_to_code

```python
def plan_to_code(
    self, plan: Plan | None = None, question: str = None, max_new_tokens=600
) -> str
```

Generates executable code based on a given plan and question.

**Parameters:**
- `plan` (Plan | None, optional): The plan to generate code for. If None, the last plan generated will be used. Defaults to None.
- `question` (str, optional): The question to resolve. If None, the last plan question will be used. Defaults to None.
- `max_new_tokens` (int, optional): The maximum number of new tokens allowed in the generated code. Defaults to 600.

**Returns:**
- str: The generated executable code that executes the plan using the exact function calls provided in the plan.

**Raises:**
- ValueError: If unable to generate the code with the given plan and question.

---

#### generate_function_call

```python
def generate_function_call(
    self,
    question: str,
    function: Callable = None,
    docstring: str = None,
    code: str = None,
    max_new_tokens: int = 500,
) -> str
```

Generates a function call in Python language based on a given question, and either the docstring of the function or a undocumented code.

**Parameters:**
- `question` (str): The question prompting the function call generation.
- `function` (Callable): The function for which documentation needs to be generated.
- `docstring` (str): The documentation string template for the function.
- `code` (str): The code snippet for which documentation needs to be generated.
- `max_new_tokens` (int, optional): The maximum number of new tokens allowed in the generated function call. Defaults to 500.

**Returns:**
- str: The Python function call generated based on the question and the provided docstring template.

**Raises:**
- RuntimeError: If an error occurs during generation.

---

#### generate_sql

```python
def generate_sql(
    self,
    schema: str,
    question: str,
    instructions: str = None,
    examples: str = None,
    max_new_tokens: int = 400,
) -> str
```

Generate SQL queries based on the provided schema and question.

**Parameters:**
- `schema` (str): The schema for the SQL query.
- `question` (str): The question related to the SQL query.
- `instructions` (str, optional): Additional instructions for generating the SQL query. Defaults to None.
- `examples` (str, optional): An examples for generating the SQL query. Defaults to None.
- `max_new_tokens` (int, optional): The maximum number of new tokens allowed in the generated SQL query. Defaults to 400.

**Returns:**
- str: The generated SQL query.

**Raises:**
- ValueError: If unable to generate the SQL query using the model.

---

#### parse_data_to_json

```python
def parse_data_to_json(
    self, data: str, question: str, max_new_tokens: int = 300
) -> str
```

Parses the configuration data to json format and generates a json response based on the given data and question.

**Parameters:**
- `data` (str): The configuration data to parse.
- `question` (str): The question related to the configuration data.
- `max_new_tokens` (int, optional): The maximum number of new tokens allowed in the generated response. Defaults to 300.

**Returns:**
- str: The generated response based on the parsed data and question.

**Raises:**
- ValueError: If unable to parse the data with the specified error.

---

## Contributing

Contributions are welcome! If you'd like to contribute to PipFlow, please fork the repository and submit a pull request.

## License

PipFlow is licensed under the MIT License.