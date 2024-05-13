# PipFlow

PipFlow is a Python package designed to facilitate pipeline planning and execution using natural language prompts and a pre-trained language model.

## Introduction

PipFlow simplifies the process of generating and executing pipelines by leveraging language models to understand natural language prompts and convert them into executable code. With PipFlow, you can create complex pipelines using simple, human-readable instructions.

## Installation

You can install PipFlow using pip:

```
pip3 install git+https://github.com/PipableAI/pip-flow.git
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
from pip_flow import PipFlow

# Initialize PipFlow object
pipflow = PipFlow()

# Add callable functions
pipflow.add_callables([func1, func2, func3])

# Generate plan
plan = pipflow.generate_plan("How to process data efficiently?")

# Visualize plan
pipflow.visualize_plan(plan)

# Generate executable code
code = pipflow.plan_to_code(plan)

# Execute code
exec(code)
```

## API Reference

### `PipFlow`

#### `__init__(model_key: str = "PipableAI/pip-code-bandit", device: str = "cloud", url: str = INFERENCE_URL)`

Initializes the PipFlow object with the provided model key, device, and URL.

- `model_key` (str): The model key for the language model.
- `device` (str): The device to use for inference (cloud or CUDA).
- `url` (str): The URL for inference when using cloud device.

#### `add_callables(functions: List[Callable], generate_docs=False)`

Registers a list of callable functions with the planner.

- `functions` (List[Callable]): A list of callable functions to be registered.
- `generate_docs` (bool, optional): A flag indicating whether to generate documentation for each function.

#### `generate_plan(question: str, max_new_tokens: int = 900) -> Plan`

Generates a plan based on the given question.

- `question` (str): The question for which the plan needs to be generated.
- `max_new_tokens` (int, optional): The maximum number of new tokens allowed in the generated plan.

#### `visualize_plan(plan: Plan | None = None)`

Visualizes a given plan using a pyvis Network.

- `plan` (Plan | None, optional): The plan to visualize. If None, the last plan generated will be used.

#### `plan_to_code(plan: Plan | None = None, question: str = None, max_new_tokens=600) -> str`

Generates executable code based on a given plan and question.

- `plan` (Plan | None, optional): The plan to generate code for. If None, the last plan generated will be used.
- `question` (str, optional): The question to resolve. If None, the last plan question will be used.
- `max_new_tokens` (int, optional): The maximum number of new tokens allowed in the generated code.

#### `generate_function_call(question: str, function: Callable = None, docstring: str = None, code: str = None, max_new_tokens: int = 500) -> str`

Generates a function call in Python language based on a given question and either the docstring of the function or a undocuemneted code.

- `question` (str): The question prompting the function call generation.
- `function` (Callable): The callable function for which to generate the function call.
- `docstring` (str, optional): The documentation string template for the function.
- `code` (str, optional): The code of the function. This can be used when the docstring is not present.
- `max_new_tokens` (int, optional): The maximum number of new tokens allowed in the generated code.

#### `generate_sql(schema: str, question: str, instructions: str = None, examples: str = None, max_new_tokens: int = 400) -> str`

Generate SQL queries based on the provided schema and question.

- `schema` (str): The schema for the SQL query.
- `question` (str): The question related to the SQL query.
- `instructions` (str, optional): Additional instructions for generating the SQL query.
- `examples` (str, optional): An examples for generating the SQL query.
- `max_new_tokens` (int, optional): The maximum number of new tokens allowed in the generated SQL query.

#### `parse_config(data: str, question: str, eos_token: str, max_new_tokens: int = 300) -> str`

Parses the configuration data and generates a response based on the given data and question.

- `data` (str): The configuration data to parse.
- `question` (str): The question related to the configuration data.
- `eos_token` (str): The end of sentence token.
- `max_new_tokens` (int, optional): The maximum number of new tokens allowed in the generated response.

For more details, refer to the [API Reference](#api-reference).

## Contributing

Contributions are welcome! If you'd like to contribute to PipFlow, please fork the repository and submit a pull request.

## License

PipFlow is licensed under the MIT License.