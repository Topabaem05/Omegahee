from lalamo.model_import.model_configs import HFDoryConfig
from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
from lalamo.model_import.model_specs.output_parser_regexes import OPTIONAL_THINKING_OUTPUT_PARSER_REGEX
from lalamo.model_import.model_specs.reasoning_configs import BOOLEAN_REASONING_DEFAULT_ON_CONFIG
from lalamo.model_import.origins import HuggingFaceOrigin

__all__ = ["NVIDIA_MODELS"]

NVIDIA_MODELS = [
    LanguageModelSpec(
        vendor="NVIDIA",
        family="Nemotron-Labs-Dory",
        name="NVIDIA-Nemotron-Labs-Dory1.0-4B",
        size="4B",
        origin=HuggingFaceOrigin(repo="nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"),
        config_type=HFDoryConfig,
        configs=ConfigMap(chat_template=FileSpec("chat_template.jinja")),
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag="</think>",
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
    ),
]
