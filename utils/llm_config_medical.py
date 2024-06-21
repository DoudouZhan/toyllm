DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

DEFAULT_SYSTEM_PROMPT_CHINESE = """\
你是一个乐于助人、尊重他人以及诚实可靠的医药咨询师。在安全的情况下，始终尽可能有帮助地回答患者的咨询。您的回答不应包含任何有害、不道德、种族主义、性别歧视、有毒、危险或非法的内容。请确保您的回答在社会上是公正的和积极的，并且符合医药领域的专业知识和伦理标准。
如果一个问题没有任何意义或与事实不符，请解释原因，而不是回答错误的问题。如果您不知道问题的答案，请不要分享虚假信息。另外，答案请使用中文。

在回答药品相关的咨询时，请基于下列示例，提供准确且有所变化的回答，并在回答后提醒用户咨询专业医生：

示例1：
用户：请介绍保济丸的主要功能。
助手：保济丸主要用于解表祛湿和中，常用于暑湿感冒、发热头痛、腹痛腹泻、恶心呕吐以及肠胃不适，也可以用于缓解晕车晕船的不适。请咨询专业医生以获取更多信息。

示例2：
用户：请介绍克感利咽口服液的主要功能。
助手：克感利咽口服液的功能是疏风清热、解毒利咽。它适用于感冒引起的风热外侵和邪热内扰，症状包括发热、微恶风、头痛、咽痛、鼻塞流涕、咳嗽痰粘、口渴和小便黄。请咨询专业医生以获取更多信息。

示例3：
用户：请介绍外感平安颗粒的主要功能。
助手：外感平安颗粒主要用于清热解表、化湿消滞。它适用于四季感冒，伴随恶寒发热、周身骨痛、头重乏力，以及感冒挟湿引起的胸闷食滞。请咨询专业医生以获取更多信息。

请记住，在回答每个问题时，始终保持诚实和尊重，并确保信息的准确性和公正性。如有需要，请提供额外的健康建议或指引患者咨询专业医生。\
"""

DEFAULT_SYSTEM_PROMPT_JAPANESE = """\
あなたは親切で、礼儀正しく、誠実なアシスタントです。 常に安全を保ちながら、できるだけ役立つように答えてください。 回答には、有害、非倫理的、人種差別的、性差別的、有毒、危険、または違法なコンテンツを含めてはいけません。 回答は社会的に偏見がなく、本質的に前向きなものであることを確認してください。
質問が意味をなさない場合、または事実に一貫性がない場合は、正しくないことに答えるのではなく、その理由を説明してください。 質問の答えがわからない場合は、誤った情報を共有しないでください。\
"""

DEFAULT_RAG_PROMPT = """\
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\
"""

DEFAULT_RAG_PROMPT_CHINESE = """\
基于以下已知信息，请简洁并专业地回答用户的问题。如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"。不允许在答案中添加编造成分。另外，答案请使用中文。\
"""

from typing import Dict, Union


class SupportedLLMConfig:
    def __init__(
        self,
        language: str = "English",
        model_id: str = "THUDM/chatglm3-6b",
        tokenizer_kwargs: Dict = {},
        model_kwargs: Dict = {},
        int4_compression_configs: Dict[str, Union[bool, float, int]] = {
            "sym": False,
            "group_size": 128,
            "ratio": 0.8,
        },
    ) -> None:
        self.language = language
        self.model_id = model_id
        self.tokenizer_kwargs = tokenizer_kwargs
        self.model_kwargs = model_kwargs
        self.int4_compression_configs = int4_compression_configs


class LLMModelGroup:
    Qwen1_5 = [
        SupportedLLMConfig(
            "Chinese",
            "Qwen/Qwen1.5-1.8B-Chat",
            tokenizer_kwargs={"add_special_tokens": False},
            model_kwargs={
                "start_message": DEFAULT_SYSTEM_PROMPT_CHINESE,
                "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
            },
        ),
        SupportedLLMConfig(
            "Chinese",
            "Qwen/Qwen1.5-4b-Chat",
            tokenizer_kwargs={"add_special_tokens": False},
            model_kwargs={
                "start_message": DEFAULT_SYSTEM_PROMPT_CHINESE,
                "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
            },
        ),
        SupportedLLMConfig(
            "Chinese",
            "Qwen/Qwen1.5-7B-Chat",
            tokenizer_kwargs={"add_special_tokens": False},
            model_kwargs={
                "start_message": DEFAULT_SYSTEM_PROMPT_CHINESE,
                "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
            },
        ),
    ]
    Qwen2 = [
        SupportedLLMConfig(
            "Chinese",
            "Qwen/Qwen2-0.5B",  # https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f
            tokenizer_kwargs={"add_special_tokens": False},
            model_kwargs={
                "start_message": DEFAULT_SYSTEM_PROMPT_CHINESE,
                "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
            },
            int4_compression_configs={
                "sym": False,
                "group_size": 128,
                "ratio": 0.8,
            },
        ),
        SupportedLLMConfig(
            "Chinese",
            "Qwen/Qwen2-0.5B-Instruct",  # https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f
            tokenizer_kwargs={"add_special_tokens": False},
            model_kwargs={
                "start_message": DEFAULT_SYSTEM_PROMPT_CHINESE,
                "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
            },
            int4_compression_configs={
                "sym": False,
                "group_size": 128,
                "ratio": 0.8,
            },
        ),
        SupportedLLMConfig(
            "Chinese",
            "Qwen/Qwen2-1.5B",  # https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f
            tokenizer_kwargs={"add_special_tokens": False},
            model_kwargs={
                "start_message": DEFAULT_SYSTEM_PROMPT_CHINESE,
                "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
            },
            int4_compression_configs={
                "sym": False,
                "group_size": 128,
                "ratio": 0.8,
            },
        ),
        SupportedLLMConfig(
            "Chinese",
            "Qwen/Qwen2-1.5B-Instruct",  # https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f
            tokenizer_kwargs={"add_special_tokens": False},
            model_kwargs={
                "start_message": DEFAULT_SYSTEM_PROMPT_CHINESE,
                "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
            },
            int4_compression_configs={
                "sym": False,
                "group_size": 128,
                "ratio": 0.8,
            },
        ),
        SupportedLLMConfig(
            "Chinese",
            "Qwen/Qwen2-7B",  # https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f
            tokenizer_kwargs={"add_special_tokens": False},
            model_kwargs={
                "start_message": DEFAULT_SYSTEM_PROMPT_CHINESE,
                "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
            },
            int4_compression_configs={
                "sym": False,
                "group_size": 128,
                "ratio": 0.8,
            },
        ),
        SupportedLLMConfig(
            "Chinese",
            "Qwen/Qwen2-7B-Instruct",  # https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f
            tokenizer_kwargs={"add_special_tokens": False},
            model_kwargs={
                "start_message": DEFAULT_SYSTEM_PROMPT_CHINESE,
                "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
            },
            int4_compression_configs={
                "sym": False,
                "group_size": 128,
                "ratio": 0.8,
            },
        ),
    ]


SUPPORTED_LLM_LIST = [
    *LLMModelGroup.Qwen1_5,
    *LLMModelGroup.Qwen2,
    # SupportedLLMConfig(
    #     "English",
    #     "THUDM/chatglm3-6b",
    #     int4_compression_configs={
    #         "sym": True,
    #         "group_size": 128,
    #         "ratio": 0.72,
    #     },
    #     tokenizer_kwargs={"add_special_tokens": False},
    # ),
    SupportedLLMConfig(
        "Chinese",
        "Qwen/Qwen1.5-0.5b-Chat",
        tokenizer_kwargs={"add_special_tokens": False},
        model_kwargs={
            "start_message": DEFAULT_SYSTEM_PROMPT_CHINESE,
            "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
        },
    ),
]

some_config = {
    "qwen2-1.5b-instruct": {
        "model_id": "Qwen/Qwen2-1.5B-Instruct",
        "remote_code": False,
        "start_message": DEFAULT_SYSTEM_PROMPT_CHINESE,
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
    },
    "qwen2-7b-instruct": {
        "model_id": "Qwen/Qwen2-7B-Instruct",
        "remote_code": False,
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
        "start_message": DEFAULT_SYSTEM_PROMPT_CHINESE,
        "rag_prompt_template": f"""<|im_start|>system
            {DEFAULT_RAG_PROMPT_CHINESE }<|im_end|>"""
        + """
            <|im_start|>user
            问题: {input} 
            已知内容: {context} 
            回答: <|im_end|><|im_start|>assistant""",
    },
}
