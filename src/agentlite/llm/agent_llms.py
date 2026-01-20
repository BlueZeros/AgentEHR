import json
from openai import OpenAI
import torch
import copy
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

OPENAI_CHAT_MODELS = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-32k",
    "gpt-4-32k-0613",
    "gpt-4-1106-preview",
    "gpt-4o-mini",
    "gpt-5-mini",
    "glm-4.5",
    "gemini-2.5-flash",
    "claude-haiku-4-5-20251001",
    "grok-4-1-fast-non-reasoning"
]
OPENAI_LLM_MODELS = ["text-davinci-003", "text-ada-001"]

LOCAL_MODEL_PATHS = {
    "qwen3_4b": "/sfs/data/ShareModels/LLMs/Qwen3-4B",
    "qwen3_8b": "/sfs/data/ShareModels/LLMs/Qwen3-8B",
    "qwen3_32b": "/sfs/rhome/liaoyusheng/data/ShareModels/LLMs/Qwen3-32B",
    "qwen3_30b_moe": "/sfs/data/ShareModels/LLMs/Qwen3-30B-A3B-Instruct-2507",
    "qwen3_80b_moe": "/sfs/data/ShareModels/LLMs/Qwen3-Next-80B-A3B-Instruct",
    "qwen3_235b_moe": "/sfs/data/ShareModels/LLMs/Qwen3-235B-A22B-Instruct-2507",
    "qwen3_235b_moe_int4": "/sfs/data/ShareModels/LLMs/Qwen3-235B-A22B-Instruct-2507-AWQ",
    "gpt_oss_20b": "/sfs/data/ShareModels/LLMs/gpt-oss-20b",
    "gpt_oss_120b": "/sfs/data/ShareModels/LLMs/gpt-oss-120b",
    "llama3.1_70b": "/sfs/data/ShareModels/LLMs/Meta-Llama-3.1-70B-Instruct",
    "glm4.5": "/sfs/data/ShareModels/LLMs/GLM-4.5-Air",
    "llama4": "/sfs/data/ShareModels/LLMs/Llama-4-Scout-17B-16E-Instruct",
    "qwen3_coder_30b_moe": "/sfs/data/ShareModels/LLMs/Qwen3-Coder-30B-A3B-Instruct"
}

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

class BaseLLM:
    def __init__(self, model_args) -> None:
        self.model_name_or_path = model_args.model_name_or_path
        self.max_new_tokens: int = model_args.max_new_tokens
        self.temperature: float = model_args.temperature
        self.top_p: float = model_args.top_p
        self.top_k: int = model_args.top_k
        self.enable_thinking: bool = model_args.enable_thinking
        self.max_seq_len: int = model_args.max_seq_len
        self.presence_penalty: float = model_args.presence_penalty

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

    def __call__(self, prompt: str) -> str:
        return self.run(prompt)
    
    def process_inputs(self, messages: str, available_tools: list = None):
        if getattr(self, 'tokenizer', None):
            if "qwen3" in self.model_name_or_path.lower():
                messages[-1]["content"] += "/think" if self.enable_thinking else "/no_think"
                inputs = self.tokenizer.apply_chat_template(messages, tools=available_tools, add_generation_prompt=True, tokenize=False, enable_thinking=self.enable_thinking)
            else:
                inputs = self.tokenizer.apply_chat_template(messages, tools=available_tools, add_generation_prompt=True, tokenize=False)
        else:
            inputs = messages
        
        return inputs
    
    def process_outputs(self, outputs: str):
        if "qwen3" in self.model_name_or_path.lower() and "</think>" in outputs:
            outputs = {
                "reasoning": (outputs.rsplit("</think>", 1)[0] + "</think>").strip(),
                "output": outputs.rsplit("</think>", 1)[-1].strip(),
            }
        else:
            outputs = {
                "reasoning": "",
                "output": outputs.strip()
            }
        
        return outputs

    def run(self, prompt: str, n: int = 1):
        # return str
        raise NotImplementedError

class HFLocalLLM(BaseLLM):
    def __init__(self, model_args):
        super().__init__(model_args=model_args)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True
        )

        self.device = self.model.device
    
    
    def run(self, messages: str, available_tools: list = None, n: int = 1):
        inputs = self.process_inputs(messages, available_tools)
        inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            model_outputs = self.model.generate(
                input_ids=inputs.input_ids.to(self.device),
                attention_mask=inputs.attention_mask.to(self.device),
                do_sample=False if self.temperature == 0.0 else True,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                tokenizer=self.tokenizer,
                return_dict_in_generate=True, 
                output_logits=True
            )

        final_model_outputs = model_outputs.sequences[:, len(inputs["input_ids"][0]):]
        outputs = self.tokenizer.batch_decode(final_model_outputs, skip_special_tokens=True)
        outputs = self.process_outputs(outputs)
        return outputs

class VLLM(BaseLLM):
    def __init__(self, model_args):
        super().__init__(model_args=model_args)

        self.gpu_memory_utilization: float = model_args.gpu_memory_utilization
        self.model = LLM(
            model=self.model_name_or_path, 
            tensor_parallel_size=torch.cuda.device_count(), 
            trust_remote_code=True, 
            enable_prefix_caching=True,
            max_model_len=self.max_seq_len, 
            gpu_memory_utilization=self.gpu_memory_utilization
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True
        )

        self.sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

    def run(self, messages: str, available_tools: list = None, n: int = 1):
        # inputs = self.process_inputs(messages, available_tools)
        self.sampling_params.n = n
        outputs = self.model.chat(
            messages, 
            sampling_params=self.sampling_params,
            tools=available_tools,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": self.enable_thinking} if "qwen3" in self.model_name_or_path.lower() else None,
        )
        outputs = self.process_outputs(outputs[0].outputs[0].text)
        return outputs

class VLLMServer(BaseLLM):
    def __init__(self, model_args):
        super().__init__(model_args=model_args)

        self.vllm_server_url = model_args.vllm_server_url
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=f"{self.vllm_server_url}/v1"
        )
        self.input_tokens = 0
        self.output_tokens = 0
    
    def token_count(self, messages: list, available_tools: list = None):
        if available_tools:
            input_ids = self.tokenizer.apply_chat_template(messages, available_tools, tokenize=True, add_generation_prompt=True)
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        token_counts = len(input_ids)
        return token_counts
    
    def _truncate_messages(self, messages: list, available_tools: list = None, max_tokens: int = 55000):
        for turn in messages:
            if turn["content"] is None:
                turn["content"] = ""
        
            if "glm" in self.model_name_or_path.lower() or "coder" in self.model_name_or_path.lower():
                if "tool_calls" in turn:
                    for tc in turn["tool_calls"]:
                        if isinstance(tc["function"]["arguments"], str):
                            tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])

        max_tokens = self.max_seq_len - self.max_new_tokens - 1000
        current_tokens = self.token_count(messages, available_tools)

        # print(f"{max_tokens=}, {current_tokens=}")
        
        if current_tokens <= max_tokens:
            return messages
        
        while current_tokens > max_tokens and len(messages) > 3:
            del messages[3]
            current_tokens = self.token_count(messages, available_tools)
            print(f"{current_tokens=}")
        
        return messages
    
    def preprocess(self, messages: list):
        # if "gpt" in self.model_name_or_path.lower() :
        #     for turn in messages:
        #         if turn["content"] is None:
        #             turn["content"] = ""
                
        #         if "tool_calls" in turn:
        #             for tc in turn["tool_calls"]:
        #                 tc["function"]["arguments"] = str(tc["function"]["arguments"])

        # if "qwen" in self.model_name_or_path.lower():
        #     for turn in messages:
        #         if "tool_calls" in turn:
        #             for tc in turn["tool_calls"]:
        #                 tc["function"]["arguments"] = str(tc["function"]["arguments"])
        
        # elif "llama" in self.model_name_or_path.lower() and "glm" in self.model_name_or_path.lower():
        
        for turn in messages:
            if turn["content"] is None:
                turn["content"] = ""
            
            if "tool_calls" in turn:
                for tc in turn["tool_calls"]:
                    tc["function"]["arguments"] = json.dumps(tc["function"]["arguments"])
                    
        
        return messages

    def run(self, messages: list, available_tools: list = None, tool_choice: str = "auto", n: int = 1):
        messages = self._truncate_messages(messages, available_tools, max_tokens=self.max_seq_len)

        messages = copy.deepcopy(messages)
        messages = self.preprocess(messages)

        if "qwen3" in self.model_name_or_path.lower():
            extra_body = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        
        elif "gpt-oss" in self.model_name_or_path.lower():
            extra_body = {"reasoning_effort": "medium" if not self.enable_thinking else "high"}
        
        else:
            extra_body = None

        response = self.client.chat.completions.create(
            model=self.model_name_or_path,
            messages=messages,
            tools=available_tools,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            n=n,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            # top_k=self.top_k,
            tool_choice=tool_choice if available_tools else None,
            extra_body=extra_body,
            stream=False
        )

        try:
            self.input_tokens += response.usage.prompt_tokens
            self.output_tokens += response.usage.completion_tokens
        except Exception as e:
            print(f"Error: {e}")

        return response
    

def get_llm_backend(model_args):
    if model_args.model_name_or_path in OPENAI_CHAT_MODELS:
        return GPT4o(model_args)

    if model_args.model_name_or_path:
        model_args.model_name_or_path = model_args.model_name_or_path if model_args.model_name_or_path not in LOCAL_MODEL_PATHS else LOCAL_MODEL_PATHS[model_args.model_name_or_path]
    
    if model_args.vllm_server_url:
        return VLLMServer(model_args)
    else:
        return VLLM(model_args)
    

class GPT4o:
    def __init__(self, model_args) -> None:
        if "grok" in model_args.model_name_or_path:
            self.client = OpenAI(
                api_key="sk-cm87qYlaJXcTs50CUkwruf2caC8fivSARVmk6xYWgwicJbKE",
                base_url=f"http://192.154.241.225:3000/v1",
            )

        else: 
            self.client = OpenAI(
                api_key="sk-3OD8o4nLS4o52so3LJ9iw2kTlrZ0aGuRiLRI4sxaSZx4c6sm",
                base_url=f"http://192.154.241.225:3000/v1",
            )

        self.model_name_or_path: str =getattr(model_args, "model_name_or_path", "gpt-4o-2024-11-20")
        self.max_new_tokens: int = getattr(model_args, "max_new_tokens", 4096)
        self.temperature: float = getattr(model_args, "temperature", 0.7)
        self.top_p: float = getattr(model_args, "top_p", 0.8)
        self.top_k: int = getattr(model_args, "top_k", 20)
        self.enable_thinking: bool = getattr(model_args, "enable_thinking", False)
        self.max_seq_len: int = getattr(model_args, "max_seq_len", 64000)
        self.presence_penalty: float = getattr(model_args, "presence_penalty", 0.5)
    
    def preprocess(self, messages: list):
        for turn in messages:
            if turn["content"] is None:
                turn["content"] = ""
            
            if "tool_calls" in turn and "gemini" not in self.model_name_or_path:
                for tc in turn["tool_calls"]:
                    tc["function"]["arguments"] = str(tc["function"]["arguments"])
                    
        return messages
    
    def token_count(self, messages):
        messages_length = sum([len(msg["content"]) for msg in messages])
        return messages_length // 3 # token len approximation

    def _truncate_messages(self, messages: list):        
        max_tokens = self.max_seq_len - self.max_new_tokens
        current_tokens = self.token_count(messages)
        
        if current_tokens <= max_tokens:
            return messages
        
        while current_tokens > max_tokens and len(messages) > 3:
            del messages[3]
            current_tokens = self.token_count(messages)
            print(f"{current_tokens=}")
        
        return messages

    def run(self, messages: list, available_tools: list = None, tool_choice: str = "auto", n: int = 1):
        messages = self.preprocess(messages)
        messages = self._truncate_messages(messages)

        if available_tools:
            response = self.client.chat.completions.create(
                model=self.model_name_or_path,
                messages=messages,
                tools=available_tools,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                n=n,
                tool_choice=tool_choice if available_tools else None,
                parallel_tool_calls=False,
                stream=False
            )

        else:
            response = self.client.chat.completions.create(
                model=self.model_name_or_path,
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                n=n,
                stream=False
            )

        return response


if __name__ == '__main__':
    model = GPT4o(model_args={})

    messages = [
        {"role": "user", "content": "hello"}
    ]
    print(model.run(messages))