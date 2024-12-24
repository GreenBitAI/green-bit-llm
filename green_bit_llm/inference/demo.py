from llm_inference import LLMInference

if __name__ == "__main__":
    # model_name = "/workspace/models/Qwen2.5-7B-Instruct"
    model_name = "/workspace/green-bit-llm/models/Qwen-1.5-14B-layer-mix-bpw-2.5"
    prompts = [
        "user:你是谁\n assistant:",
        "user:介绍一下自己\n assistant:"
        ]
    llm = LLMInference(model_name)
    llm.generate(prompts)