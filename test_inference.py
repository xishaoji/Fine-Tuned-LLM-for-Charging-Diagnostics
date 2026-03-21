from unsloth import FastLanguageModel
from transformers import TextStreamer

# 1. 加载我们微调好的模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "pile_lora_model", # 这里指向你刚才训练保存的文件夹
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# 2. 开启 Unsloth 的极速推理模式 (必须调用)
FastLanguageModel.for_inference(model)

# 3. 构造一条测试问题 (按照 messages 格式)
system_prompt = "你是一个资深的充电桩通信协议与整桩测试专家。请根据提供的报文内容或故障现象，给出专业的排查建议。"
user_input = "场景：充电桩刷卡后无响应\n报文内容：{'action': 'Authorize', 'id_tag': 'FFFFFFFF'}\n请分析该卡号被拒的原因。"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_input}
]

# 4. 使用模板格式化输入并转为张量
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # 告诉模型轮到它生成回答了
    return_tensors = "pt",
).to("cuda")

# 5. 流式输出回答 (像 ChatGPT 一样一个字一个字蹦出来)
print("\n🤖 模型解答中...\n" + "="*40)
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 512, temperature = 0.3)
print("\n" + "="*40)