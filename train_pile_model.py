from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template

# 1. 核心参数设置
max_seq_length = 2048 # 最大序列长度，Unsloth 支持 RoPE 缩放, 可以设置大点
dtype = None # 自动检测精度 (Float16 或 Bfloat16)
load_in_4bit = True # 开启 4bit QLoRA 极致省显存

# 2. 使用 Unsloth 的极速接口加载模型和 Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-7B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. 注入 LoRA 适配器 (不修改原模型权重，训练时只更新 LoRA 权重)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # LoRA 秩，通常 8 或 16
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Unsloth 的优化使得我们可以不使用 dropout 也不易过拟合
    bias = "none",
    use_gradient_checkpointing = "unsloth", # 优化显存的梯度检查点，Unsloth 专用模式
    random_state = 3407,
)

# 4. 数据格式化：将 messages 格式化为模型认识的 ChatML 模板
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-2.5", # 使用与基座模型匹配的模板
)

def formatting_prompts_func(examples):
    # Unsloth 内置了专门处理 messages 结构的函数
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

# 加载你在上一步准备好的充电桩测试数据集
dataset = load_dataset("json", data_files="pile_dataset_messages.json", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# 5. 召唤 SFTTrainer 进行训练
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2, # 数据预处理进程数
    packing = False, # 如果数据短小，设为 False 即可
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # 这里为了测试写了步数，实际可以用 num_train_epochs = 3
        learning_rate = 2e-4,
        fp16 = not getattr(model, "is_bfloat16_supported", lambda: False)(),
        bf16 = getattr(model, "is_bfloat16_supported", lambda: False)(),
        logging_steps = 1,
        optim = "adamw_8bit", # 使用 8bit 优化器进一步省显存
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 开始炼丹！
trainer_stats = trainer.train()

# 6. 保存微调后的 LoRA 权重
model.save_pretrained_merged("pile_lora_model", tokenizer, save_method = "lora")
print("✅ 模型训练与保存完毕！")