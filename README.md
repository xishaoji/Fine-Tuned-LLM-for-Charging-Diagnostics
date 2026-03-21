# 充电桩测试故障排除模型微调脚本

基于真实充电桩整桩测试与平台对接日志，采用 `Unsloth` + `vLLM` 构建的工业级垂直大模型推理服务。旨在解决自动化测试与异常排障中，协议报文解析耗时长、分析繁杂等问题。

在充电桩物联网平台对接（如 OCPP、国标协议）以及整桩集成压测中，异常报文的定位高度依赖人工经验。传统的正则匹配或自动化脚本难以应对复杂多变的非标故障场景。

结合多年全栈测试与平台交互经验，将海量心跳异常、启动交互失败、BMS充电交互等真实业务日志结构化，通过低成本云端算力调度，微调出精通协议栈的小模型，未来将以 API 的形式接入现有的自动化压测平台。

## ✨ 核心特性

- **🚀 极速低成本微调：** 深度集成 **Unsloth** 框架，通过 Triton 算子重写与极致的梯度检查点优化，在单张云端 RTX 3090 (24G) 上即可高效完成 7B/14B 级基座模型（如 Qwen2.5）的 QLoRA 指令微调。
- **⚡ 部署推理引擎：** 引入 **vLLM** 引擎。利用 `PagedAttention` 技术彻底消除显存碎片，支持高并发自动化测试脚本的同时调用。
- **🔌 动态 Adapter 热插拔：** 采用 LoRA 权重动态加载机制，无需合并上百 GB 的完整模型权重。
- **🌐 OpenAI 兼容接口：** 提供完全兼容 OpenAI 规范的 RESTful API，业务端零学习成本即可接入。

## 🏗️ 架构设计

项目遵循标准的 AI 应用工程落地范式：
1. **Data Layer:** 清洗平台交互日志，构建 `messages` (ChatML) 格式的多轮对话指令集。
2. **Training Layer:** 使用 `SFTTrainer` + `Unsloth` 在云端 Linux 环境进行 4-bit 量化微调。
3. **Serving Layer:** 通过 `vLLM` 挂载 LoRA 权重，暴露 8000 端口提供高可用 API 服务。

---

## 🚀 快速开始

### 1. 云端环境准备 (推荐 AutoDL)
建议租用配备 CUDA 12.1 的 Ubuntu 基础镜像。克隆本项目后，一键配置底层依赖：

```bash
git clone https://github.com/xishaoji/Fine-Tuned-LLM-for-Charging-Diagnostics.git
cd Fine-Tuned-LLM-for-Charging-Diagnostics
bash install_env.sh
```
### 2. 领域数据准备
将清洗好的业务数据放入根目录，确保其符合现代大模型的多轮对话格式：

样例数据文件：pile_dataset_messages.json

### 3. 开启 Unsloth 极速微调
执行训练脚本。该脚本会自动加载基座模型并注入 LoRA 适配器开始炼丹：

```Bash
python train_pile_model.py
```
训练完成后，LoRA 权重将自动保存在 ./pile_lora_model 目录下。

### 4. vLLM 高并发服务部署
利用 vLLM 引擎拉起 OpenAI 兼容的 API 服务：

```Bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --enable-lora \
    --lora-modules pile-assistant=./pile_lora_model \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.9 \
    --host 0.0.0.0 \
    --port 8000

```

服务启动后，在你的自动化测试脚本中，可以直接像调用 ChatGPT 一样请求诊断：

```Python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

response = client.chat.completions.create(
    model="pile-assistant",
    messages=[
        {"role": "system", "content": "你是一个资深的充电桩通信协议与整桩测试专家。"},
        {"role": "user", "content": "场景：平台下发远程启停指令失败。\n报文：{'action': 'RemoteStart', 'status': 'Rejected'}\n请分析排查方向。"}
    ],
    temperature=0.1
)

print(response.choices[0].message.content)
```
📌 未来规划 (TODO)
接入 RAG (检索增强生成)： 将长篇幅的国标文档 (GB/T 27930) 和车企私有协议 PDF 向量化，通过 LangChain 结合本地知识库减少模型幻觉。

Web 诊断可视化面板： 使用 Gradio 或 Streamlit 开发前端交互页面，方便现场非研发测试人员上传日志文件进行诊断。

多模态扩展： 探索支持上传充电桩示波器波形截图进行异常诊断的能力。