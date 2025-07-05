# agent/model_factory.py
import os
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI # 导入 ChatOpenAI

def get_model(name: str = "deepseek-chat", temperature: float = 0.7, streaming: bool = True, timeout: float = 120.0) -> BaseChatModel:
    if name == "deepseek-chat":
        return ChatOpenAI(
            model="deepseek-chat", # DeepSeek 的模型名称
            api_key=os.getenv("DEEPSEEK_API_KEY"), # 确保已设置DEEPSEEK_API_KEY环境变量
            base_url="https://api.deepseek.com/v1", # DeepSeek API 的基础URL
            temperature=temperature,
            streaming=streaming,
            timeout=timeout
        )
    elif name == "deepseek-reasoner":
        return ChatOpenAI(
            model="deepseek-reasoner",  # DeepSeek 的模型名称
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            temperature=temperature,
            streaming=streaming,
            timeout=timeout
        )
    elif name == "gpt-4":
        return ChatOpenAI(model_name="gpt-4", temperature=temperature, streaming=streaming)
    # 添加更多模型判断
    else:
        raise ValueError(f"未知模型名称: {name}")
