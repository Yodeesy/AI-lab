# backend/config.py

import os


class Config:
    """
    Base configuration class.
    所有配置项均尝试从环境变量获取，若未设置则使用默认值。
    """
    # Flask 应用通用配置
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "your_super_secret_key_please_change_this")  # 用于会话管理和安全
    JSON_AS_ASCII = False  # 允许返回中文 JSON，否则jsonify会转义中文

    # LLM 模型相关的配置
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "deepseek-chat")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.7))
    LLM_STREAMING = os.getenv("LLM_STREAMING", "True").lower() == "true"
    LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", 120.0))

    # 智能体执行相关的配置
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", 5))  # 代码生成和调试的最大尝试次数
    EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", 10))  # 代码执行工具的超时时间 (秒)

    # 文件上传和保存路径（这里定义为相对路径，绝对路径将在 __init__.py 中计算）
    UPLOAD_FOLDER_REL = "uploads"  # 用户上传文件的临时存放相对路径
    OUTPUT_FOLDER_REL = "out"  # AI生成代码的输出相对路径

    # 跨域请求 (CORS) 设置
    CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*")  # 生产环境中应替换为您的前端域名

    # APScheduler 配置
    SCHEDULER_API_ENABLED = True
    SCHEDULER_TIMEZONE = os.getenv("SCHEDULER_TIMEZONE", "Asia/Shanghai")  # 根据您的时区调整


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


def get_config_class():
    """
    根据 FLASK_ENV 环境变量返回相应的配置类。
    """
    env = os.getenv("FLASK_ENV", "development")
    if env == "production":
        return ProductionConfig
    return DevelopmentConfig