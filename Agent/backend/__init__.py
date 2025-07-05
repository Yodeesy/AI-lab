# backend/__init__.py

from flask import Flask
from flask_cors import CORS
from flask_apscheduler import APScheduler
import os
import logging

# 从本包导入配置获取函数
from .config import get_config_class

# 导入 routes 模块中定义的初始化函数
# 注意：我们在这里导入 init_agent_system_on_startup，它将负责全局初始化
from .routes import main_bp, cleanup_tasks_scheduled, init_agent_system_on_startup

def create_app():
    """
    创建并配置 Flask 应用实例。
    """
    # 获取当前文件 (backend/__init__.py) 的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 从当前目录向上走到项目根目录
    project_root = os.path.abspath(os.path.join(current_dir, '..'))

    # 现在构建 templates 和 static 文件夹的完整路径
    template_dir = os.path.join(project_root, 'frontend', 'templates')
    static_dir = os.path.join(project_root, 'frontend', 'static')

    # 打印这些路径以进行调试 (在生产环境中可以移除)
    # print(f"Calculated project_root: {project_root}")
    # print(f"Calculated template_dir: {template_dir}")
    # print(f"Calculated static_dir: {static_dir}")

    app = Flask(__name__,
                template_folder=template_dir,
                static_folder=static_dir,
                static_url_path='/static')

    # 加载配置
    app_config_class = get_config_class()
    app.config.from_object(app_config_class)

    # 进一步处理 UPLOAD_FOLDER 和 OUTPUT_FOLDER 的绝对路径
    app.config['UPLOAD_FOLDER'] = os.path.join(project_root, app.config['UPLOAD_FOLDER_REL'])
    app.config['OUT_DIR'] = os.path.join(project_root, app.config['OUTPUT_FOLDER_REL']) # 注意：为了统一，建议在 config.py 中将 OUTPUT_FOLDER_REL 改为 OUT_DIR_REL，或者在这里将 OUTPUT_FOLDER 改为 OUT_DIR
    app.config['OUTPUT_FOLDER'] = app.config['OUT_DIR'] # 保持兼容性，指向同一个目录

    # 确保文件上传和输出目录存在
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUT_DIR'], exist_ok=True) # 使用 OUT_DIR
    app.logger.info(f"确保上传目录存在: {app.config['UPLOAD_FOLDER']}")
    app.logger.info(f"确保输出目录存在: {app.config['OUT_DIR']}")


    # 初始化 CORS
    CORS(app, resources={r"/*": {"origins": app.config["CORS_ALLOW_ORIGINS"]}})


    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger('werkzeug').setLevel(logging.INFO)
    app.logger.info("日志系统已配置。")

    # === 新增：在应用上下文初始化智能体系统 ===
    # 这一步至关重要，确保 LLM 和 LangGraph 在所有请求处理前被初始化且全局可用
    with app.app_context():
        init_agent_system_on_startup(app)
        app.logger.info("智能体系统和LangGraph已初始化。")

    # 初始化 APScheduler
    scheduler = APScheduler()
    scheduler.init_app(app)
    scheduler.start()
    app.logger.info("APScheduler 已启动。")


    # 注册蓝图
    app.register_blueprint(main_bp)


    # 添加定时清理任务
    if not scheduler.get_job('do_cleanup_tasks'):
        scheduler.add_job(
            id='do_cleanup_tasks',
            func=cleanup_tasks_scheduled,
            trigger='interval',
            hours=1,
            timezone=app.config['SCHEDULER_TIMEZONE']
        )
        app.logger.info(f"已添加定时任务: 'do_cleanup_tasks' (每小时清理过期任务，时区: {app.config['SCHEDULER_TIMEZONE']})")


    # 打印一些最终配置信息
    app.logger.info(f"App running in {os.getenv('FLASK_ENV', 'development')} mode.")
    app.logger.info(f"LLM Model: {app.config['LLM_MODEL_NAME']}")
    app.logger.info(f"Max Retries: {app.config['MAX_RETRIES']}")
    app.logger.info(f"Upload Folder: {app.config['UPLOAD_FOLDER']}")
    app.logger.info(f"Output Folder (OUT_DIR): {app.config['OUT_DIR']}") # 统一使用 OUT_DIR
    app.logger.info(f"CORS Allowed Origins: {app.config['CORS_ALLOW_ORIGINS']}")
    app.logger.info(f"Template Folder (used by Flask): {app.template_folder}")
    app.logger.info(f"Static Folder (used by Flask): {app.static_folder}")


    return app