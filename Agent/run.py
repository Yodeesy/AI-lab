# run.py

import os
from backend import create_app # 从 backend 包导入 create_app 函数

# 创建 Flask 应用实例
app = create_app()

if __name__ == '__main__':
    # 从配置中获取端口和调试模式
    host = os.environ.get('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_RUN_PORT', 5000))
    debug = app.config.get('DEBUG', False) # 默认从 config 获取，如果 config 未设置则为 True

    # 运行 Flask 应用
    app.run(debug=debug, host=host, port=port, use_reloader=False)