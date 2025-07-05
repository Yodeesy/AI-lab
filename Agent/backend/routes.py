from flask import Blueprint, request, jsonify, current_app, render_template, send_from_directory
import os
import threading
import uuid
import time
from datetime import datetime
import json
import traceback # 新增导入，用于捕获完整的错误堆栈

# 导入智能体系统和模型工厂
from agent.model_factory import get_model
from agent.agents_core import initialize_agent_system, CodeGenerationState
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage # 确保 ToolMessage 导入

# 创建蓝图
main_bp = Blueprint('main', __name__)

# 全局变量或缓存用于存储正在进行的任务
running_tasks = {}
# 用于保护 running_tasks 的锁
tasks_lock = threading.Lock()


def init_agent_system_on_startup(app):
    """
    在 Flask 应用启动时初始化智能体系统和 LangGraph。
    """
    if not hasattr(app, 'langgraph_checkpointer'):
        app.langgraph_checkpointer = MemorySaver()
        app.logger.info("LangGraph MemorySaver initialized and attached to app.")

    app.logger.info("Initializing agent system (one-time setup)...")
    llm = get_model(
        name=app.config['LLM_MODEL_NAME'],
        temperature=app.config['LLM_TEMPERATURE'],
        streaming=False,
        timeout=app.config['LLM_TIMEOUT']
    )

    agents_tuple_local, compiled_graph_local = initialize_agent_system(llm)

    # 将初始化好的对象存储在 app 上，以便其他路由和线程访问
    app.agents_tuple = agents_tuple_local
    app.compiled_graph = compiled_graph_local

    app.logger.info("Agent system initialized.")


@main_bp.route('/')
def index():
    """根路由，提供前端 HTML 页面"""
    return render_template('index.html')


@main_bp.route('/api_test')
def api_test():
    """API 根路由，简单测试"""
    return jsonify({"message": "Welcome to the Code Generation Backend API!"})


@main_bp.route('/generate_code', methods=['POST'])
def generate_code():
    """
    处理代码生成请求。
    接收用户输入和可选的上传代码。
    此路由只负责接收请求并启动一个新线程来运行LangGraph。
    """
    user_input = request.form.get('user_input')
    uploaded_file = request.files.get('file')

    if not user_input:
        return jsonify({"error": "缺少 'user_input' 参数。"}), 400

    uploaded_code_content = ""
    uploaded_file_extension = ""

    if uploaded_file:
        try:
            uploaded_code_content = uploaded_file.read().decode('utf-8')
            filename = uploaded_file.filename
            if '.' in filename:
                uploaded_file_extension = filename.rsplit('.', 1)[1]
            current_app.logger.info(f"上传文件 '{filename}' 内容已成功读取。")
        except Exception as e:
            current_app.logger.error(f"读取上传文件失败: {e}")
            return jsonify({"error": f"读取上传文件失败: {e}"}), 500
    else:
        current_app.logger.info("没有上传文件。")

    max_retries = 3

    task_id = str(uuid.uuid4())

    # 立即初始化任务状态，并添加到全局字典
    with tasks_lock:
        running_tasks[task_id] = {
            "status": "pending",
            "start_time": time.time(),
            "last_update_time": time.time(),
            "log_file_path": os.path.join(current_app.config['OUT_DIR'], f"task_log_{task_id}.md"),
            "code_output_path": os.path.join(current_app.config['OUT_DIR'], f"generated_code_{task_id}.md"),
            "terminated": False, # <-- 明确初始化为 False
            "error_message": "",
            "current_code": "", # 用于存储最新代码，方便前端获取
            "test_results": "", # 用于存储最新测试结果
            "full_log": [] # 存储日志，以便在任务进行中也能返回
        }
    current_app.logger.info(f"新任务 {task_id} 已接收。状态: pending")

    app_instance = current_app._get_current_object()

    thread = threading.Thread(target=run_graph_in_thread,
                              args=(
                                  task_id,
                                  user_input,
                                  uploaded_code_content,
                                  uploaded_file_extension,
                                  max_retries,
                                  app_instance
                              ))
    thread.daemon = True
    thread.start()

    return jsonify({"message": "代码生成任务已启动。", "task_id": task_id}), 202


def run_graph_in_thread(task_id, user_input, uploaded_code_content, uploaded_file_extension, max_retries, app_instance):
    """
    在新线程中运行LangGraph。
    需要传入app_instance并在线程内部设置应用上下文。
    """
    with app_instance.app_context():
        compiled_graph = app_instance.compiled_graph
        checkpointer = app_instance.langgraph_checkpointer

        if compiled_graph is None or checkpointer is None:
            app_instance.logger.error("LangGraph system or checkpointer not initialized. Aborting task.")
            with tasks_lock:
                running_tasks[task_id]["status"] = "failed"
                running_tasks[task_id]["terminated"] = True # <-- 任务失败时也设为 True
                running_tasks[task_id]["error_message"] = "LangGraph 系统或检查点未初始化。任务已中止。"
                running_tasks[task_id]["full_log"].append(
                    SystemMessage(content=f"**System**: 错误：LangGraph 系统或检查点未初始化。任务已中止。")
                )
            return

        initial_state = CodeGenerationState(
            user_input=user_input,
            uploaded_code=uploaded_code_content,
            uploaded_file_extension=uploaded_file_extension,
            messages=[
                HumanMessage(content=user_input),
                SystemMessage(content=f"**System**: 用户上传了代码文件 (类型: {uploaded_file_extension})，请分析并纠正其中的错误:\n```\n{uploaded_code_content}\n```")
            ] if uploaded_code_content else [HumanMessage(content=user_input)],
            full_log=[
                SystemMessage(content=f"**System**: 收到用户请求。任务ID: {task_id}")
            ],
            max_retries=max_retries,
            round=0,
            current_speaker="user",
            terminated=False
        )

        app_instance.logger.info(f"任务 {task_id} 状态: running. 开始执行LangGraph。")

        graph_config = {
            "configurable": {
                "thread_id": task_id,
                "checkpointer": checkpointer
            },
            "recursion_limit": 100
        }

        final_state_values = {} # 用于存储最终状态

        try:
            for s in compiled_graph.stream(
                    initial_state,
                    config=graph_config,
                    stream_mode="updates"
            ):
                # 每次更新都从检查点获取最新状态，并同步到 running_tasks
                current_langgraph_state = compiled_graph.get_state(graph_config)
                if current_langgraph_state:
                    current_state_values = current_langgraph_state.values
                    with tasks_lock:
                        running_tasks[task_id]["last_update_time"] = time.time()
                        running_tasks[task_id]["status"] = "running" # 确保状态是运行中
                        running_tasks[task_id]["progress"] = current_state_values.get("round", 0)
                        running_tasks[task_id]["full_log"] = list(current_state_values.get("full_log", [])) # 复制列表
                        running_tasks[task_id]["current_code"] = current_state_values.get("code", "")
                        running_tasks[task_id]["test_results"] = current_state_values.get("test_results", "")
                        running_tasks[task_id]["error_message"] = current_state_values.get("error_message", "")
                        running_tasks[task_id]["terminated"] = current_state_values.get("terminated", False) # 同步 terminated 状态

                    # 如果 LangGraph 内部的 terminated 标志为 True，则提前退出循环
                    if current_state_values.get("terminated"):
                        app_instance.logger.info(f"任务 {task_id} 提前终止 (LangGraph内部标志)。")
                        break # 退出 stream 循环

            # 循环结束后，再次获取最终状态，确保是最新的
            final_state_obj = compiled_graph.get_state(graph_config)
            final_state_values = final_state_obj.values if final_state_obj else {}

            with tasks_lock:
                # 最终更新任务状态
                running_tasks[task_id]["status"] = "completed"
                running_tasks[task_id]["terminated"] = final_state_values.get("terminated", True) # 确保最终状态为 True
                running_tasks[task_id]["progress"] = final_state_values.get("round", running_tasks[task_id].get("progress", 0))
                running_tasks[task_id]["full_log"] = list(final_state_values.get("full_log", running_tasks[task_id]["full_log"]))
                running_tasks[task_id]["current_code"] = final_state_values.get("code", running_tasks[task_id]["current_code"])
                running_tasks[task_id]["test_results"] = final_state_values.get("test_results", running_tasks[task_id]["test_results"])
                running_tasks[task_id]["error_message"] = final_state_values.get("error_message", running_tasks[task_id]["error_message"])

            app_instance.logger.info(f"任务 {task_id} 执行完成。")

            output_dir = app_instance.config['OUT_DIR']
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # --- 保存生成的代码文件 ---
            generated_code = final_state_values.get("code", "")
            if generated_code:
                output_filename_code = f"generated_code_{timestamp}_{task_id[:8]}.md"
                output_filepath_code = os.path.join(output_dir, output_filename_code)

                file_ext_hint = uploaded_file_extension.lstrip('.') if uploaded_file_extension else ""
                markdown_lang_hint = ""
                if file_ext_hint in ["js", "py", "java", "c", "cpp", "html", "css", "ts", "json", "xml", "sh"]:
                    markdown_lang_hint = file_ext_hint
                elif generated_code.startswith("#!"):
                    markdown_lang_hint = "bash"
                elif "python" in generated_code.lower() and "import" in generated_code.lower():
                    markdown_lang_hint = "python"
                elif "#include" in generated_code.lower() or "int main" in generated_code.lower():
                    markdown_lang_hint = "c"

                code_to_save = f"```{markdown_lang_hint}\n{generated_code}\n```"

                try:
                    with open(output_filepath_code, "w", encoding="utf-8") as f:
                        f.write(code_to_save)
                    final_state_values["full_log"].append(
                        SystemMessage(content=f"**System**: 生成代码已保存至: {output_filepath_code} (Markdown 格式)")
                    )
                    app_instance.logger.info(f"任务 {task_id}: 代码已保存到 {output_filepath_code} (Markdown 格式)")
                except Exception as file_e:
                    final_state_values["full_log"].append(SystemMessage(content=f"**System**: 保存代码失败: {file_e}"))
                    app_instance.logger.error(f"任务 {task_id}: 保存代码失败: {file_e}")
            else:
                app_instance.logger.warning(f"任务 {task_id}: 未生成代码，跳过代码文件保存。")

            # --- 保存完整日志文件 ---
            output_filename_log = f"task_log_{timestamp}_{task_id[:8]}.md"
            output_filepath_log = os.path.join(output_dir, output_filename_log)
            try:
                with open(output_filepath_log, "w", encoding="utf-8") as f:
                    f.write(f"# 任务日志 - {task_id}\n\n")
                    f.write(f"用户输入: {final_state_values.get('user_input', '')}\n\n")
                    if final_state_values.get('uploaded_code'):
                        f.write(
                            f"上传代码 ({final_state_values.get('uploaded_file_extension', '')}):\n```\n{final_state_values['uploaded_code']}\n```\n\n")
                    f.write("## AI 思考过程和系统日志\n\n")

                    for msg in final_state_values.get('full_log', []): # 直接使用 msg 对象
                        if isinstance(msg, SystemMessage):
                            f.write(f"> **系统信息**: {msg.content}\n\n")
                        elif isinstance(msg, HumanMessage):
                            f.write(f"**用户**: {msg.content}\n\n")
                        elif isinstance(msg, AIMessage):
                            speaker_name = msg.name if msg.name else 'AI'
                            f.write(f"**AI ({speaker_name})**: {msg.content}\n\n")
                            if msg.tool_calls:
                                f.write("### AI 工具调用:\n\n")
                                for tool_call_obj in msg.tool_calls:
                                    f.write(f"- 调用工具: `{tool_call_obj.name}`\n")
                                    f.write(
                                        f"  参数: ```json\n{json.dumps(tool_call_obj.args, indent=2, ensure_ascii=False)}\n```\n\n")
                        elif isinstance(msg, ToolMessage): # 使用 ToolMessage
                            tool_name = msg.name if msg.name else '未知工具'
                            f.write(f"**工具结果 ({tool_name})**:\n```\n{msg.content}\n```\n\n")
                        else:
                            f.write(f"**未知消息类型 ({type(msg).__name__})**: {str(msg)}\n\n")

                final_state_values["full_log"].append(
                    SystemMessage(content=f"**System**: 完整日志已保存至: {output_filepath_log} (Markdown 格式)")
                )
                app_instance.logger.info(f"任务 {task_id}: 完整日志已保存到 {output_filepath_log} (Markdown 格式)")
            except Exception as log_e:
                app_instance.logger.error(f"任务 {task_id}: 保存完整日志失败: {log_e}")

        except Exception as e:
            error_trace = traceback.format_exc()
            app_instance.logger.error(f"任务 {task_id} 异常终止: {e}\n{error_trace}")

            with tasks_lock:
                running_tasks[task_id]["status"] = "failed"
                running_tasks[task_id]["terminated"] = True # <-- 任务异常终止时也设为 True
                running_tasks[task_id]["error_message"] = f"LangGraph 任务执行失败: {e}\n{error_trace}"

                # 尝试获取并保存当前的 LangGraph 状态到 running_tasks，以便前端可以显示
                try:
                    current_langgraph_state_on_error = compiled_graph.get_state(graph_config)
                    if current_langgraph_state_on_error:
                        error_state_values = current_langgraph_state_on_error.values
                        running_tasks[task_id]["progress"] = error_state_values.get("round", running_tasks[task_id].get("progress", 0))
                        running_tasks[task_id]["full_log"] = list(error_state_values.get("full_log", running_tasks[task_id]["full_log"]))
                        running_tasks[task_id]["current_code"] = error_state_values.get("code", running_tasks[task_id]["current_code"])
                        running_tasks[task_id]["test_results"] = error_state_values.get("test_results", running_tasks[task_id]["test_results"])
                        running_tasks[task_id]["error_message"] = running_tasks[task_id]["error_message"] # 保持上面设置的详细错误信息
                except Exception as ge_e:
                    app_instance.logger.error(f"任务 {task_id}: 尝试在错误中获取LangGraph状态失败: {ge_e}")
                    # 如果无法获取状态，确保日志中至少有错误信息
                    if not running_tasks[task_id]["full_log"]:
                        running_tasks[task_id]["full_log"].append(SystemMessage(content=f"**System**: 任务异常终止，无法加载详细日志。错误: {e}"))


                # 保存错误日志文件
                output_dir = app_instance.config['OUT_DIR']
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename_log = f"task_log_error_{timestamp}_{task_id[:8]}.md"
                output_filepath_log = os.path.join(output_dir, output_filename_log)
                try:
                    with open(output_filepath_log, "w", encoding="utf-8") as f:
                        f.write(f"# 任务日志 (错误) - {task_id}\n\n")
                        f.write(f"用户输入: {running_tasks[task_id].get('user_input', '')}\n\n") # 从 running_tasks 获取
                        f.write(f"## 错误信息\n\n```\n{running_tasks[task_id]['error_message']}\n```\n\n") # 从 running_tasks 获取
                        f.write("## AI 思考过程和系统日志 (截至错误发生时)\n\n")

                        # 遍历 running_tasks 中已有的 full_log
                        for msg in running_tasks[task_id].get('full_log', []):
                            if isinstance(msg, SystemMessage):
                                f.write(f"> **系统信息**: {msg.content}\n\n")
                            elif isinstance(msg, HumanMessage):
                                f.write(f"**用户**: {msg.content}\n\n")
                            elif isinstance(msg, AIMessage):
                                speaker_name = msg.name if msg.name else 'AI'
                                f.write(f"**AI ({speaker_name})**: {msg.content}\n\n")
                                if msg.tool_calls:
                                    f.write("### AI 工具调用:\n\n")
                                    for tool_call_obj in msg.tool_calls:
                                        f.write(f"- 调用工具: `{tool_call_obj.name}`\n")
                                        f.write(
                                            f"  参数: ```json\n{json.dumps(tool_call_obj.args, indent=2, ensure_ascii=False)}\n```\n\n")
                            elif isinstance(msg, ToolMessage): # 使用 ToolMessage
                                tool_name = msg.name if msg.name else '未知工具'
                                f.write(f"**工具结果 ({tool_name})**:\n```\n{msg.content}\n```\n\n")
                            else:
                                f.write(f"**未知消息类型 ({type(msg).__name__})**: {str(msg)}\n\n")
                    app_instance.logger.info(f"任务 {task_id}: 错误日志已保存到 {output_filepath_log}")
                except Exception as file_e:
                    app_instance.logger.error(f"任务 {task_id}: 尝试保存错误日志失败: {file_e}")


@main_bp.route('/get_task_status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """
    获取指定任务的当前状态和日志。
    """
    with tasks_lock:
        task = running_tasks.get(task_id)

    if not task:
        return jsonify({"error": "任务未找到。"}), 404

    # 直接从 running_tasks 中获取所有状态信息
    # 这些信息在 run_graph_in_thread 中会实时更新
    response = {
        "task_id": task_id,
        "status": task.get("status", "unknown"),
        "progress": task.get("progress", 0),
        "full_log": [msg.dict() if isinstance(msg, (SystemMessage, HumanMessage, AIMessage, ToolMessage)) else msg for msg in task.get("full_log", [])], # 确保消息是字典格式
        "current_code": task.get("current_code", ""),
        "test_results": task.get("test_results", ""),
        "error_message": task.get("error_message", ""),
        "terminated": task.get("terminated", False) # <-- 直接从 running_tasks 获取 terminated
    }

    return jsonify(response), 200


@main_bp.route('/get_output_file/<task_id>', methods=['GET'])
def get_output_file(task_id):
    task_info = running_tasks.get(task_id)
    if not task_info or not os.path.exists(task_info["code_output_path"]):
        return jsonify({"status": "error", "message": "Output file not found"}), 404

    file_name = os.path.basename(task_info["code_output_path"])
    return send_from_directory(current_app.config['OUT_DIR'], file_name, as_attachment=True)


@main_bp.route('/download_log/<task_id>', methods=['GET'])
def download_log_file(task_id):
    """
    提供指定任务的完整日志文件下载。
    """
    with tasks_lock:
        task_info = running_tasks.get(task_id)

    if not task_info or not os.path.exists(task_info["log_file_path"]):
        return jsonify({"status": "error", "message": "Log file not found"}, 404)

    file_name = os.path.basename(task_info["log_file_path"])
    return send_from_directory(current_app.config['OUT_DIR'], file_name, as_attachment=True)


def cleanup_tasks_scheduled():
    with tasks_lock:
        keys_to_delete = []
        for task_id, task_info in running_tasks.items():
            # 清理已终止（完成或失败）且长时间未更新的任务
            if task_info.get("terminated", False) and (time.time() - task_info["last_update_time"]) > 3600: # 1小时
                keys_to_delete.append(task_id)
            # 也可以考虑清理长时间运行但未终止的任务（例如，卡住的任务），但需要更复杂的逻辑判断

        for task_id in keys_to_delete:
            del running_tasks[task_id]
            current_app.logger.info(f"清理了过期任务: {task_id}")