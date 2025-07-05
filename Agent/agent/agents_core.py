import re
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Union, Annotated, Any, Optional
from pydantic import BaseModel
import operator
import json
import uuid

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 从本包内导入 model_factory 和 tools
from .model_factory import get_model
from .tools import execute_code, web_search, code_lint_check, fetch_web_content

# 导入嵌入工具
from backend.embedding import get_vector_store # 导入向量存储获取函数
from langchain_core.documents import Document # 导入 Document

from flask import current_app


# === 1. 定义 LangGraph 的状态 ===
# CodeGenerationState 是 LangGraph 中传递的状态对象
class CodeGenerationState(BaseModel):
    user_input: str = ""
    uploaded_code: str = ""
    uploaded_file_extension: str = ""
    code: str = ""  # 这里应该存储生成的代码
    test_code: str = ""
    test_results: str = ""
    error_message: str = ""
    messages: List[BaseMessage] = field(default_factory=list)  # 存储用于LLM对话的精简历史
    full_log: List[BaseMessage] = field(default_factory=list)  # 存储完整的、用于展示给用户的日志
    round: int = 0
    max_retries: int = 0
    current_speaker: str = "user"
    terminated: bool = False
    next_node_decision: Optional[str] = None  # <-- 决策者决定下一步流向

    class Config:
        arbitrary_types_allowed = True


# 辅助函数：从 Markdown 中提取代码
def extract_code_from_markdown(text: str) -> str:
    """
    从 Markdown 字符串中提取代码块。
    如果存在多个代码块，则将它们连接起来。
    如果没有代码块，则返回原始文本。
    """
    # 用于查找代码块的正则表达式: ```[lang]\nCODE\n```
    code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', text, re.DOTALL)
    if code_blocks:
        return "\n".join(block.strip() for block in code_blocks)
    # 如果没有代码块，则返回原始文本，假设它是直接的代码或纯文本
    return text.strip()


# === 2. 定义智能体类 ===
class CodeAgent:
    """
    一个基础的智能体类，用于代码生成工作流。
    每个智能体都有一个名称、一个LLM模型和一套可用的工具。
    """

    def __init__(self, name: str, model: BaseChatModel, tools: List, system_prompt: str,
                 include_uploaded_code_in_prompt: bool = False,
                 enable_long_term_memory: bool = False): # <-- 保持 enable_long_term_memory 参数
        self.name = name
        self.model = model
        self.tools = tools
        self.include_uploaded_code_in_prompt = include_uploaded_code_in_prompt
        self.enable_long_term_memory = enable_long_term_memory # 启用长期记忆控制

        # 绑定工具到模型
        self.model_with_tools = self.model.bind_tools(self.tools)

        # 定义智能体的 Prompt 模板
        # 根据 include_uploaded_code_in_prompt 来动态构建 system prompt
        system_messages_parts = [("system", system_prompt)]
        if self.include_uploaded_code_in_prompt:
            # 插入一个占位符，用于接收 uploaded_code_info_placeholder
            system_messages_parts.append(MessagesPlaceholder(variable_name="uploaded_code_info_placeholder"))

        # 新增一个占位符用于长期记忆检索结果
        if self.enable_long_term_memory:
            system_messages_parts.append(MessagesPlaceholder(variable_name="retrieved_memory")) # <-- 保持

        self.prompt_template = ChatPromptTemplate.from_messages(
            system_messages_parts + [
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad")  # 用于工具调用
            ]
        )

    def _retrieve_long_term_memory(self, query: str, k: int = 3) -> List[Document]:
        """根据查询从向量数据库检索相关文档"""
        app_instance = current_app._get_current_object()
        try:
            vector_store = get_vector_store()
            results = vector_store.similarity_search(query, k=k)
            app_instance.logger.info(f"DEBUG_AGENT: 长期记忆检索查询 '{query[:50]}...'，检索到 {len(results)} 条记忆。")
            return results
        except Exception as e:
            app_instance.logger.error(f"DEBUG_AGENT: 检索长期记忆失败: {e}")
            return []


    def invoke(self, state: CodeGenerationState) -> Dict[str, Any]:
        """
        核心方法：智能体根据当前状态生成响应，可能包含工具调用。
        直接修改 state 对象，而不是返回字典。
        """
        # 确保在 Agent 内部可以访问 Flask 应用上下文
        app_instance = current_app._get_current_object()

        app_instance.logger.info(f"DEBUG_AGENT: {self.name.capitalize()} Agent received state (round {state.round}): {state.model_dump_json(indent=2)}")
        if state.uploaded_code:
            app_instance.logger.info(f"DEBUG_AGENT: {self.name.capitalize()} sees uploaded_code (length: {len(state.uploaded_code)})")
        else:
            app_instance.logger.info(f"DEBUG_AGENT: {self.name.capitalize()} does NOT see uploaded_code.")


        chat_messages_for_prompt: List[BaseMessage] = []
        agent_scratchpad_content: List[BaseMessage] = []

        for msg in state.messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                # 包含工具调用的AIMessage属于scratchpad
                agent_scratchpad_content.append(msg)
            elif isinstance(msg, ToolMessage): # LangChain推荐使用 ToolMessage
                # ToolMessage (工具输出)属于scratchpad
                agent_scratchpad_content.append(msg)
            # 兼容旧的 FunctionMessage，如果模型仍然返回
            elif hasattr(msg, 'type') and msg.type == "function":
                agent_scratchpad_content.append(msg)
            elif isinstance(msg, SystemMessage) and ("**调用工具**" in msg.content or "**工具**" in msg.content):
                # 以前添加的描述工具调用的SystemMessage，这些是日志而非LLM输入，排除
                pass
            else:
                # 其他消息 (Human, System, AIMessage无工具调用) 属于主要对话
                chat_messages_for_prompt.append(msg)

        # 如果有错误信息，且不是当前智能体的工具执行结果造成的，添加到 LLM 输入
        # 避免在每次循环都重复添加
        if self.name in ["executor", "debugger"] and state.error_message:
            # 检查 messages 中是否已包含最新的错误信息，避免重复
            if not any(isinstance(msg, SystemMessage) and "当前错误信息" in msg.content and state.error_message in msg.content for msg in chat_messages_for_prompt[-5:]):
                chat_messages_for_prompt.append(SystemMessage(content=f"**当前错误信息**: {state.error_message}"))

        inputs = {
            "messages": chat_messages_for_prompt,
            "agent_scratchpad": agent_scratchpad_content,  # 传递准备好的 scratchpad
        }

        # 如果需要包含 uploaded_code_info，则构建并添加它到 inputs
        if self.include_uploaded_code_in_prompt:
            if state.uploaded_code:
                uploaded_code_info = f"用户上传了以下代码 (类型: {state.uploaded_file_extension}):\n```\n{state.uploaded_code}\n```\n请特别注意这段代码进行分析和纠错。"
                inputs["uploaded_code_info_placeholder"] = [SystemMessage(content=uploaded_code_info)]
            else:
                # 如果配置为包含但没有代码，也发送一个系统消息明确说明
                inputs["uploaded_code_info_placeholder"] = [SystemMessage(content="没有上传代码文件。")]

        # --- 长期记忆检索逻辑 ---
        if self.enable_long_term_memory:
            # 构建检索查询：可以综合用户输入、当前错误信息、最近的 AI 回复等
            retrieval_query = state.user_input
            if state.error_message:
                retrieval_query += f" 错误信息：{state.error_message}"
            # 可以根据需要添加更多上下文到查询中，例如：
            # if state.code:
            #     retrieval_query += f" 代码概要：{state.code[:200]}..." # 提取部分代码作为查询

            retrieved_docs = self._retrieve_long_term_memory(retrieval_query)
            if retrieved_docs:
                retrieved_content = "\n\n".join([
                    f"内容: {doc.page_content}\n元数据: {doc.metadata}" for doc in retrieved_docs
                ])
                retrieved_message = SystemMessage(
                    content=f"**以下是根据当前上下文从知识库中检索到的相关信息，请参考：**\n---\n{retrieved_content}\n---"
                )
                inputs["retrieved_memory"] = [retrieved_message]
            else:
                inputs["retrieved_memory"] = [] # 如果没有检索到，传入空列表

        agent_chain = self.prompt_template | self.model_with_tools
        agent_response = agent_chain.invoke(inputs)

        # 将LLM的响应添加到消息和日志中
        state.messages.append(agent_response)  # AIMessage (LLM的实际回复)
        state.full_log.append(
            AIMessage(content=f"**{self.name.capitalize()}**: {agent_response.content}", name=self.name.capitalize()))

        # 检查是否有工具调用，仅将描述工具调用的日志添加到 full_log
        if agent_response.tool_calls:
            for tool_call in agent_response.tool_calls:
                tool_name = getattr(tool_call, 'name', tool_call.get('name', '未知工具') if isinstance(tool_call, dict) else '未知工具')
                tool_args = getattr(tool_call, 'args', tool_call.get('args', {}) if isinstance(tool_call, dict) else {})

                tool_call_log = SystemMessage(
                    content=f"**{self.name.capitalize()}** 调用工具: {tool_name}，参数: {json.dumps(tool_args, ensure_ascii=False, indent=2)}")
                state.full_log.append(tool_call_log)

        # === 专门针对 coder 智能体更新 'code' 字段 ===
        if self.name == "coder":
            generated_code = extract_code_from_markdown(agent_response.content)
            state.code = generated_code

        state.current_speaker = self.name
        state.round += 1
        return state

    def execute_tools(self, state: CodeGenerationState) -> Dict[str, Any]:
        """
        执行智能体调用工具后的动作。
        直接修改 state 对象，而不是返回字典。
        """
        app_instance = current_app._get_current_object()
        tool_results: List[ToolMessage] = []
        error_message = ""

        app_instance.logger.info(f"--- {self.name.capitalize()} 正在执行工具 ---")

        # 查找最新的包含工具调用的 AIMessage
        latest_ai_message: Optional[AIMessage] = None
        for msg in reversed(state.messages):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                latest_ai_message = msg
                break

        if latest_ai_message and latest_ai_message.tool_calls:
            for tool_call in latest_ai_message.tool_calls:
                tool_name = getattr(tool_call, 'name', tool_call.get('name', '未知工具') if isinstance(tool_call, dict) else '未知工具')
                tool_args = getattr(tool_call, 'args', tool_call.get('args', {}) if isinstance(tool_call, dict) else {})
                tool_id = getattr(tool_call, 'id', tool_call.get('id', None) if isinstance(tool_call, dict) else None)

                if tool_id is None:
                    tool_id = f"dummy_id_{tool_name}_{uuid.uuid4().hex[:8]}"

                try:
                    tool_to_execute = next(
                        (t for t in self.tools if t.name == tool_name), None
                    )
                    if tool_to_execute:
                        if tool_name == "execute_code":
                            # 仅当 tool_args 中没有 language_hint 且 state 中有扩展名时添加
                            if "language_hint" not in tool_args and state.uploaded_file_extension:
                                tool_args["language_hint"] = state.uploaded_file_extension.lstrip('.')
                            app_instance.logger.info(f"执行工具: {tool_name}，参数: {tool_args}")
                            result = tool_to_execute.invoke(tool_args)
                        else:
                            app_instance.logger.info(f"执行工具: {tool_name}，参数: {tool_args}")
                            result = tool_to_execute.invoke(tool_args)

                        func_msg = ToolMessage(content=str(result), tool_call_id=tool_id, name=tool_name) # 添加name参数
                        tool_results.append(func_msg)

                        state.full_log.append(SystemMessage(
                            content=f"**{self.name.capitalize()}** 工具 '{tool_name}' (ID: {tool_id}) 返回: {json.dumps(result, ensure_ascii=False, indent=2)}"))

                        if tool_name == "execute_code":
                            if result and result.get("error"):
                                error_message = result["error"]
                                state.full_log.append(
                                    SystemMessage(
                                        content=f"**{self.name.capitalize()}** 检测到执行错误: {error_message}。"))
                            else:
                                if result and "stdout" in result:
                                    state.test_results = result["stdout"]
                                error_message = ""
                                state.full_log.append(
                                    SystemMessage(content=f"**{self.name.capitalize()}** 代码执行成功。"))
                    else:
                        error_message = f"未找到工具 '{tool_name}'。请检查工具名称是否正确注册。"
                        state.full_log.append(
                            SystemMessage(content=f"**{self.name.capitalize()}** 错误：{error_message}。"))

                except Exception as e:
                    error_message = f"执行工具 '{tool_name}' (ID: {tool_id}) 时发生错误: {e}"
                    state.full_log.append(
                        SystemMessage(content=f"**{self.name.capitalize()}** 错误：{error_message}。"))
        else:
            app_instance.logger.warning(f"{self.name.capitalize()}: 未在最新 AIMessage 中找到要执行的工具调用。")
            if not state.error_message:
                state.error_message = "执行器未能找到要执行的工具调用，可能上一个智能体没有正确发出工具调用。"

        state.messages.extend(tool_results)
        state.error_message = error_message
        state.current_speaker = self.name
        state.round += 1
        return state


# === 决策者智能体 ===
class DeciderAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(
                """你是一个智能体流程决策者。根据当前任务的状态、历史对话和最新的智能体输出，你必须决定下一步应该由哪个智能体来处理任务。
                你的响应必须是以下一个单词：'planner', 'coder', 'executor', 'debugger', 'end'。

                - 如果需要进一步的规划或用户需求理解，返回 'planner'。
                - 如果需要编写或修改代码（例如，根据规划或调试结果），返回 'coder'。
                - 如果代码已准备好运行或验证，返回 'executor'。
                - 如果 **executor 智能体刚完成其操作**（无论执行成功或失败），**下一步总是返回 'debugger'**。
                - 如果任务已完成（例如，代码成功执行且符合预期，且调试器也完成了它的职责）或无法继续，返回 'end'。

                请严格按照指令，只返回一个单词。
                """
            ),
            MessagesPlaceholder(variable_name="messages"),  # 包含之前的对话和工具结果
            HumanMessage(
                content="根据以上信息，请决定下一步应该交给哪个智能体处理（'planner', 'coder', 'executor', 'debugger', 'end'）："),
        ])
        self.runnable = self.prompt | self.llm

    def execute(self, state: CodeGenerationState) -> Dict[str, Any]:
        app_instance = current_app._get_current_object()
        app_instance.logger.info(f"--- 决策者智能体正在做决策 ---")

        # 准备传递给决策者的消息
        messages_for_decider = state.messages.copy()

        if state.error_message:
            if not any(isinstance(msg, SystemMessage) and "上次执行结果有错误" in msg.content and state.error_message in msg.content for msg in messages_for_decider[-5:]):
                messages_for_decider.append(SystemMessage(content=f"上次执行结果有错误: {state.error_message}"))

        if state.code and not any(
                isinstance(msg, SystemMessage) and "当前代码概要" in msg.content for msg in messages_for_decider[-5:]):
            messages_for_decider.append(SystemMessage(content=f"当前代码概要: 已有代码，长度 {len(state.code)}"))

        # 添加上传代码信息（如果存在）到决策者的输入中，帮助其理解上下文
        if state.uploaded_code and not any(
                isinstance(msg, SystemMessage) and "原始上传代码" in msg.content for msg in messages_for_decider[-5:]):
            messages_for_decider.append(SystemMessage(content=f"原始上传代码 (类型: {state.uploaded_file_extension}):\n```\n{state.uploaded_code}\n```"))


        messages_for_decider.append(SystemMessage(content=f"上一个发言者: {state.current_speaker.capitalize()}"))

        last_speaker = state.current_speaker

        if last_speaker == "executor":
            decision = "debugger"
            app_instance.logger.info(f"路由: 检测到 Executor 刚完成操作，强制路由到 'debugger'。")
        else:
            try:
                response = self.runnable.invoke({"messages": messages_for_decider})
                decision = response.content.strip().lower()
            except Exception as e:
                app_instance.logger.error(f"决策者LLM调用失败: {e}. 默认设置为 'end'")
                decision = "end"

        valid_decisions = ["planner", "coder", "executor", "debugger", "end"]
        if decision not in valid_decisions:
            app_instance.logger.warning(f"决策者返回了非法决策: '{decision}'. 默认设置为 'end'")
            decision = "end"

        state.full_log.append(AIMessage(content=f"决策者决定下一步是: {decision}", name="Decider"))
        state.messages.append(AIMessage(content=f"决策者决定下一步是: {decision}", name="Decider"))
        state.current_speaker = "Decider"
        state.next_node_decision = decision
        state.round += 1
        return state


# === 3. 定义 LangGraph 工作流 ===

def create_code_generation_graph(
        planner_agent: CodeAgent,
        coder_agent: CodeAgent,
        executor_agent: CodeAgent,
        debugger_agent: CodeAgent,
        decider_agent: DeciderAgent,
        checkpointer: MemorySaver
) -> StateGraph:
    """
    构建并返回 LangGraph 图。
    """
    workflow = StateGraph(CodeGenerationState)

    # 添加智能体节点
    workflow.add_node("planner", planner_agent.invoke)
    workflow.add_node("coder", coder_agent.invoke)
    workflow.add_node("executor", executor_agent.invoke)
    workflow.add_node("debugger", debugger_agent.invoke)
    workflow.add_node("decider", decider_agent.execute)

    # 添加工具执行节点
    workflow.add_node("planner_tool_executor", planner_agent.execute_tools)
    workflow.add_node("coder_tool_executor", coder_agent.execute_tools)
    workflow.add_node("executor_tool_executor", executor_agent.execute_tools)
    workflow.add_node("debugger_tool_executor", debugger_agent.execute_tools)

    # === 定义图的入口和基本流程 ===
    workflow.set_entry_point("planner")

    def custom_router(state: CodeGenerationState) -> str:
        current_speaker = state.current_speaker
        if not state.messages:
            current_app.logger.warning("路由: state.messages 为空，无法判断上一个消息类型。")
            return "decider"

        last_message = state.messages[-1]

        # 1. 如果上一个智能体的响应是工具调用，路由到其工具执行器
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            current_app.logger.info(
                f"路由: '{current_speaker.capitalize()}' 调用工具，路由到 '{current_speaker}_tool_executor'。")
            return f"{current_speaker}_tool_executor"

        # 2. 如果上一个消息是工具执行结果 (ToolMessage 或兼容 FunctionMessage)，则返回给调用工具的智能体继续思考
        elif isinstance(last_message, ToolMessage) or (hasattr(last_message, 'type') and last_message.type == 'function'):
            current_app.logger.info(f"路由: 工具执行完成，返回给 '{current_speaker.capitalize()}' 智能体继续思考。")
            return current_speaker

        # 3. 如果当前发言人是决策者，则根据决策者的结果进行路由
        elif current_speaker == "Decider":
            decision = state.next_node_decision
            current_app.logger.info(f"路由: 决策者决定下一步是 '{decision}'。")
            return decision

        # 4. 其他智能体完成思考（没有工具调用，也不是工具结果），将其路由到决策者
        current_app.logger.info(f"路由: '{current_speaker.capitalize()}' 完成思考，路由到 Decider。")
        return "decider"

    # 定义从 Planner 到 Decider 的初始路径
    workflow.add_conditional_edges(
        "planner",
        custom_router,
        {
            "planner_tool_executor": "planner_tool_executor",
            "decider": "decider",
            END: END
        }
    )
    workflow.add_edge("planner_tool_executor", "planner")

    # 定义从 Coder 到 Decider 的路径
    workflow.add_conditional_edges(
        "coder",
        custom_router,
        {
            "coder_tool_executor": "coder_tool_executor",
            "decider": "decider",
            END: END
        }
    )
    workflow.add_edge("coder_tool_executor", "coder")

    # 定义从 Executor 到 Decider 的路径
    workflow.add_conditional_edges(
        "executor",
        custom_router,
        {
            "executor_tool_executor": "executor_tool_executor",
            "decider": "decider",
            END: END
        }
    )
    workflow.add_edge("executor_tool_executor", "executor")

    # 定义从 Debugger 到 Decider 的路径
    workflow.add_conditional_edges(
        "debugger",
        custom_router,
        {
            "debugger_tool_executor": "debugger_tool_executor",
            "decider": "decider",
            END: END
        }
    )
    workflow.add_edge("debugger_tool_executor", "debugger")

    # 定义从 Decider 节点到其他智能体的路由
    workflow.add_conditional_edges(
        "decider",
        lambda state: state.next_node_decision,
        {
            "planner": "planner",
            "coder": "coder",
            "executor": "executor",
            "debugger": "debugger",
            "end": END
        }
    )

    # 编译工作流，并传入 checkpointer
    return workflow.compile(checkpointer=checkpointer)


# === 4. 系统初始化函数 ===
def initialize_agent_system(llm: BaseChatModel) -> Tuple[Any, Any]:
    """
    初始化所有智能体并构建 LangGraph。
    """
    # 定义所有可用的工具
    # 将 fetch_web_content 添加到工具列表中
    all_tools = [execute_code, web_search, code_lint_check, fetch_web_content] # <-- 新增 fetch_web_content

    # 定义各个智能体的角色提示
    planner_system_prompt = """
    你是一个代码生成任务的规划者。你的任务是接收用户需求，并根据需求和**可能从知识库中检索到的相关信息**进行初步的分解和规划。
    你的计划应该包含完成任务所需的所有逻辑步骤，但**绝不能直接编写任何代码**。
    也**不要提及任何具体的智能体名称或工具**。
    你的目标是为下一步的通用代码实现提供明确的、高层面的指导。
    如果用户上传了代码，**请你优先分析这段代码的目的和结构，然后规划如何纠正其中的错误，或者根据用户需求进行修改。**
    在规划过程中，如果需要获取更详细的信息，你可以使用 `web_search` 找到相关链接，然后使用 `fetch_web_content` 来获取网页的详细内容。
    """

    coder_system_prompt = """
    你是一个专业的代码编写者，你的任务是根据规划者的指示、用户的需求以及可能遇到的错误，编写或修改代码。
    你必须生成完整的、可运行的代码块。
    如果接收到调试器的反馈，你需要仔细分析并修正代码。
    每次只生成一个完整的代码文件内容。
    你可以使用你的工具进行代码质量检查。
    你的代码需要符合常见的最佳实践和特定语言的规范。
    请确保你的回复只包含代码（如果生成代码），或者清晰的文本解释（如果需要澄清或无法完成）。
    """

    executor_system_prompt = """
    你是一个代码执行者。你的任务是接收代码并尝试执行它，然后报告执行结果。
    如果代码执行成功，你将报告输出；如果失败，你将报告错误信息。
    你必须使用你的工具来执行代码。
    请注意，工具会返回 stdout, stderr 和 error 信息。你需要根据这些信息判断代码是否成功。
    """

    debugger_system_prompt = """
    你是一个代码调试专家。你的任务是接收执行器报告的代码执行结果（无论成功或失败），分析代码和结果，并**结合知识库中检索到的相关信息**，提出修改建议或优化方案。
    如果代码执行成功，你可以专注于代码审查、性能优化、潜在问题分析或风格改进。
    如果代码执行失败，你需要分析错误原因，并提出明确的修正指示。
    你可以使用你的工具来辅助分析代码结构和潜在问题，或者搜索错误信息或解决方案。
    当需要更详细的背景信息或解决方案时，你可以使用 `web_search` 找到相关链接，然后使用 `fetch_web_content` 来获取网页的详细内容。
    你的输出应该包含对代码的诊断，以及对代码编写者的明确修正或优化指示。
    """

    # 实例化智能体，Planner 启用 include_uploaded_code_in_prompt 和 enable_long_term_memory
    # 将 fetch_web_content 添加到 Planner 和 Debugger 的工具集中
    planner_agent = CodeAgent(name="planner", model=llm, tools=[web_search, code_lint_check, fetch_web_content], # <-- 添加 fetch_web_content
                              system_prompt=planner_system_prompt,
                              include_uploaded_code_in_prompt=True,
                              enable_long_term_memory=True)

    coder_agent = CodeAgent(name="coder", model=llm, tools=[code_lint_check], system_prompt=coder_system_prompt,
                            enable_long_term_memory=False)
    executor_agent = CodeAgent(name="executor", model=llm, tools=[execute_code], system_prompt=executor_system_prompt,
                               enable_long_term_memory=False)
    debugger_agent = CodeAgent(name="debugger", model=llm, tools=[web_search, code_lint_check, fetch_web_content], # <-- 添加 fetch_web_content
                               system_prompt=debugger_system_prompt,
                               enable_long_term_memory=True)
    decider_agent = DeciderAgent(llm=llm)

    # 从 current_app 中获取我们之前存储的 checkpointer 实例
    try:
        langgraph_checkpointer_instance = current_app.langgraph_checkpointer
    except RuntimeError:
        print("Warning: current_app not available, using a new MemorySaver for graph compilation. Ensure this is not happening in production.")
        langgraph_checkpointer_instance = MemorySaver()


    # 构建 LangGraph
    code_generation_graph = create_code_generation_graph(
        planner_agent, coder_agent, executor_agent, debugger_agent, decider_agent,
        checkpointer=langgraph_checkpointer_instance
    )

    return (planner_agent, coder_agent, executor_agent, debugger_agent, decider_agent), code_generation_graph