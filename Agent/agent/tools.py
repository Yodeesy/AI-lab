# agent/tools.py

import os
import sys
import subprocess
import tempfile
import re
import time
from flask import current_app
from langchain_core.tools import tool
from typing import Dict, Any, Tuple
from tavily import TavilyClient # 导入 Tavily 客户端库，用于WebSearch
import requests
from bs4 import BeautifulSoup
def _detect_language(code: str, file_extension: str = None) -> str:
    """
    根据代码内容和/或文件扩展名尝试检测编程语言。
    这是一个简单的启发式检测，可能不完美。
    """
    code_lower = code.lower()

    # 优先使用文件扩展名
    if file_extension:
        if file_extension.lower() == ".py":
            return "python"
        elif file_extension.lower() == ".java":
            return "java"
        elif file_extension.lower() == ".c":
            return "c"
        elif file_extension.lower() == ".cpp" or file_extension.lower() == ".cc" or file_extension.lower() == ".cxx":
            return "cpp"
        # 暂时不加 .js 和 .go 的判断，因为用户明确了只支持这四种语言

    # 如果没有文件扩展名，根据代码内容尝试检测 (更精确的启发式规则)

    # Python
    if ("def " in code_lower or "import " in code_lower) and \
            ("print(" in code_lower or "if __name__ == '__main__'" in code_lower):
        return "python"

    # Java
    if "public static void main" in code_lower and \
            ("class " in code_lower or "import java." in code_lower or "system.out.println" in code_lower):
        return "java"

    # C++
    # 检查常见的C++头文件、命名空间或输出/输入流
    if ("#include <iostream>" in code_lower or "#include <vector>" in code_lower or \
        "using namespace std;" in code_lower or "std::cout" in code_lower) and \
            "int main(" in code_lower:
        return "cpp"

    # C
    # 检查常见的C头文件和main函数结构，避免与C++混淆
    if "#include <stdio.h>" in code_lower or "#include <stdlib.h>" in code_lower:
        # 如果包含C++特有关键字，则倾向于C++
        if "std::" not in code_lower and "class " not in code_lower and "using namespace" not in code_lower:
            return "c"
        elif "int main(" in code_lower and "printf(" in code_lower:
            return "c"  # 更具体的C语言特征

    return "unknown"  # 无法识别的语言


@tool
def execute_code(code: str, language_hint: str = "auto", filename_base: str = "temp_code") -> Dict[str, Any]:
    """
    在沙盒环境中执行不同编程语言的代码。
    支持 'python', 'java', 'c', 'cpp'。
    如果 language_hint 为 'auto'，则尝试自动检测语言。

    Args:
        code (str): 要执行的代码字符串。
        language_hint (str): 预期的编程语言，例如 'python', 'java', 'c', 'cpp'。默认为 'auto'。
        filename_base (str): 用于临时文件名的基础，不包含扩展名。

    Returns:
        Dict[str, Any]: 包含 'stdout' (标准输出), 'stderr' (标准错误), 'error' (执行错误信息), 'language' (实际执行语言)。
    """
    output = {"stdout": "", "stderr": "", "error": "", "language": "unknown"}
    temp_dir = None
    temp_file_path = None
    exec_cmd = None
    compile_cmd = None
    file_extension = ""
    run_args = {}
    timeout = current_app.config.get('EXECUTION_TIMEOUT', 10)  # 默认值 10 秒

    try:
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()

        # 语言检测
        detected_language = language_hint.lower()
        if detected_language == "auto":
            detected_language = _detect_language(code)
            if detected_language == "unknown":
                output["error"] = "无法自动检测代码语言。请在 `language_hint` 参数中明确指定语言（例如 'python', 'java', 'c', 'cpp'）。"
                return output

        output["language"] = detected_language

        # 根据语言设置文件扩展名和执行命令
        if detected_language == "python":
            file_extension = ".py"
            exec_cmd = [sys.executable, filename_base + ".py"]  # <--- 将 "python3" 替换为 sys.executable
            run_args = {'check': False, 'cwd': temp_dir}  # 不raise CalledProcessError
        elif detected_language == "java":
            file_extension = ".java"
            # Java 需要编译
            compile_cmd = ["javac", filename_base + ".java"]
            exec_cmd = ["java", filename_base]  # 执行class文件，文件名不带.java后缀
            run_args = {'cwd': temp_dir, 'check': False}  # 在临时目录执行，不raise CalledProcessError
        elif detected_language == "c":
            file_extension = ".c"
            # C 需要编译
            compiled_output_name = filename_base + "_exec"
            compile_cmd = ["gcc", filename_base + ".c", "-o", compiled_output_name]
            exec_cmd = [os.path.join(".", compiled_output_name)]  # 执行编译后的可执行文件
            run_args = {'cwd': temp_dir, 'check': False}  # 在临时目录执行，不raise CalledProcessError
        elif detected_language == "cpp":
            file_extension = ".cpp"
            # C++ 需要编译
            compiled_output_name = filename_base + "_exec"
            compile_cmd = ["g++", filename_base + ".cpp", "-o", compiled_output_name]  # 使用 g++ 编译
            exec_cmd = [os.path.join(".", compiled_output_name)]
            run_args = {'cwd': temp_dir, 'check': False}
        else:
            output[
                "error"] = f"不支持的语言: {language_hint if language_hint != 'auto' else detected_language}。目前只支持 Python, Java, C, C++。"
            return output

        temp_file_path = os.path.join(temp_dir, filename_base + file_extension)

        # 将代码写入临时文件
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(code)

        # 编译代码 (如果需要)
        if compile_cmd:
            print(f"尝试编译 {detected_language} 代码...")
            compile_process = subprocess.run(compile_cmd, capture_output=True, text=True, cwd=temp_dir,
                                             timeout=timeout)
            if compile_process.returncode != 0:
                output["stderr"] = compile_process.stderr
                output["error"] = f"{detected_language} 编译错误。错误信息：\n{compile_process.stderr}"
                print(f"编译失败: {output['error']}")
                return output
            print(f"{detected_language} 代码编译成功。")

        # 执行代码
        print(f"尝试执行 {detected_language} 代码...")
        process = subprocess.run(exec_cmd, capture_output=True, text=True, timeout=timeout, **run_args)

        output["stdout"] = process.stdout
        output["stderr"] = process.stderr

        if process.returncode != 0:
            error_msg = f"{detected_language} 代码运行时出错，退出码 {process.returncode}。"
            if process.stderr:
                error_msg += f"\n错误输出：\n{process.stderr}"
            output["error"] = error_msg
            print(f"执行失败: {output['error']}")

    except subprocess.TimeoutExpired as e:
        output["error"] = f"代码执行超时 ({timeout} 秒)。\n标准输出：{e.stdout}\n标准错误：{e.stderr}"
        print(f"执行超时: {output['error']}")
    except FileNotFoundError as e:
        output["error"] = f"执行器未找到。请确保 {detected_language} 解释器/编译器已正确安装并配置在 PATH 中: {e}"
        print(f"执行器未找到: {output['error']}")
    except Exception as e:
        output["error"] = f"执行代码时发生未知错误: {e}"
        print(f"未知错误: {output['error']}")
    finally:
        # 清理临时文件和目录
        if temp_dir and os.path.exists(temp_dir):
            for root, dirs, files in os.walk(temp_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(temp_dir)
            print(f"临时目录 {temp_dir} 已清理。")

    return output


@tool
def web_search(query: str) -> str:
    """
    使用 Tavily Search API 执行网络搜索，并返回结果摘要。
    API 密钥从环境变量 TAVILY_API_KEY 中获取。

    Args:
        query (str): 搜索查询字符串。

    Returns:
        str: 搜索结果的摘要，包含标题、摘要和链接。
    """
    print(f"执行网络搜索: {query}")

    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

    if not TAVILY_API_KEY:
        current_app.logger.error("Tavily API 密钥未配置。请设置 TAVILY_API_KEY 环境变量。")
        return "Tavily API 密钥未配置。请设置 TAVILY_API_KEY 环境变量。"

    try:
        # 记录 Tavily 客户端初始化
        current_app.logger.info(
            f"DEBUG_TAVILY: Initializing TavilyClient with API Key (first 5 chars): {TAVILY_API_KEY[:5]}*****")
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

        # 记录 Tavily 搜索调用
        current_app.logger.info(f"DEBUG_TAVILY: Calling Tavily search for query: '{query}'")
        response = tavily_client.search(query=query, search_depth="basic", max_results=5)

        # 记录 Tavily 响应类型和内容
        current_app.logger.info(f"DEBUG_TAVILY: Tavily search response type: {type(response)}")
        current_app.logger.info(f"DEBUG_TAVILY: Tavily search response: {response}")

        results = response.get("results", [])
        if not results:
            current_app.logger.warning(f"DEBUG_TAVILY: No search results found for '{query}'.")
            return f"未能找到关于 '{query}' 的搜索结果。"

        results_list = []
        for i, item in enumerate(results):
            title = item.get("title", "无标题")
            content = item.get("content", "无摘要")
            url = item.get("url", "无链接")
            results_list.append(f"--- 结果 {i + 1} ---\n标题: {title}\n摘要: {content}\n链接: {url}")

        return "以下是搜索结果：\n" + "\n\n".join(results_list)

    except Exception as e:
        # 记录完整的异常信息和堆栈跟踪
        import traceback
        error_trace = traceback.format_exc()
        current_app.logger.error(f"网络搜索工具执行失败: {type(e).__name__}: {e}\n{error_trace}")
        return f"执行网络搜索时发生错误: {type(e).__name__}: {e}"


@tool
def fetch_web_content(url: str) -> str:
    """
    访问指定的 URL，并尝试提取其主要文本内容。
    注意：此工具可能会被网站的robots.txt规则或反爬机制限制。
    如果遇到问题，可能需要尝试不同的 User-Agent 或处理重定向。

    Args:
        url (str): 要访问的网页 URL。

    Returns:
        str: 提取到的网页主要文本内容，或错误信息。
    """
    print(f"尝试抓取网页内容: {url}")
    try:
        # 模拟浏览器 User-Agent，以减少被网站识别为爬虫的可能性
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # 设置超时，避免长时间等待
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # 对非200状态码（如4xx, 5xx）抛出HTTPError

        # 使用 BeautifulSoup 解析 HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # 尝试提取主要内容，例如段落、标题、列表项等
        # 这是一个简单的启发式方法，对于结构复杂的网站可能需要更具体的CSS选择器或XPath
        main_content_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span', 'div'])

        extracted_text_parts = []
        for tag in main_content_elements:
            # 尝试获取文本，并去除多余的空白
            text = tag.get_text(separator=' ', strip=True)
            if text and len(text) > 20:  # 过滤掉过短的文本片段
                extracted_text_parts.append(text)

        extracted_text = "\n".join(extracted_text_parts)

        # 如果未能提取到特定标签的内容，作为备用，尝试提取body的全部可见文本
        if not extracted_text.strip():
            body_tag = soup.find('body')
            if body_tag:
                extracted_text = body_tag.get_text(separator='\n', strip=True)

        if not extracted_text.strip():
            return f"未能从 {url} 中提取到有效文本内容。可能是页面结构复杂或内容为空。"

        # 限制返回内容长度，避免输出过长导致模型处理困难
        MAX_CONTENT_LENGTH = 3000  # 可以根据实际需求调整
        if len(extracted_text) > MAX_CONTENT_LENGTH:
            extracted_text = extracted_text[:MAX_CONTENT_LENGTH] + "...\n(内容过长，已截断)"

        return f"成功从 {url} 提取内容：\n{extracted_text}"

    except requests.exceptions.Timeout:
        return f"访问 {url} 超时 ({10} 秒)。请检查URL是否有效或网络连接。"
    except requests.exceptions.HTTPError as e:
        return f"访问 {url} 失败，HTTP 错误：{e.response.status_code} - {e.response.reason}"
    except requests.exceptions.RequestException as e:
        return f"访问 {url} 失败，网络或请求错误: {e}"
    except Exception as e:
        return f"提取 {url} 内容时发生未知错误: {e}"



@tool
def code_lint_check(code: str, language_hint: str = "auto") -> str:
    """
    对提供的代码执行静态分析（linting），检查潜在的语法错误、风格问题或常见的bug。
    这是一个模拟工具，实际使用需要集成 linting 工具（如 Pylint, ESLint, Checkstyle, cpplint）。

    Args:
        code (str): 要检查的代码字符串。
        language_hint (str): 代码的编程语言，例如 'python', 'java', 'c', 'cpp'。默认为 'auto'。

    Returns:
        str: linting 结果的报告。如果代码看起来良好，则返回“代码看起来良好，没有发现明显的linting问题。”
    """
    print(f"执行代码 linting 检查 (语言: {language_hint})...")

    code_lower = code.lower()

    detected_language = language_hint.lower()
    if detected_language == "auto":
        detected_language = _detect_language(code)

    lint_issues = []

    # 简单模拟不同语言的 linting 规则
    if detected_language == "python":
        if "    " not in code and "\t" in code:  # 混合缩进
            lint_issues.append("Python代码中发现混合使用空格和制表符进行缩进，建议统一使用4个空格。")
        if re.search(r'\bprint\s*\(', code) and "logging" not in code:
            lint_issues.append("Python代码中使用print进行调试输出，建议生产环境使用logging模块。")
        if "import os" in code and "subprocess" in code and "execute_code" not in code:
            lint_issues.append("Python代码导入了os和subprocess但未在安全函数内使用，可能存在安全风险。")
    elif detected_language == "java":
        if not re.search(r'public\s+class\s+\w+\s*\{', code):
            lint_issues.append("Java代码可能缺少公共类定义或类名不规范。")
        if "System.out.println(" in code and "Logger" not in code:
            lint_issues.append("Java代码中使用System.out.println进行调试输出，建议生产环境使用日志框架如Log4j或SLF4J。")
    elif detected_language == "c":
        if "malloc" in code and "free" not in code:
            lint_issues.append("C代码可能存在内存泄漏：使用了malloc但没有对应的free。")
        if re.search(r'while\s*\(\s*1\s*\)', code) and "break" not in code:
            lint_issues.append("C代码中发现无限循环 `while(1)`，请确保有跳出循环的条件。")
    elif detected_language == "cpp":  # <--- 新增 C++ linting 规则
        if "#include <bits/stdc++.h>" in code_lower:
            lint_issues.append("C++代码使用了非标准的 `<bits/stdc++.h>` 头文件，建议使用具体的标准头文件。")
        if re.search(r'using namespace std;', code):
            lint_issues.append(
                "C++代码中使用了 `using namespace std;`，在大项目中可能导致命名冲突，建议明确指定命名空间。")

    if lint_issues:
        return f"代码Linting问题（{detected_language}）：\n" + "\n".join([f"- {issue}" for issue in lint_issues])
    else:
        return "代码看起来良好，没有发现明显的linting问题。"