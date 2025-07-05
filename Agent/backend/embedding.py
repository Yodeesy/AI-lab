import os
from typing import List, Dict, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 全局变量或配置项
DB_DIR = "vector_db"
COLLECTION_NAME = "code_generation_knowledge"

def get_embedding_model():
    """获取嵌入模型实例"""
    # 示例使用 SentenceTransformers，你可以替换为 OpenAIEmbeddings 等
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vector_store():
    """获取 ChromaDB 实例"""
    embedding_function = get_embedding_model()
    # 创建或加载 Chroma 数据库
    vector_store = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME
    )
    return vector_store

def ingest_data_to_vector_db(texts: List[str], metadatas: Optional[List[Dict]] = None):
    """
    将文本数据摄入到向量数据库。
    texts: 要摄入的文本列表
    metadatas: 可选，与文本对应的元数据列表
    """
    vector_store = get_vector_store()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # 每个文本块的最大长度
        chunk_overlap=200 # 块之间的重叠，有助于保留上下文
    )

    documents = []
    # 确保 documents 是由 Document 对象组成的列表
    for i, text in enumerate(texts):
        doc_metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
        documents.append(Document(page_content=text, metadata=doc_metadata))

    splits = text_splitter.split_documents(documents)

    print(f"Ingesting {len(splits)} chunks into vector database...")
    if splits: # 只有当有内容时才执行添加，避免空列表报错
        vector_store.add_documents(splits)
    print("Ingestion complete.")

if __name__ == "__main__":
    # 示例：摄入一些历史代码片段或解决方案
    sample_knowledge_base = [
        "C++ 内存泄漏常见原因及检测方法：未配对的 new/delete，循环引用，智能指针使用不当。使用 Valgrind 工具进行检测。",
        "Python 异步编程最佳实践：使用 asyncio 库，await/async 关键字，避免在异步函数中执行阻塞操作。IO 密集型任务适用。",
        "修复 'Recursion limit reached' 错误：通常是由于 LangGraph 或 LangChain 的节点之间形成了无限循环。检查图的路由逻辑，确保每个节点都有明确的退出条件或路由到 END。",
        "C++ main 函数错误：'mian' 应该是 'main'。`# include {iostream}` 应该是 `#include <iostream>`。类定义需要分号结束。构造函数初始化列表语法 `num:n` 应为 `num(n)`。",
        "在 LangGraph 中将 SystemMessage 传递给 MessagesPlaceholder：如果 Placeholder 期望一个列表，则必须将 SystemMessage 包裹在列表中：`[SystemMessage(content=...)]`。",
        "如何将 Flask 与 LangGraph 集成：使用 APScheduler 管理后台任务，通过 Redis 或 MemorySaver 作为 checkpointer，并使用任务ID进行状态跟踪和轮询。",
        "常见的代码 lint 错误：未使用的变量，导入，函数或类定义错误，风格不一致，缺少文档字符串。",
        "Java 常见编译错误：类名与文件名不匹配、缺少分号、括号不匹配、未导入所需包。",
        "Java 空指针异常（NullPointerException）解决方案：在使用对象前进行空值检查，或者使用 Optional 类型。",
        "Web 开发中常见的安全漏洞：SQL 注入、XSS 攻击、CSRF、不安全的直接对象引用。",
        "设计模式：工厂模式用于创建对象而无需指定其确切类。单例模式确保一个类只有一个实例，并提供一个全局访问点。",
        "数据库连接池：可以提高数据库操作的性能，减少频繁建立和关闭连接的开销。",
        "RESTful API 设计原则：使用名词表示资源，HTTP 方法表示操作，无状态，合理使用状态码。",
        "Linux 常用命令：ls (列出文件), cd (改变目录), cp (复制), mv (移动), rm (删除), chmod (改变权限), ssh (远程连接)。"
    ]

    # 你可以为每个知识片段添加元数据，例如来源、主题、时间等
    sample_metadatas = [
        {"source": "cpp_best_practices", "topic": "memory"},
        {"source": "python_async_guide", "topic": "concurrency"},
        {"source": "langgraph_troubleshooting", "topic": "errors"},
        {"source": "cpp_common_errors", "topic": "syntax"},
        {"source": "langchain_tips", "topic": "prompts"},
        {"source": "project_architecture", "topic": "integration"},
        {"source": "code_quality", "topic": "linting"},
        {"source": "java_errors", "topic": "compilation"},
        {"source": "java_errors", "topic": "runtime"},
        {"source": "web_security", "topic": "vulnerabilities"},
        {"source": "design_patterns", "topic": "oop"},
        {"source": "database_optimization", "topic": "performance"},
        {"source": "api_design", "topic": "rest"},
        {"source": "linux_basics", "topic": "commands"},
    ]


    # 清空旧的数据库（如果存在，仅用于测试/开发阶段）
    # 在生产环境，你可能希望有一个更复杂的更新逻辑
    if os.path.exists(DB_DIR):
        import shutil
        print(f"Removing existing DB at {DB_DIR}...")
        try:
            shutil.rmtree(DB_DIR)
            print("Existing DB removed.")
        except OSError as e:
            print(f"Error removing directory {DB_DIR}: {e}. Please close any applications using these files.")
            # 如果删除失败，可能其他进程正在使用，这里可以选择退出或继续
            # 对于小项目，这里可能需要手动干预，例如重启系统或清理进程
            # 在实际部署中，数据库文件应该由容器或独立服务管理
            # return # 暂时不退出，如果目录没删干净，后面的add_documents可能会出错

    ingest_data_to_vector_db(sample_knowledge_base, sample_metadatas)
    print("Sample data ingested. You can now use get_vector_store() to retrieve.")

    # 简单测试检索（可选，可以在你运行此文件时测试）
    # vector_store = get_vector_store()
    # results = vector_store.similarity_search("java 编程中如何避免空指针", k=2)
    # print("\n--- Retrieval Test Results ---")
    # for doc in results:
    #     print(f"Retrieved: {doc.page_content}\nMetadata: {doc.metadata}\n---")