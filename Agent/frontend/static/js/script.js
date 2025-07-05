// frontend/static/js/script.js - AI 代码生成器前端逻辑

const API_BASE_URL = 'http://localhost:5000'; // 后端 API 地址

// 获取 DOM 元素
const userPromptInput = document.getElementById('user-prompt');
const codeSourceSelect = document.getElementById('code-source-select');
const pasteCodeGroup = document.getElementById('paste-code-group');
const uploadedCodeInput = document.getElementById('uploaded-code-input');
const fileExtensionPasteInput = document.getElementById('file-extension-paste'); // 针对粘贴代码的扩展名
const uploadFileGroup = document.getElementById('upload-file-group');
const codeFileInput = document.getElementById('code-file-input'); // 文件上传 input

const startGenerationBtn = document.getElementById('start-generation-btn');
const taskIdDisplay = document.getElementById('task-id-display');
const statusArea = document.getElementById('status-area');
const agentLogDiv = document.getElementById('agent-log');
const finalCodeOutputDiv = document.getElementById('final-code-output'); // 这个元素可能不再直接使用，而是由 Monaco Editor 替代
const stopPollingBtn = document.getElementById('stop-polling-btn');
const clearLogBtn = document.getElementById('clear-log-btn');
const notificationContainer = document.getElementById('notification-container');

// 新增下载按钮的引用
const downloadCodeButton = document.getElementById('download-code-button');
const downloadLogButton = document.getElementById('download-log-button');

let currentTaskId = null;
let pollingIntervalId = null;
let monacoEditor = null; // Monaco Editor 实例

// =========================================================
// Monaco Editor 初始化
// =========================================================
require.config({ paths: { 'vs': 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.27.0/min/vs' } });
require(['vs/editor/editor.main'], function () {
    monacoEditor = monaco.editor.create(document.getElementById('editor-container'), {
        value: '// AI 生成的代码将显示在这里\n',
        language: 'plaintext', // 默认语言
        theme: 'vs-dark', // 或 'vs-light'
        readOnly: true,
        automaticLayout: true // 自动调整大小
    });
});

// =========================================================
// 辅助函数
// =========================================================

/**
 * 显示通知消息
 * @param {string} message 消息内容
 * @param {string} type 消息类型 ('success', 'error', 'info', 'warning')
 */
function showNotification(message, type = 'info') {
    // 确保通知容器存在
    if (!notificationContainer) {
        console.warn("Notification container not found.");
        return;
    }
    notificationContainer.textContent = message;
    notificationContainer.className = 'show'; // 移除旧的类，只保留 show
    notificationContainer.style.backgroundColor = ''; // 清除自定义背景色

    if (type === 'success') {
        notificationContainer.style.backgroundColor = 'rgba(0, 128, 0, 0.8)';
    } else if (type === 'error') {
        notificationContainer.style.backgroundColor = 'rgba(255, 0, 0, 0.8)';
    } else if (type === 'warning') {
        notificationContainer.style.backgroundColor = 'rgba(255, 165, 0, 0.8)';
    } else { // info
        notificationContainer.style.backgroundColor = 'rgba(0, 100, 200, 0.8)';
    }

    setTimeout(() => {
        notificationContainer.classList.remove('show');
    }, 3000);
}

/**
 * HTML 转义函数，防止 XSS 攻击
 * @param {string} text
 * @returns {string} 转义后的 HTML 字符串
 */
function escapeHtml(text) {
    if (typeof text !== 'string') {
        return String(text); // 确保是字符串
    }
    var map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, function(m) { return map[m]; });
}

/**
 * 将日志消息添加到页面
 * @param {object} logEntry 后端返回的日志对象 {type, content, name, tool_calls}
 */
function appendLog(logEntry) {
    const logElement = document.createElement('div');
    logElement.classList.add('log-entry');

    // 获取当前时间戳
    const timestamp = new Date().toLocaleTimeString();
    const tsSpan = document.createElement('span');
    tsSpan.classList.add('log-timestamp');
    tsSpan.textContent = `[${timestamp}] `;
    logElement.appendChild(tsSpan);

    const speakerSpan = document.createElement('span');
    speakerSpan.classList.add('log-speaker');
    let speakerText = '';
    let contentHtml = escapeHtml(logEntry.content || ''); // 默认转义内容

    switch (logEntry.type) {
        case 'system':
            speakerText = '系统';
            logElement.classList.add('system-message');
            break;
        case 'human':
            speakerText = '用户';
            logElement.classList.add('user-message');
            break;
        case 'ai':
            speakerText = logEntry.name ? `AI (${logEntry.name})` : 'AI';
            logElement.classList.add('ai-message');
            if (logEntry.tool_calls && logEntry.tool_calls.length > 0) {
                // 拼接工具调用信息
                let toolCallsHtml = '<div class="tool-calls-block">';
                toolCallsHtml += '<strong>调用工具:</strong><ul>';
                logEntry.tool_calls.forEach(call => {
                    toolCallsHtml += `<li><code>${escapeHtml(call.name)}</code> (参数: <code>${escapeHtml(JSON.stringify(call.args, null, 2))}</code>)</li>`;
                });
                toolCallsHtml += '</ul></div>';
                contentHtml += toolCallsHtml; // 将工具调用信息追加到AI消息内容
            }
            break;
        case 'function':
            speakerText = logEntry.name ? `工具结果 (${logEntry.name})` : '工具结果';
            logElement.classList.add('tool-result-message');
            // 对于 execute_code 工具的输出进行特殊处理
            if (logEntry.name === 'execute_code' && typeof logEntry.content === 'string') {
                const stdoutMatch = logEntry.content.match(/stdout:\n([\s\S]*?)\nstderr:/);
                const stderrMatch = logEntry.content.match(/stderr:\n([\s\S]*?)\nerror:/);
                const errorStatusMatch = logEntry.content.match(/error: (True|False)/);

                let parsedOutput = '';
                if (stdoutMatch && stdoutMatch[1].trim()) parsedOutput += `<strong>STDOUT:</strong>\n<pre><code>${escapeHtml(stdoutMatch[1].trim())}</code></pre>\n`;
                if (stderrMatch && stderrMatch[1].trim()) parsedOutput += `<strong>STDERR:</strong>\n<pre><code>${escapeHtml(stderrMatch[1].trim())}</code></pre>\n`;
                if (errorStatusMatch) parsedOutput += `<strong>执行错误:</strong> ${errorStatusMatch[1] === 'True' ? '是' : '否'}\n`;

                contentHtml = parsedOutput || contentHtml; // 如果解析成功则用解析后的，否则用原始的
            }
            break;
        default:
            speakerText = logEntry.name || '未知';
            logElement.classList.add('unknown-message');
    }

    speakerSpan.textContent = `${speakerText}: `;
    logElement.appendChild(speakerSpan);

    const contentDiv = document.createElement('div');
    contentDiv.classList.add('log-content');

    // 识别 Markdown 代码块并用 <pre><code> 包裹
    contentHtml = contentHtml.replace(/```(?:\w+)?\n([\s\S]*?)\n```/g, (match, codeContent) => {
        return `<pre><code>${escapeHtml(codeContent)}</code></pre>`;
    });

    contentDiv.innerHTML = contentHtml; // 使用 innerHTML 渲染包含 HTML 标签的内容
    logElement.appendChild(contentDiv);
    agentLogDiv.appendChild(logElement);

    // 滚动到最新日志
    agentLogDiv.scrollTop = agentLogDiv.scrollHeight;
}

/**
 * 更新状态区域
 * @param {string} message
 * @param {string} type ('info', 'success', 'error', 'warning', 'terminated')
 */
function updateStatusArea(message, type = 'info') {
    statusArea.textContent = message;
    statusArea.className = `status-${type}`; // 添加类型类，用于 CSS 样式
    statusArea.style.display = 'block'; // 显示状态区域
}

/**
 * 清除所有日志和状态，重置 UI
 */
function clearAll() {
    agentLogDiv.innerHTML = '<div class="log-entry system-message"><span class="log-timestamp">[--:--:--]</span> <span class="log-speaker">系统</span>: 等待任务开始...</div>';
    updateStatusArea('请描述需求并点击“开始生成”按钮。', 'info');
    taskIdDisplay.textContent = 'N/A';
    stopPollingBtn.style.display = 'none';
    downloadCodeButton.style.display = 'none'; // 隐藏下载按钮
    downloadLogButton.style.display = 'none'; // 隐藏下载日志按钮
    startGenerationBtn.disabled = false; // 确保开始按钮可用

    if (monacoEditor) {
        monacoEditor.setValue('// AI 生成的代码将显示在这里\n');
        monacoEditor.updateOptions({ readOnly: true }); // 重置为只读
        monaco.editor.setModelLanguage(monacoEditor.getModel(), 'plaintext'); // 重置语言
    }
    currentTaskId = null;
    stopPolling();
    showNotification('界面已重置。', 'info');
    lastLogCount = 0; // 重置日志计数器
}

// =========================================================
// API 交互逻辑
// =========================================================

/**
 * 开始代码生成任务
 */
async function startGeneration() {
    const user_input = userPromptInput.value.trim();
    let fileToUpload = null; // 用于存储文件对象
    let uploadedCodeContent = ''; // 用于存储粘贴代码内容

    if (!user_input) {
        showNotification('请输入你的需求描述！', 'error');
        return;
    }

    const codeSource = codeSourceSelect.value;
    if (codeSource === 'paste') {
        uploadedCodeContent = uploadedCodeInput.value.trim();
        // 对于粘贴代码，扩展名直接从输入框获取
        const pasteExtension = fileExtensionPasteInput.value.trim();
        // 后端期望无点或有点的扩展名，这里统一处理成无点，让后端决定如何使用
        uploaded_file_extension = pasteExtension.startsWith('.') ? pasteExtension.substring(1) : pasteExtension;

    } else if (codeSource === 'upload') {
        fileToUpload = codeFileInput.files[0];
        if (!fileToUpload) {
            showNotification('请选择要上传的代码文件。', 'error');
            return;
        }
        // 对于上传文件，扩展名从文件名中提取
        const filenameParts = fileToUpload.name.split('.');
        uploaded_file_extension = filenameParts.length > 1 ? filenameParts.pop() : '';
    }

    clearAll(); // 调用 clearAll 来重置 UI
    agentLogDiv.innerHTML = '<div class="log-entry system-message"><span class="log-timestamp">[--:--:--]</span> <span class="log-speaker">系统</span>: 正在发送请求，请稍候...</div>';
    updateStatusArea('正在启动代码生成任务...', 'info');
    startGenerationBtn.disabled = true; // 禁用按钮防止重复点击

    try {
        const formData = new FormData(); // <--- 使用 FormData 对象
        formData.append('user_input', user_input);

        // 如果是粘贴代码，将内容作为普通字段发送
        if (codeSource === 'paste' && uploadedCodeContent) {
            formData.append('uploaded_code', uploadedCodeContent);
            formData.append('uploaded_file_extension', uploaded_file_extension);
        }
        // 如果是文件上传，将文件对象发送
        else if (codeSource === 'upload' && fileToUpload) {
            formData.append('file', fileToUpload); // <--- 后端 request.files.get('file') 期望这个名称
            // 对于文件上传，后端可以通过 request.files.get('file').filename 获取文件名和扩展名
            // 但为了清晰和统一，也可以明确发送 extension，虽然后端通常可以自行判断
            formData.append('uploaded_file_extension', uploaded_file_extension);
        }

        // 可以添加 max_retries
        // const maxRetries = document.getElementById('max-retries-input').value; // 如果有这个输入框
        // formData.append('max_retries', maxRetries);

        const response = await fetch(`${API_BASE_URL}/generate_code`, {
            method: 'POST',
            // **关键：不要设置 Content-Type，浏览器会自动设置 multipart/form-data**
            // headers: {
            //     'Content-Type': 'multipart/form-data', // ❌ 不要手动设置！
            // },
            body: formData // <--- 关键：发送 FormData 对象
        });

        const data = await response.json();

        if (response.ok) {
            currentTaskId = data.task_id;
            taskIdDisplay.textContent = currentTaskId;
            updateStatusArea(`任务已启动！任务ID: ${currentTaskId}. 正在轮询状态...`, 'info');
            stopPollingBtn.style.display = 'inline-block'; // 显示停止轮询按钮
            startPolling(currentTaskId);
            showNotification('代码生成任务已成功启动！', 'success');
        } else {
            updateStatusArea(`启动任务失败: ${data.error || '未知错误'}`, 'error');
            showNotification(`启动任务失败: ${data.error || '未知错误'}`, 'error');
            startGenerationBtn.disabled = false; // 重新启用按钮
        }
    } catch (error) {
        console.error('请求发送失败:', error);
        updateStatusArea(`网络请求失败: ${error.message}`, 'error');
        showNotification(`网络请求失败: ${error.message}`, 'error');
        startGenerationBtn.disabled = false; // 重新启用按钮
    }
}

/**
 * 读取文件内容为文本 (此函数在 FormData 上传文件时不再直接需要，但保留以防万一)
 * @param {File} file
 * @returns {Promise<string>}
 */
function readFileContent(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (event) => resolve(event.target.result);
        reader.onerror = (error) => reject(error);
        reader.readAsText(file);
    });
}

let lastLogCount = 0; // 记录上次日志数量，避免重复添加
/**
 * 轮询任务状态
 * @param {string} taskId
 */
async function pollTaskStatus(taskId) {
    try {
        const response = await fetch(`${API_BASE_URL}/get_task_status/${taskId}`);
        const data = await response.json();

        if (!response.ok) {
            updateStatusArea(`获取任务状态失败: ${data.error || '未知错误'}`, 'error');
            showNotification(`获取任务状态失败: ${data.error || '未知错误'}`, 'error');
            stopPolling();
            startGenerationBtn.disabled = false;
            return;
        }

        // 更新日志
        if (data.full_log && data.full_log.length > lastLogCount) {
            for (let i = lastLogCount; i < data.full_log.length; i++) {
                appendLog(data.full_log[i]); // 调用新的 appendLog 函数，直接传入整个 logEntry 对象
            }
            lastLogCount = data.full_log.length;
        }

        // 更新 Monaco Editor 中的代码
        if (monacoEditor && data.current_code) {
            monacoEditor.setValue(data.current_code);
            // 尝试根据后端返回的扩展名设置语言
            let lang = 'plaintext';
            const fileExt = data.uploaded_file_extension; // 后端返回的原始文件扩展名
            if (fileExt) {
                const ext = fileExt.replace('.', ''); // 移除点
                if (monaco.languages.getLanguages().some(l => l.id === ext)) {
                    lang = ext;
                } else {
                    // 常见语言的映射（如果Monaco没有直接的ID）
                    if (ext === 'py') lang = 'python';
                    else if (ext === 'js') lang = 'javascript';
                    else if (ext === 'java') lang = 'java';
                    else if (ext === 'cpp' || ext === 'cxx' || ext === 'h') lang = 'cpp';
                    else if (ext === 'c') lang = 'c';
                    else if (ext === 'go') lang = 'go';
                    else if (ext === 'cs') lang = 'csharp';
                    else if (ext === 'sh') lang = 'shell';
                    else if (ext === 'json') lang = 'json';
                    else if (ext === 'xml') lang = 'xml';
                    else if (ext === 'html') lang = 'html';
                    else if (ext === 'css') lang = 'css';
                    else if (ext === 'ts') lang = 'typescript'; // Added typescript
                    else if (ext === 'md') lang = 'markdown'; // Added markdown
                }
            } else if (data.current_code.includes('def ') && !data.current_code.includes('```')) { // 简单的Python启发式判断
                lang = 'python';
            }
            monaco.editor.setModelLanguage(monacoEditor.getModel(), lang);
            monacoEditor.updateOptions({ readOnly: false }); // 任务进行中或完成后允许复制
        } else {
            // 如果没有代码，保持编辑器只读
            if (monacoEditor) {
                monacoEditor.updateOptions({ readOnly: true });
            }
        }


        // 根据 terminated 字段判断任务是否最终结束
        if (data.terminated) {
            stopPolling(); // 停止轮询
            startGenerationBtn.disabled = false; // 重新启用开始按钮

            downloadCodeButton.style.display = data.current_code ? 'inline-block' : 'none'; // 有代码才显示下载按钮
            downloadLogButton.style.display = 'inline-block'; // 总是显示下载日志按钮

            if (data.status === 'completed' && !data.error_message) {
                updateStatusArea('任务成功完成！代码已生成。', 'success');
                showNotification('代码生成任务已成功完成！', 'success');
            } else if (data.status === 'failed' || data.error_message) {
                updateStatusArea(`任务失败: ${data.error_message || '未知错误'}`, 'error');
                showNotification(`代码生成任务失败: ${data.error_message || '未知错误'}`, 'error');
            } else {
                // 任务被终止但状态不是completed也不是failed，可能是达到重试上限等
                updateStatusArea(`任务已终止: ${data.error_message || '达到最大重试次数或被中断。'}`, 'warning');
                showNotification(`代码生成任务已终止: ${data.error_message || '达到最大重试次数或被中断。'}`, 'warning');
            }
        } else { // 任务还在运行
            updateStatusArea(`状态: ${data.status} (进度: ${data.progress} 轮)`, 'info');
        }

    } catch (error) {
        console.error('轮询失败:', error);
        updateStatusArea(`轮询任务状态失败: ${error.message}`, 'error');
        showNotification(`轮询任务状态失败: ${error.message}`, 'error');
        stopPolling();
        startGenerationBtn.disabled = false;
    }
}

/**
 * 启动轮询
 * @param {string} taskId
 */
function startPolling(taskId) {
    if (pollingIntervalId) {
        clearInterval(pollingIntervalId);
    }
    lastLogCount = 0; // 重置日志计数器
    // 首次立即获取状态，然后每2秒轮询
    // 注意：这里的轮询时间可以根据需要调整
    pollTaskStatus(taskId);
    pollingIntervalId = setInterval(() => pollTaskStatus(taskId), 2000); // 每2秒轮询
}

/**
 * 停止轮询
 */
function stopPolling() {
    if (pollingIntervalId) {
        clearInterval(pollingIntervalId);
        pollingIntervalId = null;
        updateStatusArea('轮询已停止。', 'info');
        stopPollingBtn.style.display = 'none'; // 隐藏停止按钮
    }
}

// =========================================================
// 事件监听器
// =========================================================
startGenerationBtn.addEventListener('click', startGeneration);
stopPollingBtn.addEventListener('click', stopPolling);
clearLogBtn.addEventListener('click', clearAll);

// 根据“现有代码来源”下拉框的选择来显示/隐藏输入组
codeSourceSelect.addEventListener('change', (event) => {
    const selected = event.target.value;
    pasteCodeGroup.style.display = (selected === 'paste') ? 'block' : 'none';
    uploadFileGroup.style.display = (selected === 'upload') ? 'block' : 'none';

    // 清空相关输入，避免提交不必要的旧数据
    uploadedCodeInput.value = '';
    fileExtensionPasteInput.value = '';
    codeFileInput.value = ''; // 清空文件选择
});

// 下载代码文件
downloadCodeButton.addEventListener('click', () => {
    if (currentTaskId) {
        // 后端 /get_output_file 路径，假设它会提供生成的代码的Markdown文件
        window.open(`${API_BASE_URL}/get_output_file/${currentTaskId}`, '_blank');
    } else {
        showNotification('没有可下载的代码文件。', 'warning');
    }
});

// 下载日志文件
downloadLogButton.addEventListener('click', () => {
    if (currentTaskId) {
        // 后端 /download_log 路径
        window.open(`${API_BASE_URL}/download_log/${currentTaskId}`, '_blank');
    } else {
        showNotification('没有可下载的日志文件。', 'warning');
    }
});


// 初始页面加载时，清空并设置默认状态
document.addEventListener('DOMContentLoaded', clearAll);