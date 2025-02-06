以下是使用 Torch、Streamlit 和 Tornado 构建 LLM 对话平台的架构实现方案，分为多个代码模块和说明：

```python
# 文件结构
.
├── frontend/                 # Streamlit 前端
│   ├── app.py                
├── backend/                  # Tornado 后台服务
│   ├── task_queue.py         # 任务队列管理
│   ├── model_handler.py      # Torch 模型处理
│   ├── server.py             # Tornado 主服务
└── shared/                   # 共享配置
    └── config.py
```

---

### 1. 共享配置 (shared/config.py)
```python
class Config:
    # Tornado 服务器配置
    SERVER_PORT = 8888
    MAX_WORKERS = 4  # 最大并发任务数
    
    # 模型配置
    MODEL_NAME = "gpt2"  # 示例使用 Huggingface 的 GPT-2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

---

### 2. 前端界面 (frontend/app.py)
```python
import streamlit as st
import requests
import time

class ChatFrontend:
    def __init__(self):
        self.api_url = f"http://localhost:{Config.SERVER_PORT}"
        self._init_session_state()
        
    def _init_session_state(self):
        if "history" not in st.session_state:
            st.session_state.history = []
            
    def _show_chat(self):
        for prompt, response in st.session_state.history:
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("ai"):
                st.write(response)
    
    def _submit_task(self, prompt):
        # 提交任务到后台
        response = requests.post(
            f"{self.api_url}/submit",
            json={"prompt": prompt}
        )
        return response.json()["task_id"]
    
    def _poll_result(self, task_id):
        # 轮询获取结果
        while True:
            result = requests.get(f"{self.api_url}/result/{task_id}").json()
            if result["status"] == "completed":
                return result["response"]
            time.sleep(0.5)
    
    def run(self):
        st.title("LLM 对话平台")
        
        if prompt := st.chat_input("输入你的问题"):
            # 显示用户输入
            with st.chat_message("user"):
                st.write(prompt)
            
            # 提交处理
            task_id = self._submit_task(prompt)
            
            # 显示临时等待信息
            with st.chat_message("ai"):
                placeholder = st.empty()
                placeholder.markdown("▌")
                
                # 获取结果
                response = self._poll_result(task_id)
                placeholder.empty()
                st.write(response)
            
            # 更新历史记录
            st.session_state.history.append((prompt, response))

if __name__ == "__main__":
    ChatFrontend().run()
```

---

### 3. 后台服务核心模块 (backend/task_queue.py)
```python
from tornado.queues import Queue
from collections import OrderedDict

class TaskManager:
    def __init__(self):
        self.task_queue = Queue()
        self.results = OrderedDict()
        self.max_results = 100  # 保留最近100条结果
        
    async def process_tasks(self):
        """Tornado 协程持续处理任务"""
        while True:
            task_id, prompt = await self.task_queue.get()
            try:
                # 实际处理会在 model_handler 完成
                response = await ModelHandler.process(prompt)
                self._store_result(task_id, response)
            finally:
                self.task_queue.task_done()
    
    def _store_result(self, task_id, response):
        # 使用有序字典管理结果
        if len(self.results) >= self.max_results:
            self.results.popitem(last=False)
        self.results[task_id] = {
            "status": "completed",
            "response": response
        }

task_manager = TaskManager()
```

---

### 4. 模型处理模块 (backend/model_handler.py)
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class ModelHandler:
    model = None
    tokenizer = None
    
    @classmethod
    def initialize(cls):
        if cls.model is None:
            cls.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
            cls.model = AutoModelForCausalLM.from_pretrained(
                Config.MODEL_NAME
            ).to(Config.DEVICE)
    
    @classmethod
    async def process(cls, prompt):
        """异步处理请求"""
        inputs = cls.tokenizer(prompt, return_tensors="pt").to(Config.DEVICE)
        outputs = cls.model.generate(
            inputs.input_ids, 
            max_length=100,
            do_sample=True
        )
        return cls.tokenizer.decode(outputs[0], skip_special_tokens=True)

# 初始化模型
ModelHandler.initialize()
```

---

### 5. Tornado 服务器 (backend/server.py)
```python
import tornado.web
import tornado.ioloop
import uuid

class SubmitHandler(tornado.web.RequestHandler):
    async def post(self):
        data = tornado.escape.json_decode(self.request.body)
        task_id = str(uuid.uuid4())
        
        # 将任务加入队列
        await task_manager.task_queue.put((task_id, data["prompt"]))
        
        self.write({
            "task_id": task_id,
            "status": "queued"
        })

class ResultHandler(tornado.web.RequestHandler):
    def get(self, task_id):
        result = task_manager.results.get(task_id, {"status": "pending"})
        self.write(result)

def make_app():
    return tornado.web.Application([
        (r"/submit", SubmitHandler),
        (r"/result/(.*)", ResultHandler),
    ])

if __name__ == "__main__":
    # 启动任务处理协程
    tornado.ioloop.IOLoop.current().spawn_callback(
        task_manager.process_tasks
    )
    
    app = make_app()
    app.listen(Config.SERVER_PORT)
    print(f"Server running on port {Config.SERVER_PORT}")
    tornado.ioloop.IOLoop.current().start()
```

---

### 系统架构说明

1. **组件交互流程**：
   - 用户通过 Streamlit 界面输入 prompt
   - 前端通过 REST API 提交任务到 Tornado
   - Tornado 将任务放入异步队列
   - 后台工作协程从队列获取任务并调用模型
   - 结果存储后，前端通过轮询获取

2. **关键技术点**：
   - **异步任务队列**：使用 Tornado 的异步队列实现非阻塞处理
   - **结果缓存**：使用有序字典自动清理旧结果
   - **模型单例**：确保模型只加载一次到内存
   - **跨进程通信**：通过 HTTP API 解耦前后端

3. **扩展方向**：
   - 添加用户认证系统
   - 实现优先级队列
   - 添加任务取消功能
   - 集成分布式任务队列（Celery/RabbitMQ）
   - 添加流式响应支持

启动顺序建议：
1. 先启动 Tornado 服务器
2. 再启动 Streamlit 前端
3. 可通过 `curl http://localhost:8888/submit -d '{"prompt":"你好"}'` 测试 API

此架构实现了基本的对话功能，同时保证了前后端的解耦和可扩展性。实际部署时建议添加错误处理、日志记录和性能监控模块。