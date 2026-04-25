# main.py
# FastAPI 主入口文件
# 提供三个路由：
#   GET  /                   — 返回前端页面
#   POST /debate             — 创建辩论会话，返回 debate_id
#   GET  /debate/{id}/stream — SSE 流，实时推送辩论内容

import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from debate_crew import run_debate

# ─────────────────────────────────────────
# 应用初始化
# ─────────────────────────────────────────

app = FastAPI(title="AI 辩论擂台")

# 线程池：CrewAI 是同步阻塞的，需要在独立线程中运行
# max_workers=3 支持最多 3 场并发辩论（本地 Ollama 实际会串行处理）
_executor = ThreadPoolExecutor(max_workers=3)

# 全局辩论会话存储
# 结构：{ debate_id: asyncio.Queue }
# Queue 中存放 SSE 消息字典，None 表示流结束
_sessions: dict[str, asyncio.Queue] = {}


# ─────────────────────────────────────────
# 数据模型
# ─────────────────────────────────────────

class DebateRequest(BaseModel):
    """POST /debate 的请求体"""
    topic: str  # 辩论话题，如 "人工智能对人类社会的影响利大于弊"


class DebateResponse(BaseModel):
    """POST /debate 的响应体"""
    debate_id: str  # 会话唯一标识，用于后续 SSE 连接


# ─────────────────────────────────────────
# 路由：前端页面
# ─────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """
    返回前端 HTML 页面。
    页面文件位于 frontend/index.html。
    """
    html_path = Path(__file__).parent / "frontend" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="前端页面文件不存在")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ─────────────────────────────────────────
# 路由：创建辩论会话
# ─────────────────────────────────────────

@app.post("/debate", response_model=DebateResponse)
async def create_debate(req: DebateRequest):
    """
    接收辩论话题，创建新的辩论会话。
    立即返回 debate_id，同时在后台线程启动 CrewAI 辩论流程。
    前端收到 debate_id 后，通过 SSE 接口实时获取辩论内容。
    """
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="辩论话题不能为空")

    # 生成唯一会话 ID
    debate_id = str(uuid.uuid4())

    # 为本次辩论创建消息队列
    # asyncio.Queue 用于在后台线程（同步）和 SSE 生成器（异步）之间安全传递消息
    q: asyncio.Queue = asyncio.Queue()
    _sessions[debate_id] = q

    # 获取当前事件循环，用于在后台线程中安全地向 Queue 放入消息
    loop = asyncio.get_event_loop()

    def on_speech(speaker: str, round_label: str, content: str):
        """
        每轮发言完成后的回调函数，由 debate_crew.run_debate 调用。
        使用 call_soon_threadsafe 将消息安全地投递到 asyncio Queue。
        """
        msg = {
            "type": "speech" if speaker != "裁判" else "judge",
            "speaker": speaker,
            "round": round_label,
            "content": content,
        }
        # 注意：此函数在后台线程中执行，不能直接 await
        # 必须通过 call_soon_threadsafe + loop.call_soon_threadsafe 跨线程操作 Queue
        loop.call_soon_threadsafe(q.put_nowait, msg)

    def run_in_thread():
        """
        在后台线程中运行完整的辩论流程。
        辩论结束后向 Queue 投递 None（结束信号）或错误消息。
        """
        try:
            run_debate(req.topic, on_speech)
            # 辩论正常结束，发送结束信号
            loop.call_soon_threadsafe(q.put_nowait, None)
        except Exception as e:
            # 发生异常，将错误消息推送给前端
            err_msg = {"type": "error", "message": str(e)}
            loop.call_soon_threadsafe(q.put_nowait, err_msg)
            loop.call_soon_threadsafe(q.put_nowait, None)

    # 提交到线程池执行，不阻塞当前 asyncio 事件循环
    loop.run_in_executor(_executor, run_in_thread)

    return DebateResponse(debate_id=debate_id)


# ─────────────────────────────────────────
# 路由：SSE 流式推送
# ─────────────────────────────────────────

@app.get("/debate/{debate_id}/stream")
async def debate_stream(debate_id: str):
    """
    Server-Sent Events 端点。
    前端通过 EventSource 连接此接口，实时接收辩论内容。

    SSE 消息格式（data 字段为 JSON 字符串）：
      发言消息：{"type": "speech", "speaker": "正方", "round": "第1轮：开场陈词", "content": "..."}
      裁判消息：{"type": "judge",  "speaker": "裁判", "round": "综合评判", "content": "..."}
      结束信号：{"type": "done"}
      错误消息：{"type": "error", "message": "..."}
    """
    if debate_id not in _sessions:
        raise HTTPException(status_code=404, detail="辩论会话不存在或已过期")

    q = _sessions[debate_id]

    async def event_generator() -> AsyncGenerator[str, None]:
        """
        异步生成器：从 Queue 中取出消息并格式化为 SSE 格式。
        当收到 None（结束信号）时停止生成并清理会话。
        """
        try:
            while True:
                # 等待队列中出现新消息（异步等待，不阻塞事件循环）
                msg = await q.get()

                if msg is None:
                    # 收到结束信号，推送 done 消息后关闭流
                    yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
                    break

                # 将消息序列化为 JSON 并格式化为 SSE 格式
                # ensure_ascii=False 保证中文字符正常输出
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"

        finally:
            # 流关闭后清理会话，释放内存
            _sessions.pop(debate_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            # 禁止缓存，确保 SSE 实时推送
            "Cache-Control": "no-cache",
            # 保持长连接
            "Connection": "keep-alive",
            # 允许跨域（开发阶段方便调试）
            "Access-Control-Allow-Origin": "*",
        },
    )
