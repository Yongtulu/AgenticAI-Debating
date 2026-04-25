# debate_crew.py
# 辩论 Crew 的核心定义文件
# 包含三个 Agent（正方、反方、裁判）和五轮辩论 Task 链的完整配置

from typing import Any, Callable

import ollama
from crewai import Agent, Crew, LLM, Process, Task
from crewai.llms.base_llm import BaseLLM


# ─────────────────────────────────────────
# 自定义 LLM：直接调用 ollama SDK，支持 think=False
#
# CrewAI 默认通过 LiteLLM 调用 Ollama 的 OpenAI 兼容接口
# （/v1/chat/completions），该接口不支持 think 参数。
# 通过继承 BaseLLM 并直接使用 ollama Python SDK 调用原生
# /api/chat 接口，可以准确传入 think=False 禁用深度思考模式。
# ─────────────────────────────────────────

class OllamaDirectLLM(BaseLLM):
    """
    使用 ollama Python SDK 直接调用本地模型的自定义 LLM。
    支持 think 参数，用于禁用 Qwen3 / Gemma4 等模型的深度思考模式。
    """

    # Pydantic 字段：是否启用模型深度思考（默认关闭）
    think: bool = False

    def call(
        self,
        messages: str | list[dict],
        tools: list | None = None,
        callbacks: list | None = None,
        available_functions: dict | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str:
        """
        调用 ollama SDK 的 chat 接口。
        messages 可能是字符串或消息列表，统一转换为列表格式。
        """
        # 如果传入的是纯字符串，包装为标准消息格式
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # 直接调用 ollama SDK，think 参数原生支持
        response = ollama.chat(
            model=self.model,
            messages=messages,
            think=self.think,
        )
        return response.message.content

    def get_context_window_size(self) -> int:
        """返回模型上下文窗口大小，CrewAI 内部会用到此方法"""
        return 8192

    def supports_function_calling(self) -> bool:
        """辩论场景不需要 function calling"""
        return False

    def supports_stop_words(self) -> bool:
        return False


# ─────────────────────────────────────────
# 模型实例
# ─────────────────────────────────────────

# 正方：glm4:9b — 通过 LiteLLM 调用（glm4 无思考模式，无需特殊处理）
LLM_PRO = LLM(
    model="ollama/glm4:9b",
    base_url="http://localhost:11434",
)

# 反方：qwen3.5:9b — 使用自定义 LLM，think=False 禁用深度思考
LLM_CON = OllamaDirectLLM(
    model="qwen3.5:9b",
    think=False,
)

# 裁判：gemma4:e4b — 使用自定义 LLM，think=False 禁用深度思考
LLM_JUDGE = OllamaDirectLLM(
    model="gemma4:e4b",
    think=False,
)


# ─────────────────────────────────────────
# Agent 定义
# ─────────────────────────────────────────

def _create_agents():
    """
    创建三个辩论 Agent，每个 Agent 绑定对应的本地模型。
    返回 (正方辩手, 反方辩手, 裁判) 元组。
    """

    # 正方：支持话题，说话接地气，不说官话
    pro_debater = Agent(
        role="正方",
        goal="支持话题观点，用大白话怼赢对方",
        backstory=(
            "你是个嘴皮子很溜的普通人，你认为这个话题说的是对的。"
            "你说话直接、接地气，喜欢举身边的例子，偶尔会用'你说的这不废话吗''你这逻辑也太离谱了吧'这类口语。"
            "不说套话，不讲大道理，就是正常人吵架的方式，语气可以有点冲。"
            "每次发言用中文，150字以内，说人话。"
        ),
        llm=LLM_PRO,
        verbose=True,
    )

    # 反方：反对话题，嘴硬不服，爱挑刺
    con_debater = Agent(
        role="反方",
        goal="反对话题观点，抓住对方漏洞死磕到底",
        backstory=(
            "你是个爱抬杠的普通人，你觉得这个话题说的完全是扯淡。"
            "你说话刻薄又犀利，喜欢抓对方话里的漏洞，常用'你这例子根本不成立''你在偷换概念吧'这类反将。"
            "不说假大空的话，就是普通人吵架时的那种劲儿，可以夹枪带棒。"
            "每次发言用中文，150字以内，说人话。"
        ),
        llm=LLM_CON,
        verbose=True,
    )

    # 裁判：像个吃瓜群众，但还是要给出判断
    judge = Agent(
        role="吃瓜裁判",
        goal="看完这场架，说说谁赢了、赢在哪",
        backstory=(
            "你是个看热闹的吃瓜群众，但你其实挺有想法。"
            "你不装专业，就用普通人的眼光评价这场架谁吵得更有道理、更有说服力。"
            "评价时说人话，可以带点主观感受，比如'说实话这个例子挺好笑的''这波反驳我觉得有点牵强'。"
            "最后明确说出你觉得谁赢了，理由是什么，用中文，不超过300字。"
        ),
        llm=LLM_JUDGE,
        verbose=True,
    )

    return pro_debater, con_debater, judge


# ─────────────────────────────────────────
# Task 构建函数
# 根据辩题动态生成 5 轮 + 裁判共 11 个 Task
# ─────────────────────────────────────────

def _build_tasks(topic: str, pro: Agent, con: Agent, judge: Agent):
    """
    根据辩题构建完整的 Task 链。
    每个 Task 通过 context 参数传入上一轮发言，实现真正的对话式辩论。
    返回所有 Task 的列表（顺序即执行顺序）。
    """

    # ── 第1轮：亮明立场 ──────────────────────
    task_pro_open = Task(
        description=(
            f"吵架话题：【{topic}】\n\n"
            "你先开口，说说你为啥支持这个观点。"
            "别废话，直接说你怎么想的，可以举身边的例子，150字以内。"
        ),
        expected_output="正方开场（150字以内，口语中文）",
        agent=pro,
    )

    task_con_open = Task(
        description=(
            f"吵架话题：【{topic}】\n\n"
            "对方刚说完了，现在轮到你了。"
            "说说你为啥觉得他说的是扯淡，直接怼，150字以内。"
        ),
        expected_output="反方开场（150字以内，口语中文）",
        agent=con,
        context=[task_pro_open],
    )

    # ── 第2轮：互相反驳 ──────────────────────
    task_pro_r2 = Task(
        description=(
            f"吵架话题：【{topic}】\n\n"
            "对方刚刚怼了你，现在你还击。"
            "抓住他说的漏洞，用你自己的话怼回去，150字以内。"
        ),
        expected_output="正方反击（150字以内，口语中文）",
        agent=pro,
        context=[task_con_open],
    )

    task_con_r2 = Task(
        description=(
            f"吵架话题：【{topic}】\n\n"
            "他又说话了，你觉得他说的还是有问题。"
            "继续怼，指出他的问题在哪，150字以内。"
        ),
        expected_output="反方反击（150字以内，口语中文）",
        agent=con,
        context=[task_pro_r2],
    )

    # ── 第3轮：举例说明 ──────────────────────
    task_pro_r3 = Task(
        description=(
            f"吵架话题：【{topic}】\n\n"
            "光说没用，举个具体的例子来证明你是对的，"
            "越贴近生活越好，150字以内。"
        ),
        expected_output="正方举例（150字以内，口语中文）",
        agent=pro,
        context=[task_con_r2],
    )

    task_con_r3 = Task(
        description=(
            f"吵架话题：【{topic}】\n\n"
            "他举了个例子，但你觉得这例子根本站不住脚。"
            "用你自己的例子或者直接拆穿他，150字以内。"
        ),
        expected_output="反方举例反驳（150字以内，口语中文）",
        agent=con,
        context=[task_pro_r3],
    )

    # ── 第4轮：激烈交锋 ──────────────────────
    task_pro_r4 = Task(
        description=(
            f"吵架话题：【{topic}】\n\n"
            "架吵到这儿了，你也有点急了，说话可以更直接更冲一点。"
            "直接说他哪里不对，150字以内。"
        ),
        expected_output="正方激烈反驳（150字以内，口语中文）",
        agent=pro,
        context=[task_con_r3],
    )

    task_con_r4 = Task(
        description=(
            f"吵架话题：【{topic}】\n\n"
            "他越说越激动，但你还是觉得他说的是错的。"
            "冷静或者更猛地怼回去，150字以内。"
        ),
        expected_output="反方激烈反驳（150字以内，口语中文）",
        agent=con,
        context=[task_pro_r4],
    )

    # ── 第5轮：穷追猛打 ──────────────────────
    task_pro_r5 = Task(
        description=(
            f"吵架话题：【{topic}】\n\n"
            "对方还在犟，你继续追着打，抓住他最大的破绽不放，150字以内。"
        ),
        expected_output="正方穷追猛打（150字以内，口语中文）",
        agent=pro,
        context=[task_con_r4],
    )

    task_con_r5 = Task(
        description=(
            f"吵架话题：【{topic}】\n\n"
            "他又来了，你顶回去，别让他得势，150字以内。"
        ),
        expected_output="反方穷追猛打（150字以内，口语中文）",
        agent=con,
        context=[task_pro_r5],
    )

    # ── 第6轮：负隅顽抗 ──────────────────────
    task_pro_r6 = Task(
        description=(
            f"吵架话题：【{topic}】\n\n"
            "架已经吵了很久了，你再补一刀，把你觉得最关键的点再强调一遍，150字以内。"
        ),
        expected_output="正方强调关键点（150字以内，口语中文）",
        agent=pro,
        context=[task_con_r5],
    )

    task_con_r6 = Task(
        description=(
            f"吵架话题：【{topic}】\n\n"
            "他在重复他的论点，你也把你最核心的理由再说一遍，让他无话可说，150字以内。"
        ),
        expected_output="反方强调关键点（150字以内，口语中文）",
        agent=con,
        context=[task_pro_r6],
    )

    # ── 第7轮：最后几句 ──────────────────────
    task_pro_close = Task(
        description=(
            f"吵架话题：【{topic}】\n\n"
            "说最后几句话，总结一下你的立场，"
            "顺便说说对方哪里始终没说服你，150字以内。"
        ),
        expected_output="正方收尾（150字以内，口语中文）",
        agent=pro,
        context=[task_con_r6],
    )

    task_con_close = Task(
        description=(
            f"吵架话题：【{topic}】\n\n"
            "最后说几句，坚持你的立场，"
            "说说对方哪里根本没说到点子上，150字以内。"
        ),
        expected_output="反方收尾（150字以内，口语中文）",
        agent=con,
        context=[task_pro_close],
    )

    # ── 裁判评判（看完全部7轮）──────────────────
    task_judge = Task(
        description=(
            f"吵架话题：【{topic}】\n\n"
            "这架吵完了，你来说说你的看法。\n"
            "不用太正式，就像跟朋友聊天一样说：\n"
            "- 哪边说的更有道理，为什么\n"
            "- 有没有哪句话让你觉得特别有意思或者特别离谱\n"
            "- 最后你觉得谁赢了这场架\n"
            "说人话，别写成作文，300字以内。"
        ),
        expected_output="裁判吃瓜点评（300字以内，口语中文）",
        agent=judge,
        context=[
            task_pro_open, task_con_open,
            task_pro_r2, task_con_r2,
            task_pro_r3, task_con_r3,
            task_pro_r4, task_con_r4,
            task_pro_r5, task_con_r5,
            task_pro_r6, task_con_r6,
            task_pro_close, task_con_close,
        ],
    )

    return [
        task_pro_open, task_con_open,
        task_pro_r2, task_con_r2,
        task_pro_r3, task_con_r3,
        task_pro_r4, task_con_r4,
        task_pro_r5, task_con_r5,
        task_pro_r6, task_con_r6,
        task_pro_close, task_con_close,
        task_judge,
    ]


# ─────────────────────────────────────────
# 每个 Task 对应的元数据（用于前端展示）
# 顺序必须与 _build_tasks 返回列表一致
# ─────────────────────────────────────────
TASK_META = [
    ("正方", "第1轮：亮明立场"),
    ("反方", "第1轮：亮明立场"),
    ("正方", "第2轮：互相反驳"),
    ("反方", "第2轮：互相反驳"),
    ("正方", "第3轮：举例说明"),
    ("反方", "第3轮：举例说明"),
    ("正方", "第4轮：激烈交锋"),
    ("反方", "第4轮：激烈交锋"),
    ("正方", "第5轮：穷追猛打"),
    ("反方", "第5轮：穷追猛打"),
    ("正方", "第6轮：负隅顽抗"),
    ("反方", "第6轮：负隅顽抗"),
    ("正方", "第7轮：最后几句"),
    ("反方", "第7轮：最后几句"),
    ("裁判", "吃瓜点评"),
]


# ─────────────────────────────────────────
# 对外接口：run_debate
# ─────────────────────────────────────────

def run_debate(topic: str, callback: Callable[[str, str, str], None]) -> None:
    """
    启动一场辩论并通过 callback 实时推送每轮结果。

    参数：
        topic    — 辩论话题字符串
        callback — 每轮结束后调用，签名为 callback(speaker, round_label, content)
    """
    pro, con, judge = _create_agents()
    tasks = _build_tasks(topic, pro, con, judge)

    # 使用列表包装整数，以便在嵌套函数中修改（Python 闭包限制）
    task_index = [0]

    def step_callback(agent_output):
        """每个 Task 完成后触发，将发言内容通过 callback 推送给 SSE"""
        idx = task_index[0]
        if idx < len(TASK_META):
            speaker, round_label = TASK_META[idx]
            content = agent_output.raw if hasattr(agent_output, "raw") else str(agent_output)
            callback(speaker, round_label, content)
            task_index[0] += 1

    crew = Crew(
        agents=[pro, con, judge],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        task_callback=step_callback,
    )

    crew.kickoff()
