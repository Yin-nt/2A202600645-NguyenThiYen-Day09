"""Stage 4: Multi-Agent System (In-Process)

Nhiều agent chuyên trách cùng phối hợp xử lý một câu hỏi pháp lý phức tạp.
Mô hình này mô phỏng kiến trúc của Stage 5 (law_agent/graph.py) nhưng chạy
hoàn toàn trong cùng một tiến trình (In-Process) — không HTTP, không giao thức A2A, không server riêng biệt.

Đồ thị (Graph): analyze_law -> check_routing -> song song [call_tax, call_compliance] -> aggregate -> END
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from common.llm import get_llm

# ---------------------------------------------------------------------------
# Công cụ (Tools) cho các sub-agent chuyên trách (Đã Việt hóa dữ liệu tri thức)
# ---------------------------------------------------------------------------

@tool
def search_tax_law(query: str) -> str:
    """Tìm kiếm trong cơ sở tri thức luật thuế về các điều luật và hình phạt liên quan.

    Args:
        query: Câu truy vấn bằng ngôn ngữ tự nhiên về luật thuế.
    """
    knowledge = [
        (
            ["thuế", "trốn thuế", "gian lận", "irs", "tax evasion"],
            "Trốn thuế (26 U.S.C. § 7201): Tội hình sự, phạt tiền lên đến $250K và phạt tù đến 5 năm. "
            "Phạt gian lận dân sự: 75% số tiền nộp thiếu (IRC § 6663). Không nộp tờ khai: phạt tiền "
            "lên đến $25K và phạt tù đến 1 year.",
        ),
        (
            ["nước ngoài", "tài khoản", "fbar", "fatca", "offshore"],
            "Hình phạt FBAR (Tài khoản nước ngoài): Lên đến $100K hoặc 50% số dư tài khoản cho mỗi vi phạm. "
            "Không tuân thủ FATCA: Khấu trừ 30% đối với các khoản thanh toán có nguồn gốc từ Mỹ. "
            "Vi phạm cố ý có thể bị truy tố hình sự.",
        ),
        (
            ["chuyển giá", "giá nội bộ", "tập đoàn", "transfer pricing"],
            "Vi phạm chuyển giá (IRC § 482): IRS có quyền phân bổ lại thu nhập giữa các bên liên kết. "
            "Hình phạt: 20-40% số tiền nộp thiếu đối với các trường hợp sai lệch giá trị nghiêm trọng/thô thiển.",
        ),
    ]
    query_lower = query.lower()
    results = []
    for keywords, text in knowledge:
        if any(kw in query_lower for kw in keywords):
            results.append(text)
    return "\n\n".join(results) if results else "Không tìm thấy điều luật thuế cụ thể nào phù hợp."


@tool
def search_compliance_law(query: str) -> str:
    """Tìm kiếm trong cơ sở tri thức về tuân thủ pháp lý và các khung quy định áp dụng.

    Args:
        query: Câu truy vấn bằng ngôn ngữ tự nhiên về tuân thủ pháp lý.
    """
    knowledge = [
        (
            ["dữ liệu", "riêng tư", "bảo mật", "gdpr", "ccpa", "đồng ý", "người dùng"],
            "CCPA: Phạt tiền lên đến $7,500 cho mỗi vi phạm cố ý. GDPR: Phạt lên đến 4% doanh thu toàn cầu "
            "hoặc 20 triệu EUR. Đạo luật FTC Mục 5 cho các hành vi không công bằng/lừa đảo. "
            "Nguy cơ đối mặt với vụ kiện tập thể theo luật bảo mật của bang ($100-$750 cho mỗi người tiêu dùng).",
        ),
        (
            ["sox", "sarbanes", "tài chính", "sec", "báo cáo"],
            "Đạo luật SOX § 906: Xác nhận sai báo cáo tài chính — phạt tiền đến $5M, tù đến 20 năm. "
            "§ 802: Tiêu hủy hồ sơ — tù đến 20 năm. § 1107: Trả đũa người tố giác — tù đến 10 năm. "
            "SEC có quyền cấm cá nhân đảm nhiệm chức vụ quản lý hoặc giám đốc.",
        ),
        (
            ["fcpa", "hối lộ", "tham nhũng", "nước ngoài", "bribery"],
            "Chống hối lộ theo FCPA: Phạt tiền lên đến $250K cho mỗi vi phạm (cá nhân), và $2M (đối với tập đoàn). "
            "Hình phạt hình sự: Lên đến 5 năm tù. Các quy định về sổ sách và hồ sơ áp dụng cho tất cả các công ty báo cáo với SEC.",
        ),
    ]
    query_lower = query.lower()
    results = []
    for keywords, text in knowledge:
        if any(kw in query_lower for kw in keywords):
            results.append(text)
    return "\n\n".join(results) if results else "Không tìm thấy quy định tuân thủ cụ thể nào phù hợp."


# ---------------------------------------------------------------------------
# Định nghĩa Trạng thái - State (Mô phỏng lại law_agent/graph.py)
# ---------------------------------------------------------------------------

from typing import Annotated, TypedDict

from langgraph.constants import Send
from langgraph.graph import END, StateGraph


def _last_wins(a: str, b: str) -> str:
    """Reducer: Giữ lại giá trị được ghi nhận muộn nhất."""
    return b if b else a


class LegalState(TypedDict):
    question: str
    law_analysis: str
    needs_tax: bool
    needs_compliance: bool
    needs_privacy: bool
    tax_result: Annotated[str, _last_wins]
    compliance_result: Annotated[str, _last_wins]
    privacy_result: Annotated[str, _last_wins]
    final_answer: str


# ---------------------------------------------------------------------------
# Triển khai các Nút xử lý - Nodes (Đã Việt hóa Prompts hướng dẫn Agent)
# ---------------------------------------------------------------------------
# agent 1
async def analyze_law(state: LegalState) -> dict:
    """Luật sư trưởng phân tích các khía cạnh pháp lý tổng quan của câu hỏi."""
    print("\n  [Node: analyze_law] Luật sư trưởng đang phân tích khía cạnh pháp lý...")
    llm = get_llm()
    messages = [
        SystemMessage(
            content=(
                "Bạn là một luật sư tranh tụng doanh nghiệp cao cấp chuyên về luật hợp đồng, "
                "luật bồi thường thiệt hại ngoài hợp đồng (tort law) và luật kinh doanh tổng quát. "
                "Hãy phân tích kỹ lưỡng các khía cạnh pháp lý của câu hỏi được đưa ra. "
                "Đưa ra câu trả lời bằng Tiếng Việt và viết dưới 200 từ."
            )
        ),
        HumanMessage(content=state["question"]),
    ]
    result = await llm.ainvoke(messages)
    print(f"  [Node: analyze_law] Hoàn thành ({len(result.content)} ký tự)")
    return {"law_analysis": result.content}

# agent 2
async def check_routing(state: LegalState) -> dict:
    """Node định tuyến: Xác định xem có cần đến các sub-agent chuyên trách nào."""
    print("\n  [Node: check_routing] Đang xác định các chuyên gia cần thiết...")
    llm = get_llm()
    messages = [
        SystemMessage(
            content=(
                "Bạn là một chuyên gia định tuyến pháp lý. Dựa trên câu hỏi, hãy quyết định xem "
                "có cần đến các sub-agent chuyên trách hay không.\n"
                "Chỉ phản hồi duy nhất định dạng JSON hợp lệ — không kèm markdown, không thêm chữ:\n"
                '{"needs_tax": <true|false>, "needs_compliance": <true|false>}\n\n'
                "needs_tax = true  → Câu hỏi có liên quan đến luật thuế, IRS, trốn thuế, phạt thuế\n"
                "needs_compliance = true → Câu hỏi liên quan đến tuân thủ quy định, SEC, SOX, AML, FCPA"
            )
        ),
        HumanMessage(content=state["question"]),
    ]
    result = await llm.ainvoke(messages)
    raw = result.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"needs_tax": True, "needs_compliance": True}

    question_lower = state["question"].lower()
    needs_tax = any(kw in question_lower for kw in ["tax", "irs", "thuế", "trốn thuế"])
    needs_compliance = any(
        kw in question_lower for kw in ["compliance", "sec", "regulation", "tuân thủ"]
    )
    needs_privacy = any(
        kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu"]
    )
    print(
        f"  [Node: check_routing] needs_tax={needs_tax}, "
        f"needs_compliance={needs_compliance}, needs_privacy={needs_privacy}"
    )
    return {
        "needs_tax": needs_tax,
        "needs_compliance": needs_compliance,
        "needs_privacy": needs_privacy,
    }


def route_to_specialists(state: LegalState) -> list[Send]:
    """Hàm định tuyến điều phối: Phân phát song song các đối tượng Send đến các node chuyên trách."""
    sends: list[Send] = []
    if state.get("needs_tax"):
        sends.append(Send("call_tax_specialist", state))
    if state.get("needs_compliance"):
        sends.append(Send("call_compliance_specialist", state))
    if state.get("needs_privacy"):
        sends.append(Send("privacy_agent", state))
    if not sends:
        sends.append(Send("aggregate", state))
    return sends

# agent 3
async def call_tax_specialist(state: LegalState) -> dict:
    """Sub-agent chuyên gia về Thuế (Chạy như một ReAct agent nội bộ)."""
    from langgraph.prebuilt import create_react_agent

    print("\n  [Node: call_tax_specialist] Agent Chuyên gia Luật Thuế bắt đầu chạy...")

    tax_prompt = (
        "Bạn là một luật sư chuyên về thuế và là một CPA (Kiểm toán viên công chứng) có chuyên môn sâu về luật thuế doanh nghiệp, "
        "phân biệt giữa trốn thuế (evasion) và tránh thuế (avoidance), thực thi pháp luật của IRS, các hình phạt theo IRC §§ 6651/6662/6663, "
        "yêu cầu về FBAR/FATCA, và các điều luật chống gian lận thuế (18 U.S.C. § 7201-7207). "
        "Sử dụng công cụ search_tax_law để làm căn cứ cho phân tích của bạn. Đưa ra câu trả lời bằng Tiếng Việt và viết dưới 200 từ."
    )

    llm = get_llm()
    agent = create_react_agent(model=llm, tools=[search_tax_law], prompt=tax_prompt)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})

    final_msg = result["messages"][-1].content
    print(f"  [Node: call_tax_specialist] Hoàn thành ({len(final_msg)} ký tự)")
    return {"tax_result": final_msg}


async def call_compliance_specialist(state: LegalState) -> dict:
    """Sub-agent chuyên gia về Tuân thủ pháp lý (Chạy như một ReAct agent nội bộ)."""
    from langgraph.prebuilt import create_react_agent

    print("\n  [Node: call_compliance_specialist] Agent Chuyên gia Tuân thủ bắt đầu chạy...")

    compliance_prompt = (
        "Bạn là một giám đốc tuân thủ pháp lý cao cấp có chuyên môn về thực thi pháp luật của SEC, "
        "tuân thủ SOX, các quy định của FTC, FCPA (Chống tham nhũng tại nước ngoài), AML/BSA (Chống rửa tiền), GDPR, CCPA và quản trị doanh nghiệp. "
        "Sử dụng công cụ search_compliance_law để làm căn cứ cho phân tích của bạn. Đưa ra câu trả lời bằng Tiếng Việt và viết dưới 200 từ."
    )

    llm = get_llm()
    agent = create_react_agent(model=llm, tools=[search_compliance_law], prompt=compliance_prompt)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})

    final_msg = result["messages"][-1].content
    print(f"  [Node: call_compliance_specialist] Hoàn thành ({len(final_msg)} ký tự)")
    return {"compliance_result": final_msg}


async def privacy_agent(state: LegalState) -> dict:
    """Agent specializing in GDPR and personal-data protection law."""
    llm = get_llm()
    prompt = f"""Bạn là chuyên gia về GDPR và luật bảo vệ dữ liệu cá nhân.

Câu hỏi gốc: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Hãy phân tích các vấn đề về privacy và GDPR (nếu có), bằng Tiếng Việt và dưới 200 từ.
"""
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"privacy_result": response.content}


async def aggregate(state: LegalState) -> dict:
    """Hợp nhất tất cả phân tích của các chuyên gia thành một câu trả lời toàn diện cuối cùng."""
    print("\n  [Node: aggregate] Đang tổng hợp phân tích từ các chuyên gia...")
    llm = get_llm()

    sections: list[str] = []
    if state.get("law_analysis"):
        sections.append(f"## Phân Tích Pháp Lý Tổng Quan\n{state['law_analysis']}")
    if state.get("tax_result"):
        sections.append(f"## Phân Tích Chuyên Sâu Về Thuế\n{state['tax_result']}")
    if state.get("compliance_result"):
        sections.append(f"## Phân Tích Tuân Thủ Quy Định\n{state['compliance_result']}")
    if state.get("privacy_result"):
        sections.append(f"## Phân Tích Quyền Riêng Tư Và GDPR\n{state['privacy_result']}")

    combined = "\n\n---\n\n".join(sections)

    messages = [
        SystemMessage(
            content=(
                "Bạn là cố vấn pháp lý cao cấp chịu trách nhiệm tổng hợp các phân tích chuyên gia thành một "
                "văn bản phản hồi toàn diện và có cấu trúc chặt chẽ. Hãy kết hợp các phần phân tích sau đây "
                "thành một câu trả lời nhất quán với các tiêu đề rõ ràng. Tránh trùng lặp thông tin. "
                "Câu trả lời bắt buộc phải viết bằng Tiếng Việt và dưới 500 từ."
            )
        ),
        HumanMessage(content=combined),
    ]
    result = await llm.ainvoke(messages)
    print(f"  [Node: aggregate] Hoàn thành ({len(result.content)} ký tự)")
    return {"final_answer": result.content}


# ---------------------------------------------------------------------------
# Xây dựng Đồ thị Graph (Mô phỏng cấu trúc của law_agent/graph.py)
# ---------------------------------------------------------------------------

def create_graph():
    """Xây dựng và biên dịch cấu trúc đồ thị StateGraph đa agent."""
    graph = StateGraph(LegalState)

    graph.add_node("analyze_law", analyze_law)
    graph.add_node("check_routing", check_routing)
    graph.add_node("call_tax_specialist", call_tax_specialist)
    graph.add_node("call_compliance_specialist", call_compliance_specialist)
    graph.add_node("privacy_agent", privacy_agent)
    graph.add_node("aggregate", aggregate)

    graph.set_entry_point("analyze_law")
    graph.add_edge("analyze_law", "check_routing")
    graph.add_conditional_edges(
        "check_routing",
        route_to_specialists,
        ["call_tax_specialist", "call_compliance_specialist", "privacy_agent", "aggregate"],
    )
    graph.add_edge("call_tax_specialist", "aggregate")
    graph.add_edge("call_compliance_specialist", "aggregate")
    graph.add_edge("privacy_agent", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()


# --- ĐÃ VIỆT HÓA CÂU HỎI VÀ SUMMARY ĐỂ TIỆN THEO DÕI LOG ---
QUESTION = "Nếu một công ty vi phạm hợp đồng và trốn thuế, họ sẽ phải đối mặt với những hậu quả pháp lý và quy định nào?"


async def main():
    print("=" * 70)
    print("STAGE 4: Hệ Thống Đa Agent - Multi-Agent System (In-Process)")
    print("=" * 70)
    print()
    print("[Cơ chế hoạt động]")
    print("  1. Agent Luật sư trưởng phân tích tổng quan câu hỏi.")
    print("  2. Router quyết định xem cần đến chuyên gia chuyên trách nào.")
    print("  3. Chuyên gia Thuế + Tuân thủ chạy SONG SONG (Thông qua Send API của LangGraph).")
    print("  4. Bộ tổng hợp (Aggregator) gom tất cả kết quả lại thành câu trả lời cuối cùng.")
    print()
    print("[Cấu trúc Đồ thị - Graph Topology]")
    print("  analyze_law -> check_routing -> [call_tax + call_compliance] -> aggregate -> END")
    print()
    print(f"Câu hỏi (Question): {QUESTION}")
    print("-" * 70)

    graph = create_graph()

    result = await graph.ainvoke({
        "question": QUESTION,
        "law_analysis": "",
        "needs_tax": False,
        "needs_compliance": False,
        "needs_privacy": False,
        "tax_result": "",
        "compliance_result": "",
        "privacy_result": "",
        "final_answer": "",
    })

    print("\n" + "=" * 70)
    print("KẾT QUẢ CUỐI CÙNG (FINAL ANSWER)")
    print("=" * 70)
    print(result["final_answer"])

    print()
    print("-" * 70)
    print("[Các điểm cải tiến so với Stage 3]")
    print("  + Tính chuyên môn hóa: Mỗi agent có một vùng tri thức chuyên biệt.")
    print("  + Chạy song song (Parallel): Agent thuế & tuân thủ chạy đồng thời giúp tiết kiệm thời gian.")
    print("  + Chất lượng cao hơn: Prompt chuyên biệt cho ra phân tích sâu sắc hơn.")
    print("  + Luồng chạy có cấu trúc rõ ràng nhờ vào định tuyến điều kiện của đồ thị.")
    print()
    print("=" * 70)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
