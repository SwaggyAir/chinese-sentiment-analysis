"""中文酒店评论情感分析系统：Streamlit 网页界面。

启动：
    python -m streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import evaluate as ev
from src import metrics as mt
from src import predict as pr
from src.config import PROCESSED_DIR

MODEL_NAMES = {
    "baseline": pr.BaselinePredictor.display_name,
    "bert": pr.BertPredictor.display_name,
}

# 示例文本为手写，不取自数据集，避免把真实标签带到预测界面。
EXAMPLES = {
    "正面·简短": "房间很干净，前台服务态度很好，性价比高，下次还会来。",
    "负面·吐槽": "位置倒是还行，就是房间又旧又吵，空调声音大得一晚没睡好。",
    "褒贬混合": "早餐种类丰富，环境也不错，但是退房时押金拖了半个月才退，体验很差。",
    "反讽语气": "呵呵，这服务真是‘到位’了，喊了三遍才有人理。",
}


@st.cache_data(show_spinner=False)
def load_meta() -> dict:
    return mt.read_json(PROCESSED_DIR / "meta.json") or {}


@st.cache_data(show_spinner=False)
def load_report(limit: int = 5) -> str:
    blocks = [ev.load_block(name) for name in ("baseline", "bert")]
    return ev.build_report([b for b in blocks if b], limit)


def sidebar(meta: dict, trained: list[str]) -> str:
    st.header("模型选择")
    options = [name for name in MODEL_NAMES if name in trained]
    if not options:
        st.error("尚未找到已训练模型，请先运行 `python -m src.train_baseline`。")
        st.stop()
    choice = st.radio("预测用模型", options, format_func=MODEL_NAMES.get)
    if "bert" not in options:
        st.info(pr.unavailable_reason("bert"))

    st.header("示例评论")
    st.caption("点击写入左侧输入框；示例为手写文本，不含数据集真实标签。")
    for index, (name, sample) in enumerate(EXAMPLES.items(), start=1):
        if st.button(name, key=f"example_{index}", width="stretch"):
            st.session_state.review_text = sample

    st.header("数据与环境")
    if meta:
        stats, sizes = meta["clean_stats"], meta["split_sizes"]
        st.write(
            f"- 文件：`{meta['source_file']}`（文本列 `{meta['detected_text_col']}`，标签列 `{meta['detected_label_col']}`）\n"
            f"- 原始 {stats['raw_rows']} 行 → 清洗后 {stats['clean_rows']} 行\n"
            f"- 标签映射：0=负面、1=正面\n"
            f"- 划分：train {sizes['train']} / val {sizes['val']} / test {sizes['test']}\n"
            f"- 随机种子：{meta['random_seed']}"
        )
    st.write(f"- 当前解释器：`{sys.executable}`")
    return choice


def predict_tab() -> None:
    text = st.text_area(
        "请输入中文酒店评论",
        key="review_text",
        height=150,
        placeholder="例如：酒店位置方便，房间干净，服务热情。",
    )
    analyze = st.button("分析", type="primary")

    if not text.strip():
        if analyze:
            st.warning("请输入中文评论文本后再点击“分析”。")
        else:
            st.caption("输入评论后点击“分析”，查看情感倾向与模型置信度。")
        return
    if not analyze:
        st.caption("输入评论后点击“分析”，查看情感倾向与模型置信度。")
        return

    choice = st.session_state.get("chosen_model", "baseline")
    try:
        result = pr.predict(text, model=choice)
    except Exception as exc:
        st.error(f"预测失败：{type(exc).__name__}: {exc}")
        return

    mark = "✅" if result.label == 1 else "❌"
    st.subheader(f"{mark} 预测结果：{result.label_name}")
    st.progress(result.confidence, text=f"模型置信度 {result.confidence:.1%}")
    st.write("类别概率：" + "　".join(f"{k} {v:.1%}" for k, v in result.probabilities.items()))
    st.caption(f"使用模型：{MODEL_NAMES[choice]}　输入长度：{len(text.strip())} 字")


def main() -> None:
    st.set_page_config(page_title="中文评论情感分析系统", layout="wide")
    st.title("中文酒店评论情感分析系统")
    st.caption("ChnSentiCorp 酒店评论二分类：正面 / 负面。指标来自本地实际训练结果。")

    meta = load_meta()
    with st.sidebar:
        st.session_state.chosen_model = sidebar(meta, pr.available_models())

    tab_predict, tab_report = st.tabs(["单条预测", "数据集与模型评估"])
    with tab_predict:
        predict_tab()
    with tab_report:
        if st.button("刷新评估结果", help="训练新模型后点此重新读取 metrics，无需重启服务"):
            st.cache_data.clear()
        if not meta:
            st.warning("未找到 data/processed/meta.json，请先运行 `python -m src.data_loader`。")
        st.markdown(load_report())


if __name__ == "__main__":
    main()
