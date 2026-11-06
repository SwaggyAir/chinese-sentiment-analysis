# 中文酒店评论情感分析系统

基于 ChnSentiCorp 酒店评论语料的中文情感二分类课程项目：输入一条中文评论，输出**正面 / 负面**
预测与模型置信度。项目包含数据检查与清洗、TF-IDF + 逻辑回归基线、中文预训练模型对比、
测试集评估与错误分析，以及一个本地 Streamlit 网页界面。

> 本 README 中的全部指标都由本机实际运行产生（见 `reports/evaluation_report.md` 与
> `models/*/metrics.json`），没有手写或估算的数字。

## 1. 功能

- 读取 CSV，自动识别文本列与标签列，兼容 utf-8 / utf-8-sig / gb18030 等编码
- 打印数据概况：行数、列名、标签分布、缺失值、重复与清洗统计
- 统一标签映射为 `0=负面`、`1=正面`，固定随机种子分层划分 80/10/10
- 基线模型：字符级 n-gram TF-IDF + Logistic Regression，在验证集上搜索超参数
- 对比模型：`hfl/chinese-roberta-wwm-ext` 微调（可选，需要 torch + transformers）
- 测试集评估：Accuracy / Precision / Recall / F1 / 混淆矩阵 + 错误样本分类分析
- Streamlit 网页：单条预测、置信度、示例评论、模型选择、数据集与评估结果展示

## 2. 环境要求

| 用途 | Python 环境 | 需要的包 |
| --- | --- | --- |
| 数据处理、基线训练、网页 | 任意装了 `requirements.txt` 的环境（本项目用 Anaconda base，Python 3.14） | pandas、scikit-learn、joblib、streamlit |
| BERT 微调、加载与网页中选择 BERT | 带 torch 的环境（本项目用 `E:/conda_envs/pytorch_CV`：torch 2.14.1+cu126、transformers 5.22.1、streamlit 1.67.0） | torch、transformers、pandas、streamlit |

基线模型和网页**不依赖** torch：BERT 相关依赖缺失、模型下载失败或训练中断时，
其余功能照常可用，脚本会打印跳过原因而不是让项目启动失败。

```bash
pip install -r requirements.txt
```

## 3. 快速开始

```bash
# 0) 准备数据：把 ChnSentiCorp_htl_all.csv 放到 data/（下载方式见 data/README.md）

# 1) 数据检查 + 清洗 + 分层划分（写出 data/processed/*.csv 与 meta.json）
python -m src.data_loader

# 2) 训练基线模型（写出 models/baseline/）
python -m src.train_baseline

# 3) 生成评估报告（写出 reports/evaluation_report.md）
python -m src.evaluate

# 4) 启动网页
python -m streamlit run app.py

# 5) 可选：在带 torch 的环境微调中文预训练模型（写出 models/bert/）
E:/conda_envs/pytorch_CV/python.exe -m src.train_bert
```

`src.train_bert` 常用参数：`--epochs 3 --batch-size 16 --max-length 256 --lr 2e-5 --seed 42`，
另有 `--model`（换底座）、`--device cpu`、`--limit-train N`、`--max-steps N`（冒烟测试）。

## 4. 目录结构

```text
.
├── app.py                    # Streamlit 网页
├── requirements.txt
├── data/
│   ├── README.md             # 数据来源与下载方式
│   ├── ChnSentiCorp_htl_all.csv        # 原始数据（不提交）
│   └── processed/                        # train/val/test.csv + meta.json（不提交）
├── models/                   # 全部为训练产物，不提交
│   ├── baseline/             # vectorizer/model.joblib + metrics + predictions
│   ├── bert/                 # 微调后的权重与 metrics
│   └── pretrained/           # 手动下载的中文预训练底座权重
├── reports/
│   ├── evaluation_report.md  # python -m src.evaluate 生成
│   └── bert_train_log.txt    # 本次 BERT 训练原始日志
└── src/
    ├── config.py             # 路径、种子、标签映射、列名候选
    ├── data_loader.py        # 读取、检查、清洗、划分
    ├── metrics.py            # 指标计算与结果落盘
    ├── train_baseline.py     # TF-IDF + 逻辑回归
    ├── train_bert.py         # 中文预训练模型微调（可选）
    ├── evaluate.py           # 指标对比与错误分析
    └── predict.py            # 统一单条预测接口
```

## 5. 数据集说明

- **来源**：[SophonPlus/ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus)
  整理的 `ChnSentiCorp_htl_all`（中文酒店评论情感分类），公开整理页描述为 7000 余条、
  正面多于负面。本项目使用实际下载到的副本，未把记录数写死进代码。
- **实际文件**：`data/ChnSentiCorp_htl_all.csv`，编码 utf-8-sig，列名 `label` 与 `review`，
  共 **7766 行**。
- **标签映射**：原始标签即为 `1` / `0`，统一为 `1 = 正面`、`0 = 负面`，与原始含义一致，
  未做重新标注。原始分布：正面 5322 条、负面 2444 条（正面约 68.5%）。
- **清洗规则**：去除首尾空白 → 删除空文本与缺失标签（本项目命中 1 行）→ 删除标签值无法
  识别的行（0 行）→ 完全相同的文本只保留第一次出现（本项目 0 行，即无重复文本）。
  清洗后 **7765 条**有效样本。
- **划分**：`train 6212 / val 776 / test 777`，`random_state=42`，两次
  `train_test_split(..., stratify=label)` 实现 80/10/10 分层抽样。
  验证集只用于选超参数，测试集只在最终评估时使用一次。
- **适用范围**：中文酒店评论领域的文档级情感二分类。语料为早年公开抓取整理，仅适合
  教学与研究用途；迁移到其它领域（电商、影视、餐饮）需要重新标注或微调。

## 6. 模型

### 6.1 基线：TF-IDF + Logistic Regression

- 特征：`TfidfVectorizer(analyzer="char_wb", ngram_range, min_df=2, sublinear_tf=True,
  max_features=100000)`。字符级 n-gram 不需要额外分词器，也避开了分词错误带来的误差。
- 分类器：`LogisticRegression(max_iter=2000, random_state=42)`。
- 超参数搜索（仅用验证集 macro-F1 选择）：`ngram_range ∈ {(1,2),(1,3)}`、
  `C ∈ {1.0, 5.0}`、`class_weight ∈ {None, balanced}`，共 8 组。
- 选中配置：`ngram_range=(1,3)`、`C=5.0`、`class_weight="balanced"`，
  验证集 macro-F1 = 0.8701。完整搜索表见 `models/baseline/metrics.json`。
- 训练只在 train 划分上进行，产物保存在 `models/baseline/`（vectorizer.joblib、
  model.joblib、model_config.json、metrics.json、predictions.jsonl）。

### 6.2 对比模型：中文预训练 Transformer 微调

- 底座：`hfl/chinese-roberta-wwm-ext`（全词遮罩预训练的中文 BERT 变体，
  12 层、768 维、约 1.02 亿参数），分类头为 2 类。
- 训练：AdamW，lr 2e-5，weight_decay 0.01，warmup 10%，batch 16，
  `max_length=256`，epochs 3，seed 42；按验证集 macro-F1 保留最佳权重后再在测试集评估。
- 权重与结果保存在 `models/bert/`。
- 权重下载：脚本默认走 `HF_ENDPOINT=https://hf-mirror.com`（`src/config.py` 设置默认值，
  已自行配置环境变量时不会被覆盖）。本机实测 huggingface.co 与 GitHub 直连均超时，
  且 hf-mirror 的 xet 断点续传会卡住，因此改用手动物理下载（`curl -L -C -`）到
  `models/pretrained/chinese-roberta-wwm-ext/`；`src.train_bert` 若检测到该目录会自动优先使用它：

  ```bash
  mkdir -p models/pretrained/chinese-roberta-wwm-ext && cd models/pretrained/chinese-roberta-wwm-ext
  base=https://hf-mirror.com/hfl/chinese-roberta-wwm-ext/resolve/main
  for f in config.json vocab.txt tokenizer_config.json special_tokens_map.json added_tokens.json pytorch_model.bin; do
    curl -L -C - -o "$f" "$base/$f"
  done
  ```

- 本次训练实际耗时：见 `models/bert/metrics.json` 的 `history`（每 epoch 秒数已记录）。

## 7. 评估结果

测试集 777 条，其中**负面 245 条、正面 532 条**（正面占 68.5%），因此除 Accuracy 外
同时给出正面类 P/R/F1 与 Macro-F1。下表数字取自 `models/*/metrics.json`。

| 指标 | TF-IDF + Logistic Regression | chinese-roberta-wwm-ext |
| --- | --- | --- |
| Accuracy | 0.9163 | **0.9279** |
| Precision（正面类） | **0.9569** | 0.9542 |
| Recall（正面类） | 0.9192 | **0.9398** |
| F1（正面类） | 0.9377 | **0.9470** |
| Macro-F1 | 0.9052 | **0.9173** |

混淆矩阵（行=真实，列=预测）：

**基线 TF-IDF + Logistic Regression**

| 真实 \ 预测 | 预测·负面 | 预测·正面 |
| --- | --- | --- |
| 真实·负面 | 223 | 22 |
| 真实·正面 | 43 | 489 |

**对比模型 chinese-roberta-wwm-ext**

| 真实 \ 预测 | 预测·负面 | 预测·正面 |
| --- | --- | --- |
| 真实·负面 | 221 | 24 |
| 真实·正面 | 32 | 500 |

- 基线错误 65 条（假阳性 22、假阴性 43）；BERT 错误 56 条（假阳性 24、假阴性 32）。
- 同一测试集对照：**都正确 687 条，都错误 31 条，仅基线正确 25 条，仅 BERT 正确 34 条**。
  两个模型的错误高度重叠（31/65），说明剩余误差主要来自语料本身的难点与标注噪声，
  而不是模型容量不足。
- 基线可解释性：逻辑回归权重最高的字符片段偏正面为「不错、很好、干净」一类，
  偏负面为「很差、旧、脏、没有、不会再」一类，完整列表见 `reports/evaluation_report.md`。
- BERT 训练耗时：单 epoch 约 124 秒，3 epoch 共约 6.5 分钟（RTX 4060 Laptop，
  batch 16 / max_length 256），按验证集 macro-F1 选用了 epoch 3 的权重。

### 7.1 错误分析

`python -m src.evaluate` 会给出完整错误样本清单。基线 65 条错误的规则归类结果：
转折结构 36、否定表达 22、褒贬混合 1、其他 6（按优先级只取第一个命中特征）。典型样例：

- **转折**：「这家酒店我已经入住多次，总体印象是好的，虽然大堂很小，但是房间整洁、温馨……
  但是这次让我有些失望」——句中出现多次转折，情感落点在最后一句，字符 n-gram 难以定位。
- **否定 + 错别字**：「基本上没什么吃的……下次在来我是不会住在这里的」（真实标签为正面，
  属于语料标注噪声，模型判负面反而更符合文本）。
- **褒贬混合**：「住过几次，感觉还可以。但这次竟然晚上冷的睡不着」——前半段褒、后半段贬。

用网页侧边栏的四个手写示例对比两个模型（这些文本不在数据集中）：

| 示例 | 基线 | BERT |
| --- | --- | --- |
| 「房间很干净，前台服务态度很好，性价比高，下次还会来。」 | 正面 0.985 | 正面 0.999 |
| 「位置倒是还行，就是房间又旧又吵，空调声音大得一晚没睡好。」 | 负面 0.638 | 负面 0.973 |
| 「早餐种类丰富，环境也不错，但是退房时押金拖了半个月才退，体验很差。」 | 正面 0.761 | 正面 0.992 |
| 「呵呵，这服务真是‘到位’了，喊了三遍才有人理。」 | 正面 0.754 | 正面 0.999 |

后两例人读应为负面，两个模型都判成正面，且 BERT 的置信度反而更高——说明预训练模型
并未真正解决强转折与反讽问题，置信度也不能等同于正确率，这是本项目暴露出的典型局限。

## 8. 网页界面

```bash
# 只用基线模型（推荐，环境要求低）
E:/Anaconda/python.exe -m streamlit run app.py

# 需要选择 BERT 对比模型时，用带 torch 的环境启动
E:/conda_envs/pytorch_CV/python.exe -m streamlit run app.py
```

浏览器会自动打开（默认 <http://localhost:8501>）。

- **单条预测**：输入中文评论 → 点击「分析」→ 显示正面/负面、置信度进度条与两类概率；
  侧边栏四个手写示例按钮可一键填入输入框（示例不取自数据集，界面上不出现任何真实标签）。
- **模型选择**：侧边栏单选框只列出「已训练且当前环境能加载」的模型，默认基线模型。
  在没装 torch 的环境里即使 `models/bert` 存在，也不会出现在选项中，而是显示原因说明。
- **数据集与模型评估**：展示数据概况、指标对比表、混淆矩阵与错误样本；
  训练新模型后点「刷新评估结果」即可重新读取，无需重启服务。
- 输入为空或全空格时提示「请输入中文评论文本后再点击“分析”」。

## 9. 已知限制

1. 数据类别不均衡（正面约 68.5%），单看 Accuracy 会偏高，因此报告同时给出 Macro-F1
   与分类别指标。
2. 语料存在标注噪声：若干被判为「错误」的样本，其人工标签本身与文本情感不一致
   （例如「房间太旧了，服务一般……」被标为正面），这限制了两个模型可达到的上限。
3. 字符级 n-gram 无法建模长距离依赖与句法结构，遇到强转折、反讽时容易出错。
4. BERT 部分为可选对比项：在无 torch/transformers 或无法下载权重的环境中不训练、不加载，
   网页与基线流程不受影响。
5. 本项目只做文档级二分类，不涉及方面级情感、情感强度或多轮对话分析。

## 10. 验收对照

| 设计文档验收标准 | 本项目对应实现 |
| --- | --- |
| 1. 按 README 可准备数据并训练基线 | §3 快速开始 + `data/README.md`；基线已在本机跑通（9 秒） |
| 2. 输出数据概况与测试集指标 | `python -m src.data_loader` 打印概况；`src.evaluate` 输出指标与混淆矩阵 |
| 3. 输入评论返回正负面与置信度 | 网页「单条预测」页，或 `python -c "from src.predict import predict; print(predict('...').as_dict())"` |
| 4. BERT 失败时基线与网页仍可运行 | 依赖缺失/权重下载失败时 `train_bert.py` 打印跳过原因并退出；无 torch 的环境里界面只显示基线并说明原因（已实测） |
| 5. 种子、划分、标签映射有文档说明 | §5 数据集说明与 `data/processed/meta.json` |
| 6. 预测界面不显示真实标签 | 预测页只输出模型预测与概率，示例为手写文本；真实标签仅出现在「模型评估」页的错误分析中，用于说明模型失误 |
| 7. 说明数据集整理版本、来源与适用范围 | §5 与 `data/README.md` |

## 11. 参考

- SophonPlus/ChineseNlpCorpus 中文自然语言处理数据集整理仓库
- 王业全．多层次文本情感分析研究．清华大学博士学位论文，2019（文档级/要素级/对话级情感分析的任务界定）
- 许峰．基于深度学习的网络舆情识别研究．北京邮电大学博士学位论文，2019（中文情感识别与深度学习方法背景）
- Devlin J, et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019
- 赵阳等．Pre-training with Complete Masked Language Models for Chinese Language Understanding（chinese-roberta-wwm-ext）. 2021

> 两篇博士论文用于研究背景与方法综述定位，正式提交前请通过学校知网入口核对作者、
> 学校、年份与学位信息，并按学校格式引用。
