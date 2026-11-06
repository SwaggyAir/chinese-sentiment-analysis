# 数据集目录

本目录存放原始数据，不提交到 Git（见根目录 `.gitignore`）。

## 需要的文件

`ChnSentiCorp_htl_all.csv`：中文酒店评论情感分类数据，两列 `label`、`review`，
标签 `1=正面`、`0=负面`。本仓库当前使用的副本为 **7766 行**（其中 1 行文本为空，
清洗后 7765 行），实际下载版本可能有差异，脚本会以文件内容为准自动识别列名。

## 来源

- 原始整理仓库：[SophonPlus/ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus)
  路径 `datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv`
- 该仓库为公开整理版，原始出处为二手酒店评论抓取，仅用于学术与教学用途，
  不含个人身份信息；使用前请遵守原作者的声明。

## 下载方式

GitHub 直连（若网络可达）：

```bash
curl -L -o data/ChnSentiCorp_htl_all.csv \
  https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv
```

GitHub 不可达时，可走 jsDelivr CDN 镜像（本项目当前副本即由此下载）：

```bash
curl -L -o data/ChnSentiCorp_htl_all.csv \
  https://cdn.jsdelivr.net/gh/SophonPlus/ChineseNlpCorpus@master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv
```

## 目录内容

- `ChnSentiCorp_htl_all.csv`：原始数据（手动准备，不提交）
- `processed/`：由 `python -m src.data_loader` 生成的 `train.csv`、`val.csv`、`test.csv`
  与 `meta.json`（记录列名识别结果、清洗统计、标签映射、随机种子和划分规模）
