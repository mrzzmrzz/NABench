# 在 MaveDB 与 GEO 上检索基因突变数据（SELEX / DMS / MPRA）——实用教程



---

## 目录
- [概述：技术与数据库](#概述技术与数据库)
- [一、在 MaveDB 检索](#一在-mavedb检索)
- [二、在 GEO 检索（涵盖 SELEX/MPRA/部分 DMS）](#二在-geo检索涵盖-selexmpra部分-dms)
- [三、在 ENCODE 检索与下载（RNA/DNA 序列变体/功能）](#三在-encode检索与下载rnadna序列变体功能)
- [四、下载与整理：字段统一与示例](#四下载与整理字段统一与示例)
- [常见关键词](#常见关键词)
---

## 概述：技术与数据库




| 内容 | MaveDB | GEO (NCBI GEO) | ENCODE |
|---|---|---|---|
| **核心定位** | **MAVEs**（多重并行变体效应实验）结果与元数据的社区数据库，偏 **DMS/MPRA** | 广谱功能基因组/转录组数据索引与归档；**SELEX**、**MPRA**、部分 **DMS** 常见，附补充表与 SRA 原始测序 | 大型功能基因组学门户；覆盖 **RBNS/eCLIP/ChIP-seq/ATAC-seq/MPRA/STARR-seq** 等，并提供 **cCRE**（候选顺式调控元件）注释与高质量元数据 |
| **数据粒度** | **Project → Experiment → Score set**（逐变体效应表） | **GSE（系列）→ GSM（样本）→ SRA（SRX/SRR）** | **Experiment（实验）→ File（文件）**；并有 **Annotation/Track**（如 cCRE）层级 |
| **典型用途** | 直接下载 **逐变体效应/打分矩阵**（CSV/TSV）用于建模评测 | 找到 **补充表（variants/sequence + score）** 或 **原始 FASTQ** 以复现分析 | 获取 **表格/峰集/信号轨** 与严格元数据；如 **RBNS k-mer 富集表**、**MPRA/STARR-seq 计数/活性**、**eCLIP/ChIP 峰与注释**、**cCRE** 清单 |
| **数据类型/文件** | CSV/TSV 的 **score set**（常含 `variant/sequence/score/stderr/count`） | 补充文件（CSV/TSV/XLSX）、SRA FASTQ、部分作者仓库链接 | TSV/CSV（富集/计数/注释）、BED/bed.gz（峰/区域）、bigWig（信号）、FASTQ（原始读段） |
| **优势** | **开箱即用的变体效应表**；分数定义清晰、易直接评测 | 覆盖广、历史数据全；多数条目有补充表或可回溯到原始测序 | 质量控制严格、元数据完备；**REST API** 稳定；多类型功能数据可交叉验证（如 cCRE + ChIP + MPRA） |
| **局限/注意** | 范围以 MAVE 为主；若需原始 reads 依赖外部链接 | 元数据规范度不一；命名/列名需统一；部分仅有原始数据需自建流程 | 某些实验仅给峰/信号需自行从序列窗口构造“序列→分数”；组装版本需统一（GRCh38 等） |
| **适配技术** | **DMS、MPRA、（部分）RNA 功能/核酶** | **SELEX/HT-SELEX、MPRA、部分 DMS**（视项目） | **RBNS（RNA 结合）、eCLIP（体内 RBP 结合）、MPRA/STARR-seq（调控活性）、ChIP-seq/ATAC-seq、cCRE 注释** |
| **“序列/变体 + 分数”可得性** | **高**：Score set 直接提供 | **中-高**：很多 GSE 提供补充表；也可能只给 FASTQ | **中-高**：RBNS/MPRA 常有 TSV/CSV；eCLIP/ChIP 多为峰/信号（需后处理） |
| **原始测序获取** | 视条目链接到 GEO/SRA 或作者仓库 | **SRA** 直接获取（prefetch/fasterq-dump） | 门户可下 FASTQ；部分也同步到外部镜像/云存储 |
| **程序化访问** | 无统一 API（以页面/DOI 为主） | NCBI **E-utilities + SRA Toolkit** | 门户 **REST API**（检索 Experiment/File、直链 `@@download`）、购物车 **manifest** 批量下载 |
| **典型检索切入** | 站内搜：`deep mutational scanning`、`MPRA`、基因/元件名 | GEO DataSets：`"deep mutational scanning" OR "saturation mutagenesis"`；`MPRA OR STARR-seq`；`SELEX OR "HT-SELEX"` | 门户筛选：**Assay=RBNS/MPRA/STARR-seq/eCLIP**；**File format=tsv/csv**；**target.label=RBP/TF 名称** |
| **适合任务示例** | 建立/评测 **变体效应预测**（蛋白/RNA/调控） | 从补充表快速组装 **序列/变体 + 分数** 评测集或从 FASTQ 复现 | 用 **RBNS** 做 **RNA 结合打分**、用 **MPRA/STARR-seq** 做 **DNA 调控活性** 建模，结合 **cCRE/ChIP/ATAC** 做多模态标签 |


---

## 常见关键词

- **DMS**：`deep mutational scanning`
- **SGE**：`Saturation Genome Editing`
- **MAVE**：`Multiplexed Assay of Variant Effects`
- **STARR-seq**：`Self-Transcribing Active Regulatory Region Sequencing`
- **MPRA**：`massively parallel reporter assay`
- **Base/Prime Editing Scanning**：`Base Editing`、`Prime Editing`
- **SELEX**：`Systematic Evolution of Ligands by Exponential Enrichment`
- **HT-SELEX**：`High-Throughput SELEX`
- **RBNS**：`RNA Binding Protein (RBP) Selection`




---

## 一、在 MaveDB 检索

**入口**：<https://www.mavedb.org/>

### 1. 基本检索
1. 打开首页顶部搜索栏，输入关键词如：
   - `homo sapiens`
   - `deep mutational scanning`
   - `massively parallel reporter assay`
    - `non_coding`
2. 点击 **Search / Browse** 后查看结果列表。

### 2. 结果页与结构认读
- **Project**（项目）：一个研究课题的集合容器  
- **Experiment**（实验）：一次/一类测定  
- **Score set**（评分集）：**逐变体的效应数值表**（通常含 `variant/sequence/score/stderr/count` 等）

> **通常你要下载的是 Score set**，它对应 “每个变体的效应分数”。
> 同时需要找到页面中的**Reference sequence**作为野生序列。

### 3. 筛选建议
- **Target Type**：'Protein Coding' 较多，可以适当选择10个进行评测，'Regulatory','Non-coding' 可以全部评测。


### 4. 下载与元数据
- 进入目标 **Score set** 页面，下载 **CSV/TSV**；同时保存：
  - **Metadata/README**（描述实验流程、打分定义、归一化方式）
  - **License/DOI**（引用用）
- 若存在多个 Score set（不同条件/轮次/筛选阈值），**分别下载并记录差异**。


---

## 二、在 GEO 检索（涵盖 SELEX/MPRA/部分 DMS）

**入口（DataSets）**：<https://www.ncbi.nlm.nih.gov/gds>

> GEO 最强在于**广谱覆盖**和**链接原始测序（SRA）**。SELEX、MPRA 相关研究常在 GEO 可查到 **GSE（系列）页面**，随后进入 **Supplementary files**（处理后矩阵/表格）或 **SRA Run Selector**（原始 FASTQ）。

### 1. 基本搜索语法与范围
- 使用布尔与短语：
  - `"deep mutational scanning" OR DMS`
  - `MPRA OR "massively parallel reporter assay" OR STARR-seq`
  - `SELEX OR "SELEX-seq" OR "HT-SELEX"`
- 结合对象与物种：
  - `("SELEX-seq" AND "RNA-binding protein") AND Homo sapiens`
  - `(MPRA AND enhancer AND Homo sapiens)`

### 2. 打开 GSE 页面关注要点
- **Overall design / Summary**：确认是否为 MPRA/DMS/SELEX  
- **Supplementary files**：通常包含**处理后矩阵**（如变体效应、位点打分、barcode 统计）

### 3. 典型场景与技巧
- **SELEX**：检索 `SELEX OR "SELEX-seq" OR "HT-SELEX"`，可配合 `transcription factor`、`RNA-binding protein`、具体蛋白名  
- **MPRA**：检索 `MPRA OR "massively parallel reporter assay" OR STARR-seq` 
- **DMS**：检索 `"deep mutational scanning" OR "saturation mutagenesis"`

---

## 三、在 ENCODE 检索与下载（RNA/DNA 序列变体/功能）

> 目标：在 ENCODE 门户中，用**网页筛选**或**REST API**快速拿到与 **RNA/DNA 序列**直接相关、可用于模型评测的**表格型文件**（如：k-mer/序列富集分数、MPRA/STARR-seq 活性、eCLIP/RBNS 结果、cCRE 注释等）。

### 1）入口与常用数据类型
- **ENCODE Data Portal**：`https://www.encodeproject.org/`
- 你最可能用到的文件类型（**File type**）与输出（**Output type**）：
  - **TSV/CSV 表格**：`k-mer enrichment`（RBNS）、`counts`/`quantification`（部分 MPRA/STARR-seq/功能表征）、`metadata.tsv`
  - **BED/bed.gz**：`peaks`（eCLIP/ChIP-seq）、`cCRE annotations`（可作 DNA 区域标签/负样本筛选）
  - **bigWig**：信号轨（供可视化/区域打分，不直接“变体→分数”，但可配合区域聚合）

---

### 2）网页检索（可视化筛选）
1. 打开 **ENCODE** 首页 → 顶部 **Search**
2. 关键过滤器（左侧或顶部栏）：
   - **Assay**：`RBNS`（RNA Bind-n-Seq）、`eCLIP`（RBP 体内结合）、`STARR-seq`/`MPRA`/`functional characterization assay`、`ChIP-seq`（TF）
   - **Organism**：`Homo sapiens` / `Mus musculus`
   - **Target of assay**：选择具体 RBP/TF（如 `RBFOX2`、`PTBP1`、`CTCF`）
   - **File format**：优先 `tsv`/`csv`，其次 `bed`/`bigWig`
   - **Assembly**：`GRCh38`（人）或 `mm10`（鼠），保持一致便于下游处理
   - **Status**：`released`
3. 进入某个 **Experiment**（实验）页面 → 切换到 **Files**（文件）页签：
   - 查看 **Output type** 与 **File type**，定位可下载的 **TSV/CSV**（例如 `k-mer enrichment`、`counts`、`quantification`）
   - 点击文件右侧 **Download** 按钮直接下载；或先 **Add to cart**（加入购物车）以便**批量**导出

> 小提示：在结果页点击右上角 **Download** 可导出当前筛选的**清单/元数据**（`metadata.tsv`）；先把筛选条件调好再导出，能直接得到一份“可追溯”的文件目录。

---

### 3）批量下载（购物车 / 清单）
- 在文件列表中把需要的条目 **Add to cart** → 打开页面右上角 **Cart**：
  - **Download manifest**（清单/清册）：得到一个包含直链 URL 的文本文件
  - 使用 `wget`/`curl` 批量下载，例如：
    ```bash
    # 清单中每行是一个可下载链接
    wget -i cart-manifest.txt -c
    # 或者：
    xargs -n 1 -P 8 curl -O -L < cart-manifest.txt
    ```
- 同时导出 **cart metadata.tsv**（元数据表），保留 `accession`、`assay_title`、`target`、`file_format`、`output_type` 等信息，便于记录来源与复现。


---

## 四、下载与整理：字段统一与示例

### 0. 目标格式

**变体效应分数表**：csv文件，每行一个变体条目，列包括:
- mutant: 如 `A123T`、`"A1G,G2C,C3T"`(多个突变用双引号括起来)
- DMS_score: 效应分数（如 log2FC）
- sequence: 变体对应的核酸序列

### 1. MaveDB（Score set）
常见列：
- `variant`（HGVS 或自定义变体编码，如 `A123T`、`c.123A>T`、`p.Ala123Thr`）
- `score`（效应分数，方向、尺度需看 README）
- `stderr` / `se`（不确定性）
- `count`（测序/条形码支持度）
- `sequence`（可选，核酸或氨基酸）



### 2. GEO（GSE/GSM 补充文件与 SRA）
- **补充文件**：TSV/CSV/Excel，常含：
  - MPRA：`oligo_id`、`sequence`、`barcode`、`counts`（输入/输出各轮）、`log2FC`、`activity_score`
  - SELEX：各轮序列频次、PWM/位点打分、k-mer 富集表
  - DMS：位点×氨基酸替换矩阵、单突变/双突变效应
- **SRA 原始数据**：使用 SRA Toolkit（`prefetch`/`fasterq-dump`）下载 FASTQ 后自建流程重算。 (复杂，待实现)

### 3. ENCODE（TSV/CSV）
- **k-mer enrichment**：`kmer`、`enrichment_score`、`stderr`、`counts`（输入/输出）
- **MPRA/STARR-seq**：`oligo_id`、`sequence`、`barcode`、`counts`（输入/输出）、`activity_score`
- **eCLIP/RBNS**：`RBP`、`kmer`、`enrichment_score`、`stderr`、`counts`