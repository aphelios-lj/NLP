# 文本分类实验（作业一）

自然语言处理课程作业一的完整实现。本项目围绕 **NYT 新闻数据集**，系统比较了三种文本表示方法在文本分类任务上的效果：

- **Bag-of-Words**（Binary BOW、Word Frequency）
- **Word Embedding**（Pre-trained GloVe、Word2Vec Trained on AG News、Word2Vec Trained on NYT）
- **Pre-trained Neural Model**（BERT-base-uncased Fine-tuning）

分类器统一使用 **Logistic Regression**（BERT 部分使用其自身的分类头）。评价指标为 **Accuracy** 与 **Macro-F1**。

---

## 项目结构

```
text-classification/
├── data/                       # 数据集目录（nyt.csv、ag.csv）
├── glove/                      # GloVe 100d 词向量目录
├── outputs/                    # 运行输出（BERT 训练日志等）
├── src/
│   ├── __init__.py
│   ├── data_utils.py           # 数据加载、划分、分词、评价
│   ├── task1_bow.py            # Task 1: Binary BOW / Word Frequency
│   ├── task2_word2vec.py       # Task 2: GloVe / Word2Vec
│   ├── task3_bert.py           # Task 3: BERT 微调
│   └── main.py                 # 主程序入口
├── requirements.txt
└── README.md
```

---

## 环境配置

推荐 Python 3.9 或 3.10。

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` 主要依赖：

```
pandas
numpy
scikit-learn
nltk
gensim
torch
transformers
datasets
tqdm
```

首次运行会自动下载 NLTK 的 `punkt_tab` 分词数据。

---

## 数据与模型准备

本项目未包含数据文件与预训练模型，请按以下方式放置：

1. **NYT 数据集**：`nyt.csv` 放入 `data/`，含 `text`、`label` 两列。
2. **AG News 数据集**：`ag.csv` 放入 `data/`，含 `text` 列。
3. **GloVe 100d**：下载 [http://nlp.stanford.edu/data/glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip)，将 `glove.6B.100d.txt` 放入 `glove/`。
4. **BERT-base-uncased**：从 HuggingFace 获取 `google-bert/bert-base-uncased`，放入项目根目录的 `bert-base-uncased/`。

---

## 运行

在项目根目录下：

```bash
python -m src.main
```

主程序依次执行：

1. 加载 NYT 与 AG News；
2. 对 NYT 按 80% / 10% / 10% 随机划分（`random_state=42`）；
3. Task 1：Binary BOW 与 Word Frequency；
4. Task 2：GloVe、Word2Vec on AG News、Word2Vec on NYT；
5. Task 3：BERT-base-uncased 微调。

每个任务结束后会打印 Validation 与 Test 集上的 Accuracy、Macro-F1。

---

## 数据划分

| 集合 | 样本数 | 占比 |
|---|---:|---:|
| Training Set | 9215 | 80% |
| Validation Set | 1152 | 10% |
| Test Set | 1152 | 10% |

所有方法使用同一划分，保证不同表示方法之间的可比性。

---

## 实验结果

### Task 1: Bag of Words（30 分）

| 方法 | Validation Accuracy | Validation Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---:|---:|---:|---:|
| Binary Bag of Words | 0.9835 | 0.9596 | 0.9887 | 0.9728 |
| Word Frequency | 0.9809 | 0.9508 | **0.9922** | **0.9820** |

### Task 2: Word Embedding（50 分）

统一使用 100 维词向量，文档向量通过对文档内所有有效词向量取平均得到。

| 方法 | Validation Accuracy | Validation Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---:|---:|---:|---:|
| Pre-trained GloVe (100d) | 0.9766 | 0.9447 | 0.9783 | 0.9491 |
| Word2Vec on AG News | 0.9748 | 0.9389 | 0.9774 | 0.9495 |
| Word2Vec on NYT | 0.9748 | 0.9365 | **0.9792** | **0.9502** |

### Task 3: BERT Fine-tuning（20 分）

配置：`bert-base-uncased`，`max_len=64`，`epochs=3`，`lr=2e-5`。

由于实验环境网络限制，HuggingFace 的模型权重下载在重定向至 Xet CDN（`cas-bridge.xethub.hf.co`）时 DNS 解析失败，模型文件无法落地，因此本部分**未获得实际运行结果**。`src/task3_bert.py` 中已包含完整实现，符合作业的模型、序列长度与训练轮数要求；若模型文件就位于本地目录，运行 `python -m src.main` 即可自动完成微调。

---

## 结果分析

1. **Bag-of-Words 表现最优**。两种 BOW 方法的 Test Accuracy 均超过 98%，Macro-F1 均超过 0.97，说明新闻类别与词汇特征高度相关，稀疏高维表示在本任务中非常有效。
2. **Word Frequency 略优于 Binary BOW**。词频保留了每个词在文档中的权重信息，Logistic Regression 能从中学习到更有区分度的特征。
3. **Word Embedding 方法整体略低于 BOW**。GloVe、Word2Vec on AG、Word2Vec on NYT 的 Test Accuracy 在 97.7%–97.9% 之间，Macro-F1 在 0.949–0.950 之间。差异主要来自**对文档内词向量取平均会丢失词序与部分语义信息**。
4. **预训练与自训练词向量差异有限**。GloVe 因大规模通用语料预训练，验证集上略好；Word2Vec on NYT 因与目标任务同分布，在 Test 集上略好，总体差距在合理波动范围内。
5. **BERT 部分代码完整但未运行**。受网络环境限制，BERT 权重无法下载，Task 3 未能取得测试结果。

---

## 实现要点

### 数据划分

```python
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
```

### 分词

统一使用 NLTK `word_tokenize`，配合小写化与非字母数字清洗：

```python
def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return word_tokenize(text)
```

### 文档向量

Word Embedding 部分对文档内所有有效词向量取平均：

```python
def document_vector(tokens, embeddings, dim=100):
    vecs = [embeddings[w] for w in tokens if w in embeddings]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0)
```

### BERT 微调

采用 HuggingFace `Trainer`，每轮在验证集上评估，按 Macro-F1 选择最优模型：

```python
training_args = TrainingArguments(
    output_dir="outputs/bert",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    metric_for_best_model="macro_f1",
    load_best_model_at_end=True,
    report_to="none"
)
```

---

## 遇到的问题与解决过程

### 1. Task 2 返回值数量不一致

**现象**：运行至 Task 2 的 GloVe 阶段时抛出：

```
ValueError: not enough values to unpack (expected 4, got 3)
```

**原因**：`main.py` 按 4 个返回值解包，而 `task2_word2vec.py` 中 `_train_and_eval_vectors` 及三个 `run_*` 函数只返回 3 个值。

**解决**：修改 `task2_word2vec.py`，让 `_train_and_eval_vectors` 统一返回 `(val_metrics, test_metrics, clf, None)`，三个 `run_*` 函数相应返回 4 个值。修改后 Task 1 与 Task 2 的接口保持一致，主程序无需再改。

### 2. HuggingFace 模型下载失败

**现象**：无论使用 `huggingface-cli`、`hf download` 还是 `curl.exe`，`model.safetensors` 的下载都会在重定向到 `cas-bridge.xethub.hf.co` 后失败，报 `getaddrinfo failed` 或 `Could not resolve host`。

**原因**：HuggingFace 自 2025 年起将大文件迁移至 Xet CDN，镜像站对元数据请求生效，但大文件仍会 302 重定向到 Xet CDN；当前网络环境无法解析该域名。

**尝试过的方案**：

- `HF_ENDPOINT=https://hf-mirror.com` 镜像；
- `HF_HUB_DISABLE_XET=1` 禁用 Xet；
- `huggingface-cli` 与新版 `hf` CLI；
- `curl.exe -L` 直接下载；
- 浏览器手动下载。

均由于同一 DNS 解析失败问题未能完成。

### 3. Git 推送失败

**现象**：`git push` 先后报 `Failed to connect to github.com:443`、`Recv failure: Connection was reset`。

**原因**：浏览器通过系统代理访问 GitHub，而 Git 默认直连，443 端口被中间设备重置。

**解决**：查询系统代理：

```powershell
Get-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings" | Select-Object ProxyEnable, ProxyServer
```

得到 `127.0.0.1:55271`，配置 Git：

```powershell
git config --global http.proxy http://127.0.0.1:55271
git config --global https.proxy http://127.0.0.1:55271
```

配置后推送成功。

---

## 结论

- 在本任务中，**BOW 表示配合 Logistic Regression 已能取得非常强的分类性能**；
- Word Embedding 提供了低维稠密表示，但在新闻分类这一类别特征明确的任务上，平均词向量相比 BOW 没有优势；
- BERT 的代码实现完整，可随时在具备模型文件的环境中运行；
- 网络环境对预训练模型的获取有显著影响，合理利用镜像与本地缓存是可行的工程手段。

---

## 许可证

本项目仅用于课程作业学习，未附带特定开源许可证。
