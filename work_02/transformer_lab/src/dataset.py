"""
dataset.py — 数据集加载与字符级分词器

本模块负责：
1. 下载并加载 Tiny Shakespeare 数据集
2. 实现字符级分词器（CharTokenizer）
3. 构建 PyTorch Dataset 用于训练和验证

Tiny Shakespeare 数据集简介：
    - 来源：Andrej Karpathy 的 char-rnn 项目
    - 内容：约 1MB 的莎士比亚戏剧全文
    - 大小：约 1,115,394 个字符
    - 特点：纯英文文本，包含约 65 个不同字符
    - 用途：非常适合字符级语言模型的教学和实验

为什么选择 Tiny Shakespeare：
    1. 数据量适中：足够训练出有意义的模型，但不需要 GPU
    2. 文本有规律：莎士比亚的戏剧有明显的格式和风格
    3. 结果直观：可以直接阅读生成的文本来判断模型质量
    4. 广泛使用：是 NLP 教学中最经典的数据集之一

任务设计：
    本项目将文本切分为固定长度的片段，每两个相邻片段构成一个（源, 目标）对。
    即：给定一段文本，预测紧接着的下一段文本。
"""

import os
import json
import torch
from torch.utils.data import Dataset


# ============================================================
# 特殊 Token 定义
# ============================================================
PAD_TOKEN = "<pad>"     # 填充 token，用于对齐不等长的序列
BOS_TOKEN = "<bos>"     # 序列开始标记 (Beginning of Sequence)
EOS_TOKEN = "<eos>"     # 序列结束标记 (End of Sequence)

# 特殊 token 对应的 ID
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


class CharTokenizer:
    """
    字符级分词器（Character-Level Tokenizer）。

    功能：
        将文本中的每个字符映射为一个整数 ID，以及反向映射。

    词表结构：
        ID 0: <pad> — 填充 token
        ID 1: <bos> — 序列开始
        ID 2: <eos> — 序列结束
        ID 3+: 数据中出现的实际字符（按排序后的顺序）

    使用示例：
        >>> tokenizer = CharTokenizer()
        >>> tokenizer.build_vocab("Hello World!")
        >>> ids = tokenizer.encode("Hello")
        >>> text = tokenizer.decode(ids)
    """

    def __init__(self):
        """初始化分词器，设置特殊 token。"""
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0

    def build_vocab(self, text):
        """
        从文本中构建字符词表。

        参数:
            text (str): 用于构建词表的完整文本。

        说明:
            词表按以下顺序构建：
            1. 先添加特殊 token（PAD, BOS, EOS）
            2. 再按字母顺序添加文本中出现的所有唯一字符
        """
        # 获取文本中所有唯一字符并排序
        chars = sorted(set(text))

        # 构建映射表
        self.char_to_id = {
            PAD_TOKEN: PAD_ID,
            BOS_TOKEN: BOS_ID,
            EOS_TOKEN: EOS_ID,
        }

        # 从 ID 3 开始分配真实字符的 ID
        for i, char in enumerate(chars):
            self.char_to_id[char] = i + 3

        # 反向映射
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}

        self.vocab_size = len(self.char_to_id)

    def encode(self, text):
        """
        将文本编码为 token ID 列表。

        参数:
            text (str): 要编码的文本。

        返回:
            list[int]: token ID 列表。
        """
        return [self.char_to_id.get(ch, PAD_ID) for ch in text]

    def decode(self, ids):
        """
        将 token ID 列表解码为文本。

        参数:
            ids (list[int]): token ID 列表。

        返回:
            str: 解码后的文本（过滤掉特殊 token）。
        """
        chars = []
        for id in ids:
            token = self.id_to_char.get(id, "")
            # 跳过特殊 token
            if token in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN):
                continue
            chars.append(token)
        return "".join(chars)

    def save(self, path):
        """
        保存词表到 JSON 文件。

        参数:
            path (str): 保存路径。
        """
        data = {
            "char_to_id": self.char_to_id,
            "vocab_size": self.vocab_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
        """
        从 JSON 文件加载词表。

        参数:
            path (str): 词表文件路径。
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.char_to_id = data["char_to_id"]
        self.id_to_char = {int(v): k for k, v in self.char_to_id.items()}
        self.vocab_size = data["vocab_size"]


class ShakespeareDataset(Dataset):
    """
    Tiny Shakespeare 数据集。

    将文本切分为固定长度的片段，每两个相邻片段构成一个（源, 目标）对：
        源序列 = text[i * seq_len : (i+1) * seq_len]
        目标序列 = text[(i+1) * seq_len : (i+2) * seq_len]

    即：给定一段文本，预测紧接着的下一段文本。

    数据格式：
        每个样本返回 (src, tgt)，其中：
        - src: [seq_len] 的 token ID 张量（源序列）
        - tgt: [seq_len] 的 token ID 张量（目标序列）
    """

    def __init__(self, text, tokenizer, seq_len=64):
        """
        初始化数据集。

        参数:
            text (str): 原始文本数据。
            tokenizer (CharTokenizer): 字符分词器。
            seq_len (int): 每个片段的长度，默认 64。
        """
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        # 将整个文本编码为 token ID
        self.data = tokenizer.encode(text)

        # 计算可用的样本数量
        # 每个样本需要 2 * seq_len 个连续 token（源 + 目标）
        self.num_samples = (len(self.data) - seq_len) // seq_len

        if self.num_samples <= 0:
            raise ValueError(
                f"文本太短！需要至少 {2 * seq_len} 个字符，"
                f"但只有 {len(self.data)} 个。"
            )

    def __len__(self):
        """返回数据集中的样本数量。"""
        return self.num_samples

    def __getitem__(self, idx):
        """
        获取第 idx 个样本。

        参数:
            idx (int): 样本索引。

        返回:
            tuple: (src, tgt)
                - src: [seq_len] 的 LongTensor
                - tgt: [seq_len] 的 LongTensor
        """
        # 计算起始位置
        start = idx * self.seq_len

        # 源序列：text[start : start + seq_len]
        src = self.data[start: start + self.seq_len]

        # 目标序列：text[start + seq_len : start + 2 * seq_len]
        # 注意：目标序列是源序列的紧接着的下一段
        tgt = self.data[start + self.seq_len: start + 2 * self.seq_len]

        # 如果目标序列不够长（到达文本末尾），用 PAD 填充
        if len(tgt) < self.seq_len:
            tgt = tgt + [PAD_ID] * (self.seq_len - len(tgt))

        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def load_data(data_dir, seq_len=64, train_ratio=0.9):
    """
    加载并预处理 Tiny Shakespeare 数据集。

    参数:
        data_dir (str): 数据目录路径（应包含 input.txt 文件）。
        seq_len (int): 序列长度，默认 64。
        train_ratio (float): 训练集占比，默认 0.9。

    返回:
        tuple: (train_dataset, val_dataset, tokenizer)
            - train_dataset: 训练集 ShakespeareDataset
            - val_dataset: 验证集 ShakespeareDataset
            - tokenizer: CharTokenizer 实例
    """
    # 读取数据文件
    data_path = os.path.join(data_dir, "input.txt")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"未找到数据文件: {data_path}\n"
            f"请先下载 Tiny Shakespeare 数据集：\n"
            f"  方法 1（推荐）：运行 python src/train.py --download\n"
            f"  方法 2（手动）：\n"
            f"    1. 访问 https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt\n"
            f"    2. 保存文件到 {data_path}"
        )

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"数据集统计:")
    print(f"  总字符数: {len(text):,}")
    print(f"  唯一字符数: {len(set(text))}")
    print(f"  前 100 个字符: {repr(text[:100])}")

    # 构建分词器
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(text)
    print(f"  词表大小: {tokenizer.vocab_size}")

    # 保存词表
    vocab_path = os.path.join(data_dir, "vocab.json")
    tokenizer.save(vocab_path)
    print(f"  词表已保存到: {vocab_path}")

    # 按比例分割训练集和验证集
    split_idx = int(len(text) * train_ratio)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    print(f"  训练集字符数: {len(train_text):,}")
    print(f"  验证集字符数: {len(val_text):,}")

    # 创建数据集
    train_dataset = ShakespeareDataset(train_text, tokenizer, seq_len)
    val_dataset = ShakespeareDataset(val_text, tokenizer, seq_len)

    print(f"  训练集样本数: {len(train_dataset):,}")
    print(f"  验证集样本数: {len(val_dataset):,}")

    return train_dataset, val_dataset, tokenizer


def download_data(data_dir):
    """
    下载 Tiny Shakespeare 数据集。

    参数:
        data_dir (str): 保存数据的目录路径。
    """
    import urllib.request

    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, "input.txt")

    if os.path.exists(data_path):
        print(f"数据文件已存在: {data_path}")
        return

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f"正在下载 Tiny Shakespeare 数据集...")
    print(f"  URL: {url}")
    print(f"  保存到: {data_path}")

    try:
        urllib.request.urlretrieve(url, data_path)
        # 验证下载
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"  下载完成！文件大小: {len(text):,} 字符")
    except Exception as e:
        print(f"  下载失败: {e}")
        print(f"  请手动下载并保存到: {data_path}")
        raise


if __name__ == "__main__":
    """模块自测：验证数据加载和分词器功能"""
    print("=" * 50)
    print("测试数据集模块")
    print("=" * 50)

    # 测试分词器
    tokenizer = CharTokenizer()
    test_text = "Hello, World! This is a test."
    tokenizer.build_vocab(test_text)

    print(f"测试文本: {repr(test_text)}")
    print(f"词表大小: {tokenizer.vocab_size}")

    encoded = tokenizer.encode("Hello")
    decoded = tokenizer.decode(encoded)
    print(f"编码 'Hello': {encoded}")
    print(f"解码回文本: {repr(decoded)}")
    assert decoded == "Hello", f"解码失败！预期 'Hello'，得到 '{decoded}'"
    print("✅ 分词器编码/解码测试通过！")

    # 测试数据集
    dataset = ShakespeareDataset(test_text * 100, tokenizer, seq_len=8)
    src, tgt = dataset[0]
    print(f"\n数据集样本:")
    print(f"  源序列形状: {src.shape}  (预期: [8])")
    print(f"  目标序列形状: {tgt.shape}  (预期: [8])")
    print(f"  源序列内容: {tokenizer.decode(src.tolist())}")
    print(f"  目标序列内容: {tokenizer.decode(tgt.tolist())}")
    assert src.shape == (8,) and tgt.shape == (8,), "形状不匹配！"
    print("✅ ShakespeareDataset 测试通过！")

    print("✅ 数据集模块全部测试通过！")
