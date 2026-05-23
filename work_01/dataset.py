# -*- coding: utf-8 -*-
"""
数据处理模块：dataset.py
本模块用于：
1. 随机生成包含多种不同自然语言/数字格式的日期数据集（如 "May 23, 2026"、"23/05/2026"、"2026-05-23" 等）。
2. 实现一个基于字符级别的轻量级分词器 CharacterTokenizer，包含词表构建、编码(encode)与解码(decode)功能。
3. 实现 PyTorch 的 Dataset 封装与 DataLoader 构建函数，方便模型进行批处理训练。
"""

import random
from datetime import datetime, timedelta
import torch
from torch.utils.data import Dataset, DataLoader

# 定义特殊字符标记
PAD_TOKEN = "<pad>"  # 填充字符，用于对齐批处理中的序列长度
SOS_TOKEN = "<sos>"  # 序列开始字符，用于指示解码器开始生成
EOS_TOKEN = "<eos>"  # 序列结束字符，用于指示生成结束
UNK_TOKEN = "<unk>"  # 未知字符，处理未见过的输入

class CharacterTokenizer:
    """
    字符级分词器 (Character-level Tokenizer)。
    将输入的字符串拆分为单个字符，并将其映射为唯一的整数 ID。
    """
    
    def __init__(self):
        """
        初始化分词器，设置特殊字符的索引，并初始化词表映射。
        """
        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        
        # 初始词表包含四个特殊字符
        self.id2char = {
            self.pad_id: PAD_TOKEN,
            self.sos_id: SOS_TOKEN,
            self.eos_id: EOS_TOKEN,
            self.unk_id: UNK_TOKEN
        }
        self.char2id = {char: idx for idx, char in self.id2char.items()}
        
    def build_vocab(self, texts):
        """
        根据给定的文本列表构建词表。
        
        Args:
            texts (list of str): 用于构建词表的文本列表。
        """
        for text in texts:
            for char in text:
                if char not in self.char2id:
                    new_id = len(self.char2id)
                    self.char2id[char] = new_id
                    self.id2char[new_id] = char
                    
    @property
    def vocab_size(self):
        """
        获取词表的大小。
        
        Returns:
            int: 词表中所有 Token 的数量。
        """
        return len(self.char2id)
        
    def encode(self, text, max_len=None, add_special=True):
        """
        将文本字符串转换为对应的 Token ID 列表。
        
        Args:
            text (str): 待编码的字符串。
            max_len (int, optional): 最大截断/填充长度。默认为 None（不限制）。
            add_special (bool): 是否在序列两端加上 <sos> 和 <eos>。默认为 True。
            
        Returns:
            list of int: 转换后的 Token ID 列表。
        """
        # 将文本转化为字符列表
        ids = [self.char2id.get(char, self.unk_id) for char in text]
        
        # 如果需要，添加特殊标记
        if add_special:
            ids = [self.sos_id] + ids + [self.eos_id]
            
        # 如果指定了最大长度，进行截断或用 <pad> 填充
        if max_len is not None:
            if len(ids) > max_len:
                # 截断时确保保留结束标记 <eos>（如果包含特殊标记的话）
                if add_special:
                    ids = ids[:max_len-1] + [self.eos_id]
                else:
                    ids = ids[:max_len]
            else:
                ids = ids + [self.pad_id] * (max_len - len(ids))
                
        return ids

    def decode(self, ids):
        """
        将 Token ID 列表还原为文本字符串，忽略填充和开始/结束特殊字符。
        
        Args:
            ids (list of int): Token ID 列表。
            
        Returns:
            str: 还原后的字符串。
        """
        chars = []
        for idx in ids:
            # 转换为常规 Python 整数
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            # 如果遇到结束标记或填充标记，则终止解码
            if idx == self.eos_id or idx == self.pad_id:
                break
            # 忽略开始标记和未知标记（或者保留）
            if idx != self.sos_id:
                chars.append(self.id2char.get(idx, UNK_TOKEN))
        return "".join(chars)


def generate_date_dataset(num_samples=10000):
    """
    随机生成日期对数据集。
    输入为多样化格式的日期字符串，输出为标准 ISO 格式 "YYYY-MM-DD" 的日期字符串。
    
    Args:
        num_samples (int): 生成的样本数量。
        
    Returns:
        tuple of (list, list): (输入日期字符串列表, 目标日期字符串列表)
    """
    # 设定生成的时间范围：从 1970-01-01 到 2035-12-31
    start_date = datetime(1970, 1, 1)
    end_date = datetime(2035, 12, 31)
    delta_days = (end_date - start_date).days
    
    # 预定义多种输入格式
    # %b: 缩写的月份 (如 May), %B: 完整的月份 (如 May), %d: 日, %Y: 四位年份, %y: 两位年份, %m: 两位月份
    input_formats = [
        "%b %d, %Y",       # e.g., "May 23, 2026"
        "%d/%m/%Y",       # e.g., "23/05/2026"
        "%Y-%m-%d",       # e.g., "2026-05-23"
        "%B %d %Y",       # e.g., "May 23 2026"
        "%d-%m-%y",       # e.g., "23-05-26"
        "%d.%m.%Y",       # e.g., "23.05.2026"
        "%b %d %y",       # e.g., "May 23 26"
        "%Y.%m.%d"        # e.g., "2026.05.23"
    ]
    
    src_texts = []
    tgt_texts = []
    
    # 设置随机数种子保证每次生成数据可复现
    random.seed(42)
    
    for _ in range(num_samples):
        # 随机挑选一天
        random_days = random.randint(0, delta_days)
        dt = start_date + timedelta(days=random_days)
        
        # 随机挑选一种输入格式
        fmt = random.choice(input_formats)
        src_str = dt.strftime(fmt)
        
        # 目标格式统一为标准 ISO 格式 (YYYY-MM-DD)
        tgt_str = dt.strftime("%Y-%m-%d")
        
        src_texts.append(src_str)
        tgt_texts.append(tgt_str)
        
    return src_texts, tgt_texts


class DateDataset(Dataset):
    """
    日期数据集封装，用于 PyTorch DataLoader 进行批量迭代。
    """
    
    def __init__(self, src_texts, tgt_texts, tokenizer, max_src_len=30, max_tgt_len=15):
        """
        初始化数据集。
        
        Args:
            src_texts (list of str): 输入的日期文本列表。
            tgt_texts (list of str): 目标的标准格式日期文本列表。
            tokenizer (CharacterTokenizer): 分词器实例。
            max_src_len (int): 输入序列的最大对齐长度。
            max_tgt_len (int): 目标序列的最大对齐长度。
        """
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        
    def __len__(self):
        """
        获取数据集的大小。
        
        Returns:
            int: 样本总数。
        """
        return len(self.src_texts)
        
    def __getitem__(self, idx):
        """
        获取单条编码后的张量数据。
        
        Args:
            idx (int): 数据索引。
            
        Returns:
            dict of Tensor: 包含编码后的 src、tgt 序列张量。
        """
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        
        # 编码为 ID
        src_ids = self.tokenizer.encode(src_text, max_len=self.max_src_len, add_special=True)
        # 目标序列在训练时，输入给 Decoder 的需要包含 <sos>，作为预测监督的需要包含 <eos>
        tgt_ids = self.tokenizer.encode(tgt_text, max_len=self.max_tgt_len, add_special=True)
        
        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long),
            "src_raw": src_text,
            "tgt_raw": tgt_text
        }


def get_dataloader(num_samples=10000, batch_size=64, val_ratio=0.2, max_src_len=30, max_tgt_len=15):
    """
    生成数据集并构建训练和验证数据加载器 (DataLoader)。
    
    Args:
        num_samples (int): 随机生成的样本数量。
        batch_size (int): 批大小。
        val_ratio (float): 验证集比例（0.0 ~ 1.0）。
        max_src_len (int): 输入序列最大对齐长度。
        max_tgt_len (int): 输出序列最大对齐长度。
        
    Returns:
        tuple: (train_loader, val_loader, tokenizer)
    """
    # 1. 生成平行文本
    src_texts, tgt_texts = generate_date_dataset(num_samples)
    
    # 2. 构建分词器
    tokenizer = CharacterTokenizer()
    # 将所有的输入文本和目标文本合并构建词表
    tokenizer.build_vocab(src_texts + tgt_texts)
    
    # 3. 划分训练集和验证集
    split_idx = int(num_samples * (1 - val_ratio))
    
    train_src, val_src = src_texts[:split_idx], src_texts[split_idx:]
    train_tgt, val_tgt = tgt_texts[:split_idx], tgt_texts[split_idx:]
    
    # 4. 构建 PyTorch Dataset
    train_dataset = DateDataset(train_src, train_tgt, tokenizer, max_src_len, max_tgt_len)
    val_dataset = DateDataset(val_src, val_tgt, tokenizer, max_src_len, max_tgt_len)
    
    # 5. 构建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, tokenizer


if __name__ == "__main__":
    # 本模块的简易功能性测试
    print("开始测试 dataset.py 模块...")
    src, tgt = generate_date_dataset(5)
    print("生成的样例数据:")
    for s, t in zip(src, tgt):
        print(f"  输入: {s:<20} -> 输出: {t}")
        
    train_loader, val_loader, tokenizer = get_dataloader(num_samples=100, batch_size=4)
    print(f"分词器词表大小: {tokenizer.vocab_size}")
    
    # 取一个 Batch 看看
    batch = next(iter(train_loader))
    print(f"Batch 中的 src 形状: {batch['src'].shape}")
    print(f"Batch 中的 tgt 形状: {batch['tgt'].shape}")
    print(f"首条 src 编码后: {batch['src'][0]}")
    print(f"首条 src 解码还原: '{tokenizer.decode(batch['src'][0])}' (原句: '{batch['src_raw'][0]}')")
    print(f"首条 tgt 解码还原: '{tokenizer.decode(batch['tgt'][0])}' (原句: '{batch['tgt_raw'][0]}')")
    print("数据处理模块测试通过！")
