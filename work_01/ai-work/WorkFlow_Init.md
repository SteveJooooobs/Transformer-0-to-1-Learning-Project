# Transformer 学习项目：环境初始化与依赖验证工作流

## 1. 初始化基本信息
- **初始化日期**：2026年5月22日
- **执行状态**：已就绪 (Completed & Verified)
- **环境路径**：`f:/PyProject/Transformer_demo`
- **Python 版本**：3.14.2
- **虚拟环境目录**：`.venv` (已更换为隐藏目录风格)

---

## 2. 虚拟环境命名习惯解析：`venv` vs `.venv`
在 Python 项目中，虚拟环境的命名主要有无点前缀的 `venv` 和带点前缀的 `.venv` 两种风格。

### 主要差异：
1. **隐藏属性 (Unix/Linux 传统)**：
   - 以 `.` 开头的文件夹（如 `.venv`）在 Unix/Linux/macOS 系统中默认被识别为“隐藏文件夹”。标准的 `ls` 命令不会直接显示它们，而 `venv` 文件夹是显式可见的。
2. **IDE 集成**：
   - 现代编辑器（如 VS Code、PyCharm）更推荐并偏好 `.venv` 风格。因为 IDE 会自动将带点前缀的配置/依赖目录进行折叠、置灰或隐藏，这可以让项目根目录文件结构更加干净，不被大量的外部依赖包文件夹干扰。
3. **Windows 表现**：
   - 在 Windows 系统中，带点前缀的文件夹不会默认变为系统级隐藏（除非手动设置），但大多数 IDE 和开发工具链依然遵循该统一的“隐藏/配置”命名标准。
4. **功能差异**：
   - **两者在功能上绝对没有任何区别**。虚拟环境的结构、Python 解释器加载逻辑、依赖库的隔离方式都是一模一样的。

---

## 3. 环境复现指南
如果您在一个全新的空项目中，可以直接复制并运行以下命令，即可完整复现当前的环境配置：

### 第一步：创建隐藏式虚拟环境
在项目根目录下创建 Python 虚拟环境 `.venv`：
```powershell
python -m venv .venv
```

### 第二步：升级 pip
升级虚拟环境中的 `pip` 到最新版本：
```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### 第三步：安装核心依赖包
一键安装项目所需的核心深度学习库 and 数据处理库（使用本地缓存或在线下载）：
```powershell
.\.venv\Scripts\python.exe -m pip install torch transformers datasets tokenizers numpy
```

### 第四步：运行依赖包验证
在项目根目录下运行 `test_imports.py` 脚本，以验证安装结果（测试脚本内容见下文）：
```powershell
.\.venv\Scripts\python.exe test_imports.py
```

---

## 4. 已安装核心依赖包及验证版本
经过 `test_imports.py` 脚本验证，以下依赖包已成功安装且能够正常导入：

| 依赖包名称 | 验证状态 | 安装版本 | 备注 |
| :--- | :--- | :--- | :--- |
| **torch** | 成功 | `2.12.0+cpu` | PyTorch 深度学习框架 (CPU 版本) |
| **transformers** | 成功 | `5.9.0` | Hugging Face Transformers 模型库 |
| **datasets** | 成功 | `4.8.5` | Hugging Face Datasets 数据集库 |
| **tokenizers** | 成功 | `0.22.2` | Hugging Face Tokenizers 分词器库 |
| **numpy** | 成功 | `2.4.6` | 基础数学与多维阵列计算库 |

### 验证输出日志：
```text
开始测试依赖包导入...
========================================
【成功】torch 导入成功，版本: 2.12.0+cpu
【成功】transformers 导入成功，版本: 5.9.0
【成功】datasets 导入成功，版本: 4.8.5
【成功】tokenizers 导入成功，版本: 0.22.2
【成功】numpy 导入成功，版本: 2.4.6
========================================
所有核心依赖包导入测试通过！
```

---

## 5. 验证脚本内容 (`test_imports.py`)
在项目根目录下建立的 `test_imports.py` 验证脚本内容如下：
```python
# -*- coding: utf-8 -*-
"""
测试 Transformer 项目核心依赖库的导入 and 版本信息
"""
import sys

def test_imports():
    packages = ['torch', 'transformers', 'datasets', 'tokenizers', 'numpy']
    all_success = True
    
    print("开始测试依赖包导入...\n" + "="*40)
    
    for pkg in packages:
        try:
            # 动态导入包
            module = __import__(pkg)
            
            # 获取版本号
            version = "未知"
            if hasattr(module, '__version__'):
                version = module.__version__
                
            print(f"【成功】{pkg} 导入成功，版本: {version}")
        except ImportError as e:
            print(f"【失败】无法导入 {pkg}。错误信息: {e}")
            all_success = False
            
    print("="*40)
    if all_success:
        print("所有核心依赖包导入测试通过！")
        sys.exit(0)
    else:
        print("部分依赖包导入失败，请检查环境配置。")
        sys.exit(1)

if __name__ == "__main__":
    test_imports()
```

---

## 6. 结语
项目开发工作区目前已准备就绪。虚拟环境已规范化为更契合现代 IDE 折叠/隐藏约定的 `.venv`。所有核心包已经成功预装并通过导入验证，可以直接开始 Transformer 架构的相关学习与代码编写任务。
