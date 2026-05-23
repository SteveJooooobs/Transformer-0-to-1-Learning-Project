# -*- coding: utf-8 -*-
"""
测试 Transformer 项目核心依赖库的导入和版本信息
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
