"""合成インスタンスの（再）生成: python make_instances.py [problem ...]
（省略時はインスタンス生成関数を持つ全問題. 同じ seed で同じものが出る）"""
import sys

import registry

if __name__ == "__main__":
    names = sys.argv[1:] or [p.name for p in registry.problems() if p.generate]
    for name in names:
        p = registry.get_problem(name)
        if p.generate is None:
            sys.exit(f"{name} はインスタンス生成関数を持ちません")
        p.generate()
