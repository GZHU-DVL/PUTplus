import importlib

# 由于 params.py 和 params_get.py 在同一目录下，直接写文件名即可

cls = getattr(importlib.import_module('test1'), 'Person1')

print(cls)


