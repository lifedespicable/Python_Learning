import os
from pathlib import Path

print(os.path.abspath('.'))
print(os.path.abspath('..'))

print(os.path.exists('D:\work'))

print(os.path.isfile('D:\work'))
print(os.path.isdir('D:\work'))

print(os.path.join('D:\work','PythonLearning'))

p = Path('.')
print(p.resolve())

print(p.is_file())
print(p.is_dir())

p = Path(r'D:\a\b\c\d\e\f\g\h\j\K')
print(Path.mkdir(p,parents=True))



