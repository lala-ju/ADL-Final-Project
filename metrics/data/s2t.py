from opencc import OpenCC
import sys

cc = OpenCC('s2tw')
s = ""
with open(sys.argv[1], 'r') as f:
    s = f.read()

with open(sys.argv[1], 'w') as f:
    f.write(cc.convert(s))
