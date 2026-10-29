# So khớp mũi tên trong file .drawio (page-2) với cột `from` của file yaml.
# Chạy:  python diagrams/check_drawio.py models/TPH-Yolov5-PFFN-neck.yaml diagrams/TPH-Yolov5-PFFN-neck.drawio
import re
import sys

import yaml

sys.stdout.reconfigure(encoding='utf-8')  # terminal Windows mặc định cp1252, không in được tiếng Việt
cfg, drawio = sys.argv[1], sys.argv[2]

# 1) yaml: layer i nhận từ những layer nào
d = yaml.safe_load(open(cfg, encoding='utf-8'))
rows = d['backbone'] + d['head']
want = {}
for i, (f, n, m, args) in enumerate(rows):
    f = f if isinstance(f, list) else [f]
    want[i] = [] if i == 0 else sorted(i + x if x < 0 else x for x in f)

# 2) drawio (page-2): mũi tên nối ô "L<a>" -> ô "L<b>"
xml = open(drawio, encoding='utf-8').read()
page2 = re.search(r'<diagram id="pffn".*?</diagram>', xml, re.S).group(0)
got = {i: [] for i in want}
for a, b in re.findall(r'source="L(\d+)" target="L(\d+)[a-d]?"', page2):  # L53a..d = 4 head của Detect
    got[int(b)].append(int(a))

# 3) so sánh từng layer
bad = 0
for i in want:
    ok = sorted(got[i]) == want[i]
    bad += not ok
    print(f'L{i:<3} {rows[i][2]:<12} yaml from={want[i]!s:<10} hình={sorted(got[i])!s:<10} {"OK" if ok else "<-- SAI"}')
print(f'\n{len(want) - bad}/{len(want)} layer khớp')
