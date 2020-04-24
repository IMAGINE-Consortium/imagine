import os
import os.path

tut_dir = '../../tutorials'

for tutorial in os.listdir(tut_dir):
    if 'tutorial' not in tutorial:
        continue
    path = os.path.join(tut_dir, tutorial)
    new = tutorial.replace('ipynb','nblink')
    with open(new, 'w') as f:
        content = '{{\n    "path": "{}"\n}}'.format(path)
        f.write(content)
