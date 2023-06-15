import re

# 原始的doc.tags字典
tags = {'no_humans': 0.9352974891662598, 'outdoors': 0.8261902332305908, 'grass': 0.7905033826828003, 'animal': 0.6697984337806702, 'animal_focus': 0.5833525061607361, 'day': 0.5270175337791443, 'scenery': 0.5263932943344116, 'baton_(conducting)': 0.55555}

# 将所有键名中的下划线替换为空格，并删除括号及其内部内容，并删除空格
new_tags = {re.sub(r'\(.*?\)', '', key.replace('_', ' ')).strip(): value for key, value in tags.items()}

# 打印输出新的tags字典
print(new_tags)