import json
import re

path = r'e:\Major_Project\Antigravity\quantum_multiorgan\models\liver\mobilenetv2_liver_final.keras\config.json'
with open(path, 'r', encoding='utf-8') as f:
    data = f.read()

# Remove quantization_config key and its value
data = re.sub(r',\s*"quantization_config"\s*:\s*null', '', data)
data = re.sub(r'"quantization_config"\s*:\s*null\s*,?', '', data)

with open(path, 'w', encoding='utf-8') as f:
    f.write(data)
print('Patched liver config.json')
