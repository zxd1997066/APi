import requests
import json

with open('Ipex Demo.json', 'r') as f:
    j = json.load(f)
response = requests.post(url='http://mlpc.intel.com/api/report_upload/ipex_cpu/common', json=j)
print(response.status_code)
print(response.text)
