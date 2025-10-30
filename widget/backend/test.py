# test_api.py
import requests

r = requests.post(
    'http://localhost:5000/api/chat',
    json={'message': 'What is fever?'},
    proxies={'http': None, 'https': None}
)

print('Status:', r.status_code)
print('Content:', r.text[:200])

print('Response:', r.json())
