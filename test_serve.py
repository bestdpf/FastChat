import requests

url = 'http://127.0.0.1:9999/get_worker_address'
myobj = {'model': 'vicuna-wizard-7b'}

x = requests.post(url, json=myobj)

print(x.text)
