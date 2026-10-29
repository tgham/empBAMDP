import requests

url = "http://127.0.0.1:8000/"

routes = ["complete/", "incomplete/", "finalise/"]

data = {
    "hi": "there",
    "some": "data"
}

for route in routes:
    full_route = f"{url}{route}"
    x = requests.post(full_route, json=data)
    print(x.text)
