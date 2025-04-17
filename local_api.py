import requests

# TODO: send a GET using the URL http://127.0.0.1:8000
url = "http://127.0.0.1:8000"
r = requests.get(url)

# Print Status messages amd welcome message
print(f"GET / -> Status: {r.status_code}")
print(f"Message: {r.json()['message']}")

response = requests.get(url)
# Print Status messages amd welcome message
# Post Data Setup
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}
# Send Post Request.
post_url = f"{url}/data/"
r = requests.post(post_url, json=data)

# Print Status messages and result.
print(f"\nPOST /data/ -> Status: {r.status_code}")
print(f"Result: {r.json()['result']}")