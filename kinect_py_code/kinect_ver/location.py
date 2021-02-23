
import os
import requests
from dotenv import load_dotenv

load_dotenv(verbose=True)

LOCATION_API_KEY = os.getenv('LOCATION_API_KEY')

url = f'https://www.googleapis.com/geolocation/v1/geolocate?key=AIzaSyBBI9hvhxn9BSa3Zb4dl3OMlBWmivQyNsU'
data = {
    'considerIp': True,
}

result = requests.post(url, data)
data = result.json()
print("data : ",data)

print(data['location']['lat'])
print(data['location']['lng'])