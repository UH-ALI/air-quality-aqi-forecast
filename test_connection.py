import os
import sys
import requests
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load your local .env file
load_dotenv()

# Simulate the exact variables GitHub Actions will use
api_key = os.getenv("HOPSWORKS_API_KEY")
project_name = os.getenv("HOPSWORKS_PROJECT", "predicting_aqi")

# Use 'app.hopsworks.ai' since that's what you added to GitHub Secrets/Variables
host = os.getenv("HOPSWORKS_HOST", "app.hopsworks.ai") 

# Build the dynamic URL using your host variable
url = f"https://{host}/hopsworks-api/api/project/getProjectInfo/{project_name}"

headers = {
    "Authorization": f"ApiKey {api_key}"
}

print(f"Secured Ping -> Connecting to Host: {host} | Project: {project_name}...")

try:
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        print("✅ Success! Your GitHub configuration is perfect. The Host, Project, and API Key match.")
    elif response.status_code == 401:
        print("❌ Authentication Failed: Check if your API Key or Hostname is typed wrong.")
    else:
        print(f"❌ Failed. Status Code: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"❌ Local Network Error: {e}")