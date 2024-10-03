# import requests

# # Define the URL for the forecast-univariate route
# url = "http://127.0.0.1:5000/forecast-univariate"

# # Define the file path to the CSV file you want to upload
# file_path = r"test\data\candy_production.csv"

# # Define other form data (based on what the endpoint expects)
# form_data = {
#     "type": "univariate",  # Time series type
#     "freq": "M",  # Frequency (e.g., daily "D", weekly "W", etc.)
#     "description": "Test forecast",  # Description of the forecast
#     "steps": "10",  # Number of forecast steps
#     "method": "without_refit",  # Forecast method (e.g., "without_refit")
# }

# # Open the CSV file and send the request
# with open(file_path, "rb") as f:
#     files = {"inputFile": f}
#     response = requests.post(url, data=form_data, files=files)

# # Print the response from the server
# print("Status Code:", response.status_code)
# print("Response JSON:", response.json())

# ======================================================================================================================
# import requests

# # Define the URL for the forecast-univariate route
# url = "http://127.0.0.1:5000/trend"

# # Define the file path to the CSV file you want to upload
# file_path = r"test\data\candy_production.csv"

# # Define other form data (based on what the endpoint expects)
# form_data = {
#     "type": "univariate",  # Time series type
#     "freq": "M",  # Frequency (e.g., daily "D", weekly "W", etc.)
#     "description": "Test forecast",  # Description of the forecast
#     "steps": "10",  # Number of forecast steps
#     "method": "without_refit",  # Forecast method (e.g., "without_refit")
# }

# # Open the CSV file and send the request
# with open(file_path, "rb") as f:
#     files = {"inputFile": f}
#     response = requests.post(url, data=form_data, files=files)

# # Print the response from the server
# print("Status Code:", response.status_code)
# print("Response JSON:", response.json())
# ======================================================================================================================


# import requests

# # Define the URL for the forecast-univariate route
# url = "http://127.0.0.1:5000/forecast-multivariate"

# # Define the file path to the CSV file you want to upload
# file_path = r"test\data\apple2.csv"

# # Define other form data (based on what the endpoint expects)
# form_data = {
#     "type": "multivariate",  # Time series type
#     "freq": "D",  # Frequency (e.g., daily "D", weekly "W", etc.)
#     "description": "Test forecast",  # Description of the forecast
#     "steps": "10",  # Number of forecast steps
#     "method": "without_refit",  # Forecast method (e.g., "without_refit")
# }


# # Open the CSV file and send the request
# with open(file_path, "rb") as f:
#     files = {"inputFile": f}
#     response = requests.post(url, data=form_data, files=files)

# # Print the response from the server
# print("Status Code:", response.status_code)
# print("Response JSON:", response.json())
# ======================================================================================================================


# import requests

# # Define the URL for the forecast-univariate route
# url = "http://127.0.0.1:5000/api/trend"

# # Define the file path to the CSV file you want to upload
# file_path = r"test/data/apple2.csv"

# # Define other form data (based on what the endpoint expects)
# form_data = {
#     "type": "multivariate",  # Time series type
#     "freq": "D",  # Frequency (e.g., daily "D", weekly "W", etc.)
#     "description": "Test forecast",  # Description of the forecast
#     "steps": "10",  # Number of forecast steps
#     "method": "without_refit",  # Forecast method (e.g., "without_refit")
# }


# # Open the CSV file and send the request
# with open(file_path, "rb") as f:
#     files = {"inputFile": f}
#     response = requests.post(url, data=form_data, files=files)

# # Print the response from the server
# print("Status Code:", response.status_code)
# print("Response JSON:", response.json())
# ======================================================================================================================
# import requests

# # Define the URL for the forecast-univariate route
# url = "http://127.0.0.1:5000/api/seasonality"

# # Define the file path to the CSV file you want to upload
# file_path = r"test/data/candy_production.csv"

# # Define other form data (based on what the endpoint expects)
# form_data = {
#     "type": "multivariate",  # Time series type
#     "freq": "M",  # Frequency (e.g., daily "D", weekly "W", etc.)
#     "description": "Test forecast",  # Description of the forecast
#     "steps": "10",  # Number of forecast steps
#     "method": "without_refit",  # Forecast method (e.g., "without_refit")
# }


# # Open the CSV file and send the request
# with open(file_path, "rb") as f:
#     files = {"inputFile": f}
#     response = requests.post(url, data=form_data, files=files)

# # Print the response from the server
# print("Status Code:", response.status_code)
# print("Response JSON:", response.json())

# ===================================================

# import requests

# # Define the URL for the forecast-univariate route
# url = "http://127.0.0.1:5000/api/llm"

# # Define other form data (based on what the endpoint expects)
# form_data = {
#     "message": "what is trend in time series?",
#     "about": "trend",
# }


# response = requests.post(url, data=form_data)

# # Print the response from the server
# print("Status Code:", response.status_code)
# print("Response JSON:", response.json())

# ========================================================

# import requests
# import json
# import time

# # Define the URL for the forecast-univariate route
# url = "http://127.0.0.1:5000/api/seasonality"

# # Define the file path to the CSV file you want to upload
# # file_path = r"test\data\temperature_2m_max-2024-09-24.csv"
# file_path = r"test\data\apple2.csv"


# # Define other form data (based on what the endpoint expects)
# form_data = {
#     "type": "multivariate",  # Time series type
#     "freq": "D",  # Frequency (e.g., daily "D", weekly "W", etc.)
#     "description": "Test forecast",  # Description of the forecast
#     "steps": "10",  # Number of forecast steps
#     "method": "without_refit",  # Forecast method (e.g., "without_refit")
# }


# # Open the CSV file and send the request
# with open(file_path, "rb") as f:
#     files = {"inputFile": f}
#     response = requests.post(url, data=form_data, files=files)

# # Print the response from the server
# print("Status Code:", response.status_code)
# print("Response JSON:", response.json())
# response_json = response.json()
# print("Response JSON:", response_json)

# # Store the response JSON into a file
# json_file_path = f"response_new4.json"
# with open(json_file_path, "w") as json_file:
#     json.dump(response_json, json_file, indent=4)
#     print(f"Response saved to {json_file_path}")


# ===============================================================
# import requests

# # Define the URL for the forecast-univariate route
# url = "http://127.0.0.1:5000/api/forecast-multivariate"

# # Define the file path to the CSV file you want to upload
# file_path = r"test/data/apple2.csv"

# # Define other form data (based on what the endpoint expects)
# form_data = {
#     "type": "multivariate",  # Time series type
#     "freq": "D",  # Frequency (e.g., daily "D", weekly "W", etc.)
#     "description": "Test forecast",  # Description of the forecast
#     "steps": "10",  # Number of forecast steps
#     "method": "without_refit",  # Forecast method (e.g., "without_refit")
# }


# # Open the CSV file and send the request
# with open(file_path, "rb") as f:
#     files = {"inputFile": f}
#     response = requests.post(url, data=form_data, files=files)

# # Print the response from the server
# print("Status Code:", response.status_code)
# print("Response JSON:", response.json())


# # ===============================================================
# import requests

# # Define the URL for the forecast-univariate route
# url = "http://127.0.0.1:5000/api/forecast-univariate"

# # Define the file path to the CSV file you want to upload
# file_path = r"test\data\open-meteo-52.55N13.41E38m.csv"

# # Define other form data (based on what the endpoint expects)
# form_data = {
#     "type": "univariate",  # Time series type
#     "freq": "D",  # Frequency (e.g., daily "D", weekly "W", etc.)
#     "description": "Test forecast",  # Description of the forecast
#     "steps": "15",  # Number of forecast steps
#     "method": "without_refit",  # Forecast method (e.g., "without_refit")
# }


# # Open the CSV file and send the request
# with open(file_path, "rb") as f:
#     files = {"inputFile": f}
#     response = requests.post(url, data=form_data, files=files)

# # Print the response from the server
# print("Status Code:", response.status_code)
# print("Response JSON:", response.json())


# # ===============================================================
# import requests

# # Define the URL for the forecast-univariate route
# url = "http://127.0.0.1:5000/api/forecast-multivariate"

# # Define the file path to the CSV file you want to upload
# file_path = r"test\data\open-meteo-52.55N13.41E38m (1).csv"

# # Define other form data (based on what the endpoint expects)
# form_data = {
#     "type": "multivariate",  # Time series type
#     "freq": "D",  # Frequency (e.g., daily "D", weekly "W", etc.)
#     "description": "Test forecast",  # Description of the forecast
#     "steps": "15",  # Number of forecast steps
#     "method": "without_refit",  # Forecast method (e.g., "without_refit")
# }


# # Open the CSV file and send the request
# with open(file_path, "rb") as f:
#     files = {"inputFile": f}
#     response = requests.post(url, data=form_data, files=files)

# # Print the response from the server
# print("Status Code:", response.status_code)
# print("Response JSON:", response.json())

# ===============================================================
import requests

# Define the URL for the forecast-univariate route
url = "http://127.0.0.1:5000/api/trend"

# Define the file path to the CSV file you want to upload
file_path = r"test/data/apple2.csv"

# Define other form data (based on what the endpoint expects)
form_data = {
    "type": "multivariate",  # Time series type
    "freq": "D",  # Frequency (e.g., daily "D", weekly "W", etc.)
    "description": "Test forecast",  # Description of the forecast
    "steps": "15",  # Number of forecast steps
    "method": "without_refit",  # Forecast method (e.g., "without_refit")
}


# Open the CSV file and send the request
with open(file_path, "rb") as f:
    files = {"inputFile": f}
    response = requests.post(url, data=form_data, files=files)

# Print the response from the server
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
