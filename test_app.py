# import requests

# # Define the URL for the forecast-univariate route
# url = "http://127.0.0.1:5000/forecast-univariate"

# # Define the file path to the CSV file you want to upload
# file_path = "test\data\candy_production.csv"

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


import requests

# Define the URL for the forecast-univariate route
url = "http://127.0.0.1:5000/forecast-multivariate"

# Define the file path to the CSV file you want to upload
file_path = r"test\data\apple2.csv"

# Define other form data (based on what the endpoint expects)
form_data = {
    "type": "multivariate",  # Time series type
    "freq": "D",  # Frequency (e.g., daily "D", weekly "W", etc.)
    "description": "Test forecast",  # Description of the forecast
    "steps": "10",  # Number of forecast steps
    "method": "without_refit",  # Forecast method (e.g., "without_refit")
}


# Open the CSV file and send the request
with open(file_path, "rb") as f:
    files = {"inputFile": f}
    response = requests.post(url, data=form_data, files=files)

# Print the response from the server
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
