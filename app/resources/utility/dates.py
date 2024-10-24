from datetime import datetime, timedelta


def generate_list_of_dates():
    # Start date (January 1st)
    start_date = datetime(2024, 1, 1)

    # End date (December 31st)
    end_date = datetime(2024, 12, 31)

    # List to hold the formatted date strings
    date_list = []

    # Iterate through the range of dates
    current_date = start_date
    while current_date <= end_date:
        # Format the date without the year and add to list (e.g., "Jan. 24")
        date_str = current_date.strftime("%b %d")
        date_list.append(date_str)

        # Move to the next day
        current_date += timedelta(days=1)

    return date_list
