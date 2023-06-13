"""
	
PYTHON TO DATABASE
	
"""

# Import necessary libraries
import datetime, time
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Establish connection with Google Sheets 
scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('/home/pi/project/codes/project_to_database.json', scope) 
client = gspread.authorize(creds)
sheet = client.open("project_database").sheet1

# Send data 
time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
print(time)
data = 10000
values =[time,data]
sheet.append_row(values)

"""

Bibliography: 
    https://towardsdatascience.com/turn-google-sheets-into-your-own-database-with-python-4aa0b4360ce7 
	
"""