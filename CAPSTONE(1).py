#!/usr/bin/env python
# coding: utf-8

# In[13]:


import re
import pandas as pd
import os

# Directory containing your .nmfs log files
log_directory = r"C:\Users\HP\Desktop\Capstone\Data_Capstone"
output_csv = r"C:\Users\HP\Desktop\Capstone\Data_Capstone\Combine53.CSV"  # Output CSV file path

# Initialize lists to store log information
log_info = []

# Function to calculate file size in KB
def get_file_size(file_path):
    return os.path.getsize(file_path) / 1024  # Convert bytes to KB

# Iterate through all .nmfs files in the specified directory
for filename in os.listdir(log_directory):
    if filename.endswith(".nmfs"):
        file_path = os.path.join(log_directory, filename)

        # Determine label based on the file suffix and file size
        suffix = filename[-7:]  # Assuming the suffix is the last 8 characters
        file_size_kb = get_file_size(file_path)

        if suffix == ".1.nmfs" and file_size_kb < 1000:
            label = "MOBILE_ORIGINATING"
        elif suffix == ".2.nmfs" and file_size_kb < 1000:
            label = "MOBILE_TERMINATING"
        elif suffix == ".1.nmfs" and file_size_kb > 1000:
            label = "4G_DATA"
        elif suffix == ".2.nmfs" and file_size_kb > 1000:
            label = "3G_DATA"
        else:
            label = "UNKNOWN"  # Handle any other cases

        # Extract date and time from the log name (adjust this part based on your log naming convention)
        log_name = os.path.splitext(filename)[0]  # Remove file extension
        log_date = log_name[0:7]  # Assuming the date is the first 6 characters
        log_time = log_name[8:12]  # Assuming the time is the next 6 characters
        Slice=log_name[15:16]

        log_info.append([filename, label, log_date, log_time, file_size_kb, Slice])

# Create a DataFrame from the log information
log_df = pd.DataFrame(log_info, columns=['Log Name', 'Label', 'Date', 'Time', 'Size_Of_Log_KB', 'Slice'])

# Export the log information to a CSV file
log_df.to_csv(output_csv, index=False)

print("Log information exported to CSV file:", output_csv)


# In[5]:


get_ipython().system('pip install pandas')


# In[14]:


import pandas as pd

file_path = r'C:\Users\HP\Desktop\Capstone\Data_Capstone\Combine53.CSV'
df = pd.read_csv(file_path)
df


# In[15]:


df.head()


# In[16]:


df.Size_Of_Log_KB


# In[44]:


df.Size_Of_Log_KB

average_size_of_log_kb = df.Size_Of_Log_KB.mean()
print( "Average Value of Size of log in KB =", average_size_of_log_kb)


# In[45]:


maximum_size_of_Log_kb=df.Size_Of_Log_KB.max()
print( "Max Value of Size of log in KB =", maximum_size_of_Log_kb)


# In[46]:


Minimum_size_of_Log_kb=df.Size_Of_Log_KB.min()
print( "Minimun Value of Size of log in KB =", Minimum_size_of_Log_kb)


# In[47]:


Total_size_of_Log_kb=df.Size_Of_Log_KB.sum()
print( "Total Size of log in KB=", Total_size_of_Log_kb)


# In[57]:


# Assuming you have a DataFrame 'df' with a 'Log Name' column
# Replace 'df' with your actual DataFrame name if different

# Use the .str slice notation to extract the desired slice
df['Slice'] = df['Log Name'].str[15:16]

# Now, 'Slice' contains the sliced values from 'Log Name'
df.Slice



# In[ ]:





# In[55]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Assuming you have already loaded your data into a Pandas DataFrame called 'df'
# 'Label', 'size of log', and 'Slice' are columns in your DataFrame

# Define your features (X) and target (y)
X = df[['Size_Of_Log_KB', 'Slice']]
y = df['Label']

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a classification model (Random Forest classifier in this example)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)


# In[ ]:




