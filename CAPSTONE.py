#!/usr/bin/env python
# coding: utf-8

# In[17]:


import os
import pandas as pd

# Directory containing your .nmfs log files
log_directory = r"C:\Users\HP\Desktop\ufone roaming"
output_csv = os.path.expanduser("~\combined_logs.csv")


# Initialize a list to store log names and labels
log_info = []

# Regular expressions to identify MO and MT entries based on labels
mo_pattern = r'^Mobile Originating:'
mt_pattern = r'^Mobile Terminating:'

# Iterate through all .nmfs files in the specified directory
for filename in os.listdir(log_directory):
    if filename.endswith(".nmfs"):
        label = "Mobile Originating" if filename.endswith("2.nmfs") else "Mobile Terminating"
        log_info.append((filename, label))

# Create a DataFrame from the log names and labels
log_df = pd.DataFrame(log_info, columns=['Log Name', 'Label'])

# Export the log names and labels to a CSV file
log_df.to_csv(output_csv, index=False)

print("Log names and labels exported to CSV file:", output_csv)


# In[27]:


import os
import re
import pandas as pd

# Directory containing your .nmfs log files
log_directory = r"C:\Users\HP\Desktop\ufone roaming"
output_csv = os.path.join(log_directory, "combined_logs5.csv")

# Initialize a list to store log names, labels, and dates
log_info = []

# Regular expressions to identify MO and MT entries based on labels
mo_pattern = r'^Mobile Originating:'
mt_pattern = r'^Mobile Terminating:'

# Function to read file with multiple encodings
def read_file_with_encodings(file_path, encodings):
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                lines = file.readlines()
            return lines
        except UnicodeDecodeError:
            continue
    return []

# Iterate through all .nmfs files in the specified directory
for filename in os.listdir(log_directory):
    if filename.endswith(".nmfs"):
        file_path = os.path.join(log_directory, filename)
        lines = read_file_with_encodings(file_path, ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252'])

        label = "Mobile Originating" if filename.endswith("2.nmfs") else "Mobile Terminating"

        # Extract date from the filename using a regular expression
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        if date_match:
            date = date_match.group(1)
        else:
            date = "Date Not Found"  # Handle if date is not found in the filename

        current_entry = [filename, label, date]

        log_info.append(current_entry)

# Create a DataFrame from the log names, labels, and dates
log_df = pd.DataFrame(log_info, columns=['Log Name', 'Label', 'Date'])

# Export the log names, labels, and dates to a CSV file
log_df.to_csv(output_csv, index=False)

print("Log names, labels, and dates exported to CSV file:", output_csv)


# In[20]:


import os
import re
import pandas as pd

# Directory containing your .nmfs log files
log_directory = r"C:\Users\HP\Desktop\ufone roaming"
output_csv = os.path.expanduser("~\combined_logs.csv")


# Initialize a list to store log names, labels, and dates
log_info = []

# Regular expressions to identify MO and MT entries based on labels
mo_pattern = r'^Mobile Originating:'
mt_pattern = r'^Mobile Terminating:'

# Function to read file with multiple encodings
def read_file_with_encodings(file_path, encodings):
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                lines = file.readlines()
            return lines
        except UnicodeDecodeError:
            continue
    return []

# Iterate through all .nmfs files in the specified directory
for filename in os.listdir(log_directory):
    if filename.endswith(".nmfs"):
        file_path = os.path.join(log_directory, filename)
        lines = read_file_with_encodings(file_path, ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252'])

        label = "Mobile Originating" if filename.endswith("2.nmfs") else "Mobile Terminating"

        # Extract date from the filename using a regular expression
        date_match = re.search(r'(\d{2}[A-Za-z]{3}\d{2})', filename)
        if date_match:
            date = date_match.group(1)
        else:
            date = "Date Not Found"  # Handle if date is not found in the filename

        current_entry = [filename, label, date]

        log_info.append(current_entry)

# Create a DataFrame from the log names, labels, and dates
log_df = pd.DataFrame(log_info, columns=['Log Name', 'Label', 'Date'])

# Export the log names, labels, and dates to a CSV file
log_df.to_csv(output_csv, index=False)

print("Log names, labels, and dates exported to CSV file:", output_csv)


# In[21]:


import os
import re
import pandas as pd

# Directory containing your .nmfs log files
log_directory = r"C:\Users\HP\Desktop\ufone roaming"
output_csv = os.path.join(log_directory, "combined_logs.csv")

# Initialize a list to store log names, labels, dates, and times
log_info = []

# Regular expressions to identify MO and MT entries based on labels
mo_pattern = r'^Mobile Originating:'
mt_pattern = r'^Mobile Terminating:'

# Function to read file with multiple encodings
def read_file_with_encodings(file_path, encodings):
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                lines = file.readlines()
            return lines
        except UnicodeDecodeError:
            continue
    return []

# Iterate through all .nmfs files in the specified directory
for filename in os.listdir(log_directory):
    if filename.endswith(".nmfs"):
        file_path = os.path.join(log_directory, filename)
        lines = read_file_with_encodings(file_path, ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252'])

        label = "Mobile Originating" if filename.endswith("2.nmfs") else "Mobile Terminating"

        # Extract date and time from the filename using regular expressions
        date_match = re.search(r'(\d{2}[A-Za-z]{3}\d{2})', filename)
        time_match = re.search(r'(\d{6})', filename)
        if date_match:
            date = date_match.group(1)
        else:
            date = "Date Not Found"  # Handle if date is not found in the filename
        if time_match:
            time = time_match.group(1)
        else:
            time = "Time Not Found"  # Handle if time is not found in the filename

        current_entry = [filename, label, date, time]

        log_info.append(current_entry)

# Create a DataFrame from the log names, labels, dates, and times
log_df = pd.DataFrame(log_info, columns=['Log Name', 'Label', 'Date', 'Time'])

# Export the log names, labels, dates, and times to a CSV file
log_df.to_csv(output_csv, index=False)

print("Log names, labels, dates, and times exported to CSV file:", output_csv)


# In[23]:


import os
import re
import pandas as pd

# Directory containing your .nmfs log files
log_directory = r"C:\Users\HP\Desktop\ufone roaming"
output_csv = os.path.join(log_directory, "combined_logs1.csv")

# Initialize a list to store log names, labels, dates, and times
log_info = []

# Regular expressions to identify MO and MT entries based on labels
mo_pattern = r'^Mobile Originating:'
mt_pattern = r'^Mobile Terminating:'

# Function to read file with multiple encodings
def read_file_with_encodings(file_path, encodings):
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                lines = file.readlines()
            return lines
        except UnicodeDecodeError:
            continue
    return []

# Iterate through all .nmfs files in the specified directory
for filename in os.listdir(log_directory):
    if filename.endswith(".nmfs"):
        file_path = os.path.join(log_directory, filename)
        lines = read_file_with_encodings(file_path, ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252'])

        label = "Mobile Originating" if filename.endswith("2.nmfs") else "Mobile Terminating"

        # Extract date and time from the filename using regular expressions
        date_match = re.search(r'(\d{2}[A-Za-z]{3}\d{2})', filename)
        time_match = re.search(r'(\d{6})', filename)
        if date_match:
            date = date_match.group(1)
        else:
            date = "Date Not Found"  # Handle if date is not found in the filename
        if time_match:
            time = time_match.group(1)
        else:
            time = "Time Not Found"  # Handle if time is not found in the filename

        current_entry = [filename, label, date, time]

        log_info.append(current_entry)

# Create a DataFrame from the log names, labels, dates, and times
log_df = pd.DataFrame(log_info, columns=['Log Name', 'Label', 'Date', 'Time'])

# Export the log names, labels, dates, and times to a CSV file
log_df.to_csv(output_csv, index=False)

print("Log names, labels, dates, and times exported to CSV file:", output_csv)


# In[2]:


import re
import pandas as pd
import os

# Directory containing your .nmfs log files
log_directory = r"C:\Users\HP\Desktop\Capstone\Data_Capstone"
output_csv = r"C:\Users\HP\Desktop\Capstone\Data_Capstone\Combine.CSV"  # Output CSV file path

# Initialize lists to store log information
log_info = []

# Regular expressions to identify MO and MT entries based on labels
mo_pattern = r'^Mobile Originating:'
mt_pattern = r'^Mobile Terminating:'

# Function to calculate file size in KB
def get_file_size(file_path):
    return os.path.getsize(file_path) / 1024  # Convert bytes to KB

# Iterate through all .nmfs files in the specified directory
for filename in os.listdir(log_directory):
    if filename.endswith(".nmfs"):
        file_path = os.path.join(log_directory, filename)

        # Determine label based on the filename
        label = "Mobile Originating" if filename.endswith("2.nmfs") else "Mobile Terminating"

        # Calculate the size of the log in KB
        log_size_kb = get_file_size(file_path)

        # Extract date and time from the log name (adjust this part based on your log naming convention)
        log_name = os.path.splitext(filename)[0]  # Remove file extension
        log_date = log_name[0:7]  # Assuming the date is the last 6 characters
        log_time = log_name[8:14]  # Assuming the time is the last 6 characters

        log_info.append([filename, label, log_date, log_time, log_size_kb])

# Create a DataFrame from the log information
log_df = pd.DataFrame(log_info, columns=['Log Name', 'Label', 'Date', 'Time', 'Size_Of_Log (KB)'])

# Export the log information to a CSV file
log_df.to_csv(output_csv, index=False)

print("Log information exported to CSV file:", output_csv)


# In[34]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Assuming you have loaded your data into log_df

# Define your feature columns (excluding 'Log Name' and 'Label')
feature_columns = ['Log Name','Date', 'Time', 'Size_Of_Log (KB)']

# Define your target column
target_column = 'Label'

# Split the data into training and testing sets (80% train, 20% test)
X = log_df[feature_columns]
y = log_df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM model
svm_classifier = SVC(kernel='linear')  # You can try different kernels ('linear', 'rbf', etc.)
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)


# In[39]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load your dataset
data = pd.read_csv(r'C:\Users\HP\Desktop\combined_logs10.csv')


# Separate the target variable (y) and features (X)
X = data[feature_columns]
y = data[target_column]

# Define preprocessing steps for numerical and categorical columns
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder())])

# Identify the categorical and numerical columns
categorical_columns = ['Date', 'Time']  # Update with your categorical feature columns
numeric_columns = ['Size_Of_Log (KB)']  # Update with your numerical feature columns

# Use ColumnTransformer to apply the preprocessing steps to the correct columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Create the SVM model
svm_classifier = SVC(kernel='linear')  # You can choose a different kernel if needed

# Create a pipeline that includes preprocessing and the SVM model
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', svm_classifier)])

from sklearn.preprocessing import OneHotEncoder

# Initialize the OneHotEncoder with handle_unknown='ignore'
encoder = OneHotEncoder(handle_unknown='ignore')

# Fit and transform the encoder on the training data
X_train_encoded = encoder.fit_transform(X_train)

# Transform the test data using the same encoder
X_test_encoded = encoder.transform(X_test)


# Evaluate the model (you can use appropriate metrics based on your classification task)
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test)
report = classification_report(y_test)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)


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




