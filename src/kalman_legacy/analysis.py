import os
import pandas as pd
import matplotlib.pyplot as plt

# Get a list of all files in the folder
folder_path = './'  # Replace with the actual folder path
files = os.listdir(folder_path)

# Initialize empty lists to store the error and measurement error data
error_data = []
measurement_error_data = []

# Iterate over the files and extract the error and measurement error data
for file in files:
    if file.endswith('.csv') and file.startswith('log_file'):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)

        # Append the 'Error' and 'MeasurementError' data to the respective lists
        error_data.extend(df['Error'])
        measurement_error_data.extend(df['MeasurementError'])

# Create a single figure with subplots for 'Error' and 'MeasurementError' boxplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Plot 'Error' boxplot
ax1.boxplot(error_data)
ax1.set_title('Error Boxplot')
ax1.set_xlabel('Error')
ax1.set_ylabel('Value')

# Plot 'MeasurementError' boxplot
ax2.boxplot(measurement_error_data)
ax2.set_title('MeasurementError Boxplot')
ax2.set_xlabel('MeasurementError')

plt.tight_layout()
plt.show()
