# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 13:18:49 2024

@author: NIYATI
"""
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('C:\\Users\\NIYATI\\OneDrive\\Desktop\\study material\\PROJECT\\end to end project\\Amazon Sale Report.csv')
# Display the first few rows to inspect the data
"""
"""
print("First 5 rows of the dataset:")
print(df.head())

# Display basic information about the dataset, including column names and data types
print("\nDataset Info:")
print(df.info())

# Display summary statistics for numerical columns
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values in each column
print("\nMissing Values:")
print(df.isnull().sum())


df.dropna(subset=['Order ID'], inplace=True)
df['ship-postal-code'].fillna('Unknown', inplace=True)

# Drop columns with a high percentage of missing values (if any)
df.drop(columns=['New', 'PendingS'], inplace=True)  # Example columns with many missing values

# 2. Correct Data Types
# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%y', errors='coerce')

# Convert 'Amount' to float for numerical operations
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

# 3. Remove Duplicates
# Remove duplicate rows based on 'Order ID'
df.drop_duplicates(subset=['Order ID'], keep='first', inplace=True)

# 4. Process Columns
# Standardize the 'Status' column to be in lowercase
df['Status'] = df['Status'].str.lower()

# Handle categorical data by encoding (if needed for analysis or ML models)
df['Fulfilment'] = df['Fulfilment'].astype('category')
df['Category'] = df['Category'].astype('category')

# Display cleaned dataset info and check for any remaining issues
print("Cleaned Dataset Info:")
print(df.info())

print("\nSample of the cleaned data:")
print(df.head())
"""
""" DESCRIPTIVE analysis"""

"""

# Summary statistics for numerical columns
print("Summary statistics:")
print(df.describe())

# Summary statistics for categorical columns
print("\nValue counts for 'Status':")
print(df['Status'].value_counts())

print("\nValue counts for 'Category':")
print(df['Category'].value_counts())

import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of 'Amount'
plt.figure(figsize=(8, 5))
sns.histplot(df['Amount'], bins=30, kde=True)
plt.title('Distribution of Sales Amount')
plt.xlabel('Sales Amount (INR)')
plt.ylabel('Frequency')
plt.show()


# Count plot for 'Status'
plt.figure(figsize=(10, 8))
sns.countplot(data=df, y='Status', palette='pastel')  # Changed to horizontal

# Title and labels
plt.title('Order Status Distribution', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Order Status', fontsize=14)

# Adding data labels next to bars
for p in plt.gca().patches:
    plt.gca().annotate(f'{int(p.get_width())}', 
                       (p.get_width(), p.get_y() + p.get_height() / 2.),
                       ha='left', va='center', fontsize=12)

plt.grid(axis='x')  # Add grid for better readability
plt.tight_layout()  # Adjust layout for better fit
plt.show()




# Calculate mean, median, and mode for 'Amount'
mean_amount = df['Amount'].mean()
median_amount = df['Amount'].median()
mode_amount = df['Amount'].mode()[0]  # Mode returns a series, get the first mode

print(f"Mean Sales Amount: {mean_amount}")
print(f"Median Sales Amount: {median_amount}")
print(f"Mode Sales Amount: {mode_amount}")

# Calculate variance and standard deviation
variance_amount = df['Amount'].var()
std_dev_amount = df['Amount'].std()

print(f"Variance of Sales Amount: {variance_amount}")
print(f"Standard Deviation of Sales Amount: {std_dev_amount}")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.countplot(data=df, y='ship-state', palette='pastel')  # Use 'ship-state' for counting orders by state

# Title and labels with increased font size
plt.title('Number of Orders by Shipping State', fontsize=16)
plt.xlabel('Number of Orders', fontsize=14)
plt.ylabel('Shipping State', fontsize=14)

# Adding data labels next to bars
for p in plt.gca().patches:
    plt.gca().annotate(f'{int(p.get_width())}', 
                       (p.get_width(), p.get_y() + p.get_height() / 2.),
                       ha='left', va='center', fontsize=12)

plt.grid(axis='x')  # Add grid for better readability
plt.tight_layout()  # Adjust layout for better fit
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Assuming 'Date' is already converted
df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%y', errors='coerce')
daily_sales = df.groupby('Date')['Amount'].sum()

# Plot time series trend
plt.figure(figsize=(10, 5))
daily_sales.plot()

# Formatting the date on the x-axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Set interval for major ticks

plt.title('Daily Sales Trend')
plt.xlabel('Date')
plt.ylabel('Total Sales Amount (INR)')
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout for better fit
plt.show()
"""



# TIME SERIES ANALYSIS

"""
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

# Load the data
file_path = 'C:\\Users\\NIYATI\\OneDrive\\Desktop\\study material\\PROJECT\\end to end project\\Amazon Sale Report.csv'
data = pd.read_csv(file_path)

# Display basic information
print("Data Overview:")
print(data.info())
print("\nFirst Few Rows:")
print(data.head())

# Step 1: Preprocessing
# Convert 'Date' column to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data.dropna(subset=['Date'], inplace=True)  # Remove rows with invalid dates
data.set_index('Date', inplace=True)

# Fill missing values in key columns
data['Amount'] = data['Amount'].fillna(0)
data['Qty'] = data['Qty'].fillna(0)

# Step 2: Descriptive Analysis
# Plot overall sales trend
plt.figure(figsize=(10, 6))
data['Amount'].resample('M').sum().plot(title='Monthly Sales Trend', ylabel='Sales Amount', xlabel='Date')
plt.show()

# Analyze sales by category
category_sales = data.groupby('Category')['Amount'].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
category_sales.plot(kind='bar', title='Total Sales by Category')
plt.ylabel('Sales Amount')
plt.xlabel('Category')
plt.show()

# Step 3: Seasonal Decomposition
# Resample to daily and decompose
daily_sales = data['Amount'].resample('D').sum()
decomposition = seasonal_decompose(daily_sales, model='additive', period=30)

# Plot decomposition
decomposition.plot()
plt.show()

# Step 4: Time-based Feature Engineering
data['Month'] = data.index.month
data['Year'] = data.index.year
data['Day_of_Week'] = data.index.dayofweek

# Plot sales trends by year
plt.figure(figsize=(10, 6))
yearly_sales = data.groupby('Year')['Amount'].sum()
yearly_sales.plot(kind='bar', title='Yearly Sales Trend')
plt.ylabel('Total Sales')
plt.xlabel('Year')
plt.show()

# Step 5: Forecasting using Prophet
# Prepare data for Prophet
prophet_data = daily_sales.reset_index()
prophet_data.columns = ['ds', 'y']  # Prophet expects 'ds' for date and 'y' for value

# Fit Prophet model
model = Prophet()
model.fit(prophet_data)

# Make future predictions
future = model.make_future_dataframe(periods=30)  # Forecast for the next 30 days
forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.title('Sales Forecast')
plt.show()

# Step 6: Anomaly Detection
# Calculate z-scores for anomaly detection
data['Z-Score'] = (data['Amount'] - data['Amount'].mean()) / data['Amount'].std()
anomalies = data[data['Z-Score'].abs() > 3]

# Display anomalies
print("\nAnomalies in Sales Data:")
print(anomalies)

# Step 7: Rolling Averages and Lag Features
data['Rolling_Avg'] = data['Amount'].rolling(window=7).mean()
data['Lag_1'] = data['Amount'].shift(1)

# Plot rolling average
plt.figure(figsize=(10, 6))
data['Amount'].plot(label='Original', alpha=0.5)
data['Rolling_Avg'].plot(label='7-Day Rolling Average', alpha=0.8)
plt.legend()
plt.title('Sales with Rolling Average')
plt.show()

# Save processed data
data.to_csv('Processed_Amazon_Sale_Report.csv')
print("\nProcessed data saved to 'Processed_Amazon_Sale_Report.csv'")

"""
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = 'C:\\Users\\NIYATI\\OneDrive\\Desktop\\study material\\PROJECT\\end to end project\\Amazon Sale Report.csv'
data = pd.read_csv(file_path)

# Convert 'Date' column to datetime and handle missing values
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data.dropna(subset=['Date'], inplace=True)  # Remove rows with invalid dates
data['Amount'] = data['Amount'].fillna(0)
data['Qty'] = data['Qty'].fillna(0)

# Aggregation Analysis
# 1. Time-Based Aggregations
print("\nTime-Based Aggregations:")
# Monthly Sales
monthly_sales = data.groupby(data['Date'].dt.to_period('M'))['Amount'].sum()
print(monthly_sales)

# Plot Monthly Sales
monthly_sales.plot(kind='line', title='Monthly Sales Trend', ylabel='Sales Amount', xlabel='Month', figsize=(10, 6))
plt.show()

# Daily Sales
daily_sales = data.groupby(data['Date'].dt.to_period('D'))['Amount'].sum()
print(daily_sales.head())

# 2. Category-Based Aggregations
print("\nCategory-Based Aggregations:")
category_sales = data.groupby('Category')['Amount'].agg(['sum', 'mean', 'count'])
print(category_sales)

# Plot Sales by Category
category_sales['sum'].plot(kind='bar', title='Total Sales by Category', ylabel='Total Sales', xlabel='Category', figsize=(10, 6))
plt.show()

# 3. Geographical Aggregations
print("\nGeographical Aggregations:")
geo_sales = data.groupby('ship-country')['Amount'].agg(['sum', 'mean', 'count']).sort_values(by='sum', ascending=False)
print(geo_sales)

# Plot Top 10 Countries by Sales
geo_sales.head(10)['sum'].plot(kind='bar', title='Top 10 Countries by Total Sales', ylabel='Total Sales', xlabel='Country', figsize=(10, 6))
plt.show()

# 4. Fulfillment and Status-Based Aggregations
print("\nFulfillment and Status-Based Aggregations:")
fulfillment_status = data.groupby(['Fulfilment', 'Status'])['Amount'].agg(['sum', 'mean', 'count'])
print(fulfillment_status)

# 5. Advanced Metrics
# Average Order Value (AOV)
print("\nAverage Order Value (AOV):")
aov = data['Amount'].sum() / data['Order ID'].nunique()  # Assuming 'Order ID' represents unique orders
print(f"AOV: {aov}")

# Top Performing Categories
print("\nTop Performing Categories:")
top_categories = category_sales.sort_values(by='sum', ascending=False).head(5)
print(top_categories)

# Save Aggregation Results to a CSV File
results = {
    'Monthly Sales': monthly_sales,
    'Category Sales': category_sales,
    'Geographical Sales': geo_sales,
    'Fulfillment Status Sales': fulfillment_status,
}
with pd.ExcelWriter('Aggregation_Analysis_Results.xlsx') as writer:
    monthly_sales.to_frame(name='Monthly Sales').to_excel(writer, sheet_name='Monthly Sales')
    category_sales.to_excel(writer, sheet_name='Category Sales')
    geo_sales.to_excel(writer, sheet_name='Geographical Sales')
    fulfillment_status.to_excel(writer, sheet_name='Fulfillment Status Sales')

print("\nAggregation results saved to 'Aggregation_Analysis_Results.xlsx'")