import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


'''def fetch_and_transform_data(filename):
    # Read the CSV file into a DataFrame
    original_df = pd.read_csv(filename)

    # Reshape the DataFrame to have a better structure
    yeardata = original_df.set_index(['Country Name', 'Indicator Name']).stack().unstack(0).reset_index()
    yeardata.columns.name = None

    return original_df, yeardata


# Provide the file path
file_path = 'Indicatorsdata.csv'

# Call the function and unpack the results
original_df, yeardata = fetch_and_transform_data(file_path)


# Converting and Reading the Transformed data to Csv
yeardata.to_csv('filtered.csv')'''
countrydf = pd.read_csv('filtered.csv')
countrydf=countrydf.dropna()


# 1 Descriptive statistics.
stat = countrydf.describe()
print(stat)

# 2 Correlation Heatmap
# Filter the DataFrame for selected indicators
'''selected_data = countrydf[countrydf['Indicator Name'].isin(['Access to electricity (% of population)', 'Electric power consumption (kWh per capita)'])]
# Pivot the data for better visualization
pivot_data = selected_data.pivot(index='Year', columns='Indicator Name', values='Pakistan')
pivot_data.dropna(inplace=True)
# Calculate the correlation matrix
correlation_matrix = pivot_data.corr()
# Plot the correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap for electricity consumption and access in Pakistan')
plt.show()'''


# 3 Line Chart For Agricultural Land

# Filter the DataFrame for selected indicator and valid years
'''selected_data = countrydf[
    (countrydf['Indicator Name'] == 'Agricultural irrigated land (% of total agricultural land)') &
    (countrydf['Year'].notna())
]

# Select only the relevant columns
selected_data = selected_data[['Year', 'Oman', 'Nepal', 'Norway']]

# Group by Year and sum the values
grouped_data = selected_data.groupby('Year').sum()

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(grouped_data.index, grouped_data['Oman'], label='Oman')
plt.plot(grouped_data.index, grouped_data['Nepal'], label='Nepal')
plt.plot(grouped_data.index, grouped_data['Norway'], label='Norway')

# Set plot labels and title
plt.title('Agricultural Irrigated Land - Nepal, Oman, Norway')
plt.xlabel('Year')
plt.ylabel('Agricultural Irrigated Land (% of total)')
plt.legend()
plt.grid(True)
plt.show()'''


# 4 Boxplot For Urban Population
'''indicator_name = 'Urban population'
selected_data = countrydf[countrydf['Indicator Name'] == indicator_name][['Austria', 'Belgium']]
melted_data = pd.melt(selected_data, var_name='Country', value_name='Value')

# Create a boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Country', y='Value', data=melted_data)
plt.title(f'Boxplot for {indicator_name}')
plt.xlabel('Country')
plt.ylabel(f'{indicator_name} (%)')  # Adjusted ylabel for clarity
plt.show()'''


# 5 Bar chart for Electricity production comparison
'''barchart = countrydf[
    countrydf['Indicator Name'] == 'Electricity production from renewable sources, excluding hydroelectric (% of total)'
][['New Zealand', 'South Africa']]

# Sum the values for each country
aggregated_data = barchart.sum()

# Create a bar chart
aggregated_data.plot(kind='bar', color=['blue', 'orange'], figsize=(10, 6))  # Adjusted color list
plt.title('Renewable Electricity Production - South Africa vs New Zealand')
plt.xlabel('Country')
plt.ylabel('Electricity Production (%)')  # Adjusted ylabel for clarity
plt.show()'''


# TOP 10 VERTICAL BAR CHART

# Filter the DataFrame for the 'Cereal yield (kg per hectare)' indicator
'''population_df = original_df[original_df['Indicator Name'] == 'Cereal yield (kg per hectare)']

# Sum the values for each country
population_sum = population_df.drop(['Indicator Name'], axis=1).groupby('Country Name').sum()

# Sum the values across all countries and select the top 10
top_countries = population_sum.sum(axis=1).sort_values(ascending=False).head(10)

# Create a bar chart
plt.figure(figsize=(10, 6))
bar_chart = sns.barplot(x=top_countries.values, y=top_countries.index, palette='magma')
plt.title('Top 10 Countries by Cereal Yield')
plt.xlabel('Total Cereal Yield (kg per hectare)')
plt.ylabel('Country')
plt.show()'''


# Pie Chart
# Filter the DataFrame for the 'Primary completion rate, total (% of relevant age group)' indicator
'''filtered_df = original_df[original_df['Indicator Name'] == 'Primary completion rate, total (% of relevant age group')]

# Select relevant columns
selected_columns = ['Country Name', '2020', '2021']
filtered_df = filtered_df[selected_columns]

# Convert the columns '2020' and '2021' to numeric, handling errors
filtered_df[['2020', '2021']] = filtered_df[['2020', '2021']].apply(pd.to_numeric, errors='coerce')

# Calculate the sum of '2020' and '2021' for each country
filtered_df['Sum'] = filtered_df[['2020', '2021']].sum(axis=1)

# Select the top 5 countries based on the sum
top_countries = filtered_df.sort_values(by='Sum', ascending=False).head(5)

# Extract labels and sizes for the pie chart
labels = top_countries['Country Name']
sizes = top_countries['Sum']

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Top 5 Countries - Primary Completion Rate (% of relevant age group) - 2020 and 2021')
plt.show()'''
