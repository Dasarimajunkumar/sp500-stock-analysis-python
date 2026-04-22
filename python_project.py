# =========================================
# 1. IMPORT LIBRARIES
# =========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# =========================================
# 2. LOAD DATA
# =========================================
df = pd.read_csv(r"C:\Users\Majun\Downloads\S&P+500+Stock+Prices+2014-2017.csv\S&P 500 Stock Prices 2014-2017.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# =========================================
# 3. DATA CLEANING
# =========================================
df['date'] = pd.to_datetime(df['date'])

df['Year'] = df['date'].dt.year
df['Month'] = df['date'].dt.month

df['Daily_Return'] = df.groupby('symbol')['close'].pct_change() * 100
df['Price_Range'] = df['high'] - df['low']

# =========================================
# 4. BASIC EDA
# =========================================
print(df.info())
print(df.describe())
print(df.isnull().sum())

# =========================================
# 5. VISUALIZATIONS
# =========================================

# Top 10 Stocks by Average Close Price
top10 = df.groupby('symbol')['close'].mean().sort_values(ascending=False).head(10)

plt.figure()
sns.barplot(x=top10.index, y=top10.values, hue=top10.index, palette="Blues", legend=False)
plt.title("Top 10 Stocks by Average Close Price")
plt.xlabel("Stock Symbol")
plt.ylabel("Average Close Price")
plt.show()


# Pie Chart - Volume Share
top6 = df.groupby('symbol')['volume'].sum().sort_values(ascending=False).head(6)

plt.figure()
plt.pie(top6.values,
        labels=top6.index,
        autopct='%1.1f%%')
plt.title("Top 6 Stocks Volume Share")
plt.show()


# Box Plot - Daily Return by Year
plt.figure()
sns.boxplot(x='Year', y='Daily_Return', data=df)
plt.title("Daily Return Distribution by Year")
plt.show()


# Scatter Plot - Open vs Close Price
plt.figure()
sns.scatterplot(x='open', y='close', data=df)
plt.title("Open vs Close Price")
plt.xlabel("Open Price")
plt.ylabel("Close Price")
plt.show()


# Line Chart - Stock Trend
stocks = ['AAPL', 'AMZN', 'NFLX']
df_line = df[df['symbol'].isin(stocks)]

plt.figure()

for stock in stocks:
    data = df_line[df_line['symbol'] == stock]
    plt.plot(data['date'], data['close'], label=stock)

plt.title("Stock Price Trend")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()

# =========================================
# TOP 5 TRADING PROFIT COMPANIES (2014–2017)
# =========================================

# Create profit column
df['profit'] = df['close'] - df['open']

# Calculate total profit for each company
top_profit = df.groupby('symbol')['profit'].sum().sort_values(ascending=False).head(5)

print("Top 5 Trading Profit Companies:")
print(top_profit)

# Visualization
plt.figure()

sns.barplot(
    x=top_profit.values,
    y=top_profit.index,
    hue=top_profit.index,
    palette="viridis",
    legend=False
)

plt.title("Top 5 Trading Profit Companies (2014–2017)")
plt.xlabel("Total Trading Profit")
plt.ylabel("Company Symbol")

plt.show()


# Histogram - Daily Returns
plt.figure()
sns.histplot(df['Daily_Return'].dropna(), bins=100, kde=True)
plt.title("Daily Return Distribution")
plt.xlabel("Daily Return")
plt.show()


# Heatmap - Correlation
numeric_df = df.select_dtypes(include=np.number)

plt.figure()
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# =========================================
# 6. SKEWNESS
# =========================================
print("Skewness:")
print(numeric_df.skew())


# =========================================
# 7. OUTLIERS
# =========================================
Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1

outliers = ((numeric_df < (Q1 - 1.5 * IQR)) |
            (numeric_df > (Q3 + 1.5 * IQR)))

print("Outliers per column:")
print(outliers.sum())

print("Total rows with outliers:")
print(outliers.any(axis=1).sum())
