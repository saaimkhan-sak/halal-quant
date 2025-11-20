# lists, dicts, functions, pandas
import pandas as pd

nums = [10, 12, 15, 21]
def pct_change(series):
    out = []
    for i in range(1, len(series)):
        out.append( (series[i] - series[i-1]) / series[i-1] )
    return out

print("Percent changes:", pct_change(nums))

# tiny DataFrame exercise
df = pd.DataFrame({"ticker":["AAPL","MSFT","NVDA"], "price":[100, 120, 90]})
print(df.head())
df.to_csv("output.csv", index=False)
print("Wrote output.csv")