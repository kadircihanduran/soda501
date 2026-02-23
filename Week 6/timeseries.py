#Q1:

#Time leakage occurs when information from the future leaks into the training phase. In time-indexed data, random train test splits usually inflate performance because they allow the model to learn from data points that occurred after the events it is trying to predict, essentially giving the model answers from the future. Consequently, train on past, test on future is the default evaluation since it replicates the actual scenario where a model must predict unknown future outcomes based only on historical observations. While a single time split provides a baseline, rolling-origin backtesting adds rigor by producing multiple out-of-sample errors across different time periods. This approach ensures that the model’s performance is stable over time and not just an accident of one specific split.

import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA 
import statsmodels.api as sm 

np.random.seed(19880106) 
n = 600 
dates = pd.date_range(start="2024-01-01", periods=n, freq="D") 
t = np.arange(1, n + 1) 
trend = 0.02 * t 
weekly = 1.2 * np.sin(2 * np.pi * t / 7) 
phi = 0.65 
eps = np.random.normal(loc=0.0, scale=1.0, size=n) 
ar_noise = np.empty(n) 
ar_noise[0] = eps[0] 
for i in range(1, n):
   ar_noise[i] = phi * ar_noise[i - 1] + eps[i] 
 
y = 10 + trend + weekly + ar_noise 
df = pd.DataFrame({"date": dates, "t": t, "y": y}) 
df.set_index("date", inplace=True)
 
# Q3
 
os.makedirs("Week 6/outputs/figures", exist_ok=True) 

stl = STL(df['y'], period=7)
result = stl.fit() 

fig = result.plot()
fig.set_size_inches(10, 8) 
plt.suptitle("STL Decomposition: Observed, Trend, Seasonal, and Residual", fontsize=14) 
plt.tight_layout() 
plt.savefig("Week 6/outputs/figures/decomposition.png") 
plt.show() 
 
#Q4
 
os.makedirs("Week 6/outputs/tables", exist_ok=True) 
 
initial_window = 300
n_total = len(df) 
 
dates_test = [] 
y_actual = [] 
y_hat = [] 
errors = [] 
 
for t in range(initial_window, n_total):
    train_data = df['y'].iloc[:t].to_numpy()
    model = ARIMA(train_data, order=(1, 0, 0)) 
    fit_model = model.fit()
    forecast = fit_model.forecast(steps=1)[0]
    actual = df['y'].iloc[t] 
    error = actual - forecast 
    dates_test.append(df.index[t] if isinstance(df.index, pd.DatetimeIndex) else df['date'].iloc[t]) 
    y_actual.append(actual) 
    y_hat.append(forecast) 
    errors.append(error) 
 
results_df = pd.DataFrame({'date': dates_test, 'y': y_actual, 'yhat': y_hat, 'error': errors })
 
rmse_backtest = np.sqrt(np.mean(results_df['error'] ** 2)) 
print(f"Backtest RMSE (Rolling-origin): {rmse_backtest:.6f}") 

results_df.to_csv("Week 6/outputs/tables/backtest_errors.csv", index=False) 

plt.figure(figsize=(10, 6)) 
plt.plot(results_df['date'], results_df['y'], label='Observed (y)', color='blue') 
plt.plot(results_df['date'], results_df['yhat'], label='Forecast (yhat)', color='orange', linestyle='--') 
plt.title("Rolling-Origin Backtest: 1-Step Ahead Forecasts vs Observed") 
plt.xlabel("Date") 
plt.ylabel("y") 
plt.legend() 
plt.tight_layout() 
plt.savefig("Week 6/outputs/figures/backtest_forecast.png") 
plt.show() 

#5

np.random.seed(19880106)
n = 600 
dates = pd.date_range(start="2024-01-01", periods=n, freq="D") 
t = np.arange(1, n + 1) 
 
phi = 0.5 
eps = np.random.normal(0, 2, n) 
noise = np.empty(n) 
noise[0] = eps[0] 
for i in range(1, n):
    noise[i] = phi * noise[i-1] + eps[i] 
 
t0 = 300 
alpha = 10
delta = 0.05
tau1_true = 8.0
tau2_true = 0.03
indicator = (t >= t0).astype(int)
post_time = (t - t0) * indicator
 
y = alpha + delta * t + tau1_true * indicator + tau2_true * post_time + noise 
df_its = pd.DataFrame({"date": dates, "t": t, "y": y, "indicator": indicator, "post_time": post_time}) 
X = sm.add_constant(df_its[["t", "indicator", "post_time"]]) 
model_real = sm.OLS(df_its["y"], X).fit() 

df_its["fitted"] = model_real.predict(X) 
X_cf = X.copy() 
X_cf["indicator"] = 0 
X_cf["post_time"] = 0 
df_its["counterfactual"] = model_real.predict(X_cf) 

plt.figure(figsize=(10, 6)) 
plt.plot(df_its["date"], df_its["y"], label="Observed y_t", alpha=0.5, color="gray") 
plt.plot(df_its["date"], df_its["fitted"], label="Fitted ITS", color="blue", linewidth=2) 
plt.plot(df_its["date"], df_its["counterfactual"], label="Counterfactual (Extended Pre-trend)", color="red", linestyle="--", linewidth=2) 
plt.axvline(df_its.loc[t0-1, "date"], color="black", linestyle=":", label=f"Intervention (t0={t0})") 
plt.title("Interrupted Time Series Analysis (Level & Trend Change)") 
plt.xlabel("Date") 
plt.ylabel("y") 
plt.legend() 
plt.tight_layout() 
plt.savefig("Week 6/outputs/figures/its_plot.png" ) 
plt.show() 

df_pre = df_its[df_its["t"] < t0].copy() 
t_fake = 150
df_pre["placebo_indicator"] = (df_pre["t"] >= t_fake).astype(int) 
df_pre["placebo_post_time"] = (df_pre["t"] - t_fake) * df_pre["placebo_indicator"] 
X_placebo = sm.add_constant(df_pre[["t", "placebo_indicator", "placebo_post_time"]]) 
model_placebo = sm.OLS(df_pre["y"], X_placebo).fit()

results = pd.DataFrame({ 
"Model": ["Real ITS", "Real ITS", "Real ITS", "Real ITS",
"Placebo ITS", "Placebo ITS", "Placebo ITS", "Placebo ITS"], 
"Parameter": ["Intercept", "Pre-Trend", "Level Change (tau1)", "Trend Change (tau2)", 
"Intercept", "Pre-Trend", "Fake Level Change", "Fake Trend Change"], 
"Estimate": [ 
model_real.params["const"], model_real.params["t"], model_real.params["indicator"], model_real.params["post_time"], 
model_placebo.params["const"], model_placebo.params["t"], model_placebo.params["placebo_indicator"], model_placebo.params["placebo_post_time"]],"P-Value": [ 
model_real.pvalues["const"], model_real.pvalues["t"], model_real.pvalues["indicator"], model_real.pvalues["post_time"], 
model_placebo.pvalues["const"], model_placebo.pvalues["t"], model_placebo.pvalues["placebo_indicator"], model_placebo.pvalues["placebo_post_time"]]})
 
results.to_csv("Week 6/outputs/tables/its_results.csv", index=False) 

print(f"Fake Level Change P-Value: {model_placebo.pvalues['placebo_indicator']:.4f}") 
print(f"Fake Trend Change P-Value: {model_placebo.pvalues['placebo_post_time']:.4f}") 

#Because both placebo p-values are much larger than the standard 0.05 threshold, we fail to reject the null hypothesis that there was an effect at the fake date. This demonstrates that the model does not find significant changes where none exist, strengthening the claim that the effects found at t0=300 are truly due to the intervention rather than random noise or pre-existing instability in the trend.
