import numpy as np
import pandas as pd
import dyson
from dyson import DysonRouter
import os

os.environ["dyson_api"] = "your_dyson_api_key_here"  # Replace with your actual Dyson API key
router = DysonRouter()


def financial_data_analysis(n_stocks=50, n_days=252):
    """
    Analyze synthetic financial data
    """
    print(f"Analyzing {n_stocks} stocks over {n_days} days")

    # Generate synthetic stock data
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, (n_days, n_stocks))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))

    # Create DataFrame
    stock_names = [f"STOCK_{i:03d}" for i in range(n_stocks)]
    df = pd.DataFrame(prices, columns=stock_names)

    # Calculate metrics
    daily_returns = df.pct_change().dropna()

    # Risk metrics
    volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
    sharpe_ratio = (daily_returns.mean() * 252) / volatility

    # Correlation analysis
    correlation_matrix = daily_returns.corr()
    avg_correlation = correlation_matrix.values[
        np.triu_indices_from(correlation_matrix.values, k=1)
    ].mean()

    # Portfolio optimization (equal weight)
    portfolio_returns = daily_returns.mean(axis=1)
    portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
    portfolio_return = portfolio_returns.mean() * 252

    # Value at Risk (95% confidence)
    var_95 = np.percentile(daily_returns.values.flatten(), 5)

    return {
        "n_stocks": n_stocks,
        "n_days": n_days,
        "avg_volatility": volatility.mean(),
        "avg_sharpe_ratio": sharpe_ratio.mean(),
        "avg_correlation": avg_correlation,
        "portfolio_return": portfolio_return,
        "portfolio_volatility": portfolio_volatility,
        "portfolio_sharpe": portfolio_return / portfolio_volatility,
        "var_95_percent": var_95,
        "best_performing_stock": stock_names[daily_returns.mean().idxmax()],
        "worst_performing_stock": stock_names[daily_returns.mean().idxmin()],
    }


# Route the function to hardware
hardware = router.route_hardware(
    financial_data_analysis,
    mode="cost-effective",
    judge=5,
    run_type="log",
    complexity="medium",
    n_stocks=50,
    n_days=252,
)
print("Hardware Specification:", hardware["spec"])
print("Hardware Type:", hardware["hardware_type"])
# Print the results
result = hardware["compiled_function"](n_stocks=50, n_days=252)
