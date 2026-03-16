import numpy as np
import pandas as pd

# 注意：參數新增了 factor_pass_data
def run_backtest(test_dates, backtest_prices, final_signals, ma20_data, factor_pass_data, initial_capital, stop_loss_pct, take_profit_pct):
    capital = initial_capital
    position = 0
    trade_log = []
    equity_curve = [initial_capital]
    trade_profits = []
    entry_price = 0

    for i in range(len(final_signals) - 1):
        current_price = backtest_prices[i]
        next_price = backtest_prices[i+1]
        signal = final_signals[i]
        current_ma20 = ma20_data[i]
        current_factor_pass = factor_pass_data[i] # 取得當天的多因子審查結果

        # === 1. 持股狀態下的賣出判斷 ===
        if position > 0:
            current_ret = (current_price - entry_price) / entry_price

            if current_ret <= -stop_loss_pct:
                capital += position * current_price
                profit = (current_price - entry_price) * position
                trade_profits.append(profit)
                trade_log.append((test_dates[i], "STOP LOSS", current_price, capital, 0))
                position = 0

            elif current_ret >= take_profit_pct:
                capital += position * current_price
                profit = (current_price - entry_price) * position
                trade_profits.append(profit)
                trade_log.append((test_dates[i], "TAKE PROFIT", current_price, capital, 0))
                position = 0

            elif not signal:
                capital += position * current_price
                profit = (current_price - entry_price) * position
                trade_profits.append(profit)
                trade_log.append((test_dates[i], "SELL (AI Signal)", current_price, capital, 0))
                position = 0

        # === 2. 空手狀態下的買進判斷 (極度嚴謹的進場邏輯) ===
        elif signal and position == 0:
            # 必須同時滿足：站上 MA20 且 通過三維度多因子過濾 (目前實作了爆量)
            if (current_price > current_ma20) and current_factor_pass:
                position = capital // current_price
                if position > 0:
                    capital -= position * current_price
                    entry_price = current_price
                    trade_log.append((test_dates[i], "BUY (AI + Multi-Factor)", current_price, capital, position))

        current_equity = capital + position * next_price
        equity_curve.append(current_equity)

    if position > 0:
        final_price = backtest_prices[-1]
        capital += position * final_price
        profit = (final_price - entry_price) * position
        trade_profits.append(profit)
        trade_log.append((test_dates[-1], "FINAL SELL", final_price, capital, 0))

    return capital, equity_curve, trade_log, trade_profits

def calculate_metrics(initial_capital, final_capital, equity_curve, trade_profits):
    # (這部分與之前提供的一樣，保持不變即可)
    total_return_pct = (final_capital / initial_capital - 1) * 100
    equity_series = pd.Series(equity_curve)
    strategy_returns = equity_series.pct_change().fillna(0).values 
    
    mean_ret = np.mean(strategy_returns)
    std_ret = np.std(strategy_returns) + 1e-9
    if hasattr(mean_ret, 'item'): mean_ret = mean_ret.item()
    if hasattr(std_ret, 'item'): std_ret = std_ret.item()
    sharpe_val = (mean_ret / std_ret) * np.sqrt(252)
    if np.isnan(sharpe_val): sharpe_val = 0.0

    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100 
    if hasattr(max_drawdown, 'item'): max_drawdown = max_drawdown.item()

    if len(trade_profits) > 0:
        win_rate = np.mean(np.array(trade_profits) > 0) * 100
    else:
        win_rate = 0.0
    if hasattr(win_rate, 'item'): win_rate = win_rate.item()

    return {
        "Total Return (%)": float(np.array(total_return_pct).item()),
        "Sharpe Ratio": float(sharpe_val),
        "Max Drawdown (%)": float(max_drawdown),
        "Win Rate (%)": float(win_rate),
        "Strategy Returns": strategy_returns
    }