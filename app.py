import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 匯入我們自建的核心模組
import data_loader
import model_core
import backtest_core

# --- 1. 網頁標題與側邊欄設定 ---
st.set_page_config(page_title="AI 量化回測平台", layout="wide")
st.title("📈 AI 雙因子量化回測系統")
st.markdown("結合 **LSTM 動能預測** 與 **順勢布林通道濾網** 的跨市場回測平台")

st.sidebar.header("⚙️ 策略參數設定")
ticker = st.sidebar.text_input("股票代號 (支援跨市場, 如 2330.TW, AAPL)", "2330.TW")
look_back = st.sidebar.slider("滑動視窗天數 (Look Back)", 30, 90, 60)
epochs = st.sidebar.slider("AI 訓練輪數 (Epochs)", 10, 200, 50)
initial_capital = st.sidebar.number_input("初始本金 ($)", min_value=1000, value=100000, step=1000)
start_date = st.sidebar.date_input("開始日期", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("結束日期", pd.to_datetime("today"))

st.sidebar.subheader("🛡️ 風險控管模組")
stop_loss_pct = st.sidebar.slider("停損比例 (%)", 1, 20, 5) / 100
take_profit_pct = st.sidebar.slider("停利比例 (%)", 1, 100, 15) / 100
predict_btn = st.sidebar.button("🚀 執行模型訓練與回測")

# --- 主程式邏輯 ---
if predict_btn:
    # [資料存取層]
    with st.spinner('📊 正在從資料庫獲取跨市場金融數據與技術指標...'):
        df = data_loader.load_and_preprocess_data(ticker, start_date, end_date)
        if df is None or df.empty:
            st.error("找不到該股票代號或數據不足，請重新輸入。")
            st.stop()
            
        all_dates = df.index
        raw_close_prices = df['Close'].values
        ma20_values = df['MA20'].values
        factor_pass_values = df['Factor_Pass'].values # 提取多因子審查結果

    # [邏輯運算層 - AI 訓練]
    # (這段與之前完全相同，保持不變)
    with st.spinner('🧠 AI 正在進行深度學習與特徵萃取...'):
        X, y, scaler, scaled_data = model_core.prepare_model_data(df, look_back)
        prediction_dates = all_dates[look_back:]
        
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        test_dates = prediction_dates[train_size:]
        
        model, device = model_core.train_lstm_model(X_train, y_train, epochs=epochs)
        
        test_predict_scaled = model_core.predict_model(model, X_test, device)
        test_predict_returns = model_core.get_inverse_price(test_predict_scaled, scaler, scaled_data.shape[1])
        y_test_actual_returns = model_core.get_inverse_price(y_test, scaler, scaled_data.shape[1])

    # [邏輯運算層 - 回測引擎]
    with st.spinner('💰 正在執行雙因子策略回測與績效結算...'):
        final_signals = test_predict_returns > 0 
        
        backtest_prices = raw_close_prices[look_back + train_size:]
        backtest_ma20 = ma20_values[look_back + train_size:]
        backtest_factor_pass = factor_pass_values[look_back + train_size:] # 對齊多因子資料
        
        # 呼叫回測引擎 (傳入 backtest_factor_pass)
        final_capital, equity_curve, trade_log, trade_profits = backtest_core.run_backtest(
            test_dates, backtest_prices, final_signals, backtest_ma20, backtest_factor_pass,
            initial_capital, stop_loss_pct, take_profit_pct
        )
        
        metrics = backtest_core.calculate_metrics(initial_capital, final_capital, equity_curve, trade_profits)

    # --- 應用層 UI 展示 ---
    st.success("✅ 回測完成！以下為策略績效評估報告：")

    # 1. 明日走勢預測
    feature_count = scaled_data.shape[1] 
    last_window_data = scaled_data[-look_back:]
    last_window_tensor = np.expand_dims(last_window_data, axis=0) 
    
    next_pred_scaled = model_core.predict_model(model, last_window_tensor, device)
    next_return_val = float(model_core.get_inverse_price(next_pred_scaled, scaler, feature_count)[0])
    last_actual_close = float(raw_close_prices[-1])
    next_price_val = last_actual_close * (1 + next_return_val)

    st.subheader("🔮 明日 AI 預測與策略建議")
    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("預測漲跌幅", f"{next_return_val * 100:.2f}%")
    col_p2.metric("預測目標價", f"${next_price_val:.2f}")
    
    # 明日建議邏輯更新：判斷是否符合多因子條件
    current_ma20_val = ma20_values[-1]
    current_factor_val = factor_pass_values[-1]
    is_bullish = (next_return_val > 0) and (last_actual_close > current_ma20_val) and current_factor_val
    signal_text = "🟢 建議買進 (AI看多 + 站上MA20 + 爆量表態)" if is_bullish else "🔴 建議觀望 (未達多維度進場標準)"
    col_p3.info(f"明日策略建議：\n**{signal_text}**")

    # (下方的績效儀表板、資金曲線圖、歷史交易紀錄的程式碼保持不變)
    # ...

    # 2. 專業評估儀表板
    st.divider()
    st.subheader("📊 策略績效儀表板")
    
    # 基礎誤差
    rmse = np.sqrt(mean_squared_error(y_test_actual_returns, test_predict_returns))
    actual_direction = y_test_actual_returns > 0
    direction_acc = np.mean(actual_direction == final_signals) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AI 方向準確度", f"{direction_acc:.1f}%")
    col2.metric("策略總報酬率", f"{metrics['Total Return (%)']:.2f}%")
    col3.metric("最大回撤 (MDD)", f"{metrics['Max Drawdown (%)']:.2f}%")
    col4.metric("策略勝率", f"{metrics['Win Rate (%)']:.1f}%")

    # 3. 資金曲線對比圖
    st.subheader("💰 累積收益率對比 (策略 vs 大盤買入持有)")
    
    # 計算買入持有(Buy & Hold)的累積報酬
    market_returns = pd.Series(y_test_actual_returns).fillna(0).values
    cumulative_market = (1 + market_returns).cumprod()
    cumulative_strategy = (1 + metrics['Strategy Returns']).cumprod()
    
    fig_perf, ax_perf = plt.subplots(figsize=(10, 4))
    ax_perf.plot(test_dates, cumulative_market, label="Market (Buy & Hold)", color="gray", alpha=0.5)
    ax_perf.plot(test_dates, cumulative_strategy, label="AI + Bollinger Bands Strategy", color="gold", linewidth=2)
    ax_perf.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
    ax_perf.set_ylabel("Cumulative Return")
    ax_perf.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_perf.legend()
    st.pyplot(fig_perf)

    # 4. 交易紀錄
    st.subheader("📜 歷史交易紀錄")
    if trade_log:
        trade_df = pd.DataFrame(trade_log, columns=["Date", "Action", "Price", "Cash Left", "Shares"])
        trade_df['Date'] = pd.to_datetime(trade_df['Date']).dt.strftime('%Y-%m-%d')
        st.dataframe(trade_df, use_container_width=True)
    else:
        st.write("測試期間無觸發交易訊號。")
