import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- 網頁標題 ---
st.title("AI 股價預測系統")
st.write("輸入股票代號，LSTM 模型預測趨勢並進行回測。")

# --- 側邊欄輸入 ---
st.sidebar.header("設定參數")
ticker = st.sidebar.text_input("請輸入股票代號 (如: 2330.TW, AAPL)", "2330.TW")
look_back = st.sidebar.slider("滑動視窗天數 (Look Back)", 30, 90, 60)
epochs = st.sidebar.slider("訓練輪數 (Epochs)", 10, 200, 100)
initial_capital = st.sidebar.number_input("初始本金 ($)", min_value=1000, value=100000, step=1000)
start_date = st.sidebar.date_input("開始日期", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("結束日期", pd.to_datetime("today"))
predict_btn = st.sidebar.button("執行 AI 訓練與預測")
st.sidebar.subheader("風險控管")
stop_loss_pct = st.sidebar.slider("停損比例 (%)", 1, 20, 5) / 100
take_profit_pct = st.sidebar.slider("停利比例 (%)", 1, 100, 15) / 100

if "is_trained" not in st.session_state:
    st.session_state.is_trained = False

if predict_btn:
    st.session_state.is_trained = True

if st.session_state.is_trained:
    # 把原本 if predict_btn: 底下的程式碼全部放在這裡，這樣就不會因為按鈕被按了又按了而重複執行訓練流程了
    # --- 2. 抓取數據 (替換原本這段) ---
    with st.spinner('正在獲取金融數據...'):
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            st.error("找不到該股票代號，請重新輸入。")
            st.stop()
        
        # 關鍵：新增報酬率作為預測目標
        df['Return'] = df['Close'].pct_change()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df = df.dropna() 
        
        # 關鍵：保留一份原始收盤價供最後計算使用
        raw_close_prices = df['Close'].values 
        
        # 這裡的特徵數量變成 4 (Return, Close, Volume, RSI)
        features = df[['Return', 'Close', 'Volume', 'RSI']].values
        all_dates = df.index
        raw_close_prices = df['Close'].values # 保留原始收盤價，等等回測要用！

    # --- 3. 數據預處理 ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_features = features[:int(len(features) * 0.8)]
    scaler.fit(train_features)
    scaled_data = scaler.transform(features)

    def create_dataset(dataset, look_back=60):
        X, y = [], []
        for i in range(look_back, len(dataset)):
            X.append(dataset[i - look_back:i, :])
            y.append(dataset[i, 0])
        return np.array(X), np.array(y)

    def get_inverse_price(pred_array):
        dummy = np.zeros((len(pred_array), scaled_data.shape[1]))
        dummy[:, 0] = pred_array.flatten()
        return scaler.inverse_transform(dummy)[:, 0]

    # --- 3. 數據預處理與切分 ---
    X, y = create_dataset(scaled_data, look_back)
    prediction_dates = all_dates[look_back:]

    # 設定切分比例 (80% 訓練, 20% 測試)
    train_size = int(len(X) * 0.8)

    # 分割 X (特徵) 與 y (目標)
    X_train_raw, X_test_raw = X[:train_size], X[train_size:]
    y_train_raw, y_test_raw = y[:train_size], y[train_size:]

    # 分割日期 (以便後續繪圖對齊)
    train_dates = prediction_dates[:train_size]
    test_dates = prediction_dates[train_size:]

    # 轉換為 PyTorch Tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_tensor = torch.from_numpy(X_train_raw).float().to(device)
    y_train_tensor = torch.from_numpy(y_train_raw).float().to(device)
    X_test_tensor = torch.from_numpy(X_test_raw).float().to(device)
    y_test_tensor = torch.from_numpy(y_test_raw).float().to(device)

    st.write(f"數據切分完成：訓練集 {len(X_train_raw)} 筆，測試集 {len(X_test_raw)} 筆。")

    class LSTMModel(nn.Module):
        def __init__(self):
            super(LSTMModel, self).__init__()
            # input_size 改成 4 (Return, Close, Volume, RSI)
            self.lstm = nn.LSTM(input_size=4, hidden_size=50, num_layers=2, batch_first=True, dropout=0.2)
            self.linear = nn.Linear(50, 1)

        def forward(self, x):
            h0 = torch.zeros(2, x.size(0), 50).to(device)
            c0 = torch.zeros(2, x.size(0), 50).to(device)
            out, _ = self.lstm(x, (h0, c0))
            return self.linear(out[:, -1, :])

    model = LSTMModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    from torch.utils.data import TensorDataset, DataLoader

    batch_size = 32

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # --- 5. 訓練模型 ---
    with st.spinner('AI 正在學習股價規律...'):
        #X_train_tensor = torch.from_numpy(X_train_raw).float().to(device)
        #y_train_tensor = torch.from_numpy(y_train_raw).float().to(device)

        for e in range(epochs):
            model.train()
            epoch_loss = 0

            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb.view(-1, 1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            if e % 10 == 0:
                st.write(f"Epoch {e}, Loss: {epoch_loss:.6f}")

            if epoch_loss < 0.0005:
                st.write(f"Early stopping at epoch {e}")
                break

    # --- 6. 預測與結果展示 ---
    # --- 6. 預測與結果展示 (修正：區分訓練與測試) ---
    model.eval()
    with torch.no_grad():
        # 分別對訓練集和測試集進行預測
        train_predict = model(X_train_tensor).cpu().numpy()
        test_predict = model(X_test_tensor).cpu().numpy()

    # 轉回原始價格
    train_predict_plot = get_inverse_price(train_predict)
    test_predict_plot = get_inverse_price(test_predict)
    y_train_actual = get_inverse_price(y_train_raw)
    y_test_actual = get_inverse_price(y_test_raw)

    # --- 繪製圖表 (修正：畫出盲測線) ---
    st.subheader(f"📊 {ticker} 股價預測圖 (含盲測分界線)")
    fig, ax = plt.subplots(figsize=(12, 6))

    # 1. 畫訓練集 (用淡色)
    ax.plot(train_dates, y_train_actual, label='Train Actual', color='blue', alpha=0.3)
    ax.plot(train_dates, train_predict_plot, label='Train Predict', color='red', linestyle='--', alpha=0.3)

    # 2. 畫測試集 (用深色，這才是重點)
    ax.plot(test_dates, y_test_actual, label='Test Actual (Blind Test)', color='blue', linewidth=2)
    ax.plot(test_dates, test_predict_plot, label='Test Predict (Blind Test)', color='orange', linewidth=2)

    # 3. 畫分界線
    if len(test_dates) > 0:
        ax.axvline(x=test_dates[0], color='black', linestyle='--', label='Training/Test Split')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    ax.legend()
    st.pyplot(fig)

    # 預測明天
    # --- 預測明天 (雲端部屬防呆版) ---
    feature_count = scaled_data.shape[1] 
    last_window_data = scaled_data[-look_back:]
    last_window_tensor = torch.from_numpy(last_window_data).float().view(1, look_back, feature_count).to(device)
    
    model.eval()
    with torch.no_grad():
        next_pred_raw = model(last_window_tensor).cpu().numpy()

    # 1. 轉回實際報酬率：確保只抓取第一個純數值
    inv_pred = get_inverse_price(next_pred_raw)
    next_return_val = float(np.array(inv_pred).flatten()[0])
    
    # 2. 抓取最後一天的真實收盤價：同樣確保它是純數值
    # .flatten() 能把任何多維陣列壓平，[ -1] 抓最後一個
    last_actual_close = float(np.array(raw_close_prices).flatten()[-1])
    
    # 3. 換算預測收盤價
    # 計算公式： $next\_price = last\_close \times (1 + next\_return)$
    next_price_val = last_actual_close * (1 + next_return_val)

    # 4. 顯示結果 (使用轉好的純 float 變數)
    st.divider()
    st.subheader("🔮 明日走勢預測")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.metric("預測漲跌幅", f"{next_return_val * 100:.2f}%")
    with col_p2:
        # 直接使用計算好的數字，不再在 f-string 裡面包 float()
        st.success(f"🎯 換算預測收盤價約為： **${next_price_val:.2f}**")

    # --- 7. 指標計算 (修正：僅針對測試集進行評估) ---

    # A. 基礎誤差指標 (針對測試集)
    # A. 基礎誤差指標
    rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict_plot))
    mae = mean_absolute_error(y_test_actual, test_predict_plot)
    # 移除 mape，因為報酬率接近 0 時會導致數值異常

    # B. 方向準度
    actual_direction = y_test_actual > 0
    final_signals = test_predict_plot > 0 
    direction_acc = np.mean(actual_direction == final_signals) * 100

    # C. 金融指標 (針對測試集)
    # === 交易模擬 ===
    # C. 金融指標 (針對測試集)
    # === 交易模擬 (完全無未來函數，且使用真實價格) ===
    capital = initial_capital
    position = 0
    trade_log = []
    equity_curve = [initial_capital]
    trade_profits = []

    # 提取與測試集對齊的真實收盤價
    backtest_prices = raw_close_prices[look_back + train_size:]

    entry_price = 0

    # 留最後一天計算資產，所以迴圈跑 len(final_signals) - 1
    for i in range(len(final_signals) - 1):
        current_price = backtest_prices[i]
        next_price = backtest_prices[i+1]
        signal = final_signals[i]

        # === 持股狀態下的判斷 ===
        if position > 0:
            # 計算當前報酬率
            current_ret = (current_price - entry_price) / entry_price

            # 1. 檢查停損 (Stop Loss)
            if current_ret <= -stop_loss_pct:
                capital += position * current_price
                profit = (current_price - entry_price) * position
                trade_profits.append(profit)
                trade_log.append((test_dates[i], "STOP LOSS", current_price, capital, 0))
                position = 0

            # 2. 檢查停利 (Take Profit)
            elif current_ret >= take_profit_pct:
                capital += position * current_price
                profit = (current_price - entry_price) * position
                trade_profits.append(profit)
                trade_log.append((test_dates[i], "TAKE PROFIT", current_price, capital, 0))
                position = 0

            # 3. AI 賣出訊號 (當預測明天會跌時)
            elif not signal:
                capital += position * current_price
                profit = (current_price - entry_price) * position
                trade_profits.append(profit)
                trade_log.append((test_dates[i], "SELL (AI Signal)", current_price, capital, 0))
                position = 0

        # === 空手狀態下的買進判斷 ===
        elif signal and position == 0:
            position = capital // current_price
            if position > 0:
                capital -= position * current_price
                entry_price = current_price
                trade_log.append((test_dates[i], "BUY", current_price, capital, position))

        # 每日更新資產價值
        current_equity = capital + position * next_price
        equity_curve.append(current_equity)

    # 最後強制平倉
    if position > 0:
        final_price = backtest_prices[-1]
        capital += position * final_price
        profit = (final_price - entry_price) * position
        trade_profits.append(profit)
        trade_log.append((test_dates[-1], "FINAL SELL", final_price, capital, 0))
        position = 0

    final_capital = capital
    total_return_pct = (final_capital / initial_capital - 1) * 100

    
    # === 修正：補上缺失的 strategy_returns ===
    equity_series = pd.Series(equity_curve)
    strategy_returns = equity_series.pct_change().fillna(0).values 
    
    # 計算夏普比率與最大回撤
    # 修正: np.mean/np.std 回傳型態，確保 sharpe_val 為純 float
    # 再次修正: np.mean/np.std 回傳型態，item() 取純 float
    mean_ret = np.mean(strategy_returns)
    std_ret = np.std(strategy_returns) + 1e-9
    if hasattr(mean_ret, 'item'):
        mean_ret = mean_ret.item()
    if hasattr(std_ret, 'item'):
        std_ret = std_ret.item()
    sharpe_val = mean_ret / std_ret * np.sqrt(252)

# 如果計算結果是 NaN (例如沒交易)，給它一個 0.0
    if np.isnan(sharpe_val):
        sharpe_val = 0.0
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100    
    if len(trade_profits) > 0:
        win_rate = np.mean(np.array(trade_profits) > 0) * 100
    else:
        win_rate = 0
    # --- 8. 在 Streamlit 上顯示儀表板 ---
    st.divider()
    st.header("📈 模型專業評估儀表板")

    # 建立三欄佈局
    col1, col2, col3 = st.columns(3)

    with col1:
        # 顯示 RMSE
        st.metric("RMSE (均方根誤差)", f"{rmse:.2f}")
        st.caption("數值越低，代表預測價格與實際股價越接近。")

    with col2:
        # 顯示方向準確度
        # 加上顏色箭頭：如果大於 50% 顯示綠色 (normal)，否則顯示紅色 (inverse)
        delta_val = f"{direction_acc - 50:.1f}%" if direction_acc != 0 else None
        st.metric(
            "方向準確度 (HIT Rate)",
            f"{direction_acc:.1f}%",
            delta=delta_val,
            delta_color="normal" if direction_acc >= 50 else "inverse"
        )
        st.caption("預測明天『漲跌方向』的成功率（>50% 具備參考價值）。")

    with col3:
        # 顯示夏普比率
        st.metric("夏普比率 (Sharpe Ratio)", f"{sharpe_val:.2f}")
        st.caption("風險調整後的報酬，通常 >1.0 代表策略表現良好。")

    # --- 額外增加：回測曲線圖 ---
    # 這能讓儀表板更完整，看到「跟著 AI 買」的累積損益
    st.subheader("💰 AI 策略模擬累積收益率")
    
    # 修正：直接使用 y_test_actual 作為大盤對照組

    # Clean cumulative returns to ensure all values are finite
    # Ensure inputs are 1D float arrays

    # Ensure all elements are float scalars (not arrays)
    def flatten_to_float(arr):
        return np.array([float(x) if np.ndim(x) == 0 else float(np.asarray(x).flatten()[0]) for x in arr])

    strategy_returns_clean = flatten_to_float(strategy_returns)
    y_test_actual_clean = flatten_to_float(y_test_actual)

    cumulative_strategy = (1 + pd.Series(strategy_returns_clean)).cumprod()
    cumulative_strategy = pd.Series(cumulative_strategy).replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
    cumulative_strategy = np.asarray(cumulative_strategy).astype(float).flatten()

    cumulative_actual = (1 + pd.Series(y_test_actual_clean)).cumprod()
    cumulative_actual = pd.Series(cumulative_actual).replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
    cumulative_actual = np.asarray(cumulative_actual).astype(float).flatten()

    fig_perf, ax_perf = plt.subplots(figsize=(10, 4))
    # 讓兩條線都從 1.0 開始（代表 100% 原始本金）
    ax_perf.plot(cumulative_actual, label="Market Return (Buy & Hold)", color="gray", alpha=0.5)
    ax_perf.plot(cumulative_strategy, label="AI Strategy Return", color="gold", linewidth=2)
    ax_perf.axhline(y=1.0, color='black', linestyle='--', alpha=0.3) # 增加一條 1.0 的基準線
    ax_perf.legend()
    ax_perf.set_ylabel("Cumulative Return (Multiple)")
    st.pyplot(fig_perf)

    # 顯示指標解釋
    with st.expander("💡 如何解讀這些指標？"):
        st.write("""
        1. **RMSE**: 衡量價格預測的「絕對誤差」，適合看預測值是否偏離現實太多。
        2. **方向準確度**: 這是實戰中最關鍵的指標。即使價格預測不準，只要「方向」對了就能獲利。
        3. **夏普比率**: 考慮了波動風險。如果數值很高但方向準確度低，可能代表該模型只是運氣好抓到了幾次大暴漲。
        """)
    st.subheader("📜 交易紀錄")
    if trade_log:
        trade_df = pd.DataFrame(trade_log, columns=["Date", "Action", "Price", "Cash Left", "Shares"])
        st.dataframe(trade_df)
    else:
        st.write("沒有產生交易訊號")
    # --- 修正：策略最終結果型態轉換 ---
    # 使用 np.array(x).item() 就像是把多層包裝的禮物拆開，直到剩下核心的純數字
    final_capital_val = float(np.array(final_capital).item())
    total_return_pct_val = float(np.array(total_return_pct).item())
    max_drawdown_val = float(np.array(max_drawdown).item())
    win_rate_val = float(np.array(win_rate).item())

    st.subheader("💰 策略最終結果")
    colA, colB = st.columns(2)

    with colA:
        # 顯示初始本金
        st.metric("初始本金", f"${initial_capital:,.0f}")

    with colB:
        # 顯示最終資金與總報酬變動
        st.metric(
            "最終資金", 
            f"${final_capital_val:,.0f}", 
            delta=f"{total_return_pct_val:.2f}%"
        )
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("最終報酬率", f"{total_return_pct_val:.2f}%")

    with col2:
        st.metric("最大回撤 (MDD)", f"{max_drawdown_val:.2f}%")

    with col3:
        st.metric("勝率 (Win Rate)", f"{win_rate_val:.2f}%")  
    # --- 預測明天 (修正型態問題) ---
    feature_count = scaled_data.shape[1] 
    last_window_data = scaled_data[-look_back:]
    last_window_tensor = torch.from_numpy(last_window_data).float().view(1, look_back, feature_count).to(device)  
    model.eval()
    with torch.no_grad():
        next_pred_raw = model(last_window_tensor).cpu().numpy()

    # 1. 轉回實際數值並用 .item() 轉成純 Python 數字
    # get_inverse_price 回傳的是陣列，[0] 取出第一個，.item() 確保它是純數字
    next_return_val = float(get_inverse_price(next_pred_raw)[0])
    # 2. 抓取最後一天的真實收盤價，同樣確保它是純數字
    # raw_close_prices[-1] 有時會是個陣列，強制轉成純 float
    last_actual_close = float(np.asarray(raw_close_prices[-1]).flatten()[0])
    # 3. 換算預測收盤價
    next_price_val = last_actual_close * (1 + next_return_val)

    # 4. 顯示結果
    st.divider()
    st.subheader("🔮 明日走勢預測")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        # 這裡就不會報錯了，因為 next_return_val 是數字
        st.metric("預測漲跌幅", f"{next_return_val * 100:.2f}%")
    with col_p2:
        # 這裡也不會報錯了，因為 next_price_val 是數字
        st.metric("預測目標收盤價", f"${next_price_val:.2f}")
