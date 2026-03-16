import yfinance as yf
import pandas as pd

def load_and_preprocess_data(ticker, start_date, end_date):
    """
    資料存取層：獲取數據並建構「AI 特徵」與「多因子過濾器」
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            return None

        # 🌟 【關鍵修復】解決 yfinance 新版雙層欄位 (MultiIndex) 導致的報錯
        if isinstance(df.columns, pd.MultiIndex):
            # 只保留第一層欄位名稱 (Close, Volume 等)，把股票代號那一層清掉
            df.columns = df.columns.get_level_values(0)

        # --- A. AI 模型所需的特徵 ---
        df['Return'] = df['Close'].pct_change()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['MA20'] = df['Close'].rolling(window=20).mean()

        # --- B. 策略核心：三維度多因子過濾器 (Multi-Factor Filter) ---
        
        # 1. 技術面 (Technical): 當日成交量大於 5日均量
        df['Vol_MA5'] = df['Volume'].rolling(window=5).mean()
        df['Tech_Pass'] = df['Volume'] > df['Vol_MA5']

        # 2. 基本面 (Fundamental): 連續3個月月營收 YOY > 20%
        # TODO: 未來串接 FinMind/TEJ 解決日月資料對齊後補上實質邏輯
        df['Fund_Pass'] = True 

        # 3. 籌碼面 (Chip): 外資連續買超
        # TODO: 未來串接 FinMind/TEJ 獲取三大法人數據後補上實質邏輯
        df['Chip_Pass'] = True 

        # 綜合評估：三個維度皆須通過，才允許進場
        df['Factor_Pass'] = df['Tech_Pass'] & df['Fund_Pass'] & df['Chip_Pass']

        # --- C. 清理缺失值 ---
        df = df.dropna()
        
        return df
        
    except Exception as e:
        print(f"獲取 {ticker} 數據時發生錯誤: {e}")
        return None