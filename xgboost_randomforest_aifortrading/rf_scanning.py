import MetaTrader5 as mt5
import pandas as pd
import numpy as np

from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from backtesting import Backtest, Strategy

# ==========================================
# CONNECT MT5
# ==========================================

if not mt5.initialize():
    print("MT5 gagal connect")
    quit()

print("MT5 berhasil connect")

# ==========================================
# SETTING
# ==========================================

symbol = "XAUUSDc"
timeframe = mt5.TIMEFRAME_M1

start_date = datetime(2026, 2, 15, 17, 0)
end_date = datetime.now()

# Modal akun cent
modal_awal = 10000  # USC

# Risk lebih aman
risk_per_trade = 0.01  # 1%

sl_multiplier = 1
reward_ratio = 1.46

future = 10
confidence = 0.63
komisi = 0.0

leverage = 100

# Filter optimasi
max_allowed_drawdown = 30
min_total_trade = 50

# ==========================================
# AMBIL DATA
# ==========================================

rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
df = pd.DataFrame(rates)

if df.empty:
    print("Data kosong. Cek symbol atau history MT5.")
    mt5.shutdown()
    quit()

df['time'] = pd.to_datetime(df['time'], unit='s')

print("Jumlah candle:", len(df))
print("Tanggal awal :", df['time'].iloc[0])
print("Tanggal akhir:", df['time'].iloc[-1])

# ==========================================
# INDIKATOR DASAR
# ==========================================

df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()

delta = df['close'].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)

gain = pd.Series(gain, index=df.index).rolling(14).mean()
loss = pd.Series(loss, index=df.index).rolling(14).mean()

rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

low14 = df['low'].rolling(14).min()
high14 = df['high'].rolling(14).max()
df['stoch_k'] = ((df['close'] - low14) / (high14 - low14)) * 100

high_low = df['high'] - df['low']
high_close = abs(df['high'] - df['close'].shift())
low_close = abs(df['low'] - df['close'].shift())

true_range = pd.concat(
    [high_low, high_close, low_close],
    axis=1
).max(axis=1)

df['atr'] = true_range.rolling(14).mean()

# ==========================================
# FEATURE ENGINEERING
# ==========================================

df['body'] = abs(df['close'] - df['open'])
df['range'] = df['high'] - df['low']

df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']

df['body_ratio'] = df['body'] / df['range'].replace(0, np.nan)
df['upper_wick_ratio'] = df['upper_wick'] / df['range'].replace(0, np.nan)
df['lower_wick_ratio'] = df['lower_wick'] / df['range'].replace(0, np.nan)

df['momentum_3'] = df['close'] - df['close'].shift(3)
df['momentum_5'] = df['close'] - df['close'].shift(5)
df['momentum_10'] = df['close'] - df['close'].shift(10)

df['momentum_3_atr'] = df['momentum_3'] / df['atr']
df['momentum_5_atr'] = df['momentum_5'] / df['atr']
df['momentum_10_atr'] = df['momentum_10'] / df['atr']

df['ema_slope_5'] = df['ema20'] - df['ema20'].shift(5)
df['ema_slope_10'] = df['ema20'] - df['ema20'].shift(10)

df['ema_gap_20_50'] = (df['ema20'] - df['ema50']) / df['ema50']
df['ema_gap_50_100'] = (df['ema50'] - df['ema100']) / df['ema100']

df['distance_ema20'] = (df['close'] - df['ema20']) / df['ema20']
df['distance_ema50'] = (df['close'] - df['ema50']) / df['ema50']
df['distance_ema100'] = (df['close'] - df['ema100']) / df['ema100']

df['volatility_10'] = df['close'].rolling(10).std()
df['volatility_20'] = df['close'].rolling(20).std()
df['volatility_ratio'] = df['volatility_10'] / df['volatility_20']

df['return_1'] = df['close'].pct_change(1)
df['return_3'] = df['close'].pct_change(3)
df['return_5'] = df['close'].pct_change(5)
df['return_10'] = df['close'].pct_change(10)

df['volume_change'] = df['tick_volume'].pct_change()
df['volume_ma20'] = df['tick_volume'].rolling(20).mean()
df['volume_ratio'] = df['tick_volume'] / df['volume_ma20']

df['ema_cross'] = (df['ema20'] > df['ema50']).astype(int)
df['price_above_ema20'] = (df['close'] > df['ema20']).astype(int)
df['price_above_ema50'] = (df['close'] > df['ema50']).astype(int)
df['price_above_ema100'] = (df['close'] > df['ema100']).astype(int)

df['rsi_above_50'] = (df['rsi'] > 50).astype(int)
df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
df['rsi_oversold'] = (df['rsi'] < 30).astype(int)

df['stoch_above_50'] = (df['stoch_k'] > 50).astype(int)
df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)

# ==========================================
# TARGET BERDASARKAN SL TP
# ==========================================

targets = []

for i in range(len(df)):

    if i + future >= len(df):
        targets.append(np.nan)
        continue

    entry = df['close'].iloc[i]
    atr = df['atr'].iloc[i]

    if pd.isna(atr) or atr <= 0:
        targets.append(np.nan)
        continue

    future_data = df.iloc[i + 1:i + future + 1]

    buy_sl = entry - (atr * sl_multiplier)
    buy_tp = entry + (atr * reward_ratio)

    buy_hit_tp = future_data[future_data['high'] >= buy_tp]
    buy_hit_sl = future_data[future_data['low'] <= buy_sl]

    buy_win = False

    if not buy_hit_tp.empty and not buy_hit_sl.empty:
        buy_win = buy_hit_tp.index[0] < buy_hit_sl.index[0]
    elif not buy_hit_tp.empty:
        buy_win = True

    sell_sl = entry + (atr * sl_multiplier)
    sell_tp = entry - (atr * reward_ratio)

    sell_hit_tp = future_data[future_data['low'] <= sell_tp]
    sell_hit_sl = future_data[future_data['high'] >= sell_sl]

    sell_win = False

    if not sell_hit_tp.empty and not sell_hit_sl.empty:
        sell_win = sell_hit_tp.index[0] < sell_hit_sl.index[0]
    elif not sell_hit_tp.empty:
        sell_win = True

    if buy_win and not sell_win:
        targets.append(1)
    elif sell_win and not buy_win:
        targets.append(0)
    else:
        targets.append(np.nan)

df['target'] = targets

# ==========================================
# BERSIHKAN DATA
# ==========================================

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df = df[df['target'].isin([0, 1])]

if df.empty:
    print("Data habis setelah filter target.")
    mt5.shutdown()
    quit()

# ==========================================
# FEATURE LIST
# ==========================================

features = [
    'ema20', 'ema50', 'ema100',
    'rsi', 'stoch_k', 'atr',
    'body', 'range',
    'upper_wick', 'lower_wick',
    'body_ratio', 'upper_wick_ratio', 'lower_wick_ratio',
    'momentum_3', 'momentum_5', 'momentum_10',
    'momentum_3_atr', 'momentum_5_atr', 'momentum_10_atr',
    'ema_slope_5', 'ema_slope_10',
    'ema_gap_20_50', 'ema_gap_50_100',
    'distance_ema20', 'distance_ema50', 'distance_ema100',
    'volatility_10', 'volatility_20', 'volatility_ratio',
    'return_1', 'return_3', 'return_5', 'return_10',
    'volume_change', 'volume_ratio',
    'ema_cross',
    'price_above_ema20', 'price_above_ema50', 'price_above_ema100',
    'rsi_above_50', 'rsi_overbought', 'rsi_oversold',
    'stoch_above_50', 'stoch_overbought', 'stoch_oversold'
]

X = df[features]
y = df['target'].astype(int)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# ==========================================
# RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=8,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance_df = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)

print()
print("===================================")
print("TOP 15 FEATURE TERPENTING")
print("===================================")
print(importance_df.head(15).to_string(index=False))

importance_df.to_csv("feature_importance_randomforest.csv", index=False)

# ==========================================
# PREDICTION + OPTIMASI CONFIDENCE BARU
# ==========================================

probability = model.predict_proba(X_test)

sell_probability = probability[:, 0]
buy_probability = probability[:, 1]

test_df = df.iloc[split_index:].copy()
test_df['buy_probability'] = buy_probability
test_df['sell_probability'] = sell_probability


def manual_backtest_dual_confidence(data, buy_conf, sell_conf):

    saldo_temp = modal_awal
    peak_saldo = modal_awal
    max_drawdown = 0

    win_temp = 0
    loss_temp = 0
    total_trade_temp = 0

    buy_trade = 0
    sell_trade = 0

    profit_total_temp = 0
    loss_total_temp = 0

    i = 0

    while i < len(data) - future:

        row = data.iloc[i]

        if row['buy_probability'] >= buy_conf:
            signal = 1
        elif row['sell_probability'] >= sell_conf:
            signal = 0
        else:
            i += 1
            continue

        entry = row['close']
        atr = row['atr']

        if pd.isna(atr) or atr <= 0:
            i += 1
            continue

        risk_amount = saldo_temp * risk_per_trade
        next_data = data.iloc[i + 1:i + future + 1]

        result = None

        if signal == 1:
            sl = entry - (atr * sl_multiplier)
            tp = entry + (atr * reward_ratio)

            hit_tp = next_data[next_data['high'] >= tp]
            hit_sl = next_data[next_data['low'] <= sl]

            if not hit_tp.empty and not hit_sl.empty:
                result = "WIN" if hit_tp.index[0] < hit_sl.index[0] else "LOSS"
            elif not hit_tp.empty:
                result = "WIN"
            elif not hit_sl.empty:
                result = "LOSS"

        else:
            sl = entry + (atr * sl_multiplier)
            tp = entry - (atr * reward_ratio)

            hit_tp = next_data[next_data['low'] <= tp]
            hit_sl = next_data[next_data['high'] >= sl]

            if not hit_tp.empty and not hit_sl.empty:
                result = "WIN" if hit_tp.index[0] < hit_sl.index[0] else "LOSS"
            elif not hit_tp.empty:
                result = "WIN"
            elif not hit_sl.empty:
                result = "LOSS"

        if result == "WIN":
            profit = risk_amount * reward_ratio
            saldo_temp += profit
            profit_total_temp += profit
            win_temp += 1
            total_trade_temp += 1

            if signal == 1:
                buy_trade += 1
            else:
                sell_trade += 1

            i += future

        elif result == "LOSS":
            loss_amount = risk_amount
            saldo_temp -= loss_amount
            loss_total_temp += loss_amount
            loss_temp += 1
            total_trade_temp += 1

            if signal == 1:
                buy_trade += 1
            else:
                sell_trade += 1

            i += future

        else:
            i += 1
            continue

        if saldo_temp > peak_saldo:
            peak_saldo = saldo_temp

        drawdown_now = ((peak_saldo - saldo_temp) / peak_saldo) * 100
        if drawdown_now > max_drawdown:
            max_drawdown = drawdown_now

    winrate_temp = (win_temp / total_trade_temp) * 100 if total_trade_temp > 0 else 0
    profit_factor_temp = profit_total_temp / loss_total_temp if loss_total_temp > 0 else 0
    net_profit_temp = saldo_temp - modal_awal
    roi_temp = (net_profit_temp / modal_awal) * 100

    score = 0

    if total_trade_temp > 0:
        score += profit_factor_temp * 40
        score += winrate_temp * 0.5
        score += roi_temp * 0.2
        score -= max_drawdown * 1.5

        if buy_trade > 0 and sell_trade > 0:
            score += 20

    return {
        "buy_conf": buy_conf,
        "sell_conf": sell_conf,
        "total_trade": total_trade_temp,
        "buy_trade": buy_trade,
        "sell_trade": sell_trade,
        "win": win_temp,
        "loss": loss_temp,
        "winrate": winrate_temp,
        "profit_factor": profit_factor_temp,
        "max_drawdown": max_drawdown,
        "net_profit": net_profit_temp,
        "roi": roi_temp,
        "saldo_akhir": saldo_temp,
        "score": score
    }


confidence_list = np.arange(0.50, 0.86, 0.01)

optimization_results = []

for buy_conf in confidence_list:
    for sell_conf in confidence_list:

        result = manual_backtest_dual_confidence(
            test_df,
            buy_conf,
            sell_conf
        )

        if result["total_trade"] < min_total_trade:
            continue

        if result["profit_factor"] < 1.10:
            continue

        if result["max_drawdown"] > max_allowed_drawdown:
            continue

        optimization_results.append(result)

optimization_df = pd.DataFrame(optimization_results)

if optimization_df.empty:
    print()
    print("Tidak ada hasil optimasi yang memenuhi filter aman.")
    print("Pakai confidence default.")

    best_buy_confidence = confidence
    best_sell_confidence = confidence

else:
    optimization_df = optimization_df.sort_values(
        by=["score", "profit_factor", "roi"],
        ascending=[False, False, False]
    )

    best_result = optimization_df.iloc[0]

    best_buy_confidence = float(best_result["buy_conf"])
    best_sell_confidence = float(best_result["sell_conf"])

    print()
    print("===================================")
    print("HASIL OPTIMASI CONFIDENCE BARU")
    print("===================================")
    print(optimization_df.head(10).to_string(index=False))
    print("===================================")
    print(f"Best BUY confidence  : {best_buy_confidence:.2f}")
    print(f"Best SELL confidence : {best_sell_confidence:.2f}")
    print(f"Total Trade          : {int(best_result['total_trade'])}")
    print(f"BUY Trade            : {int(best_result['buy_trade'])}")
    print(f"SELL Trade           : {int(best_result['sell_trade'])}")
    print(f"Winrate              : {best_result['winrate']:.2f}%")
    print(f"Profit Factor        : {best_result['profit_factor']:.2f}")
    print(f"Max Drawdown         : {best_result['max_drawdown']:.2f}%")
    print(f"ROI                  : {best_result['roi']:.2f}%")
    print(f"Saldo akhir simulasi : {best_result['saldo_akhir']:.2f} USC")
    print("===================================")

    optimization_df.to_csv("optimization_confidence_balanced.csv", index=False)


predictions = np.where(
    buy_probability >= best_buy_confidence,
    1,
    np.where(
        sell_probability >= best_sell_confidence,
        0,
        np.nan
    )
)

valid_index = ~np.isnan(predictions)

print()

if valid_index.sum() > 0:
    accuracy = accuracy_score(
        y_test.iloc[valid_index],
        predictions[valid_index]
    )

    print("Accuracy Random Forest khusus signal valid:")
    print(round(accuracy * 100, 2), "%")
    print("Jumlah signal valid:", valid_index.sum())
else:
    print("Tidak ada signal valid.")

test_df['prediction'] = predictions

# ==========================================
# DATA UNTUK BACKTESTING.PY
# ==========================================

bt_data = test_df.copy()

bt_data = bt_data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'tick_volume': 'Volume'
})

bt_data.set_index('time', inplace=True)

bt_data = bt_data[
    [
        'Open',
        'High',
        'Low',
        'Close',
        'Volume',
        'prediction',
        'atr',
        'buy_probability',
        'sell_probability'
    ]
]

bt_data.replace([np.inf, -np.inf], np.nan, inplace=True)

bt_data.dropna(
    subset=[
        'Open',
        'High',
        'Low',
        'Close',
        'Volume',
        'atr',
        'buy_probability',
        'sell_probability'
    ],
    inplace=True
)

print()
print("===================================")
print("CEK DATA UNTUK BACKTESTING.PY")
print("===================================")
print("Jumlah data bt_data :", len(bt_data))
print("Jumlah sinyal BUY   :", int((bt_data['prediction'] == 1).sum()))
print("Jumlah sinyal SELL  :", int((bt_data['prediction'] == 0).sum()))
print("Jumlah NO TRADE     :", int(bt_data['prediction'].isna().sum()))
print("===================================")

# ==========================================
# STRATEGY BACKTESTING.PY
# ==========================================

class TradingStrategyRF(Strategy):

    reward_ratio_param = reward_ratio
    sl_multiplier_param = sl_multiplier
    risk_per_trade_param = risk_per_trade

    def init(self):
        pass

    def next(self):

        signal = self.data.prediction[-1]
        atr = self.data.atr[-1]
        price = self.data.Close[-1]

        if np.isnan(signal):
            return

        if atr <= 0:
            return

        if self.position:
            return

        risk_amount = self.equity * self.risk_per_trade_param
        sl_distance = atr * self.sl_multiplier_param

        if sl_distance <= 0:
            return

        size_by_risk = risk_amount / sl_distance
        max_size_by_margin = (self.equity * leverage * 0.95) / price

        size = int(min(size_by_risk, max_size_by_margin))

        if size < 1:
            return

        if signal == 1:

            sl = price - sl_distance
            tp = price + (atr * self.reward_ratio_param)

            if sl >= price or tp <= price:
                return

            self.buy(
                size=size,
                sl=sl,
                tp=tp
            )

        elif signal == 0:

            sl = price + sl_distance
            tp = price - (atr * self.reward_ratio_param)

            if sl <= price or tp >= price:
                return

            self.sell(
                size=size,
                sl=sl,
                tp=tp
            )

# ==========================================
# RUN BACKTESTING.PY
# ==========================================

if bt_data.empty:
    print()
    print("Data backtesting kosong.")
else:

    bt = Backtest(
        bt_data,
        TradingStrategyRF,
        cash=modal_awal,
        commission=komisi,
        margin=1 / leverage,
        exclusive_orders=True,
        finalize_trades=True
    )

    stats = bt.run()

    print()
    print("===================================")
    print("SUMMARY BACKTESTING.PY")
    print("===================================")
    print(stats)
    print("===================================")

    bt.plot()

# ==========================================
# SIGNAL TERBARU
# ==========================================

latest_data = X.iloc[-1:]
latest_prob = model.predict_proba(latest_data)

latest_sell_prob = latest_prob[0][0]
latest_buy_prob = latest_prob[0][1]

print()

if latest_buy_prob >= best_buy_confidence:
    print(f"SIGNAL SEKARANG = BUY ({latest_buy_prob:.2%})")
elif latest_sell_prob >= best_sell_confidence:
    print(f"SIGNAL SEKARANG = SELL ({latest_sell_prob:.2%})")
else:
    print("SIGNAL SEKARANG = NO TRADE")

print(f"BUY Confidence yang dipakai : {best_buy_confidence:.2f}")
print(f"SELL Confidence yang dipakai: {best_sell_confidence:.2f}")
print(f"Modal testing               : {modal_awal} USC")

# ==========================================
# SHUTDOWN MT5
# ==========================================

mt5.shutdown()