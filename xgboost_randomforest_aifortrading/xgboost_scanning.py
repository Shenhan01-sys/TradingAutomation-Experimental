import MetaTrader5 as mt5
import pandas as pd
import numpy as np

from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

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

modal_awal = 10000
risk_per_trade = 0.01

sl_multiplier = 1
reward_ratio = 1.46

future = 10
confidence = 0.63
komisi = 0.0
leverage = 100

max_allowed_drawdown = 30
min_total_trade = 50

min_adx = 20

confidence_list = np.arange(0.50, 0.86, 0.05)


# ==========================================
# AMBIL DATA
# ==========================================

rates = mt5.copy_rates_range(
    symbol,
    timeframe,
    start_date,
    end_date
)

df = pd.DataFrame(rates)

if df.empty:
    print("Data kosong. Cek symbol atau history MT5.")
    mt5.shutdown()
    quit()

df["time"] = pd.to_datetime(df["time"], unit="s")

print("Jumlah candle:", len(df))
print("Tanggal awal :", df["time"].iloc[0])
print("Tanggal akhir:", df["time"].iloc[-1])


# ==========================================
# INDIKATOR DASAR
# ==========================================

df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
df["ema100"] = df["close"].ewm(span=100, adjust=False).mean()

delta = df["close"].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)

gain = pd.Series(gain, index=df.index).rolling(14).mean()
loss = pd.Series(loss, index=df.index).rolling(14).mean()

rs = gain / (loss + 1e-9)
df["rsi"] = 100 - (100 / (1 + rs))

low14 = df["low"].rolling(14).min()
high14 = df["high"].rolling(14).max()

df["stoch_k"] = (
    (df["close"] - low14) /
    (high14 - low14 + 1e-9)
) * 100

high_low = df["high"] - df["low"]
high_close = abs(df["high"] - df["close"].shift())
low_close = abs(df["low"] - df["close"].shift())

true_range = pd.concat(
    [high_low, high_close, low_close],
    axis=1
).max(axis=1)

df["atr"] = true_range.rolling(14).mean()


# ==========================================
# ADX FILTER
# ==========================================

plus_dm = df["high"].diff()
minus_dm = -df["low"].diff()

plus_dm = np.where(
    (plus_dm > minus_dm) & (plus_dm > 0),
    plus_dm,
    0
)

minus_dm = np.where(
    (minus_dm > plus_dm) & (minus_dm > 0),
    minus_dm,
    0
)

plus_di = 100 * (
    pd.Series(plus_dm, index=df.index).rolling(14).mean()
    /
    (df["atr"] + 1e-9)
)

minus_di = 100 * (
    pd.Series(minus_dm, index=df.index).rolling(14).mean()
    /
    (df["atr"] + 1e-9)
)

dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)

df["adx"] = dx.rolling(14).mean()


# ==========================================
# FEATURE ENGINEERING
# ==========================================

df["body"] = abs(df["close"] - df["open"])
df["range"] = df["high"] - df["low"]

df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

df["body_ratio"] = df["body"] / df["range"].replace(0, np.nan)
df["upper_wick_ratio"] = df["upper_wick"] / df["range"].replace(0, np.nan)
df["lower_wick_ratio"] = df["lower_wick"] / df["range"].replace(0, np.nan)

df["momentum_3"] = df["close"] - df["close"].shift(3)
df["momentum_5"] = df["close"] - df["close"].shift(5)
df["momentum_10"] = df["close"] - df["close"].shift(10)

df["momentum_3_atr"] = df["momentum_3"] / (df["atr"] + 1e-9)
df["momentum_5_atr"] = df["momentum_5"] / (df["atr"] + 1e-9)
df["momentum_10_atr"] = df["momentum_10"] / (df["atr"] + 1e-9)

df["ema_slope_5"] = df["ema20"] - df["ema20"].shift(5)
df["ema_slope_10"] = df["ema20"] - df["ema20"].shift(10)

df["ema_gap_20_50"] = (df["ema20"] - df["ema50"]) / (df["ema50"] + 1e-9)
df["ema_gap_50_100"] = (df["ema50"] - df["ema100"]) / (df["ema100"] + 1e-9)

df["distance_ema20"] = (df["close"] - df["ema20"]) / (df["ema20"] + 1e-9)
df["distance_ema50"] = (df["close"] - df["ema50"]) / (df["ema50"] + 1e-9)
df["distance_ema100"] = (df["close"] - df["ema100"]) / (df["ema100"] + 1e-9)

df["volatility_10"] = df["close"].rolling(10).std()
df["volatility_20"] = df["close"].rolling(20).std()
df["volatility_ratio"] = df["volatility_10"] / (df["volatility_20"] + 1e-9)

df["return_1"] = df["close"].pct_change(1)
df["return_3"] = df["close"].pct_change(3)
df["return_5"] = df["close"].pct_change(5)
df["return_10"] = df["close"].pct_change(10)

df["volume_change"] = df["tick_volume"].pct_change()
df["volume_ma20"] = df["tick_volume"].rolling(20).mean()
df["volume_ratio"] = df["tick_volume"] / (df["volume_ma20"] + 1e-9)

df["ema_cross"] = (df["ema20"] > df["ema50"]).astype(int)

df["price_above_ema20"] = (df["close"] > df["ema20"]).astype(int)
df["price_above_ema50"] = (df["close"] > df["ema50"]).astype(int)
df["price_above_ema100"] = (df["close"] > df["ema100"]).astype(int)

df["rsi_above_50"] = (df["rsi"] > 50).astype(int)
df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
df["rsi_oversold"] = (df["rsi"] < 30).astype(int)

df["stoch_above_50"] = (df["stoch_k"] > 50).astype(int)
df["stoch_overbought"] = (df["stoch_k"] > 80).astype(int)
df["stoch_oversold"] = (df["stoch_k"] < 20).astype(int)

# Market session
df["hour"] = df["time"].dt.hour

df["asia_session"] = (
    (df["hour"] >= 0) &
    (df["hour"] < 8)
).astype(int)

df["london_session"] = (
    (df["hour"] >= 8) &
    (df["hour"] < 16)
).astype(int)

df["newyork_session"] = (
    (df["hour"] >= 16) &
    (df["hour"] < 24)
).astype(int)


# ==========================================
# TARGET BERDASARKAN SL TP
# ==========================================

targets = []

for i in range(len(df)):

    if i + future >= len(df):
        targets.append(np.nan)
        continue

    entry = df["close"].iloc[i]
    atr = df["atr"].iloc[i]

    if pd.isna(atr) or atr <= 0:
        targets.append(np.nan)
        continue

    future_data = df.iloc[i + 1:i + future + 1]

    buy_sl = entry - (atr * sl_multiplier)
    buy_tp = entry + (atr * reward_ratio)

    buy_hit_tp = future_data[future_data["high"] >= buy_tp]
    buy_hit_sl = future_data[future_data["low"] <= buy_sl]

    buy_win = False

    if not buy_hit_tp.empty and not buy_hit_sl.empty:
        buy_win = buy_hit_tp.index[0] < buy_hit_sl.index[0]
    elif not buy_hit_tp.empty:
        buy_win = True

    sell_sl = entry + (atr * sl_multiplier)
    sell_tp = entry - (atr * reward_ratio)

    sell_hit_tp = future_data[future_data["low"] <= sell_tp]
    sell_hit_sl = future_data[future_data["high"] >= sell_sl]

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

df["target"] = targets


# ==========================================
# BERSIHKAN DATA
# ==========================================

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df = df[df["target"].isin([0, 1])]

if df.empty:
    print("Data habis setelah filter target.")
    mt5.shutdown()
    quit()


# ==========================================
# FEATURE LIST
# ==========================================

features = [
    "ema20", "ema50", "ema100",
    "rsi", "stoch_k", "atr", "adx",

    "body", "range",
    "upper_wick", "lower_wick",
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio",

    "momentum_3", "momentum_5", "momentum_10",
    "momentum_3_atr", "momentum_5_atr", "momentum_10_atr",

    "ema_slope_5", "ema_slope_10",
    "ema_gap_20_50", "ema_gap_50_100",
    "distance_ema20", "distance_ema50", "distance_ema100",

    "volatility_10", "volatility_20", "volatility_ratio",

    "return_1", "return_3", "return_5", "return_10",

    "volume_change", "volume_ratio",

    "ema_cross",
    "price_above_ema20", "price_above_ema50", "price_above_ema100",

    "rsi_above_50", "rsi_overbought", "rsi_oversold",

    "stoch_above_50", "stoch_overbought", "stoch_oversold",

    "hour", "asia_session", "london_session", "newyork_session"
]

X = df[features]
y = df["target"].astype(int)


# ==========================================
# TIME SERIES SPLIT VALIDATION
# ==========================================

print()
print("===================================")
print("TIME SERIES SPLIT VALIDATION")
print("===================================")

tscv = TimeSeriesSplit(n_splits=5)
accuracy_list = []

for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):

    X_train_cv = X.iloc[train_index]
    X_test_cv = X.iloc[test_index]

    y_train_cv = y.iloc[train_index]
    y_test_cv = y.iloc[test_index]

    jumlah_class_0_cv = (y_train_cv == 0).sum()
    jumlah_class_1_cv = (y_train_cv == 1).sum()

    if jumlah_class_1_cv == 0:
        scale_pos_weight_cv = 1
    else:
        scale_pos_weight_cv = jumlah_class_0_cv / jumlah_class_1_cv

    model_cv = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.025,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.2,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight_cv,
        random_state=42,
        n_jobs=-1
    )

    model_cv.fit(X_train_cv, y_train_cv)

    pred_cv = model_cv.predict(X_test_cv)

    acc_cv = accuracy_score(y_test_cv, pred_cv)

    accuracy_list.append(acc_cv)

    print(f"Fold {fold} Accuracy: {acc_cv * 100:.2f}%")

print("Rata-rata Accuracy:", round(np.mean(accuracy_list) * 100, 2), "%")


# ==========================================
# TRAIN TEST SPLIT FINAL
# ==========================================

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# ==========================================
# XGBOOST MODEL FINAL
# ==========================================

jumlah_class_0 = (y_train == 0).sum()
jumlah_class_1 = (y_train == 1).sum()

if jumlah_class_1 == 0:
    scale_pos_weight = 1
else:
    scale_pos_weight = jumlah_class_0 / jumlah_class_1

model = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.025,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.2,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

pred_test = model.predict(X_test)

print()
print("Accuracy XGBoost Test Data:")
print(round(accuracy_score(y_test, pred_test) * 100, 2), "%")


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance_df = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print()
print("===================================")
print("TOP 15 FEATURE TERPENTING XGBOOST")
print("===================================")
print(importance_df.head(15).to_string(index=False))

importance_df.to_csv("feature_importance_xgboost.csv", index=False)


# ==========================================
# PREDICTION + SCANNING CONFIDENCE
# ==========================================

probability = model.predict_proba(X_test)

sell_probability = probability[:, 0]
buy_probability = probability[:, 1]

test_df = df.iloc[split_index:].copy()
test_df["buy_probability"] = buy_probability
test_df["sell_probability"] = sell_probability


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

        # FILTER ANTI OVERTRADE
        if row["adx"] < min_adx:
            i += 1
            continue

        if row["buy_probability"] >= buy_conf:
            signal = 1
        elif row["sell_probability"] >= sell_conf:
            signal = 0
        else:
            i += 1
            continue

        entry = row["close"]
        atr = row["atr"]

        if pd.isna(atr) or atr <= 0:
            i += 1
            continue

        risk_amount = saldo_temp * risk_per_trade
        next_data = data.iloc[i + 1:i + future + 1]

        result = None

        if signal == 1:

            sl = entry - (atr * sl_multiplier)
            tp = entry + (atr * reward_ratio)

            hit_tp = next_data[next_data["high"] >= tp]
            hit_sl = next_data[next_data["low"] <= sl]

            if not hit_tp.empty and not hit_sl.empty:
                result = "WIN" if hit_tp.index[0] < hit_sl.index[0] else "LOSS"
            elif not hit_tp.empty:
                result = "WIN"
            elif not hit_sl.empty:
                result = "LOSS"

        else:

            sl = entry + (atr * sl_multiplier)
            tp = entry - (atr * reward_ratio)

            hit_tp = next_data[next_data["low"] <= tp]
            hit_sl = next_data[next_data["high"] >= sl]

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


optimization_results = []

print()
print("===================================")
print("MULAI SCANNING CONFIDENCE XGBOOST")
print("===================================")

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
    print("HASIL SCANNING CONFIDENCE XGBOOST")
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

    optimization_df.to_csv("optimization_confidence_xgboost.csv", index=False)


# ==========================================
# PREDICTION DARI CONFIDENCE TERBAIK
# ==========================================

predictions = np.where(
    (buy_probability >= best_buy_confidence) &
    (test_df["adx"].values >= min_adx),
    1,
    np.where(
        (sell_probability >= best_sell_confidence) &
        (test_df["adx"].values >= min_adx),
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

    print("Accuracy XGBoost khusus signal valid:")
    print(round(accuracy * 100, 2), "%")
    print("Jumlah signal valid:", valid_index.sum())

else:
    print("Tidak ada signal valid.")

test_df["prediction"] = predictions


# ==========================================
# DATA UNTUK BACKTESTING.PY
# ==========================================

bt_data = test_df.copy()

bt_data = bt_data.rename(columns={
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "tick_volume": "Volume"
})

bt_data.set_index("time", inplace=True)

bt_data = bt_data[
    [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "prediction",
        "atr",
        "adx",
        "buy_probability",
        "sell_probability"
    ]
]

bt_data.replace([np.inf, -np.inf], np.nan, inplace=True)

bt_data.dropna(
    subset=[
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "atr",
        "adx",
        "buy_probability",
        "sell_probability"
    ],
    inplace=True
)

print()
print("===================================")
print("CEK DATA UNTUK BACKTESTING.PY")
print("===================================")
print("Jumlah data bt_data :", len(bt_data))
print("Jumlah sinyal BUY   :", int((bt_data["prediction"] == 1).sum()))
print("Jumlah sinyal SELL  :", int((bt_data["prediction"] == 0).sum()))
print("Jumlah NO TRADE     :", int(bt_data["prediction"].isna().sum()))
print("===================================")


# ==========================================
# STRATEGY BACKTESTING.PY
# ==========================================

class TradingStrategyXGBoost(Strategy):

    reward_ratio_param = reward_ratio
    sl_multiplier_param = sl_multiplier
    risk_per_trade_param = risk_per_trade

    def init(self):
        pass

    def next(self):

        signal = self.data.prediction[-1]
        atr = self.data.atr[-1]
        adx = self.data.adx[-1]
        price = self.data.Close[-1]

        if np.isnan(signal):
            return

        if atr <= 0:
            return

        if adx < min_adx:
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
        TradingStrategyXGBoost,
        cash=modal_awal,
        commission=komisi,
        margin=1 / leverage,
        exclusive_orders=True,
        finalize_trades=True
    )

    stats = bt.run()

    print()
    print("===================================")
    print("SUMMARY BACKTESTING.PY XGBOOST")
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

latest_adx = df["adx"].iloc[-1]

print()
print("===================================")
print("SIGNAL TERBARU XGBOOST")
print("===================================")

if latest_adx < min_adx:
    print("SIGNAL SEKARANG = NO TRADE")
    print(f"Alasan: ADX rendah ({latest_adx:.2f})")

elif latest_buy_prob >= best_buy_confidence:
    print(f"SIGNAL SEKARANG = BUY ({latest_buy_prob:.2%})")

elif latest_sell_prob >= best_sell_confidence:
    print(f"SIGNAL SEKARANG = SELL ({latest_sell_prob:.2%})")

else:
    print("SIGNAL SEKARANG = NO TRADE")

print(f"BUY Probability          : {latest_buy_prob:.2%}")
print(f"SELL Probability         : {latest_sell_prob:.2%}")
print(f"ADX                      : {latest_adx:.2f}")
print(f"MIN ADX                  : {min_adx}")
print(f"BUY Confidence dipakai   : {best_buy_confidence:.2f}")
print(f"SELL Confidence dipakai  : {best_sell_confidence:.2f}")
print(f"Modal testing            : {modal_awal} USC")
print("===================================")


# ==========================================
# SHUTDOWN MT5
# ==========================================

mt5.shutdown()