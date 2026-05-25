import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime,timedelta

mt5.initialize()

symbol="XAUUSDc"

start_date = datetime(2026, 4, 23)
end_date = datetime.now()

rates = mt5.copy_rates_range(
    symbol,
    mt5.TIMEFRAME_M1,
    start_date,
    end_date
)

df=pd.DataFrame(rates)

print("Jumlah candle:",len(df))

if not df.empty:

    df['time']=pd.to_datetime(
        df['time'],
        unit='s'
    )

    print(df['time'].min())
    print(df['time'].max())

mt5.shutdown()