# Імпорт необхідних бібліотек
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import math
import logging
from binance import AsyncClient
from binance.exceptions import BinanceAPIException
from binance.enums import *
import warnings
warnings.filterwarnings("ignore")

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Підключення до Binance API
api_key = ""
api_secret = ""

# Символ для торгівлі
symbol = 'BTCUSDT'

# Глобальні параметри
leverage = 10
stop_loss_percent = 0.02
target_profit_percent = 0.05
trailing_stop_percent = 0.02
entry_threshold = 70  # 70% для входу
quantity_percentage = 0.2  # Використовуємо 20% від балансу

# Глобальна змінна для клієнта
client = None

# Функція для отримання історичних даних з Binance
async def get_binance_data(client, interval='1m', limit=1000):
    """
    Отримує історичні дані kline з Binance.
    """
    try:
        klines = await client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        data = pd.DataFrame(klines, columns=[
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ])
        data = data[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]
        data.rename(columns={'Open time': 'Timestamp'}, inplace=True)
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='ms')
        data[['Open', 'High', 'Low', 'Close', 'Volume']] = data[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        logger.info(f"Історичні дані ({interval}) завантажено успішно.")
        print(f"Історичні дані ({interval}) завантажено успішно.")
        return data
    except BinanceAPIException as e:
        logger.error(f"Помилка при отриманні даних з Binance: {e}")
        print(f"Помилка при отриманні даних з Binance: {e}")
        return pd.DataFrame()

# Функція для обчислення ATR
def calculate_atr(data, period=14):
    """
    Розраховує ATR для оцінки волатильності.
    """
    data['TR'] = np.maximum(data['High'] - data['Low'],
                            np.maximum(abs(data['High'] - data['Close'].shift(1)),
                                       abs(data['Low'] - data['Close'].shift(1))))
    data['ATR'] = data['TR'].rolling(window=period).mean()
    data.dropna(inplace=True)
    logger.info("ATR розраховано успішно.")
    print("ATR розраховано успішно.")
    return data

# Функція для обчислення MACD
def calculate_macd(data):
    """
    Розраховує MACD і сигнальну лінію.
    """
    fast_ema = data['Close'].ewm(span=12, adjust=False).mean()
    slow_ema = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = fast_ema - slow_ema
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    logger.info("MACD розраховано успішно.")
    print("MACD розраховано успішно.")
    return data

# Функція для обчислення індикаторів
def calculate_indicators(data):
    """
    Обчислює всі необхідні індикатори і додає їх до даних.
    """
    # Просте ковзне середнє (SMA)
    data['MA7'] = data['Close'].rolling(window=7).mean()
    data['MA25'] = data['Close'].rolling(window=25).mean()

    # Експоненціальне ковзне середнє (EMA)
    data['EMA7'] = data['Close'].ewm(span=7, adjust=False).mean()
    data['EMA25'] = data['Close'].ewm(span=25, adjust=False).mean()

    # Індекс відносної сили (RSI)
    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    avg_gain = up.rolling(window=14).mean()
    avg_loss = down.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # ATR
    data = calculate_atr(data)

    # MACD
    data = calculate_macd(data)

    # Видалити рядки з NaN значеннями
    data.dropna(inplace=True)
    logger.info("Індикатори розраховані і додані до даних.")
    print("Індикатори розраховані і додані до даних.")
    return data

# Функція для визначення напрямку та сили тренду
def determine_trend(data):
    """
    Визначає напрямок і силу тренду.
    """
    data['Trend'] = data['MA25'].diff()
    if data['Trend'].iloc[-1] > 0:
        trend_direction = 'uptrend'
    elif data['Trend'].iloc[-1] < 0:
        trend_direction = 'downtrend'
    else:
        trend_direction = 'sideways'

    trend_strength = abs(data['Trend'].iloc[-1])
    logger.info(f"Напрямок тренду: {trend_direction}, сила тренду: {trend_strength}")
    print(f"Напрямок тренду: {trend_direction}, сила тренду: {trend_strength}")
    return trend_direction, trend_strength

# Функція для відкриття позиції
async def open_position(client, symbol, side, quantity, leverage, entry_price):
    """
    Відкриває позицію на Binance Futures і встановлює стоп-лос та трейлінг-стоп.
    """
    try:
        await client.futures_change_leverage(symbol=symbol, leverage=leverage)
        position_side = 'LONG' if side == SIDE_BUY else 'SHORT'
        order_params = {
            'symbol': symbol,
            'side': side,
            'type': FUTURE_ORDER_TYPE_MARKET,
            'quantity': quantity
        }
        order = await client.futures_create_order(**order_params)
        logger.info(f"Позиція {position_side} відкрита: {order}")
        print(f"Позиція {position_side} відкрита: {order}")

        # Встановлюємо стоп-лос
        stop_loss_price = entry_price * (1 - stop_loss_percent) if side == SIDE_BUY else entry_price * (1 + stop_loss_percent)
        stop_loss_params = {
            'symbol': symbol,
            'side': SIDE_SELL if side == SIDE_BUY else SIDE_BUY,
            'type': FUTURE_ORDER_TYPE_STOP_MARKET,
            'stopPrice': round(stop_loss_price, 2),
            'quantity': quantity
        }
        stop_order = await client.futures_create_order(**stop_loss_params)
        logger.info(f"Стоп-лос встановлено: {stop_order}")
        print(f"Стоп-лос встановлено: {stop_order}")

        # Встановлюємо трейлінг-стоп
        trailing_stop_params = {
            'symbol': symbol,
            'side': SIDE_SELL if side == SIDE_BUY else SIDE_BUY,
            'type': FUTURE_ORDER_TYPE_TRAILING_STOP_MARKET,
            'callbackRate': trailing_stop_percent * 100,  # У відсотках
            'quantity': quantity
        }
        trailing_order = await client.futures_create_order(**trailing_stop_params)
        logger.info(f"Трейлінг-стоп встановлено: {trailing_order}")
        print(f"Трейлінг-стоп встановлено: {trailing_order}")

        return order
    except BinanceAPIException as e:
        logger.error(f"Помилка при відкритті позиції: {e}")
        print(f"Помилка при відкритті позиції: {e}")
        return None

# Функція для закриття позиції
async def close_position(client, symbol, side, quantity):
    """
    Закриває відкриту позицію на Binance Futures.
    """
    try:
        order_params = {
            'symbol': symbol,
            'side': side,
            'type': FUTURE_ORDER_TYPE_MARKET,
            'quantity': quantity
        }
        order = await client.futures_create_order(**order_params)
        logger.info(f"Позиція закрита: {order}")
        print(f"Позиція закрита: {order}")
        return order
    except BinanceAPIException as e:
        logger.error(f"Помилка при закритті позиції: {e}")
        print(f"Помилка при закритті позиції: {e}")
        return None

# Функція для запуску бота
async def run_trading_bot():
    """
    Основна функція для запуску торгового бота.
    """
    global client
    client = await AsyncClient.create(api_key, api_secret)
    open_order = None

    try:
        while True:
            # Отримання даних
            data_1h = await get_binance_data(client, interval='1h', limit=100)
            data_30m = await get_binance_data(client, interval='30m', limit=100)
            data_15m = await get_binance_data(client, interval='15m', limit=100)
            data_5m = await get_binance_data(client, interval='5m', limit=100)
            data_1m = await get_binance_data(client, interval='1m', limit=100)

            # Розрахунок індикаторів
            data_1h = calculate_indicators(data_1h)
            data_30m = calculate_indicators(data_30m)
            data_15m = calculate_indicators(data_15m)
            data_5m = calculate_indicators(data_5m)
            data_1m = calculate_indicators(data_1m)

            # Аналіз тренду
            trend_direction, trend_strength = determine_trend(data_1h)

            # Логіка прийняття рішень
            conditions_met = 0

            # ATR для 1г
            if data_1h['ATR'].iloc[-1] > data_1h['ATR'].mean():
                conditions_met += 20  # ATR показує волатильність
                logger.info("ATR волатильність: +20%")
                print("ATR волатильність: +20%")
                print(f"Поточна вага умов: {conditions_met}%")

            # MA7 понад MA25 для 30хв
            if data_30m['MA7'].iloc[-1] > data_30m['MA25'].iloc[-1]:
                conditions_met += 20
                logger.info("MA7 понад MA25 (таймфрейм 30хв): +20%")
                print("MA7 понад MA25 (таймфрейм 30хв): +20%")
                print(f"Поточна вага умов: {conditions_met}%")

            # RSI в зоні перепроданості для 15хв
            if data_15m['RSI'].iloc[-1] < 30:
                conditions_met += 15
                logger.info("RSI в зоні перепроданості (таймфрейм 15хв): +15%")
                print("RSI в зоні перепроданості (таймфрейм 15хв): +15%")
                print(f"Поточна вага умов: {conditions_met}%")

            # MACD перетинає сигнальну лінію для 5хв
            if data_5m['MACD'].iloc[-1] > data_5m['Signal_Line'].iloc[-1]:
                conditions_met += 15
                logger.info("MACD перетин сигнальної лінії (таймфрейм 5хв): +15%")
                print("MACD перетин сигнальної лінії (таймфрейм 5хв): +15%")
                print(f"Поточна вага умов: {conditions_met}%")

            # Відображення загальної ваги
            logger.info(f"Загальна вага умов: {conditions_met}%")
            print(f"Загальна вага умов: {conditions_met}%")

            # Відкриття позиції
            if conditions_met >= entry_threshold and open_order is None:
                entry_price = data_1m['Close'].iloc[-1]
                account_info = await client.futures_account_balance()
                balance_info = [asset for asset in account_info if asset['asset'] == 'USDT']
                if balance_info:
                    balance = float(balance_info[0]['balance'])
                    risk_amount = balance * quantity_percentage
                    quantity = (risk_amount * leverage) / entry_price

                    # Відкриття позиції
                    side = SIDE_BUY if trend_direction == 'uptrend' else SIDE_SELL
                    open_order = await open_position(client, symbol, side, quantity, leverage, entry_price)

            # Закриття позиції на основі зворотних умов
            elif open_order is not None:
                current_price = data_1m['Close'].iloc[-1]
                position_side = open_order['side']
                opposite_side = SIDE_SELL if position_side == SIDE_BUY else SIDE_BUY

                # Закриття позиції, якщо зворотні умови виконуються
                if (position_side == SIDE_BUY and trend_direction == 'downtrend') or \
                   (position_side == SIDE_SELL and trend_direction == 'uptrend'):
                    await close_position(client, symbol, opposite_side, open_order['origQty'])
                    open_order = None
                    logger.info("Позиція закрита через зміну тренду.")
                    print("Позиція закрита через зміну тренду.")

            # Затримка між перевірками
            await asyncio.sleep(60)

    except asyncio.CancelledError:
        logger.info("Бот зупинено.")
        print("Бот зупинено.")
    except Exception as e:
        logger.error(f"Виникла помилка: {e}")
        print(f"Виникла помилка: {e}")
    finally:
        if client:
            await client.close_connection()

if __name__ == "__main__":
    try:
        asyncio.run(run_trading_bot())
    except KeyboardInterrupt:
        logger.info("Бот зупинено користувачем.")
        print("Бот зупинено користувачем.")
    except Exception as e:
        logger.error(f"Несподівана помилка: {e}")
        print(f"Несподівана помилка: {e}")
