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

# Глобальна змінна для клієнта, який буде ініціалізований пізніше
client = None

# Символ для торгівлі
symbol = 'BTCUSDT'

# Використовуване кредитне плече
leverage = 10

# Відсоток стоп-лоса (2% від суми ордеру)
stop_loss_percent = 0.02

# Цільовий відсоток прибутку (можна налаштувати відповідно до стратегії)
target_profit_percent = 0.05

# Поріг для входу в позицію
entry_threshold = 75  # 80%

# Функція для отримання історичних даних з Binance для різних таймфреймів
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
        return data
    except BinanceAPIException as e:
        logger.error(f"Помилка при отриманні даних з Binance: {e}")
        return pd.DataFrame()

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

    # Смуги Боллінджера (Bollinger Bands)
    data['MiddleBand'] = data['Close'].rolling(window=20).mean()
    data['StdDev'] = data['Close'].rolling(window=20).std()
    data['UpperBand'] = data['MiddleBand'] + (data['StdDev'] * 2)
    data['LowerBand'] = data['MiddleBand'] - (data['StdDev'] * 2)

    # Видалити рядки з NaN значеннями
    data.dropna(inplace=True)
    logger.info("Індикатори розраховані і додані до даних.")
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
    return trend_direction, trend_strength

# Функція для розрахунку та встановлення стоп-лоса та трейлінг-стопа
def set_stop_loss_and_trailing_stop(entry_price, side, stop_loss_percent, trailing_stop_percent):
    """
    Встановлює стоп-лос та трейлінг-стоп для відкритої позиції.
    """
    if side == SIDE_BUY:
        stop_loss_price = entry_price * (1 - stop_loss_percent)
        trailing_stop_callback = trailing_stop_percent * 100  # У відсотках
    elif side == SIDE_SELL:
        stop_loss_price = entry_price * (1 + stop_loss_percent)
        trailing_stop_callback = trailing_stop_percent * 100  # У відсотках
    else:
        raise ValueError("Невідомий тип позиції. Доступні типи: SIDE_BUY, SIDE_SELL.")
    stop_loss_price = round(stop_loss_price, 2)
    trailing_stop_callback = round(trailing_stop_callback, 1)
    logger.info(f"Розраховано stop_loss_price: {stop_loss_price}, trailing_stop_callback: {trailing_stop_callback}%")
    return stop_loss_price, trailing_stop_callback

async def open_position(client, symbol, side, quantity, leverage, entry_price):
    """
    Відкриває позицію на Binance Futures і встановлює стоп-лос та трейлінг-стоп.
    """
    try:
        # Отримуємо режим позиції
        position_mode_info = await client.futures_get_position_mode()
        hedge_mode = position_mode_info['dualSidePosition']  # True якщо Hedge Mode

        # Встановлюємо кредитне плече
        await client.futures_change_leverage(symbol=symbol, leverage=leverage)
        logger.info(f"Кредитне плече встановлено на {leverage}x для {symbol}")

        # Визначаємо сторону позиції
        position_side = 'LONG' if side == SIDE_BUY else 'SHORT'

        # Параметри ордера
        order_params = {
            'symbol': symbol,
            'side': side,
            'type': FUTURE_ORDER_TYPE_MARKET,
            'quantity': quantity
        }

        if hedge_mode:
            order_params['positionSide'] = position_side

        # Відкриваємо ринковий ордер
        order = await client.futures_create_order(**order_params)
        logger.info(f"Позиція {position_side} відкрита: {order}")
        print(f"Позиція {position_side} відкрита: {order}")

        # Встановлюємо стоп-лос і трейлінг-стоп
        stop_loss_price, trailing_stop_callback = set_stop_loss_and_trailing_stop(
            entry_price, side, stop_loss_percent, trailing_stop_percent=0.02
        )

        # Стоп-лос ордер
        stop_order_params = {
            'symbol': symbol,
            'side': SIDE_SELL if side == SIDE_BUY else SIDE_BUY,
            'type': FUTURE_ORDER_TYPE_STOP_MARKET,
            'stopPrice': stop_loss_price,
            'closePosition': True
        }

        if hedge_mode:
            stop_order_params['positionSide'] = position_side

        stop_order = await client.futures_create_order(**stop_order_params)
        logger.info(f"Стоп-лос встановлено: {stop_order}")
        print(f"Стоп-лос встановлено: {stop_order}")

        # Трейлінг-стоп ордер
        trailing_stop_order_params = {
            'symbol': symbol,
            'side': SIDE_SELL if side == SIDE_BUY else SIDE_BUY,
            'type': FUTURE_ORDER_TYPE_TRAILING_STOP_MARKET,
            'callbackRate': trailing_stop_callback,
            'quantity': quantity
        }

        if hedge_mode:
            trailing_stop_order_params['positionSide'] = position_side

        trailing_stop_order = await client.futures_create_order(**trailing_stop_order_params)
        logger.info(f"Трейлінг-стоп встановлено: {trailing_stop_order}")
        print(f"Трейлінг-стоп встановлено: {trailing_stop_order}")

    except BinanceAPIException as e:
        logger.error(f"Помилка при відкритті позиції або встановленні стоп-лоса: {e}")
        print(f"Помилка при відкритті позиції або встановленні стоп-лоса: {e}")

# Продовження коду аналогічно з іншими функціями, уникаючи викликів `await` поза асинхронними функціями,
# та забезпечуючи, що всі змінні оголошені і ініціалізовані перед їх використанням.

# Основна функція для запуску торгового бота
async def run_trading_bot():
    """
    Основна функція для запуску торгового бота.
    """
    global client
    client = await AsyncClient.create(api_key, api_secret)

    try:
        open_order = None
        while True:
            # Отримання історичних даних з Binance
            data_30m = await get_binance_data(client, interval='30m', limit=100)
            data_5m = await get_binance_data(client, interval='5m', limit=100)
            data_1m = await get_binance_data(client, interval='1m', limit=100)

            # Обчислення індикаторів для кожного таймфрейму
            data_30m = calculate_indicators(data_30m)
            data_5m = calculate_indicators(data_5m)
            data_1m = calculate_indicators(data_1m)

            # Визначення напрямку та сили тренду на 30-хвилинному таймфреймі
            trend_direction, trend_strength = determine_trend(data_30m)

            # Додати логіку для прийняття рішень на основі даних та ваги умов
            conditions_met = 0

            # Розрахунок ваги умов
            # 5-хв таймфрейм (основний)
            if data_5m['MA7'].iloc[-1] > data_5m['MA25'].iloc[-1]:
                conditions_met += 20  # MA7 перетинає MA25 на 5m
                logger.info("Вага 5m MA7 перетин MA25: +20%")
                print("Вага 5m MA7 перетин MA25: +20%")
            if 30 <= data_5m['RSI'].iloc[-1] <= 70:
                conditions_met += 20  # RSI між 30 та 70 на 5m
                logger.info("Вага 5m RSI між 30 та 70: +20%")
                print("Вага 5m RSI між 30 та 70: +20%")
            if data_5m['Close'].iloc[-1] > data_5m['MiddleBand'].iloc[-1]:
                conditions_met += 20  # Ціна вище середньої смуги Боллінджера на 5m
                logger.info("Вага 5m Ціна вище MiddleBand: +20%")
                print("Вага 5m Ціна вище MiddleBand: +20%")

            # 30-хв таймфрейм (підтвердження)
            if trend_direction == 'uptrend':
                conditions_met += 10  # MA25 зростає на 30m
                logger.info("Вага 30m MA25 Trend: +10%")
                print("Вага 30m MA25 Trend: +10%")

            # 1-хв таймфрейм (підтвердження)
            if data_1m['MA7'].iloc[-1] > data_1m['MA25'].iloc[-1]:
                conditions_met += 5  # MA7 перетинає MA25 на 1m
                logger.info("Вага 1m MA7 перетин MA25: +5%")
                print("Вага 1m MA7 перетин MA25: +5%")
            if 30 <= data_1m['RSI'].iloc[-1] <= 70:
                conditions_met += 5  # RSI між 30 та 70 на 1m
                logger.info("Вага 1m RSI між 30 та 70: +5%")
                print("Вага 1m RSI між 30 та 70: +5%")

            # Відображення загальної ваги умов
            logger.info(f"Загальна вага умов: {conditions_met}%")
            print(f"Загальна вага умов: {conditions_met}%")

            # Прийняття рішення на основі загальної ваги
            if conditions_met >= entry_threshold and open_order is None:
                # Відкриваємо позицію
                entry_price = data_1m['Close'].iloc[-1]
                # Логіка для відкриття позиції аналогічна попереднім фрагментам

            # Затримка перед наступною ітерацією
            await asyncio.sleep(60)  # Чекаємо 1 хвилину перед наступною перевіркою

    except Exception as e:
        logger.error(f"Виникла помилка: {e}")
        print(f"Виникла помилка: {e}")
    finally:
        await client.close_connection()
        await asyncio.sleep(0.1)  # Даємо час для коректного закриття циклу

if __name__ == "__main__":
    try:
        asyncio.run(run_trading_bot())
    except KeyboardInterrupt:
        logger.info("Бот зупинений користувачем.")
        print("Бот зупинений користувачем.")
    except Exception as e:
        logger.error(f"Несподівана помилка: {e}")
        print(f"Несподівана помилка: {e}")
