import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime

logging.basicConfig(
    filename='gb_model.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_features(df, lags=3):
    """Создание признаков для временного ряда"""
    try:
        df = df.copy()
        for lag in range(1, lags+1):
            df[f'lag_{lag}'] = df['amount'].shift(lag)
        df['month'] = df.index.month
        return df.dropna()
    except Exception as e:
        logging.error(f"Ошибка создания признаков: {str(e)}")
        raise

def process_category(category_df):
    """Обработка одной категории"""
    try:
        category_name = category_df['category'].iloc[0]
        logging.info(f"Начало обработки категории: {category_name}")
        
        # Подготовка данных
        processed = create_features(category_df.set_index('date')[['amount']])
        if len(processed) < 4:
            raise ValueError("Недостаточно данных для обучения")
            
        X = processed.drop('amount', axis=1)
        y = processed['amount']
        
        # Обучение модели
        start_time = datetime.now()
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1
        )
        model.fit(X, y)
        
        # Прогноз
        last_values = X.iloc[-1].values.tolist()[1:] + [X.iloc[-1].values[0]]
        forecast = model.predict([last_values])[0]
        
        exec_time = datetime.now() - start_time
        logging.info(f"Категория {category_name} обработана за {exec_time.total_seconds():.1f} сек.")
        return forecast
        
    except Exception as e:
        logging.error(f"Ошибка в категории {category_name}: {str(e)}")
        return None

if __name__ == "__main__":
    try:
        # Пример данных
        data = {
            'date': pd.date_range(start='2022-01-01', periods=24, freq='M').repeat(6),
            'category': np.tile([f'category_{i}' for i in range(1,7)], 24),
            'amount': np.random.randint(500, 5000, 144)
        }
        df = pd.DataFrame(data)
        
        # Обработка по категориям
        results = {}
        for category in df['category'].unique():
            category_data = df[df['category'] == category]
            results[category] = process_category(category_data)
        
        # Вывод результатов
        for category, value in results.items():
            if value:
                print(f"{category}: {value:.2f} руб.")
                
    except Exception as e:
        logging.critical(f"Критическая ошибка: {str(e)}")
        raise