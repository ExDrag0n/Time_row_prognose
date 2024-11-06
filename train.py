import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from datetime import timedelta

# Загрузка данных
triggers = pd.read_csv('triggers.csv')
actions = pd.read_csv('actions.csv')

# Объединение данных
data = triggers.merge(actions, on=['guid', 'date'], how='left').fillna(0)

# Преобразование столбца 'date' в формат datetime
data['date'] = pd.to_datetime(data['date'])

# Создание временных признаков
data['day_of_week'] = data['date'].dt.dayofweek
data['hour'] = data['date'].dt.hour

# Проверка типов данных после преобразования
print("Типы данных:")
print(data.dtypes)

# Кодирование категориальных переменных
le = LabelEncoder()
data['trigger_type_encoded'] = le.fit_transform(data['type'])

# Вычисление времени между последним триггером и текущим
data['time_since_last_trigger'] = data.groupby('guid')['date'].diff().dt.total_seconds()

# Формирование признаков и целевой переменной
X = data[['trigger', 'trigger_type_encoded', 'day_of_week', 'hour', 'time_since_last_trigger']]
y = data['result']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Оптимизация гиперпараметров модели
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1_macro')
grid_search.fit(X_train, y_train)

# Обучение модели с оптимальными параметрами
best_model = grid_search.best_estimator_

# Оценка производительности на тестовых данных
y_pred = best_model.predict(X_test)
print(f"F1-score на тестовой выборке: {f1_score(y_test, y_pred)}")


# Функция для фильтрации результатов с учетом ограничений
def filter_results(guids, dates, predictions):
    filtered_results = []
    last_interaction = {}

    for guid, date, prediction in zip(guids, dates, predictions):
        if guid not in last_interaction or (date - last_interaction[guid]).days >= 14:
            filtered_results.append(prediction)
            last_interaction[guid] = date
        else:
            filtered_results.append(0)

    return filtered_results


# Применение модели к новым данным
new_data = pd.DataFrame({
    'guid': ['user1', 'user2', 'user3'],
    'date': pd.to_datetime(['2024-03-11', '2024-03-12', '2024-03-13']),
    'trigger': [1, 2, 3],
    'type': ['email', 'sms', 'push'],
    'day_of_week': [0, 1, 2],
    'hour': [10, 12, 14],
    'time_since_last_trigger': [86400, 172800, 259200]  # Примерные значения
})

new_data['trigger_type_encoded'] = le.transform(new_data['type'])
predictions = best_model.predict(
    new_data[['trigger', 'trigger_type_encoded', 'day_of_week', 'hour', 'time_since_last_trigger']])
filtered_predictions = filter_results(new_data['guid'], new_data['date'], predictions)

print("Прогнозы для новых данных:", filtered_predictions)
