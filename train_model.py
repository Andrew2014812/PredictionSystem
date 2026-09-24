import sys
import os
import pandas as pd
from xgboost import XGBRegressor
# Импортируем space_eval для расшифровки параметров
from hyperopt import hp, fmin, tpe, space_eval

# Добавляем путь, чтобы Python видел папку src
sys.path.append(os.getcwd())

from src.storage.tables import AzureBlobTable
from src.modelling.experiment import HyperParamExperiment
from src.utils.config import Config as cfg
from src.utils.functions import split_features_target


# --- ЛЕЧЕНИЕ ОШИБКИ ---
# Мы создаем свой класс, который наследует оригинальный эксперимент,
# но исправляет ошибку с индексами (0 вместо 3).
class PatchedHyperParamExperiment(HyperParamExperiment):
    def run(self, score, space, algo=tpe.suggest, n_iter=10):
        self.score = score
        self.X_score, self.y_score = split_features_target(self.score)
        self.space = space
        self.algo = algo
        self.n_iter = n_iter

        # Запускаем поиск лучших параметров
        best_indices = fmin(fn=self.objective_function,
                            space=self.space,
                            algo=self.algo,
                            max_evals=self.n_iter)

        # ГЛАВНОЕ ИСПРАВЛЕНИЕ:
        # Превращаем индексы (0, 1...) обратно в реальные значения (3, 4, 5...)
        best_params = space_eval(self.space, best_indices)

        return self.get_trained_model(best_params)


# ----------------------

def main():
    print("🚀 1. Загрузка данных...")
    try:
        loader = AzureBlobTable(cfg.AZURE_PROCESSED_TABLE, ftype="csv")
        train_df = loader.read("train")
        valid_df = loader.read("valid")

        # Исправляем формат дат
        train_df['F_DATE'] = pd.to_datetime(train_df['F_DATE'])
        valid_df['F_DATE'] = pd.to_datetime(valid_df['F_DATE'])
    except Exception as e:
        print(f"❌ Данные не найдены. Ошибка: {e}")
        print("   Сначала запустите 'py run_pipelines.py'")
        return

    print(f"   Готово. Строк: {len(train_df)} (train), {len(valid_df)} (valid)")

    print("⏳ 2. Обучение модели...")
    space = {
        'n_estimators': 100,
        'max_depth': hp.choice('max_depth', [3, 4, 5]),
        'learning_rate': hp.uniform('learning_rate', 0.05, 0.2),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
        'sample_weight_lam': hp.loguniform('sample_weight_lam', -10, -3)
    }

    # ИСПОЛЬЗУЕМ НАШ ИСПРАВЛЕННЫЙ КЛАСС
    exp = PatchedHyperParamExperiment(XGBRegressor, train_df)
    model = exp.run(valid_df, space, n_iter=2)

    print("💾 3. Сохранение...")
    AzureBlobTable(cfg.AZURE_MODELS_FOLDER, ftype="pkl").upload({"model": model})
    print("✅ УСПЕХ! Модель 'model.pkl' создана.")


if __name__ == "__main__":
    main()