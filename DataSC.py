import warnings
warnings.filterwarnings('ignore')
# 1. Импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Настройка отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.set_style("whitegrid")

print("=" * 70)
print("ЧАСТЬ 1A: ЗАГРУЗКА И ПЕРВИЧНЫЙ ОБЗОР")
print("=" * 70)

# 2. Загрузка данных
# Проверяем, есть ли файл в текущей директории
current_dir = os.getcwd()
csv_file = 'Sleep_health_and_lifestyle_dataset.csv'

# Создаем путь к файлу
file_path = os.path.join(current_dir, csv_file)

# Проверяем существование файла
if os.path.exists(file_path):
    print(f"Файл найден: {file_path}")
    df = pd.read_csv(file_path)
else:
    # Если файл не найден, создаем путь к папке проекта
    print(f"Файл не найден по пути: {file_path}")
    print(f"1. Текущая папка: {current_dir}")

    # Предполагаемый путь (подставьте ваш реальный путь)
    project_path = r'C:\Users\777\PycharmProjects\PythonProject1'
    alternative_path = os.path.join(project_path, csv_file)

    if os.path.exists(alternative_path):
        print(f"\n✅ Файл найден по альтернативному пути: {alternative_path}")
        df = pd.read_csv(alternative_path)
    else:
        print(" Файл не найден. Пожалуйста, укажите правильный путь:")
        print("1. Переместите файл CSV в папку:", current_dir)
        exit()

# 3. Вывод первых строк
print(f"ДАННЫЕ УСПЕШНО ЗАГРУЖЕНЫ!")
print(f"Размер данных: {df.shape[0]} строк, {df.shape[1]} столбцов")

print("ПЕРВЫЕ 5 СТРОК:")
print(df.head())

print("ПОСЛЕДНИЕ 5 СТРОК:")
print(df.tail())

# 4. Структура данных
print("ИНФОРМАЦИЯ О СТРУКТУРЕ ДАННЫХ:")
print(df.info())

# 5. Типы данных
print("ТИПЫ ДАННЫХ В КАЖДОМ СТОЛБЦЕ:")
print(df.dtypes)

print("УНИКАЛЬНЫЕ ЗНАЧЕНИЯ В КАТЕГОРИАЛЬНЫХ СТОЛБЦАХ:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"{col}: {df[col].nunique()} уникальных значений")
    print(f"Примеры: {df[col].unique()[:5]}")

print("=" * 70)
print("ЧАСТЬ 1B: ОБРАБОТКА ДАННЫХ")
print("=" * 70)

# 1. Пропущенные значения
print("ПРОВЕРКА НА ПРОПУСКИ:")
missing_values = df.isnull().sum()
total_missing = missing_values.sum()

if total_missing == 0:
    print("Пропущенных значений нет!")
else:
    print(f"Обнаружено {total_missing} пропусков:")
    print(missing_values[missing_values > 0])

    # Стратегия заполнения
    print("ВЫБОР СТРАТЕГИИ ЗАПОЛНЕНИЯ:")

    # Для числовых столбцов - медиана
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"   {col}: заполнено медианой ({median_val:.2f})")

    # Для категориальных - мода
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"   {col}: заполнено модой ('{mode_val}')")

    print("Все пропуски заполнены!")

# 2. Дубликаты
print("ПРОВЕРКА НА ДУБЛИКАТЫ:")
duplicates_count = df.duplicated().sum()
print(f"Найдено {duplicates_count} полных дубликатов строк")

if duplicates_count > 0:
    print(f"Удаляем {duplicates_count} дубликатов...")
    initial_rows = len(df)
    df = df.drop_duplicates()
    final_rows = len(df)
    print(f"Удалено {initial_rows - final_rows} дубликатов")
    print(f"   Было: {initial_rows} строк")
    print(f"   Стало: {final_rows} строк")
else:
    print("Дубликатов нет!")

# 3. Преобразование типов данных
print("ПРЕОБРАЗОВАНИЕ ТИПОВ ДАННЫХ:")

# Разделяем Blood Pressure на два числовых столбца
if 'Blood Pressure' in df.columns:
    print("Разделяем 'Blood Pressure' на систолическое и диастолическое...")
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    df = df.drop('Blood Pressure', axis=1)
    print("'Blood Pressure' разделен на Systolic_BP и Diastolic_BP")

# Приводим категориальные столбцы к правильному типу
categorical_columns = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].astype('category')
        print(f"   {col}: преобразован в category")

print("Типы данных после преобразования:")
print(df.dtypes)

# 4. Выбросы (z-score)
print("АНАЛИЗ ВЫБРОСОВ (z-score):")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Удаляем Person ID из анализа выбросов
if 'Person ID' in numeric_cols:
    numeric_cols.remove('Person ID')

print(f"Анализируем {len(numeric_cols)} числовых столбцов:")
print(", ".join(numeric_cols))

from scipy import stats

outliers_info = {}
for col in numeric_cols:
    z_scores = np.abs(stats.zscore(df[col]))
    outliers = (z_scores > 3).sum()
    if outliers > 0:
        outliers_info[col] = outliers
        print(f"   {col}: {outliers} выбросов (z-score > 3)")

if outliers_info:
    print(f"Обнаружены выбросы в {len(outliers_info)} столбцах")
    # Можно удалить строки с выбросами или обработать их
    # Например, заменить на граничные значения
    for col in outliers_info.keys():
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Заменяем выбросы на граничные значения
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        print(f"   {col}: выбросы обработаны (метод IQR)")
else:
    print("Значительных выбросов не обнаружено")

# 5. Описательная статистика
print("ОПИСАТЕЛЬНАЯ СТАТИСТИКА:")
print(df.describe().round(2))

# 6. Подсчет уникальных значений
print("ПОДСЧЕТ УНИКАЛЬНЫХ ЗНАЧЕНИЙ:")
for col in df.columns:
    unique_count = df[col].nunique()
    if df[col].dtype == 'category' or unique_count < 20:
        print(f"{col} ({df[col].dtype}): {unique_count} уникальных значений")
        if unique_count <= 10:
            value_counts = df[col].value_counts()
            for val, count in value_counts.items():
                print(f"   '{val}': {count} ({count / len(df) * 100:.1f}%)")

# 7. Корреляционный анализ
print("КОРРЕЛЯЦИОННЫЙ АНАЛИЗ:")
# Выбираем только числовые столбцы
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

print("Матрица корреляций (первые 5x5):")
print(correlation_matrix.iloc[:5, :5].round(3))

# Находим сильные корреляции (|r| > 0.5)
print("СИЛЬНЫЕ КОРРЕЛЯЦИИ (|r| > 0.5):")
strong_correlations = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        corr = correlation_matrix.iloc[i, j]
        if abs(corr) > 0.5:
            col1 = correlation_matrix.columns[i]
            col2 = correlation_matrix.columns[j]
            strong_correlations.append((col1, col2, corr))
            print(f"   {col1} ↔ {col2}: {corr:.3f}")

if not strong_correlations:
    print("   Сильных корреляций не обнаружено")
print("=" * 70)
print("ЧАСТЬ 1C: ВИЗУАЛИЗАЦИЯ ДАННЫХ")
print("=" * 70)

# Создаем папку для графиков, если её нет
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

print("📊 Создаем графики...")

# График 1: Гистограмма распределения качества сна
plt.figure(figsize=(10, 6))
plt.hist(df['Quality of Sleep'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
plt.title('Распределение качества сна', fontsize=14, fontweight='bold')
plt.xlabel('Качество сна (оценка от 1 до 10)', fontsize=12)
plt.ylabel('Количество людей', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/histogram_sleep_quality.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ График 1: Гистограмма сохранена")

# График 2: Линейный график - зависимость качества сна от возраста
plt.figure(figsize=(10, 6))
# Группируем по возрасту и считаем среднее
age_sleep = df.groupby('Age')['Quality of Sleep'].mean().reset_index()
plt.plot(age_sleep['Age'], age_sleep['Quality of Sleep'],
         marker='o', linewidth=2, markersize=6, color='darkgreen')
plt.title('Среднее качество сна по возрастам', fontsize=14, fontweight='bold')
plt.xlabel('Возраст', fontsize=12)
plt.ylabel('Среднее качество сна', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(range(min(df['Age']), max(df['Age'])+1, 5))
plt.tight_layout()
plt.savefig('visualizations/line_age_sleep.png', dpi=150, bbox_inches='tight')
plt.show()
print("Линейный график сохранен")

# График 3: Столбчатая диаграмма - количество людей по профессиям
plt.figure(figsize=(12, 6))
occupation_counts = df['Occupation'].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(occupation_counts)))
bars = plt.bar(occupation_counts.index, occupation_counts.values, color=colors)
plt.title('Распределение людей по профессиям', fontsize=14, fontweight='bold')
plt.xlabel('Профессия', fontsize=12)
plt.ylabel('Количество людей', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

# Добавляем значения на столбцы
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{int(height)}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('visualizations/bar_occupations.png', dpi=150, bbox_inches='tight')
plt.show()
print("Столбчатая диаграмма сохранена")

# График 4: Boxplot - качество сна по категориям ИМТ
plt.figure(figsize=(10, 6))
bmi_categories = df['BMI Category'].cat.categories
data_to_plot = [df[df['BMI Category'] == cat]['Quality of Sleep'] for cat in bmi_categories]

box = plt.boxplot(data_to_plot, labels=bmi_categories, patch_artist=True)

# Раскрашиваем boxplot
colors = ['lightblue', 'lightgreen', 'lightcoral', 'wheat']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.title('Качество сна по категориям ИМТ', fontsize=14, fontweight='bold')
plt.xlabel('Категория ИМТ', fontsize=12)
plt.ylabel('Качество сна', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('visualizations/boxplot_bmi_sleep.png', dpi=150, bbox_inches='tight')
plt.show()
print("Boxplot сохранен")

# График 5: Heatmap корреляций
plt.figure(figsize=(10, 8))
# Выбираем самые интересные столбцы для heatmap
selected_cols = ['Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                 'Stress Level', 'Heart Rate', 'Daily Steps', 'Age']
if 'Systolic_BP' in df.columns and 'Diastolic_BP' in df.columns:
    selected_cols.extend(['Systolic_BP', 'Diastolic_BP'])

corr_selected = df[selected_cols].corr()

mask = np.triu(np.ones_like(corr_selected, dtype=bool))
sns.heatmap(corr_selected, mask=mask, annot=True, fmt=".2f",
            cmap='coolwarm', center=0, square=True,
            cbar_kws={"shrink": 0.8}, linewidths=0.5)
plt.title('Тепловая карта корреляций', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/heatmap_correlations.png', dpi=150, bbox_inches='tight')
plt.show()
print("Heatmap сохранен")

# Бонусный график 6: Рассеяние - физическая активность vs качество сна
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['Physical Activity Level'], df['Quality of Sleep'],
                      c=df['Stress Level'], cmap='viridis',
                      alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
plt.title('Физическая активность и качество сна', fontsize=14, fontweight='bold')
plt.xlabel('Уровень физической активности (мин/день)', fontsize=12)
plt.ylabel('Качество сна', fontsize=12)
plt.colorbar(scatter, label='Уровень стресса')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/scatter_activity_sleep.png', dpi=150, bbox_inches='tight')
plt.show()
print("Рассеяние сохранен")

print(f"Все графики сохранены в папке 'visualizations/'")
print("=" * 70)
print("ЧАСТЬ 1D: ГРУППИРОВКИ И АГРЕГАЦИИ")
print("=" * 70)

print("📊 ГРУППИРОВКИ И АГРЕГАЦИИ:")

# Агрегация 1: Средние показатели по профессиям
print("\n1️⃣ СРЕДНИЕ ПОКАЗАТЕЛИ ПО ПРОФЕССИЯМ:")
occupation_stats = df.groupby('Occupation').agg({
    'Age': ['mean', 'count'],
    'Sleep Duration': 'mean',
    'Quality of Sleep': 'mean',
    'Stress Level': 'mean',
    'Physical Activity Level': 'mean'
}).round(2)

occupation_stats.columns = ['Средний возраст', 'Количество',
                            'Средняя длительность сна', 'Среднее качество сна',
                            'Средний уровень стресса', 'Средняя активность']

print(occupation_stats.sort_values('Среднее качество сна', ascending=False))

# Агрегация 2: Подсчет количества нарушений сна по полу и ИМТ
print("КОЛИЧЕСТВО НАРУШЕНИЙ СНА ПО ПОЛУ И КАТЕГОРИИ ИМТ:")
if 'Sleep Disorder' in df.columns:
    sleep_disorder_counts = df[df['Sleep Disorder'] != 'None'].groupby(
        ['Gender', 'BMI Category']
    ).size().unstack(fill_value=0)

    print(sleep_disorder_counts)

    # Визуализация
    plt.figure(figsize=(10, 6))
    sleep_disorder_counts.plot(kind='bar', stacked=True, colormap='Set2')
    plt.title('Нарушения сна по полу и категориям ИМТ', fontsize=14, fontweight='bold')
    plt.xlabel('Пол', fontsize=12)
    plt.ylabel('Количество нарушений', fontsize=12)
    plt.legend(title='Категория ИМТ', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('visualizations/stacked_bar_sleep_disorders.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("График распределения нарушений сна сохранен")

# Агрегация 3: Сводная таблица (pivot_table)
print("СВОДНАЯ ТАБЛИЦА: Качество сна и стресс по полу и возрасту")
# Создаем возрастные группы
df['Age_Group'] = pd.cut(df['Age'],
                         bins=[20, 30, 40, 50, 60, 70],
                         labels=['20-29', '30-39', '40-49', '50-59', '60+'])

pivot_table = pd.pivot_table(df,
                             values=['Quality of Sleep', 'Stress Level', 'Physical Activity Level'],
                             index='Age_Group',
                             columns='Gender',
                             aggfunc='mean').round(2)

print("Средние показатели по возрастным группам и полу:")
print(pivot_table)

# Агрегация 4: Процент людей с нарушениями сна по профессиям
print("ПРОЦЕНТ ЛЮДЕЙ С НАРУШЕНИЯМИ СНА ПО ПРОФЕССИЯМ:")
if 'Sleep Disorder' in df.columns:
    sleep_disorder_percentage = df.groupby('Occupation').apply(
        lambda x: (x['Sleep Disorder'] != 'None').sum() / len(x) * 100
    ).round(2).sort_values(ascending=False)

    sleep_disorder_df = pd.DataFrame({
        'Профессия': sleep_disorder_percentage.index,
        '% с нарушениями сна': sleep_disorder_percentage.values
    })

    print(sleep_disorder_df)

# Агрегация 5: Корреляция физической активности и качества сна по возрастным группам
print("КОРРЕЛЯЦИЯ АКТИВНОСТИ И КАЧЕСТВА СНА ПО ВОЗРАСТНЫМ ГРУППАМ:")
correlation_by_age = df.groupby('Age_Group').apply(
    lambda x: x[['Physical Activity Level', 'Quality of Sleep']].corr().iloc[0, 1]
).round(3)

print("Корреляция между физической активностью и качеством сна:")
for age_group, corr in correlation_by_age.items():
    print(f"  {age_group}: {corr}")

# Агрегация 6: Статистика по давлению и ЧСС
print("СТАТИСТИКА АРТЕРИАЛЬНОГО ДАВЛЕНИЯ И ЧСС:")
if 'Systolic_BP' in df.columns and 'Diastolic_BP' in df.columns:
    bp_stats = df.groupby('BMI Category').agg({
        'Systolic_BP': ['mean', 'std', 'min', 'max'],
        'Diastolic_BP': ['mean', 'std', 'min', 'max'],
        'Heart Rate': 'mean'
    }).round(2)

    print("Статистика по категориям ИМТ:")
    print(bp_stats)

# Агрегация 7: Топ-3 профессии по разным показателям
print("ТОП-3 ПРОФЕССИИ ПО РАЗНЫМ ПОКАЗАТЕЛЯМ:")
top_metrics = {
    'Лучшее качество сна': 'Quality of Sleep',
    'Наименьший стресс': 'Stress Level',
    'Высокая физическая активность': 'Physical Activity Level',
    'Самая длинная продолжительность сна': 'Sleep Duration'
}

for metric_name, column in top_metrics.items():
    if column in df.columns:
        top_3 = df.groupby('Occupation')[column].mean().nlargest(3).round(2)
        print(f"{metric_name}:")
        for i, (occupation, value) in enumerate(top_3.items(), 1):
            print(f"  {i}. {occupation}: {value}")

print("=" * 70)
print("📋 ИТОГОВАЯ СТАТИСТИКА АНАЛИЗА")
print("=" * 70)

# Финальная статистика
summary = {
    'Общее количество записей': len(df),
    'Количество столбцов': len(df.columns),
    'Количество профессий': df['Occupation'].nunique(),
    'Средний возраст': round(df['Age'].mean(), 1),
    'Средняя продолжительность сна': round(df['Sleep Duration'].mean(), 2),
    'Среднее качество сна': round(df['Quality of Sleep'].mean(), 2),
    'Средний уровень стресса': round(df['Stress Level'].mean(), 2),
    'Средняя физическая активность': round(df['Physical Activity Level'].mean(), 1),
    'Среднее дневное количество шагов': round(df['Daily Steps'].mean(), 0)
}

if 'Sleep Disorder' in df.columns:
    sleep_disorder_percent = round((df['Sleep Disorder'] != 'None').sum() / len(df) * 100, 1)
    summary['Процент с нарушениями сна'] = f"{sleep_disorder_percent}%"

for key, value in summary.items():
    print(f"{key}: {value}")

print("=" * 70)
print("АНАЛИЗ ДАННЫХ ЗАВЕРШЕН!")
print("=" * 70)
print("Созданы файлы:")
print("   - visualizations/histogram_sleep_quality.png")
print("   - visualizations/line_age_sleep.png")
print("   - visualizations/bar_occupations.png")
print("   - visualizations/boxplot_bmi_sleep.png")
print("   - visualizations/heatmap_correlations.png")
print("   - visualizations/scatter_activity_sleep.png")
print("   - visualizations/stacked_bar_sleep_disorders.png")

df.to_csv('sleep_analysis_for_datalens.csv', index=False, encoding='utf-8')
print("Файл для DataLens сохранен: sleep_analysis_for_datalens.csv")