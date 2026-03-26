import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

df = pd.read_csv("Set - 21 Dataset_Anime.csv")

print("Shape:", df.shape)
print("\nColumn Info:")
print(df.info())
print("\nFirst 5 Rows:")
display(df.head())

essential_cols = ['title','genre','aired','episodes','members','popularity','ranked','score']
df = df[essential_cols]

print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())

df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')
df['score'] = pd.to_numeric(df['score'], errors='coerce')
df['members'] = pd.to_numeric(df['members'], errors='coerce')
df.dropna(subset=['score','members','episodes'], inplace=True)
df.drop_duplicates(inplace=True)
print("\nCleaned Shape:", df.shape)

num_cols = ['episodes','members','popularity','ranked','score']
desc_stats = df[num_cols].describe().T
desc_stats['range'] = desc_stats['max'] - desc_stats['min']
display(desc_stats)

for col in num_cols:
    mean = df[col].mean()
    median = df[col].median()
    print(f"{col}: Mean={mean:.2f}, Median={median:.2f}, Best={ 'Median' if abs(mean-median)>1 else 'Mean'}")

plt.figure(figsize=(8,5))
sns.histplot(df['score'], bins=20, kde=True)
plt.title('Score Distribution')
plt.show()

df['primary_genre'] = df['genre'].str.split(',').str[0]
top_genres = df['primary_genre'].value_counts().head(5).index
plt.figure(figsize=(10,6))
sns.boxplot(data=df[df['primary_genre'].isin(top_genres)], x='primary_genre', y='score')
plt.title('Score by Top 5 Genres')
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df['episodes'], bins=30)
plt.title('Episodes Distribution')
plt.show()

def remove_outliers(col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

plt.figure(figsize=(8,5))
sns.boxplot(df['score'])
plt.title("Before Outlier Removal: Score")
plt.show()

df_no_outliers = remove_outliers('score')

plt.figure(figsize=(8,5))
sns.boxplot(df_no_outliers['score'])
plt.title("After Outlier Removal: Score")
plt.show()

sm.qqplot(df['score'], line='s')
plt.title("Normal Q-Q Plot for Score")
plt.show()

corr = df[num_cols].corr()
plt.figure(figsize=(8,5))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

print("\nCorrelation with Score:\n", corr['score'].sort_values(ascending=False))

sem = stats.sem(df['score'])
ci = stats.t.interval(0.95, len(df['score'])-1, loc=mean_score, scale=sem)
print(f"\n95% Confidence Interval for Mean Score: {ci}")

def margin_of_error(conf, col):
    z = stats.norm.ppf((1+conf)/2)
    std_err = stats.sem(df[col])
    return z * std_err

for c in [0.90, 0.95, 0.99]:
    print(f"Margin of Error ({int(c*100)}%):", margin_of_error(c, 'members'))

genres = ['Action','Romance']
subsets = [df[df['genre'].str.contains(g, case=False, na=False)]['score'] for g in genres]

t_stat, p_val = stats.ttest_ind(subsets[0], subsets[1], nan_policy='omit')
print(f"\nT-Test (Action vs Romance): t={t_stat:.3f}, p={p_val:.3f}")
if p_val < 0.05:
    print("Reject H₀ → Significant difference between genres.")
else:
    print("Fail to reject H₀ → No significant difference.")

corr_val = stats.pearsonr(df['members'], df['score'])
print(f"\nPearson Correlation (members vs score): {corr_val[0]:.3f}, p={corr_val[1]:.3f}")

sns.lmplot(data=df, x='members', y='score', scatter_kws={'alpha':0.5})
plt.title("Members vs Score with Regression Line")
plt.show()

X = df[['members']]
y = df['score']
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print(f"\nRegression Equation: Score = {model.intercept_:.3f} + {model.coef_[0]:.8f} * Members")
print(f"R² Score: {r2:.3f}")
