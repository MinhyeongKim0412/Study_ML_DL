#%%
import seaborn as sb
import matplotlib.pyplot as plt

#%%
# Seaborn의 Titanic 데이터셋 로드
titanic = sb.load_dataset('titanic')

#%%
# 결측치 확인
print("결측치 확인:")
print(titanic.isna().sum())

#%%
# 데이터 프레임의 첫 5행 확인
print("데이터 프레임의 첫 5행:")
print(titanic.head())

#%%
# 데이터 프레임의 열 이름 확인
print("열 이름 확인:")
print(titanic.columns)

#%%
# 데이터 프레임 정보 확인
print("데이터 프레임 정보:")
print(titanic.info())

#%%
# 데이터 프레임의 행과 열 수 확인
print("데이터 프레임의 행과 열 수:")
print(titanic.shape)

#%%
# 각 열의 데이터 타입 확인
print("각 열의 데이터 타입:")
print(titanic.dtypes)

#%%
# 기초 통계량 확인
print("기초 통계량:")
print(titanic.describe(include='all'))

#%%
# 'age'와 'fare'의 분포를 시각화
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sb.histplot(titanic['age'].dropna(), kde=True)
plt.title('Age Distribution')

plt.subplot(1, 2, 2)
sb.histplot(titanic['fare'].dropna(), kde=True)
plt.title('Fare Distribution')

plt.tight_layout()
plt.show()

#%%
# 성별에 따른 생존율 시각화
plt.figure(figsize=(8, 6))
sb.barplot(x='sex', y='survived', data=titanic, ci=None)
plt.title('Survival Rate by Sex')
plt.show()

#%%
# Pclass와 Survived 간의 관계를 시각화
plt.figure(figsize=(8, 6))
sb.countplot(x='pclass', hue='survived', data=titanic)
plt.title('Survival Count by Pclass')
plt.show()
