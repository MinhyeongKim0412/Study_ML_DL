# %%
import numpy as np  # 수치 연산 라이브러리
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier  # 랜덤 포레스트 분류기
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # 서포트 벡터 머신 분류기
from sklearn.tree import DecisionTreeClassifier  # 결정 트리 분류기
import pandas as pd  # 데이터 분석 라이브러리

# %%
class Stacking:
    
    def __init__(self, *args):  # 모델 초기화
        self.clfs = [arg for arg in args]  # 입력된 분류기들 저장
        self.clfs_fit = {}  # 학습된 분류기들 저장
        self.meta_clf = None  # 메타 분류기
        self.pred = None  # 예측 결과 저장소 초기화

    def fit(self, x, y):  # 모델 학습
        row = x.shape[0]  # 데이터 샘플 수
        col = np.unique(y).size * (len(self.clfs) - 1)  # 예측 결과 행렬의 열 수
        self.pred = np.zeros((row, col))  # 예측 결과 저장 행렬 초기화
        for i, clf in enumerate(self.clfs[:-1]):  # 마지막 분류기 제외하고 학습
            self.clfs_fit[repr(clf)] = clf.fit(x, y)  # 학습된 분류기 저장
            self.pred[:, i*len(np.unique(y)):(i+1)*len(np.unique(y))] = clf.predict_proba(x)  # 예측 확률값 저장
        self.meta_clf = self.clfs[-1]  # 마지막 분류기 = 메타 분류기
        self.meta_clf.fit(self.pred, y)  # 메타 분류기 학습

    def predict(self):  # 예측 메서드 (미구현)
        pass
    
    def score(self):  # 정확도 계산 메서드 (미구현)
        pass
    
    def confusion_matrix(self):  # 혼동 행렬 계산 메서드 (미구현)
        pass
    
    def precision(self):  # 정밀도 계산 메서드 (미구현)
        pass
    
    def recall(self):  # 재현율 계산 메서드 (미구현)
        pass
    
    # 중복된 메서드를 제거 (이전에는 recall이 중복됨)
    # 미구현된 메서드들은 추후에 필요 시 작성하면 됩니다.

# -----------------------------------------------------------------------
# %%
def make_df(x):
    df = x.data
    df = pd.DataFrame(df)
    df.columns = x.feature_names
    df['target'] = x.target
    return df

# %%
# Iris 데이터셋을 로드하여 DataFrame으로 변환
iris = make_df(load_iris())

# %%
# 특성과 타겟 변수 분리
x = iris.iloc[:, :-1]  # 특성 데이터
y = iris.iloc[:, -1]  # 타겟 데이터

# %%
# 학습 데이터와 테스트 데이터로 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  # 학습 및 테스트 데이터 분할

# -----------------------------------------------------------------------

# %%
stack = Stacking(DecisionTreeClassifier(), SVC(probability=True), RandomForestClassifier())  # 스태킹 모델 생성

# %%
stack.fit(x_train, y_train)  # 스태킹 모델 학습

# %%
stack.meta_clf  # 메타 분류기 확인

# %%
stack.clfs_fit  # 학습된 분류기들 확인

# %%
np.unique(y_train)  # y_train의 고유값 확인

# %%
# stack.pred 배열의 실제 행 수에 맞춰 크기를 조정
n_samples = stack.pred.shape[0]  # stack.pred의 샘플 수를 가져옴

# %%
stack.pred[:, :3] = np.ones((n_samples, 3))  # 예측값 배열 첫 3열을 1로 채움

# %%
stack.pred[:, 3:6] = np.ones((n_samples, 3)) * 2  # 예측값 배열 다음 3열을 2로 채움

# %%
stack.pred  # 최종 예측값 배열 확인
