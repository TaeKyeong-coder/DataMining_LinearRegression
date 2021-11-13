#파이썬 머신러닝 라이브러리 싸이킷런을 불러오기
from sklearn.linear_model import LinearRegression
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.model_selection import train_test_split

import pandas as pd
#배열을 바꿀 때 필요한 numpy
import numpy as np
#시각화를 위한 matplotlib
import matplotlib.pyplot as plt


#파일 불러오기 (https://bigdaheta.tistory.com/40)
df = pd.read_csv("student_health_2.csv", encoding = "cp949")
#df1 = pd.read_csv("student_health_2.csv", encoding = "cp949", index_col='키')
#df2 = pd.read_csv("student_health_2.csv", encoding = "cp949", index_col='몸무게')

df.head()


x = df[["키", "몸무게"]]
y = df[["학년"]]

#점으로 찍어보기 (https://dailyheumsi.tistory.com/36)
#plt.plot(x,y, marker='o')
#plt.show()

#데이터 학습시키기 (다중선열)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, test_size=0.2)

mlr = LinearRegression()
mlr.fit(x_train, y_train)

#예측하기
my_body = [[150, 45]]
my_predict = mlr.predict(my_body)
