#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[8]:


a = np.array([1,2,3])


# In[9]:


s = pd.Series([1,2,3])


# In[22]:


s.index = ['민형','신희', '소라']


# In[34]:


s = pd.Series({'소라':[100,100],'소라':100,'소라':[100,50]})


# In[27]:


s.index = ['소라','소라','소라',]


# In[31]:


s.mean()


# In[37]:


s.values


# In[40]:


dic  = {'a':1,'b':2,'c':3}
pd.Series(dic)
pd.Series([1,2,3],index= [i for i in 'abc'])


# In[41]:


list_data = ['2019-01-02',3.14,' ABC',100,True]
sr = pd.Series(list_data)
sr


# In[45]:


sr.index = np.arange(1,6)


# In[52]:


sr.index = ['동연','민형','혜림','신희','소라']


# In[57]:


sr[['동연','혜림']]
sr['동연':'신희']


# In[65]:


tup_data = ('영인','2010-05-01','여',True)
ar1 = pd.Series(tup_data,index=['이름','생년월일','성별','학생여부'])
print(ar1)
## dictionary이용
dic = {'이름':'영인','생년월일':'2010-05-01','성별':'여','학생여부':True}
ar2 = pd.Series(dic)
print("=============================")
print(ar2)
pd.Series(dict(zip(ar2.index,ar2.values)))
ar2.to_dict()


# In[66]:


pd.Series(ar1.to_dict())


# In[67]:


dir(ar1)


# In[68]:


ar1.생년월일


# In[69]:


ar1['age'] = 30


# In[71]:


ar1.age


# In[72]:


arr = np.random.randint(1,100,30)
sr = pd.Series(arr,index = np.random.choice(['a','b','c'],30))


# In[81]:


sr.var()
sr.std()
sr.describe()['25%']


# In[83]:


np.percentile(sr.values,25)


# In[87]:


np.random.seed(100)
arr = np.random.randint(60,101,30)
score = pd.Series(arr)
score.index = np.random.choice(['태우','광우','영훈'],30)


# In[91]:


pd.Series(score.index).value_counts()


# In[97]:


#태우,광우,영훈 세분의 평균
score['태우'].max()
score['광우'].max()
score['영훈'].max()


# In[101]:


dic = {}
for cat in score.index.unique():
    dic[cat] = score[cat].mean()
score_mean = pd.Series(dic)
score_mean


# In[110]:


score.groupby(level=0).count()


# In[111]:


pd.Series(score.index).value_counts()


# In[125]:


def mean(x):
    return x.mean()
def max(x):
    return x.max()
def sum(x):
    return x.sum()
def min(x):
    return x.min()
def median(x):
    return x.median()
def quarter1(x):
    return np.percentile(x,25)
def quarter2(x):
    return np.percentile(x,75)
                         
def group_by(x,method):
    method_dict = {
        'mean':mean,
        'max':max,
        'sum':sum,
        'min':min,
        'median':median,
        '25%':quarter1,
        '75%':quarter2
    }
    dic = {}
    for cat in x.index.unique():
        dic[cat] = method_dict[method](x[cat])
    x = pd.Series(dic)
    return x


# In[129]:


group_by(score,'75%')


# In[127]:


# group_by(score,'mean',max,sum,min,median,25%,75%)
score.groupby(level=0).median()


# In[130]:


group_by(score,'mean')


# In[136]:


# column 추가 
df = pd.DataFrame(np.random.randint(60,101,size=(100,4)))
df.index = np.arange(1,101)
df.columns = ['kor','eng','math','sci']


# In[145]:


dic = {
    'kor':np.random.randint(60,101,100),
     'eng':np.random.randint(60,101,100),
     'math':np.random.randint(60,101,100),
     'sci':np.random.randint(60,101,100),
}
df1 = pd.DataFrame(dic,index=np.arange(1,101))


# In[154]:


df.to_dict().get('eng').get(20)


# In[156]:


###  원소를 3개씩 담고 있는 리스트를 다섯개 만들고 각 리스트에 딕셔너리의  키는 c0~c4
dic = {
    'c0':[1,2,3],
    'c1':[1,2,3],
    'c2':[1,2,3],
    'c3':[1,2,3],
    'c4':[1,2,3],
}
pd.DataFrame(dic)


# In[170]:


df = pd.DataFrame(np.ones((3,5),dtype=np.int_)*np.array([1,2,3]).reshape(-1,1))
df.columns = ['c'+str(i) for i in range(5)]
df.columns = ['a']*5
df.index = np.arange(1,4)


# In[177]:


df.columns = ['c '+str(i) for i in range(5)]


# In[176]:


df.c0


# In[180]:


df['c 0']


# In[182]:


df.columns = ['c' +str(i) for i in range(5)]


# In[184]:


df.columns = ['d' +str(i) for i in range(5)]


# In[196]:


df.columns = np.hstack([df.columns[:4],['e4']])


# In[200]:


df.rename(columns={'d2':'f2'},inplace=True)
df


# In[195]:


np.hstack([df.columns[:4],['e4']])


# In[192]:


df.columns[:4]


# In[201]:


df = pd.DataFrame(np.random.randint(1,10,(5,11)))


# In[203]:


df.columns = ['col '+str(i) for i in range(11)]


# In[208]:


df.rename(columns={'col_1':'col_0'},inplace=True)


# In[224]:


df.rename(columns={ i:i.replace(' ', '_') for i in df.columns  if ' ' in i },inplace=True)


# In[222]:


'col 1'.replace(' ','_')


# In[228]:


df = pd.DataFrame(df.values)
df = pd.DataFrame(np.random.randint(60,101,(100,4)))
df.columns = ['Korean','Enlgish','Mathmatic','Science']


# In[238]:


df.columns.str.lower().str.slice(0,3)


# In[239]:


dict(zip(df.columns,df.columns.str.lower().str.slice(0,3)))


# In[241]:


df.rename(columns=dict(zip(df.columns,df.columns.str.lower().str.slice(0,3))),inplace=True)


# In[242]:


df


# In[250]:


df.columns[df.columns != 'mat']


# In[252]:


df1 = df[['kor','enl','mat']]
df2 = df.drop(['sci'],axis=1)
# df.drop(['sci'],axis=1,inplace=True)


# In[255]:


df.drop([96],axis=0,inplace=True)


# In[260]:


df = pd.DataFrame(np.random.randint(60,101,(30,4)),columns=['k','e','m','s'])
df.index = np.random.choice(['시형','태우','소라'],30)


# In[264]:


df.drop(['태우','소라'],axis=0)


# In[266]:


df.loc['태우','k']


# In[268]:


df = pd.DataFrame(np.random.randint(60,101,(100,3)))


# In[269]:


df.columns = ['kor','eng','math']


# In[272]:


df.drop(0,axis=0,inplace=True)


# In[283]:


df.iloc[0,:]


# In[285]:


df.loc[1,:'math']


# In[294]:


df.kor > df.kor.mean()


# In[298]:


# 국어 성적이 국어성적 평균보다 큰 값
df.loc[df.kor > df.kor.mean(),:]
df.iloc[np.array((df.kor > df.kor.mean())),:]


# In[302]:


df.loc[df.kor>df.kor.mean(),[True,False,True]]
df.iloc[(df.kor>df.kor.mean()).values,[True,False,True]]


# In[326]:


np.random.seed(100)
df = pd.DataFrame(np.random.randint(60,101,(100,4)))


# In[327]:


df.columns = ['c1','c2','c3','c4']


# In[307]:


df['y'] = np.random.choice([0,1,2],100) 


# In[309]:


X = df.iloc[:,:-1]
y = df.iloc[:,-1]


# In[313]:


df.drop(['y'],axis=1,inplace=True)


# In[325]:


# 전체 평균 보다 못한 사람 제외
df.loc[df.mean(1) >= df.mean(1).mean(),:]


# In[321]:


df.values.mean()


# In[339]:


# 1 iloc사용 2 loc사용
# c2 과목의 중간값 보다 c2 과목의 성적이 큰 사람들의  c3과목의 평균
df.loc[df.c2 > df.c2.median(),'c3'].mean()
df.iloc[(df.c2 > df.c2.median()).values,np.where(df.columns == 'c3')[0]].mean()


# In[355]:


## 과목중 평균이 가장 높은 과목
df.columns[max(df.mean(0)) == df.mean(0)][0]
df.mean(0).argsort().head(1).index[0]


# In[359]:


np.random.seed(100)
df = pd.DataFrame(np.random.choice([0,1],(1000,8)))


# In[366]:


## 1 여성호르몬,  0 남성 호르몬 여성호르몬이 짝수면  여성 그렇지 않으면 남성
## 여성의 총 호르몬의 개수
df.loc[df.sum(1) % 2 == 0,:].sum().sum()
df.iloc[(df.sum(1) % 2 == 0).values,:].sum().sum()


# In[371]:


df[0]
df.iloc[:,0]
df.loc[:,0]
df.rename(columns={0:'c0'},inplace=True)


# In[372]:


df.c0


# In[377]:


df[8] = np.ones(1000,dtype=np.int_)


# In[378]:


df


# In[380]:


df.loc[1000] = 1


# In[381]:


df


# In[383]:


df.loc[1001] = [1,1,1,0,0,1,0,1,0]


# In[384]:


df


# In[385]:


df.loc[1003] = 1


# In[386]:


df


# In[391]:


df.iloc[0,0] = 10
df.loc[0,1:3] = 10
df.iloc[1,1:4] = 10
df.drop(index=[3],inplace=True)


# In[395]:


df.loc[2:5,1:4] = 0


# In[400]:


df.iloc[2:5,1:5] = 10
df


# In[401]:


df = pd.DataFrame(np.arange(12).reshape(3,4))
df.columns = ['kor','eng','math','sci']
df.index = ['광우','태우', '시형']


# In[403]:


df['total'] = df.sum(1)


# In[409]:


df.transpose()


# In[424]:


np.random.seed(100)
df = pd.DataFrame(np.random.randint(5,100,(365,3)))


# In[425]:


df.columns = ['태우','영훈','광우']


# In[426]:


df['sales_date'] = np.datetime64('2023-08-05') + np.arange(365)


# In[427]:


df = df[['sales_date','태우','영훈','광우']]


# In[ ]:


2023-08


# In[455]:


## 현재까지 가장많이 판 사람은?
df[['태우','영훈','광우']].sum(0).argmin()
df[['태우','영훈','광우']].sum(0).index[df[['태우','영훈','광우']].sum(0).argmin()]
## 2023년도 8월에 가장 많이 판 사람은?
new_data = df.loc[df.sales_date.astype('str').str.slice(0,7) == '2023-08',["태우","영훈","광우"]].sum(0)


# In[471]:


new_data.reset_index().loc[new_data.argmax(),'index']
new_data.index[new_data.argmax()]
df.loc[df.sales_date.astype('str').str.slice(0,7) == '2023-08',['태우','영훈','광우']].sum(0).index[df.loc[df.sales_date.astype('str').str.slice(0,7) == '2023-08',['태우','영훈','광우']].sum(0).argmin()]


# In[473]:


df['sales_month'] = df.sales_date.astype('str').str[0:7]


# In[475]:


## 월별 가장 많이 판 사람 
df_month_sales = pd.DataFrame(columns=['sales_month','best_seller'])
df_month_sales


# In[493]:


df_month_sales
#pd.DataFrame(tmp_dic)


# In[495]:


man_list = []
for idx, val in enumerate(df.sales_month.unique()):
    man = df.iloc[(df.sales_month==val).values,[1,2,3]].sum(0).index[df.iloc[(df.sales_month==val).values,[1,2,3]].sum(0).argmax()]
    man_list.append(man)
df_sales_month = pd.DataFrame()
df_sales_month['sales_month'] = df.sales_month.unique()
df_sales_month['best_seller'] = man_list


# In[503]:


# 각 월별 가장 많이 판 사람을 우승자라 할때 제일 많은 우승은?
x, y = np.unique(df_sales_month.best_seller,return_counts=True)
x[np.argmax(y)]


# In[507]:


df.set_index('sales_date',inplace=True)


# In[513]:


df.reset_index(inplace=True)
df


# In[516]:


df['new_date'] = np.random.choice(df.sales_date,365,replace=False)


# In[518]:


df.set_index('new_date',inplace=True)


# In[521]:


df.drop(columns=['sales_date'],inplace=True)


# In[523]:


df.reset_index(inplace=True)


# In[529]:


df = df.set_index('new_date').sort_index().reset_index()


# In[530]:


df


# In[538]:


arr1 = np.arange(3) 
arr2 = np.arange(100)


# In[541]:


pd.Series(arr1,index=[100,101,102]) + pd.Series(arr2)


# In[545]:


arr1 = np.random.randn(2,3)
arr2 = np.random.randn(2,2)


# In[547]:


df1 = pd.DataFrame(arr1) 
df2 = pd.DataFrame(arr2)


# In[548]:


df1


# In[549]:


df2


# In[550]:


df1 + df2


# In[551]:


df1.columns = ['a','b','c']


# In[553]:


df1.shape


# In[554]:


df2.shape


# In[556]:


df1.columns = np.arange(3)


# In[559]:


df2 + df1


# In[561]:


df1.add(df2,fill_value=0)


# In[562]:


df1


# In[563]:


df2


# In[564]:


df2.index = [2,3]


# In[566]:


df1.add(df2,fill_value=0)


# In[567]:


import seaborn as sns


# In[571]:


titanic = sns.load_dataset('titanic')


# In[572]:


type(titanic)


# In[576]:


titanic.head(3)
titanic.tail(3)


# In[577]:


titanic.columns


# In[578]:


titanic.shape


# In[581]:


titanic.info()


# In[585]:


titanic.dtypes['age']


# In[586]:


titanic.describe()


# In[588]:


titanic.isna().sum()


# In[589]:


titanic.head()
titanic.columns
titanic.info()
titanic.shape
titanic.dtypes
titanic.describe()
titanic.isna().sum()


# In[590]:


df = pd.DataFrame(np.random.randint(60,101,(10,3)))
df.columns = ['kor','eng','math']
df['name'] = np.random.choice(['a','b','c'],10)


# In[592]:


df.set_index('name',inplace=True)


# In[594]:


df1 = pd.DataFrame(np.random.randint(60,101,(10,3)))
df1.columns = ['kor','eng','math']
df1['name'] = np.random.choice(['a','b','c'],10)
df1.set_index('name',inplace=True)


# In[597]:


df # 1 학기에 시험점수
df1 # 2 학기에 시험점수


# In[630]:


score1 = pd.DataFrame([100,90,70])
score1.columns = ['first']
score1['name'] = ['c','b','a']


# In[631]:


score2 = pd.DataFrame([100,90,70])
score2.columns = ['second']
score2['name'] = ['b','a','c']


# In[633]:


score1.set_index('name')['first'] +  score2.set_index('name')['second'] 


# In[611]:


for n in np.unique(score1.name):
    f = score1.loc[score1.name == n,'first']
    s = score2.loc[score2.name == n, 'second']
    print(f"{n} ==> {f} {s}")


# In[627]:


score1.set_index('name',inplace=True)
score1


# In[629]:


score1['first']


# In[634]:


df  = pd.DataFrame(np.random.randint(60,101,(100,4)))


# In[635]:


df.columns = ['kor','eng','math','sci']


# In[640]:


df.to_csv("score.csv",index=None)
df.to_excel("score.xlsx",index=None)


# In[641]:


score = pd.read_csv('score.csv')


# In[643]:


iris_df = pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")


# In[644]:


iris_df


# In[645]:


iris_df.to_excel('iris.xlsx')


# In[646]:


sample = pd.read_clipboard()


# In[649]:


import json


# In[654]:


dic  = {'a':1,'b':2}


# In[661]:


type(eval((str(dic))))


# In[666]:


url = "https://gist.githubusercontent.com/omarish/5687264/raw/7e5c814ce6ef33e25d5259c1fe79463c190800d9/mpg.csv"
mpg_df = pd.read_csv(url)


# In[668]:


mpg_df.head()


# In[670]:


mpg_df.columns


# In[671]:


mpg_df.info()


# In[673]:


mpg_df.shape[0]


# In[676]:


mpg_df.mpg.value_counts()
x, y = np.unique(mpg_df.mpg,return_counts=True)


# In[681]:


mpg_df.mpg.mean()
mpg_df.iloc[:,2].mean()


# In[684]:


mpg_df.iloc[:,:3].mean()


# In[697]:


mpg_df[mpg_df.dtypes.index[mpg_df.dtypes != 'object']].median()


# In[696]:


mpg_df.iloc[:,np.arange(mpg_df.shape[1])[mpg_df.dtypes != 'object']].median()


# In[701]:


mpg_df.describe().loc['std',:]


# In[702]:


import matplotlib.pyplot as plt


# In[703]:


mpg_df


# In[704]:


plt.hist(mpg_df.weight)


# In[709]:


np.unique(mpg_df.cylinders)


# In[712]:


cylinder = mpg_df.groupby('cylinders')['mpg'].mean()


# In[714]:


plt.bar(cylinder.index,cylinder.values)


# In[718]:


plt.pie(cylinder)


# In[719]:


import plotly.express as px


# In[720]:


df = px.data.tips()
fig = px.pie(df, values='tip', names='day')
fig.show()


# In[726]:


fig= px.pie(df, values='tip', names='day')
fig


# In[729]:


df.groupby('smoker')['tip'].mean()


# In[730]:


fig = px.pie(df,values='tip',names='smoker')
fig.show


# In[734]:


x = np.random.choice([1,2,50],1000,p=[(99/100)*0.5,(99/100)*0.5,1/100])


# In[735]:


plt.boxplot(x)


# In[738]:


x = np.random.randn(1000)


# In[739]:


x = np.hstack([x,[10]])


# In[740]:


plt.boxplot(x)


# In[742]:


df = pd.DataFrame(np.random.randint(30,50,365))


# In[744]:


df['date'] = np.datetime64('2023-08-05') + np.arange(365)


# In[760]:


d = np.random.choice(df.date,df.date.size,replace=False)
df['d'] = d


# In[761]:


x = df.loc[df.d.astype('str').str.slice(0,7) == '2023-08',:]
plt.plot(x.d,x[0])


# In[766]:


titanic.isna().sum()
titanic_raw = titanic.copy()


# In[769]:


titanic.drop(columns=['deck'],inplace=True)


# In[776]:


titanic1 = titanic.loc[titanic.loc[~titanic.embark_town.isna(),:].index,:]


# In[782]:


titanic.drop(index=titanic.loc[titanic.embark_town.isna(),:].index,inplace=True)


# In[786]:


titanic.age[888] = titanic.age.mean()


# In[792]:


titanic.age[titanic.loc[titanic.age.isna(),:].index] = titanic.age.mean()


# In[794]:


titanic.isna().sum().sum()


# In[800]:


np.random.seed(100)
subject = ['kor','eng','math','sci']
df = pd.DataFrame(np.random.choice([90,80,70,60,50,np.nan],(100,4)),columns = subject)


# In[828]:


type(warnings)


# In[830]:


import warnings
warnings.filterwarnings(action='ignore')


# In[831]:


## 열에 nan 25%이상이면 컬럼을 지우고 (컬럼)
## 행에 nan 2개 이상아면  행을 지우고 (인덱스)
## 나머지 nan 은 각 컬럼의 평균으로 채우기 
## 각 컬럼의 평균을 구하면?
df.columns[df.isna().sum() > df.shape[0]*0.25]
df.loc[df.isna().sum(1) < 2, :].mean()
for col in df.columns:
    df.loc[df[col].isna(),col] = df.loc[:,col].mean()
df.mean()


# In[814]:


df = df.loc[df.isna().sum(1) < 2, :]


# In[833]:


for col in df.columns:
    df.loc[df[col].isna(),col] = df.loc[:,col].mean()


# In[820]:


df.isna().sum()


# In[834]:


data = np.hstack([np.random.randint(1,10,100),np.nan])
df = pd.DataFrame(np.random.choice(data,size=(1000,10)))


# In[839]:


df.columns = 'col'+df.columns.astype('str')


# In[862]:


# df.col0 = df.col0.fillna(df.col0.mode())
# df.col1 = df.col1.fillna(df.col1.mode())
for col in df.columns:
    df[col] = df[col].fillna(df[col].mean())


# In[864]:


df = pd.DataFrame({
    'c1':['a','a','b','a','b'],
    'c2':[1,1,1,2,2],
    'c3':[1,1,2,2,2]
})


# In[874]:


df.duplicated(keep='last')


# In[871]:


df.loc[~df.duplicated(),:]


# In[875]:


df = pd.DataFrame(np.random.randint(60,101,size=(5,3)))
df['name'] = np.random.choice(['a','b','c'],5)


# In[879]:


df.name.duplicated()


# In[907]:


## 1. normalize (x - x.mean())/x.std()
## 2. minmax (x-x.min())/(x.max()-x.min())
## 3. robust ( x-x.median()/np.percentile(x,75) - np.percentile(x,25))
# 
df = pd.DataFrame()
df['height']  = np.random.randint(160,190,10)
df['weight'] = np.random.randint(40,90,10)


# In[886]:


(df.height - df.height.mean())/df.height.std()


# In[887]:


(df.weight - df.weight.mean())/df.weight.std()


# In[899]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# In[889]:


sc = StandardScaler()


# In[891]:


sc.fit(df[['height']])
aa = sc.transform(df[['height']])


# In[894]:


mc = MinMaxScaler()
mc.fit(df[['height']])
aa = mc.transform(df[['height']])


# In[896]:


bb = (df.height - df.height.min())/(df.height.max()-df.height.min())


# In[900]:


rc = RobustScaler()
aa = rc.fit_transform(df[['weight']])


# In[901]:


aa


# In[905]:


bb = (df.weight - df.weight.median())/(np.percentile(df.weight,75)-np.percentile(df.weight,25))


# In[906]:


bb


# In[910]:


titanic.age


# In[912]:





# In[913]:


url = "https://gist.githubusercontent.com/omarish/5687264/raw/7e5c814ce6ef33e25d5259c1fe79463c190800d9/mpg.csv"
mpg_df = pd.read_csv(url)


# In[914]:


mpg_df


# In[923]:


mpg_df.dtypes


# In[934]:


idx = []
for i in range(mpg_df.horsepower.size):
    try:
        float(mpg_df.horsepower[i])
    except:
        idx.append(i)


# In[936]:


mpg_df.drop(index=idx,inplace=True)


# In[937]:


mpg_df.horsepower = mpg_df.horsepower.astype(np.float_)


# In[944]:


count, bin_dividers = np.histogram(mpg_df.horsepower,bins=4)


# In[945]:


bin_name = ['저출력','보통출력','고출력']
mpg_df['horse_bin'] = pd.cut(mpg_df.horsepower,
                         bins=bin_dividers,
                         labels=['저출력','보통','고출력','초고출력'],
                         include_lowest=True)


# In[947]:


np.unique(mpg_df.horse_bin,return_counts=True)


# In[953]:


data


# In[993]:


#data = np.hstack([np.linspace(0,3,0.2),np.nan])
df = pd.DataFrame(np.random.choice(data,(30,24)))


# In[994]:


df.columns = df.columns.astype('str')+'hour'


# In[1010]:


## 0hour에 nan이 있으면 시간대의  평균으로 값을 채움
## 이전시간으로 값을 채움
df.fillna(0,inplace=True)


# In[1023]:


len([1,23,'add'])


# In[1027]:


## 각 날자별로 날자의 전기사용량의  중간값 보나 낮게 쓴 시간 
def find_hour(x):
    return list(df.columns[x.median() < x])
df.apply(find_hour,1)
df.apply(lambda x: list(df.columns[x.median() < x]),1)


# In[1037]:


df[df.columns[np.arange(24) % 2 == 1][::-1]]


# In[1042]:


titanic.loc[(titanic['class'] == 'Third') | (titanic['class'] == 'second'),:].shape


# In[1045]:


titanic.loc[titanic['class'].isin(['Third','second']),:].shape


# In[1050]:


df1 = pd.DataFrame()
df1['a'] = [ 'a'+str(i) for i in range(4)]
df1['b'] = [ 'b'+str(i) for i in range(4)]
df1['c'] = [ 'c'+str(i) for i in range(4)]
df1.index = [0,1,2,3]


# In[1055]:


df2 = pd.DataFrame()
df2['a'] = [ 'a'+str(i) for i in range(2,6)]
df2['b'] = [ 'b'+str(i) for i in range(2,6)]
df2['c'] = [ 'c'+str(i) for i in range(2,6)]
df2['d'] = [ 'd'+str(i) for i in range(2,6)]
df2.index = np.arange(2,6)


# In[1062]:


pd.concat([df1,df2,df1,df1,df1],axis=1) # outerjoin


# In[1069]:


pd.merge(df1,df2,how='right')


# In[1070]:


df = titanic[['age','sex','class','fare','survived']]


# In[1084]:


df.groupby(['class','sex'])[['fare','age']].mean().reset_index()


# In[1086]:
import pandas as pd
import numpy as np

## 5개의 지점  a,b,c,d,e 
## sales
np.random.seed(10)
df = pd.DataFrame()
df['date'] = (np.datetime64('2023-01-01') + np.arange(365)).astype('str')
df['location'] = np.random.choice([i for i in 'abcde'], 365)
df['sales'] = np.random.randint(1,30,365)


# In[1088]:


df = pd.DataFrame(columns= ['date','location','sales'])
for date in (np.datetime64('2023-01-01') + np.arange(365)):
    for loc in 'abcde':
        sales = np.random.randint(0,30,1)
        tmp_df = pd.DataFrame({'date':date,'location':loc,'sales':sales})
        df = pd.concat([df,tmp_df])


# In[1093]:


## 월별 지점별 판매량
df.groupby(df.location)['sales'].sum()


# In[1097]:
df.groupby([df.date.astype('str').str.slice(0,7),'location'])['sales'].sum()


#%%
import pandas as pd

# 예시 데이터 프레임 (df) 생성
data = {
    'month': [1, 1, 2, 2, 3, 3],
    'location': ['A', 'B', 'A', 'B', 'A', 'B'],
    'sales': [100, 200, 150, 250, 200, 300]
}
df = pd.DataFrame(data)

# 월별, 지점별 판매량 합계
df_month = df.groupby(['month', 'location'])['sales'].sum().reset_index()

# 월별 판매량 1등인 지점
df_month_best = df_month.loc[df_month.groupby('month')['sales'].idxmax()].reset_index(drop=True)

# 결과 데이터프레임 초기화
df_monthly_best_location = pd.DataFrame(columns=['month', 'location', 'sales'])

# 각 월별로 가장 판매량이 높은 지점 찾기
for i in range(len(df_month_best)):
    month = df_month_best.loc[i, 'month']
    sales = df_month_best.loc[i, 'sales']
    location = df_month_best.loc[i, 'location']
    tmp_df = pd.DataFrame({'month': [month], 'location': [location], 'sales': [sales]})
    df_monthly_best_location = pd.concat([df_monthly_best_location, tmp_df], ignore_index=True)

print(df_monthly_best_location)
# %%
date = np.datetime64('2024-01-01','h') + np.arange(24*365)
# %%
date_time = np.random.choice(date,100_000)
# %%
df = pd.DataFrame()
df['usage'] = np.abs(np.random.randn(100_000)*3 +1)
# %%
