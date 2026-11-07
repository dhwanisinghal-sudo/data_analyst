import pandas as pd, numpy as np, sqlite3, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score, mean_absolute_error
sns.set_theme(style="whitegrid")
URL="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
df=pd.read_csv(URL); R={}
R["shape"]=df.shape; R["missing"]={k:int(v) for k,v in df.isna().sum().items() if v>0}
num=["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]
# 1 Pandas
g=df.groupby("species")[num].mean().round(1); R["group_means"]=g.reset_index().to_dict("records")
piv=df.pivot_table(index="island",columns="species",values="body_mass_g",aggfunc="count",fill_value=0); R["pivot"]=piv.reset_index().to_dict("records")
sexm=df.dropna(subset=["sex"]).pivot_table(index="species",columns="sex",values="body_mass_g",aggfunc="mean").round(0); R["sexmass"]=sexm.reset_index().to_dict("records")
# 2 NumPy
X=df[num].dropna().to_numpy(); z=(X-X.mean(axis=0))/X.std(axis=0)
R["z_out"]=int((np.abs(z)>3).any(axis=1).sum()); R["np_n"]=int(X.shape[0])
cm=np.corrcoef(X,rowvar=False).round(2); R["np_corr_flip_mass"]=float(cm[2,3]); R["np_corr_depth_len"]=float(cm[0,1])
# 3 SQL
con=sqlite3.connect(":memory:"); df.to_sql("penguins",con,index=False)
q="""SELECT species, island, COUNT(*) AS n, ROUND(AVG(body_mass_g),0) AS avg_mass, ROUND(AVG(flipper_length_mm),1) AS avg_flipper
FROM penguins WHERE body_mass_g IS NOT NULL GROUP BY species, island ORDER BY avg_mass DESC"""
R["sql"]=pd.read_sql(q,con).to_dict("records"); R["sql_query"]=q
# 4 SciPy
grp=[d.body_mass_g.dropna() for _,d in df.groupby("species")]
F,p=stats.f_oneway(*grp); R["anova"]=[float(F),float(p)]
m=df.dropna(subset=["sex"]); t,pt=stats.ttest_ind(m[m.sex=="MALE"].body_mass_g,m[m.sex=="FEMALE"].body_mass_g,equal_var=False); R["ttest"]=[float(t),float(pt)]
r,pr=stats.pearsonr(*df[["flipper_length_mm","body_mass_g"]].dropna().T.values); R["pearson"]=[float(r),float(pr)]
# 5 Seaborn charts
fig,ax=plt.subplots(1,2,figsize=(10,4))
sns.scatterplot(data=df,x="flipper_length_mm",y="body_mass_g",hue="species",ax=ax[0]);ax[0].set_title("Flipper Length vs Body Mass")
sns.boxplot(data=df,x="species",y="body_mass_g",hue="species",ax=ax[1],legend=False);ax[1].set_title("Body Mass by Species")
plt.tight_layout();plt.savefig("t1.png",dpi=150);plt.close()
fig,ax=plt.subplots(1,2,figsize=(10,4))
sns.histplot(data=df,x="bill_length_mm",hue="species",kde=True,ax=ax[0]);ax[0].set_title("Bill Length Distribution")
sns.scatterplot(data=df,x="bill_length_mm",y="bill_depth_mm",hue="species",ax=ax[1]);ax[1].set_title("Bill Length vs Bill Depth")
plt.tight_layout();plt.savefig("t2.png",dpi=150);plt.close()
# 6 scikit-learn classification
d=df.dropna(subset=num+["species"]); Xc=d[num+["island","sex"]]; y=d["species"]
pre=ColumnTransformer([("n",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),num),
 ("c",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore"))]),["island","sex"])])
Xtr,Xte,ytr,yte=train_test_split(Xc,y,test_size=.25,stratify=y,random_state=42)
res={}
for name,mdl in [("Logistic Regression",LogisticRegression(max_iter=1000)),("Random Forest",RandomForestClassifier(n_estimators=200,random_state=42))]:
    pl=Pipeline([("p",pre),("m",mdl)]); cv=cross_val_score(pl,Xc,y,cv=StratifiedKFold(5,shuffle=True,random_state=42))
    pl.fit(Xtr,ytr); pr_=pl.predict(Xte)
    res[name]={"cv":float(cv.mean()),"cv_std":float(cv.std()),"test":float(accuracy_score(yte,pr_))}
    if name=="Random Forest":
        labs=sorted(y.unique()); cmx=confusion_matrix(yte,pr_,labels=labs)
        fig,ax=plt.subplots(figsize=(5,4));sns.heatmap(cmx,annot=True,fmt="d",cmap="Blues",xticklabels=labs,yticklabels=labs,ax=ax);ax.set_xlabel("Predicted");ax.set_ylabel("Actual");ax.set_title("Random Forest Confusion Matrix")
        plt.tight_layout();plt.savefig("t3.png",dpi=150);plt.close()
        names=num+list(pl.named_steps["p"].named_transformers_["c"].named_steps["o"].get_feature_names_out(["island","sex"]))
        imp=pd.Series(pl.named_steps["m"].feature_importances_,index=names).sort_values()
        fig,ax=plt.subplots(figsize=(6,4));imp.plot.barh(ax=ax,color="#2e75b6");ax.set_title("Random Forest Feature Importance");plt.tight_layout();plt.savefig("t4.png",dpi=150);plt.close()
        R["imp"]=imp.sort_values(ascending=False).round(3).head(4).to_dict()
R["clf"]=res; R["test_n"]=len(yte)
# regression
dr=df.dropna(subset=num+["species"]); Xr=pd.get_dummies(dr[["flipper_length_mm","bill_length_mm","bill_depth_mm","species"]],drop_first=True).astype(float); yr=dr.body_mass_g
Xa,Xb,ya,yb=train_test_split(Xr,yr,test_size=.25,random_state=42); lr=LinearRegression().fit(Xa,ya); pr2=lr.predict(Xb)
R["reg"]={"r2":float(r2_score(yb,pr2)),"mae":float(mean_absolute_error(yb,pr2))}
json.dump(R,open("tres.json","w"),indent=1,default=str)
