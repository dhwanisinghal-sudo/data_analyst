import pandas as pd, numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid")
URL="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df=pd.read_csv(URL)
R={}
R["shape"]=df.shape
R["dups"]=int(df.duplicated().sum())
R["missing"]={k:int(v) for k,v in df.isna().sum().items() if v>0}
R["dtypes"]={k:str(v) for k,v in df.dtypes.items()}
# cleaning
d=df.copy()  # duplicates flagged but retained: no unique passenger ID
R["rows_after_dedup"]=len(d)
d=d.drop(columns=["deck","alive","class","who","embark_town","adult_male"])
d["embarked"]=d["embarked"].fillna(d["embarked"].mode()[0])
d["age_missing"]=d["age"].isna()
d["age"]=d.groupby(["pclass","sex"])["age"].transform(lambda s:s.fillna(s.median()))
d["sex"]=d["sex"].astype("category"); d["embarked"]=d["embarked"].astype("category")
d["survived"]=d["survived"].astype(int)
# outliers fare IQR
q1,q3=d.fare.quantile([.25,.75]); iqr=q3-q1; hi=q3+1.5*iqr
R["fare_outliers"]=int((d.fare>hi).sum()); R["fare_hi"]=float(hi)
R["fare_max"]=float(d.fare.max())
d["fare_capped"]=d.fare.clip(upper=d.fare.quantile(.99))
d["family"]=d.sibsp+d.parch+1
d["age_group"]=pd.cut(d.age,[0,12,18,35,60,100],labels=["Child","Teen","Young adult","Adult","Senior"])
d.to_csv("titanic_clean.csv",index=False)
R["desc"]=d[["age","fare","family"]].describe().round(2).to_dict()
R["overall"]=round(d.survived.mean()*100,1)
R["sex"]=(d.groupby("sex",observed=True).survived.mean()*100).round(1).to_dict()
R["pclass"]=(d.groupby("pclass").survived.mean()*100).round(1).to_dict()
R["agegrp"]=(d.groupby("age_group",observed=True).survived.mean()*100).round(1).to_dict()
R["embarked"]=(d.groupby("embarked",observed=True).survived.mean()*100).round(1).to_dict()
R["family"]=(d.groupby(pd.cut(d.family,[0,1,4,20],labels=["Alone","2-4","5+"]),observed=True).survived.mean()*100).round(1).to_dict()
R["sexclass"]=(d.groupby(["sex","pclass"],observed=True).survived.mean()*100).round(1).unstack().to_dict()
corr=d[["survived","pclass","age","sibsp","parch","fare","family"]].corr().round(2)
R["corr_surv"]=corr["survived"].to_dict()
R["n"]=len(d)
json.dump(R,open("results.json","w"),indent=1,default=str)
# charts
fig,ax=plt.subplots(1,2,figsize=(10,4))
sns.barplot(x="sex",y="survived",data=d,ax=ax[0],palette="Set2",errorbar=None);ax[0].set_title("Survival Rate by Sex");ax[0].set_ylabel("Survival rate")
sns.barplot(x="pclass",y="survived",data=d,ax=ax[1],palette="Set2",errorbar=None);ax[1].set_title("Survival Rate by Passenger Class");ax[1].set_ylabel("Survival rate")
plt.tight_layout();plt.savefig("c1_bar.png",dpi=150);plt.close()
fig,ax=plt.subplots(figsize=(7,4))
sns.histplot(data=d,x="age",hue="survived",bins=25,kde=True,multiple="layer",ax=ax,palette=["#d9534f","#2e8b57"])
ax.set_title("Age Distribution by Survival (0 = died, 1 = survived)");plt.tight_layout();plt.savefig("c2_hist.png",dpi=150);plt.close()
fig,ax=plt.subplots(1,2,figsize=(10,4))
sns.boxplot(x="pclass",y="fare",data=df,ax=ax[0],palette="Set2");ax[0].set_title("Fare by Class (raw, outliers visible)")
sns.boxplot(x="pclass",y="age",data=d,ax=ax[1],palette="Set2");ax[1].set_title("Age by Class (cleaned)")
plt.tight_layout();plt.savefig("c3_box.png",dpi=150);plt.close()
fig,ax=plt.subplots(figsize=(7,4))
sns.scatterplot(data=d,x="age",y="fare_capped",hue="survived",style="pclass",alpha=.7,ax=ax,palette=["#d9534f","#2e8b57"])
ax.set_title("Age vs Fare (capped at 99th percentile)");plt.tight_layout();plt.savefig("c4_scatter.png",dpi=150);plt.close()
fig,ax=plt.subplots(figsize=(7,4))
order=["Child","Teen","Young adult","Adult","Senior"]
s=d.groupby("age_group",observed=True).survived.mean().reindex(order)*100
ax.plot(order,s.values,marker="o",color="#1f6feb");ax.set_title("Survival Rate Across Age Groups");ax.set_ylabel("Survival rate (%)")
plt.tight_layout();plt.savefig("c5_line.png",dpi=150);plt.close()
fig,ax=plt.subplots(figsize=(6.5,5))
sns.heatmap(corr,annot=True,cmap="coolwarm",center=0,ax=ax);ax.set_title("Correlation Heatmap")
plt.tight_layout();plt.savefig("c6_heat.png",dpi=150);plt.close()
