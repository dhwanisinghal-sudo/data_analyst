"""Capstone: Predicting diamond prices (data cleaning, EDA, ML, evaluation, insights)."""
import pandas as pd, numpy as np, json, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
sns.set_theme(style="whitegrid"); RS=42; R={}
URL="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
raw=pd.read_csv(URL); R["raw_n"]=len(raw)
# ---------- data quality
R["dups"]=int(raw.duplicated().sum()); R["zero_dim"]=int(((raw[["x","y","z"]]==0).any(axis=1)).sum())
R["big_dim"]=int(((raw.y>30)|(raw.z>30)).sum())
d=raw.drop_duplicates()
d=d[(d.x>0)&(d.y>0)&(d.z>0)&(d.y<30)&(d.z<30)].copy()
R["n"]=len(d); R["removed"]=R["raw_n"]-R["n"]
# depth consistency check (depth% = 2z/(x+y))
d["depth_calc"]=2*d.z/(d.x+d.y)*100
R["depth_mismatch"]=float(((d.depth-d.depth_calc).abs()>1).mean()*100)
cut_o={"Fair":1,"Good":2,"Very Good":3,"Premium":4,"Ideal":5}
col_o={"J":1,"I":2,"H":3,"G":4,"F":5,"E":6,"D":7}
cla_o={"I1":1,"SI2":2,"SI1":3,"VS2":4,"VS1":5,"VVS2":6,"VVS1":7,"IF":8}
d["cut_n"]=d.cut.map(cut_o); d["color_n"]=d.color.map(col_o); d["clarity_n"]=d.clarity.map(cla_o)
d["volume"]=d.x*d.y*d.z; d["log_carat"]=np.log(d.carat); d["log_volume"]=np.log(d.volume)
d["log_price"]=np.log(d.price)
R["price_skew"]=float(d.price.skew()); R["logprice_skew"]=float(d.log_price.skew())
R["desc"]=d[["carat","price","depth","table"]].describe().round(2).to_dict()
# ---------- EDA
R["sp_carat"]=float(stats.spearmanr(d.carat,d.price)[0])
R["sp"]={c:round(float(stats.spearmanr(d[c],d.price)[0]),2) for c in ["carat","cut_n","color_n","clarity_n","depth","table"]}
cp=d.groupby("cut").agg(price=("price","mean"),carat=("carat","mean"),n=("price","size")).reindex(["Fair","Good","Very Good","Premium","Ideal"])
R["cut_tbl"]={k:{"price":round(float(v.price)),"carat":round(float(v.carat),2),"n":int(v.n)} for k,v in cp.iterrows()}
band=d[(d.carat>=0.9)&(d.carat<=1.1)].groupby("cut").price.mean().reindex(cp.index); R["cut_1ct"]={k:round(float(v)) for k,v in band.items()}
kw=stats.kruskal(*[g.price.values for _,g in d.groupby("clarity")]); R["kruskal_clarity"]=[float(kw[0]),float(kw[1])]
R["carat_peak"]={"share_round":float(d.carat.isin([0.3,0.4,0.5,0.7,1.0,1.5,2.0]).mean()*100)}
ones=d[(d.carat>=0.95)&(d.carat<1.05)]; R["n_1ct"]=len(ones)
R["clarity_1ct"]={k:round(float(v)) for k,v in ones.groupby("clarity").price.mean().reindex(list(cla_o)).items()}
R["color_1ct"]={k:round(float(v)) for k,v in ones.groupby("color").price.mean().reindex(list(col_o)).items()}
# charts EDA
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
sns.histplot(d.price,bins=60,ax=ax[0],color="#2e75b6");ax[0].set_title("Price (skewed right)");ax[0].set_xlabel("Price ($)")
sns.histplot(d.log_price,bins=60,ax=ax[1],color="#4f9d69");ax[1].set_title("log(Price) (near symmetric)");ax[1].set_xlabel("log price")
plt.tight_layout();plt.savefig("c1.png",dpi=150);plt.close()
s=d.sample(6000,random_state=RS)
fig,ax=plt.subplots(1,2,figsize=(10,4))
sns.scatterplot(data=s,x="carat",y="price",hue="clarity",hue_order=list(cla_o),s=10,alpha=.6,ax=ax[0],legend=False);ax[0].set_title("Price vs Carat (curved)")
sns.scatterplot(data=s,x="log_carat",y="log_price",hue="clarity",hue_order=list(cla_o),s=10,alpha=.6,ax=ax[1]);ax[1].set_title("log Price vs log Carat (near linear)")
ax[1].legend(title="Clarity",fontsize=7,title_fontsize=8,ncol=2)
plt.tight_layout();plt.savefig("c2.png",dpi=150);plt.close()
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
ax[0].bar(cp.index,cp.price,color="#ed7d31");ax[0].set_title("Average Price by Cut ($)")
ax[1].bar(cp.index,cp.carat,color="#7f6000");ax[1].set_title("Average Carat by Cut")
plt.tight_layout();plt.savefig("c3.png",dpi=150);plt.close()
fig,ax=plt.subplots(figsize=(6.5,5));cm=d[["price","carat","cut_n","color_n","clarity_n","depth","table","volume"]].corr(method="spearman")
sns.heatmap(cm,annot=True,fmt=".2f",cmap="coolwarm",center=0,ax=ax);ax.set_title("Spearman Correlation");plt.tight_layout();plt.savefig("c4.png",dpi=150);plt.close()
# ---------- modelling
feats=["carat","log_carat","volume","log_volume","x","y","z","depth","table","cut_n","color_n","clarity_n"]
X=d[feats]; y=d.log_price
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=RS)
R["train_n"]=len(Xtr); R["test_n"]=len(Xte)
def metrics(y_true_log,pred_log):
    yt=np.exp(y_true_log); yp=np.exp(pred_log)
    return {"rmse":float(np.sqrt(mean_squared_error(yt,yp))),"mae":float(mean_absolute_error(yt,yp)),"r2":float(r2_score(yt,yp)),
            "r2_log":float(r2_score(y_true_log,pred_log)),"mdape":float(np.median(np.abs(yp-yt)/yt)*100),"mape":float(np.mean(np.abs(yp-yt)/yt)*100)}
models={"Baseline (typical price)":DummyRegressor(),"Linear Regression":LinearRegression(),"Ridge":Ridge(alpha=1.0),
        "Random Forest":RandomForestRegressor(n_estimators=100,min_samples_leaf=2,n_jobs=-1,random_state=RS),
        "Gradient Boosting":HistGradientBoostingRegressor(random_state=RS)}
kf=KFold(3,shuffle=True,random_state=RS); res={}; fitted={}
for n,m in models.items():
    t=time.time(); cv=cross_val_score(m,Xtr,ytr,cv=kf,scoring="r2"); m.fit(Xtr,ytr); p=m.predict(Xte)
    res[n]=metrics(yte.values,p); res[n]["cv_r2_log"]=float(cv.mean()); res[n]["cv_std"]=float(cv.std()); fitted[n]=p
    print(n,round(time.time()-t,1),res[n]["r2"],res[n]["mae"],flush=True)
# tuning
t=time.time()
rs=RandomizedSearchCV(HistGradientBoostingRegressor(random_state=RS),{"learning_rate":[.03,.05,.1,.2],"max_iter":[200,400,600],"max_leaf_nodes":[15,31,63],"min_samples_leaf":[10,20,40],"l2_regularization":[0,.1,1]},
                      n_iter=8,cv=3,scoring="r2",random_state=RS,n_jobs=1)
rs.fit(Xtr,ytr); best=rs.best_estimator_; R["best_params"]=rs.best_params_; R["tuned_cv"]=float(rs.best_score_)
pt=best.predict(Xte); res["Gradient Boosting (tuned)"]=metrics(yte.values,pt); res["Gradient Boosting (tuned)"]["cv_r2_log"]=float(rs.best_score_); res["Gradient Boosting (tuned)"]["cv_std"]=float(rs.cv_results_["std_test_score"][rs.best_index_])
fitted["Gradient Boosting (tuned)"]=pt; print("tuned",round(time.time()-t,1),res["Gradient Boosting (tuned)"],flush=True)
R["models"]=res
# ---------- evaluation of final model
yt=np.exp(yte.values); yp=np.exp(pt); err=yp-yt; ape=np.abs(err)/yt*100
ev=pd.DataFrame({"y":yt,"p":yp,"ape":ape,"err":err,"carat":Xte.carat.values,"cut":d.loc[Xte.index,"cut"].values,"clarity":d.loc[Xte.index,"clarity"].values})
ev["pband"]=pd.cut(ev.y,[0,1000,2500,5000,10000,20000],labels=["<$1k","$1k-2.5k","$2.5k-5k","$5k-10k","$10k+"])
eb=ev.groupby("pband",observed=True).agg(n=("y","size"),mdape=("ape","median"),mae=("err",lambda x:np.abs(x).mean()))
R["err_band"]={str(k):{"n":int(v.n),"mdape":round(float(v.mdape),1),"mae":round(float(v.mae),0)} for k,v in eb.iterrows()}
ec=ev.groupby("cut").ape.median().reindex(["Fair","Good","Very Good","Premium","Ideal"]); R["err_cut"]={k:round(float(v),1) for k,v in ec.items()}
R["within10"]=float((ape<=10).mean()*100); R["within5"]=float((ape<=5).mean()*100); R["within20"]=float((ape<=20).mean()*100)
R["over20_n"]=int((ape>20).sum()); w=ev.sort_values("ape",ascending=False).head(5)[["carat","cut","clarity","y","p"]].copy(); w["carat"]=w.carat.round(2); w["y"]=w.y.round(0); w["p"]=w.p.round(0); R["worst"]=w.to_dict("records")
R["bias_mean_pct"]=float(((yp-yt)/yt*100).mean()); R["under_share"]=float((yp<yt).mean()*100)
ev_big=ev[ev.y>=15000]; R["n_15k"]=len(ev_big); R["mdape_15k"]=float(ev_big.ape.median()) if len(ev_big) else None
# permutation importance
idx=np.random.RandomState(RS).choice(len(Xte),3000,replace=False)
pi=permutation_importance(best,Xte.iloc[idx],yte.iloc[idx],n_repeats=3,random_state=RS,scoring="r2")
imp=pd.Series(pi.importances_mean,index=feats).sort_values(ascending=False); R["perm_imp"]={k:round(float(v),3) for k,v in imp.items()}
# charts model
mm=pd.DataFrame(res).T.drop(index="Baseline (typical price)")
fig,ax=plt.subplots(1,2,figsize=(10.5,4))
ax[0].barh(mm.index[::-1],mm.mae[::-1],color="#2e75b6");ax[0].set_title("Test MAE ($), lower is better")
ax[1].barh(mm.index[::-1],mm.mdape[::-1],color="#ed7d31");ax[1].set_title("Median % error, lower is better")
plt.tight_layout();plt.savefig("c5.png",dpi=150);plt.close()
fig,ax=plt.subplots(1,2,figsize=(10.5,4.2))
ax[0].hexbin(np.log10(ev.y),np.log10(ev.p),gridsize=60,cmap="Blues",mincnt=1,bins="log");lim=[2.4,4.4];ax[0].plot(lim,lim,"r--",lw=1);ax[0].set_xlabel("log10 actual price");ax[0].set_ylabel("log10 predicted price");ax[0].set_title("Predicted vs Actual (tuned GB)")
sns.histplot((np.log(yp)-np.log(yt)),bins=60,ax=ax[1],color="#4f9d69");ax[1].set_title("Residuals (log scale)");ax[1].set_xlabel("log(pred) - log(actual)")
plt.tight_layout();plt.savefig("c6.png",dpi=150);plt.close()
fig,ax=plt.subplots(1,2,figsize=(10.5,4))
ax[0].bar(eb.index.astype(str),eb.mdape,color="#7f6000");ax[0].set_title("Median % Error by Price Band");ax[0].set_ylabel("%")
imp.head(8).iloc[::-1].plot.barh(ax=ax[1],color="#2e75b6");ax[1].set_title("Permutation Importance (drop in R²)")
plt.tight_layout();plt.savefig("c7.png",dpi=150);plt.close()
json.dump(R,open("cres.json","w"),indent=1,default=str)
