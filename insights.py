"""Analytics Report & Insights: NYC taxi operations, March 2019."""
import pandas as pd, numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid")
URL="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv"
raw=pd.read_csv(URL,parse_dates=["pickup","dropoff"]); R={}
# ---- data quality & cleaning
d=raw.copy(); d["mins"]=(d.dropoff-d.pickup).dt.total_seconds()/60
R["raw_n"]=len(d)
bad=(d.distance<=0)|(d.mins<=0)|(d.mins>180)|(d.fare<=0)
d["mph"]=d.distance/(d.mins/60)
bad|=d.mph>80
R["removed"]=int(bad.sum()); d=d[~bad].copy(); R["n"]=len(d)
R["miss_pay"]=int(d.payment.isna().sum()); R["miss_zone"]=int(d.pickup_zone.isna().sum())
d["hour"]=d.pickup.dt.hour; d["dow"]=d.pickup.dt.day_name(); d["date"]=d.pickup.dt.date
d["weekend"]=d.pickup.dt.dayofweek>=5
d["rev_per_min"]=d.total/d.mins; d["rev_per_mile"]=d.total/d.distance
d["tip_pct"]=np.where(d.fare>0,d.tip/d.fare*100,np.nan)
# ---- KPIs
R["rev"]=float(d.total.sum()); R["avg_total"]=float(d.total.mean()); R["avg_fare"]=float(d.fare.mean())
R["avg_dist"]=float(d.distance.mean()); R["avg_mins"]=float(d.mins.mean()); R["rpm"]=float(d.total.sum()/d.mins.sum())
R["rpmile"]=float(d.total.sum()/d.distance.sum())
card=d[d.payment=="credit card"]; R["card_share"]=float((d.payment=="credit card").sum()/d.payment.notna().sum()*100)
R["tip_pct_card"]=float(card.tip.sum()/card.fare.sum()*100); R["tip_zero_card"]=float((card.tip==0).mean()*100)
R["tip_avg_card"]=float(card.tip.mean()); R["cash_tip_total"]=float(d[d.payment=="cash"].tip.sum())
R["tips_total"]=float(d.tip.sum()); R["tips_share"]=float(d.tip.sum()/d.total.sum()*100)
R["tolls_total"]=float(d.tolls.sum())
# ---- time
h=d.groupby("hour").agg(rides=("total","size"),rev=("total","sum"),rpm=("rev_per_min","median"),avg=("total","mean"))
tot_rpm=d.groupby("hour").apply(lambda x:x.total.sum()/x.mins.sum()); h["rpm_w"]=tot_rpm
R["peak_hour"]=int(h.rides.idxmax()); R["peak_rides"]=int(h.rides.max()); R["low_hour"]=int(h.rides.idxmin()); R["low_rides"]=int(h.rides.min())
R["best_rpm_hour"]=int(h.rpm_w.idxmax()); R["best_rpm"]=float(h.rpm_w.max()); R["worst_rpm_hour"]=int(h.rpm_w.idxmin()); R["worst_rpm"]=float(h.rpm_w.min())
R["rev_6_9pm"]=float(h.loc[17:20,"rev"].sum()/h.rev.sum()*100)  # 17-20 = 5pm-8:59pm
R["rides_5_9pm"]=float(h.loc[17:20,"rides"].sum()/h.rides.sum()*100)
R["hours_tbl"]={int(k):{"rides":int(v.rides),"rpm":round(float(v.rpm_w),2),"avg":round(float(v.avg),2)} for k,v in h.iterrows()}
dw=d.groupby("dow").agg(rides=("total","size"),rev=("total","sum"),avg=("total","mean"),dist=("distance","mean"))
order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]; dw=dw.reindex(order)
R["dow"]={k:{"rides":int(v.rides),"avg":round(float(v.avg),2),"rev":round(float(v.rev),0)} for k,v in dw.iterrows()}
we=d.groupby("weekend").agg(rides=("total","size"),avg=("total","mean"),dist=("distance","mean"),mins=("mins","mean"),rpm=("rev_per_min","median"))
R["weekend"]={str(k):{c:round(float(v[c]),2) for c in we.columns} for k,v in we.iterrows()}
# ---- geography
b=d.groupby("pickup_borough").agg(rides=("total","size"),rev=("total","sum"),avg=("total","mean"),dist=("distance","mean"))
b["share"]=b.rev/b.rev.sum()*100; b=b.sort_values("rev",ascending=False)
R["borough"]={k:{"rides":int(v.rides),"rev":round(float(v.rev),0),"avg":round(float(v.avg),2),"dist":round(float(v.dist),2),"share":round(float(v.share),1)} for k,v in b.iterrows()}
z=d.groupby("pickup_zone").agg(rides=("total","size"),rev=("total","sum"),avg=("total","mean")).sort_values("rev",ascending=False)
R["top_zones"]=[{"zone":k,"rides":int(v.rides),"rev":round(float(v.rev),0),"avg":round(float(v.avg),2)} for k,v in z.head(8).iterrows()]
R["top10_zone_share"]=float(z.head(10).rev.sum()/z.rev.sum()*100); R["n_zones"]=int(len(z))
# airports
ap=d[d.pickup_zone.fillna("").str.contains("Airport|JFK|LaGuardia",case=False)|d.dropoff_zone.fillna("").str.contains("Airport|JFK|LaGuardia",case=False)]
R["airport_n"]=len(ap); R["airport_share_rides"]=float(len(ap)/len(d)*100); R["airport_share_rev"]=float(ap.total.sum()/d.total.sum()*100); R["airport_avg"]=float(ap.total.mean()); R["nonap_avg"]=float(d.drop(ap.index).total.mean())
# ---- trip length bands
d["band"]=pd.cut(d.distance,[0,1,2,5,10,100],labels=["<1 mi","1-2 mi","2-5 mi","5-10 mi","10+ mi"])
bd=d.groupby("band",observed=True).agg(rides=("total","size"),rev=("total","sum"),avg=("total","mean"),rpm=("rev_per_min","median"),rpmile=("rev_per_mile","median"))
bd["share_rides"]=bd.rides/bd.rides.sum()*100; bd["share_rev"]=bd.rev/bd.rev.sum()*100
R["band"]={str(k):{c:round(float(v[c]),2) for c in bd.columns} for k,v in bd.iterrows()}
# ---- fleet colour
cl=d.groupby("color").agg(rides=("total","size"),rev=("total","sum"),avg=("total","mean"),dist=("distance","mean"),rpm=("rev_per_min","median"))
R["color"]={k:{c:round(float(v[c]),2) for c in cl.columns} for k,v in cl.iterrows()}
# ---- payment & tipping
pm=d.dropna(subset=["payment"]).groupby("payment").agg(rides=("total","size"),avg_fare=("fare","mean"),avg_total=("total","mean"))
R["pay"]={k:{c:round(float(v[c]),2) for c in pm.columns} for k,v in pm.iterrows()}
card=d[d.payment=="credit card"]
tb=card.groupby("band",observed=True).apply(lambda x:x.tip.sum()/x.fare.sum()*100)
R["tip_by_band"]={str(k):round(float(v),1) for k,v in tb.items()}
th=card.groupby("hour").apply(lambda x:x.tip.sum()/x.fare.sum()*100); R["tip_hr_best"]=[int(th.idxmax()),float(th.max())]; R["tip_hr_worst"]=[int(th.idxmin()),float(th.min())]
tw=card.groupby("pickup_borough").apply(lambda x:x.tip.sum()/x.fare.sum()*100); R["tip_by_borough"]={k:round(float(v),1) for k,v in tw.items()}
pp=d.groupby(d.passengers.clip(upper=4)).total.mean().round(2); R["pax"]={int(k):float(v) for k,v in pp.items()}
R["solo_share"]=float((d.passengers==1).mean()*100)
corr=d[["distance","mins","fare","tip","tolls","total","passengers"]].corr(); R["corr_fare_dist"]=float(corr.loc["fare","distance"])
# ---- charts (dashboard)
fig,ax=plt.subplots(2,2,figsize=(11,8))
ax[0,0].bar(h.index,h.rides,color="#2e75b6");ax[0,0].set_title("Rides by Hour of Day");ax[0,0].set_xlabel("Pickup hour");ax[0,0].set_ylabel("Rides")
ax[0,1].plot(h.index,h.rpm_w,marker="o",color="#c0504d");ax[0,1].set_title("Revenue per Minute by Hour ($)");ax[0,1].set_xlabel("Pickup hour")
ax[1,0].bar(dw.index.str[:3],dw.rides,color="#4f9d69");ax[1,0].set_title("Rides by Day of Week");ax[1,0].set_ylabel("Rides")
bb=b.copy();ax[1,1].barh(bb.index[::-1],bb.rev[::-1],color="#7f6000");ax[1,1].set_title("Revenue by Pickup Borough ($)")
plt.tight_layout();plt.savefig("i1.png",dpi=150);plt.close()
fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].bar(bd.index.astype(str),bd.share_rides,label="% of rides",color="#9dc3e6",width=.4,align="edge")
ax[0].bar(bd.index.astype(str),bd.share_rev,label="% of revenue",color="#1f4e79",width=-.4,align="edge")
ax[0].set_title("Share of Rides vs Revenue by Trip Length");ax[0].set_ylabel("%");ax[0].legend()
ax[1].bar(bd.index.astype(str),bd.rpm,color="#ed7d31");ax[1].set_title("Median Revenue per Minute by Trip Length ($)")
plt.tight_layout();plt.savefig("i2.png",dpi=150);plt.close()
fig,ax=plt.subplots(1,2,figsize=(11,4))
tz=z.head(8).iloc[::-1];ax[0].barh(tz.index,tz.rev,color="#2e75b6");ax[0].set_title("Top 8 Pickup Zones by Revenue ($)")
tbp=pd.Series(R["tip_by_band"]);ax[1].bar(tbp.index,tbp.values,color="#70ad47");ax[1].set_title("Tip % of Fare by Trip Length (card payments)");ax[1].set_ylabel("%")
plt.tight_layout();plt.savefig("i3.png",dpi=150);plt.close()
json.dump(R,open("ires.json","w"),indent=1,default=str)
