# %% [markdown]
# # Part E: alternative models
# Same data, same train/test split (train Jan 2021-Jun 2025, test Jul 2025-Jun 2026)
# and same MASE definition as Part C, so the numbers slot into the validation table.

# %%
import numpy as np, pandas as pd, warnings
import statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from pathlib import Path
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd().parent
data = pd.read_csv(ROOT/'part_c/handoff/model_input.csv', encoding='utf-8-sig', parse_dates=['date'])
data['log_tows'] = np.log1p(data.tows)
train = data[data.date <= '2025-06-30'].copy()
test = data[data.date >= '2025-07-01'].copy()
# MASE scale: each borough's in-sample seasonal-naive (lag 7) MAE, as in Part C
scales = train.assign(e=(train.tows - train.groupby('borough').tows.shift(7)).abs()).groupby('borough').e.mean()
def mase(pred):
    err = (test.tows - pred).abs().groupby(test.borough).mean()
    return (err/scales).mean(), (test.tows - pred).abs().mean()

M3 = 'C(borough) + C(dow) + C(month) + trend + moving_day + snow_7c + snow_8_14 + snow_1 + rain_1 + temp_7 + C(borough):snow_7c'
rows = []

# %% [markdown]
# ## 1. OLS on log(tows+1): the Part C model, for reference

# %%
ols = smf.ols('log_tows ~ ' + M3, train).fit()
rows.append(['OLS log(tows+1) (Part C model 3)', *mase(np.maximum(np.expm1(ols.predict(test)), 0))])

# %% [markdown]
# ## 2. Count models: Poisson and Negative Binomial GLMs
# Model tow counts directly (no log+1 and no back-transform). NB adds an
# overdispersion parameter because winter spikes make variance >> mean.

# %%
pois = smf.glm('tows ~ ' + M3, train, family=sm.families.Poisson()).fit()
nb = smf.negativebinomial('tows ~ ' + M3, train).fit(method='bfgs', maxiter=2000, disp=0)
alpha = nb.params['alpha']
print(f'NB overdispersion alpha = {alpha:.2f} (0 would mean Poisson is fine)')
print(f'Pearson dispersion of Poisson fit = {pois.pearson_chi2/pois.df_resid:.1f} (should be ~1)')
for name, m in [('Poisson GLM', pois), ('Negative Binomial GLM', nb)]:
    mu = m.predict(test)
    rows.append([name + ' (mean)', *mase(mu)])
# NB predictive median (MAE is minimised by the median, so this is the fair point forecast)
from scipy.stats import nbinom
mu = nb.predict(test); n = 1/alpha
rows.append(['Negative Binomial GLM (median)', *mase(nbinom.median(n, n/(n+mu)))])

# %% [markdown]
# ### Do the borough snow effects agree with OLS?
# % change in tows per +10 cm of past-week snow = exp(10*slope) - 1

# %%
def borough_pct(params):
    base = params['snow_7c']
    return {b: 100*np.expm1(10*(base + params.get(f'C(borough)[T.{b}]:snow_7c', 0))) for b in sorted(data.borough.unique())}
eff = pd.DataFrame({'OLS_pct': borough_pct(ols.params), 'NB_pct': borough_pct(nb.params)}).round(0)
eff['OLS_rank'] = eff.OLS_pct.rank(ascending=False).astype(int)
eff['NB_rank'] = eff.NB_pct.rank(ascending=False).astype(int)
eff = eff.sort_values('OLS_rank')
rho = spearmanr(eff.OLS_pct, eff.NB_pct).correlation

# Extra cars per +10 cm, same scenario as Part C: Dec-Mar test calendar, other weather = 0, 0 vs 10 cm
base = test[test.month.isin([12,1,2,3])].copy()
for c in ['snow_7c','snow_8_14','snow_1','rain_1','temp_7']: base[c] = 0.0
plus = base.assign(snow_7c=10.0)
eff['NB_extra_cars'] = (nb.predict(plus) - nb.predict(base)).groupby(base.borough).mean().round(1)
pc = pd.read_csv(ROOT/'part_c/tables/extra_cars_scenario.csv', encoding='utf-8-sig').set_index('borough')
eff['OLS_extra_cars'] = pc.extra_tows.round(1)   # from Part C output (run part_c_analysis.py first)
rho_cars = spearmanr(eff.OLS_extra_cars, eff.NB_extra_cars).correlation
print(eff.to_string())
print(f'Spearman rank correlation OLS vs NB -- % effect: {rho:.2f} | extra cars: {rho_cars:.2f}')

# %% [markdown]
# ## 3. Logistic regression: "will tomorrow be a big towing day citywide?"
# Surge day = 150+ tows citywide (the plan's reference line). Uses only past weather + calendar,
# so it can flag the FIRST day of an operation (which "yesterday's count" never can).

# %%
city = data.groupby('date').agg(tows=('tows','sum'), snow_7c=('snow_7c','first'), snow_8_14=('snow_8_14','first'),
        snow_1=('snow_1','first'), rain_1=('rain_1','first'), temp_7=('temp_7','first'), dow=('dow','first'),
        month=('month','first'), trend=('trend','first'), moving_day=('moving_day','first'))
city['surge'] = (city.tows >= 150).astype(int)
quiet3 = pd.concat([city.tows.shift(k) < 150 for k in [1,2,3]], axis=1).all(axis=1)
city['start'] = (city.surge.eq(1) & quiet3).astype(int)        # same onset rule as Part C extension
ctr, cte = city[city.index <= '2025-06-30'], city[city.index >= '2025-07-01']
logit = smf.logit('surge ~ snow_7c + snow_8_14 + snow_1 + rain_1 + temp_7 + C(dow) + C(month) + trend + moving_day',
                  ctr).fit(disp=0, method='bfgs', maxiter=2000)
p = logit.predict(cte); flag = p >= 0.5
yest = cte.tows.shift(1).fillna(ctr.tows.iloc[-1]).ge(150)
def clf(f):
    t = cte.surge.astype(bool)
    return dict(AUC=round(roc_auc_score(cte.surge, f.astype(float) if f.dtype==bool else f),3),
                surge_days=int(t.sum()), caught=int((f>=0.5 if f.dtype!=bool else f)[t].sum()),
                false_alarms=int(((f>=0.5 if f.dtype!=bool else f) & ~t).sum()),
                starts=int(cte.start.sum()), starts_caught=int(((f>=0.5 if f.dtype!=bool else f) & cte.start.astype(bool)).sum()))
logit_res = pd.DataFrame({'Logistic (weather+calendar)': clf(p), 'Naive: surge if surge yesterday': clf(yest)}).T
print(logit_res.to_string())
print('Logit odds ratio per +10 cm past-week snow:', round(np.exp(10*logit.params['snow_7c']), 1))

# %% [markdown]
# ## 4. Regression with ARIMA errors (citywide, dynamic regression)
# log(city tows+1) on past weather + calendar, with ARMA(1,1) errors to soak up the
# day-to-day persistence that makes OLS residuals autocorrelated. One-step-ahead forecasts.

# %%
X = (city[['snow_7c','snow_8_14','snow_1','rain_1','temp_7','moving_day','trend']]
     .join(pd.get_dummies(city.dow, prefix='d', drop_first=True))
     .join(pd.get_dummies(city.month, prefix='m', drop_first=True))).astype(float)
y = np.log1p(city.tows)
ntr = len(ctr)
arx = SARIMAX(y.iloc[:ntr], exog=X.iloc[:ntr], order=(1,0,1), trend='c').fit(disp=0, maxiter=500)
full = arx.apply(y, exog=X)                            # same parameters, filtered through test
one_step = np.maximum(np.expm1(full.get_prediction(start=ntr).predicted_mean), 0)
ols_city = sm.OLS(y.iloc[:ntr], sm.add_constant(X.iloc[:ntr])).fit()
ols_city_pred = np.maximum(np.expm1(ols_city.predict(sm.add_constant(X.iloc[ntr:]))), 0)
cscale = (ctr.tows - ctr.tows.shift(7)).abs().mean()
lb_ols = sm.stats.acorr_ljungbox(ols_city.resid, lags=[7]).lb_pvalue.iloc[0]
lb_arx = sm.stats.acorr_ljungbox(arx.resid, lags=[7]).lb_pvalue.iloc[0]
city_res = pd.DataFrame({
    'Citywide OLS (weather+calendar)': [(cte.tows-ols_city_pred).abs().mean()/cscale, (cte.tows-ols_city_pred).abs().mean(), lb_ols],
    'Regression + ARMA(1,1) errors':   [(cte.tows-one_step).abs().mean()/cscale, (cte.tows-one_step).abs().mean(), lb_arx],
    'Naive: yesterday':                [(cte.tows-cte.tows.shift(1).fillna(ctr.tows.iloc[-1])).abs().mean()/cscale,
                                        (cte.tows-cte.tows.shift(1).fillna(ctr.tows.iloc[-1])).abs().mean(), np.nan],
}, index=['MASE','MAE (cars/day)','Ljung-Box p (lag 7)']).T.round(3)
print(city_res.to_string())
arx_flag = pd.Series(one_step.values >= 150, index=cte.index)
print('ARIMA-errors model, surge starts flagged:', int((arx_flag & cte.start.astype(bool)).sum()), 'of', int(cte.start.sum()))

# %% [markdown]
# ## Summary: borough-day accuracy on the test year (lower = better)

# %%
summary = pd.DataFrame(rows, columns=['model','test_MASE','test_MAE']).round(3)
summary.loc[len(summary)] = ['Naive: yesterday (Part C extension)', 0.926, 10.16]
summary.loc[len(summary)] = ['Boosted correction (Part C extension)', 0.700, 7.71]
print(summary.to_string(index=False))
out = ROOT/'part_e_alt_models'
summary.to_csv(out/'borough_day_scores.csv', index=False); eff.to_csv(out/'borough_effects_ols_vs_nb.csv')
logit_res.to_csv(out/'surge_classifier.csv'); city_res.to_csv(out/'citywide_arima_errors.csv')
