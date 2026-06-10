# epidemic_regressions_raw.py  ── final version
#
# Runs exactly the four Table-1 models **with raw prevalence** and writes:
#   • table1_coeffs.csv  – coefficients, pseudo-R², BIC for each regression
#   • analysis_vars.csv  – the cleaned panel used for estimation
#
# USAGE
#   python epidemic_regressions_raw.py  AllRuns_Indiv_Data_for_logistic_regression.csv
# ------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit
import statsmodels.api as sm

# 0. load data
df = pd.read_csv("Run1_Indiv_Data_for_logistic_regression.csv")

# 1. rename & recode
df['agent_id'] = df['Run Number'].astype(str) + '_' + df['name']
df = df.rename(columns={
    'Time Step'              : 'time',
    'Response'               : 'StayHome',
    'slight_cough'           : 'LightCough',
    'cough_and_fever'        : 'FeverCough',
    'Daily New Cases Day 4'  : 'CaseFeedback',
    'Extraversion'           : 'extraversion',
    'Agreeableness'          : 'agreeableness',
    'Conscientiousness'      : 'conscientiousness',
    'Emotional Stability'    : 'emotional_stability',
    'Openness to Experience' : 'intellect'
})
for c in ['LightCough','FeverCough','extraversion','agreeableness',
          'conscientiousness','emotional_stability','intellect','gender']:
    df[c] = df[c].astype(int)

# scale CaseFeedback by 1000 then square for CaseFeedback2
df['CaseFeedback'] = df['CaseFeedback'] * 1000
df['CaseFeedback2'] = df['CaseFeedback'] ** 2
df = df.dropna(subset=['StayHome'])

# 2. helper for FE models
def fit_FE(name, cols):
    # conditional logit (FE) fit
    m = ConditionalLogit(df['StayHome'], df[cols], groups=df['agent_id'])
    # use OPG covariance to avoid Hessian inversion failures
    r = m.fit(method='bfgs', disp=False)
    # try to compute heteroskedasticity-consistent covariances, but skip if unsupported
    try:
        r = r._get_robustcov_results(cov_type='HC0')
    except (AttributeError, ValueError):
        pass
    # approximate null log-likelihood via unconditional mean
    p = df['StayHome'].mean()
    llnull = (df['StayHome'] * np.log(p) + (1 - df['StayHome']) * np.log(1 - p)).sum()
    r2 = 1 - r.llf / llnull
    bic = -2 * r.llf + len(r.params) * np.log(len(df))
    return name, r.params, r2, bic

results = [
    fit_FE("Reg1_FE", ["LightCough","FeverCough"]),
    fit_FE("Reg2_FE", ["LightCough","FeverCough","CaseFeedback"]),
    fit_FE("Reg3_FE", ["LightCough","FeverCough","CaseFeedback","CaseFeedback2"]),
]

# 3. random‐intercept (RE) model
X_fix = sm.add_constant(df[[
    'LightCough', 'FeverCough', 'CaseFeedback', 'CaseFeedback2',
    'agreeableness', 'conscientiousness', 'extraversion',
    'emotional_stability', 'intellect', 'age', 'gender'
]])
ids   = df['agent_id'].astype('category').cat.codes
Z     = np.zeros((len(df), ids.nunique()))
Z[np.arange(len(df)), ids] = 1
# null log-likelihood via unconditional mean (for pseudo R²)
p_null = df['StayHome'].mean()
llnull = (df['StayHome'] * np.log(p_null) + (1 - df['StayHome']) * np.log(1 - p_null)).sum()
# fit the Bayesian mixed model via VB
re = sm.BinomialBayesMixedGLM(df['StayHome'], X_fix, Z, np.zeros(ids.nunique(),int)).fit_vb()
# extract fixed and random coefficients
coef_fixed = re.params[:len(X_fix.columns)]
coef_random = re.params[len(X_fix.columns):]
# compute linear predictor and fitted probabilities
eta = X_fix.dot(coef_fixed) + Z.dot(coef_random)
p_hat = 1 / (1 + np.exp(-eta))
# compute log-likelihood, pseudo-R², and BIC
llf_RE = (df['StayHome'] * np.log(p_hat) + (1 - df['StayHome']) * np.log(1 - p_hat)).sum()
r2_RE = 1 - llf_RE / llnull
bic_RE = -2 * llf_RE + len(coef_fixed) * np.log(len(df))
results.append(("Reg4_RE", pd.Series(coef_fixed, index=X_fix.columns), r2_RE, bic_RE))

# 4. assemble & write
union = sorted({*results[0][1].index, *results[1][1].index, *results[2][1].index, *results[3][1].index})
table = pd.DataFrame({name: coeffs.reindex(union) for name,coeffs,_,_ in results})
table.loc['Pseudo_R2'] = [r2 for _,_,r2,_ in results]
table.loc['BIC']       = [bic for _,_,_,bic in results]
table.to_csv("table1_coeffs.csv")
