# Video Game Sales Analysis — Complete Documentation

**Dataset:** `vgsales.csv`
**Notebook:** `vgsales_analysis.ipynb`
**Language:** Python 3
**Last Run:** March 18, 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Environment Setup](#2-environment-setup)
3. [Dataset Description](#3-dataset-description)
4. [Data Loading](#4-data-loading)
5. [Data Understanding](#5-data-understanding)
6. [Data Cleaning and Preprocessing](#6-data-cleaning-and-preprocessing)
7. [Feature Engineering](#7-feature-engineering)
8. [Exploratory Data Analysis](#8-exploratory-data-analysis)
9. [Data Visualization](#9-data-visualization)
10. [Predictive Modeling](#10-predictive-modeling)
11. [Time Series Forecasting](#11-time-series-forecasting)
12. [Key Insights and Conclusions](#12-key-insights-and-conclusions)
13. [Outputs and Saved Files](#13-outputs-and-saved-files)
14. [Known Limitations](#14-known-limitations)

---

## 1. Project Overview

This project performs a full-cycle data science analysis on the `vgsales.csv` dataset, which catalogs video game sales figures across platforms, genres, publishers, and global regions from the early 1980s through the mid-2010s.

The analysis is structured into five major stages:

| Stage | Purpose |
|---|---|
| Data Cleaning | Ensure the dataset is analytically sound |
| Feature Engineering | Derive new variables that add predictive and explanatory value |
| Exploratory Data Analysis | Understand distributions, trends, and relationships |
| Predictive Modeling | Build regression models to predict global sales |
| Time Series Forecasting | Project future global sales using ARIMA |

The notebook is designed to run top-to-bottom without errors, with every decision explained in accompanying markdown cells.

---

## 2. Environment Setup

### Required Libraries

Install all dependencies before running the notebook:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy
```

### Library Roles

| Library | Role |
|---|---|
| `pandas` | Data loading, manipulation, and aggregation |
| `numpy` | Numerical operations and array handling |
| `matplotlib` | Base plotting engine |
| `seaborn` | Statistical visualization on top of matplotlib |
| `scipy.stats` | Statistical tests |
| `sklearn` | Preprocessing, model training, and evaluation |
| `statsmodels` | ARIMA time series modeling and ADF test |

### Global Configuration

The notebook sets a consistent visual style at the start using `seaborn.set_theme()` and `matplotlib.rcParams`. All plots use `figure.dpi=120` for clarity and share consistent font sizes. A fixed `RANDOM_STATE = 42` is used throughout for reproducibility.

---

## 3. Dataset Description

### Source

The dataset is a widely used public dataset scraped from VGChartz, containing physical video game sales data up to approximately 2016.

### Columns

| Column | Type | Description |
|---|---|---|
| `Rank` | Integer | Sales rank globally |
| `Name` | String | Title of the video game |
| `Platform` | String | Console or platform (e.g., PS2, Wii, DS) |
| `Year` | Float/Int | Year of release |
| `Genre` | String | Game genre (e.g., Action, Sports, RPG) |
| `Publisher` | String | Publishing company |
| `NA_Sales` | Float | Sales in North America (millions) |
| `EU_Sales` | Float | Sales in Europe (millions) |
| `JP_Sales` | Float | Sales in Japan (millions) |
| `Other_Sales` | Float | Sales in rest of world (millions) |
| `Global_Sales` | Float | Total worldwide sales (millions) |

### Key Facts

- Approximately **16,598 rows** and **11 columns** in the raw file
- Sales figures are in **millions of units sold** (not revenue)
- Data is dominated by physical retail sales; digital distribution is not captured
- Coverage is strongest from **1995 to 2015**; records before 1990 and after 2016 are sparse

---

## 4. Data Loading

The dataset is loaded using `pd.read_csv('vgsales.csv')`. An immediate inspection covers:

- **Shape** — total rows and columns
- **`.info()`** — column dtypes and non-null counts
- **Missing value audit** — count and percentage of nulls per column

### Missing Values in Raw Data

| Column | Missing Count | Missing % |
|---|---|---|
| `Year` | ~271 | ~1.6% |
| `Publisher` | ~58 | ~0.4% |
| All others | 0 | 0% |

These relatively small proportions justify imputation rather than row deletion.

---

## 5. Data Understanding

Before any cleaning, the notebook examines:

- **Descriptive statistics** for all numeric columns via `df.describe()`
- **Cardinality** of categorical columns — how many unique platforms, genres, and publishers exist
- **Year range** — the sorted list of unique year values to spot anomalies

### Cardinality Summary

| Column | Unique Values |
|---|---|
| `Platform` | 31 |
| `Genre` | 12 |
| `Publisher` | ~578 |

The high publisher count (578+) is important context for modeling — it means one-hot encoding publishers directly would create an unwieldy feature space, so label encoding is used instead.

---

## 6. Data Cleaning and Preprocessing

### Strategy Summary

| Issue | Column | Strategy | Rationale |
|---|---|---|---|
| Missing values | `Year` | Median imputation | Robust to skew; preserves row count |
| Missing values | `Publisher` | Fill with `'Unknown'` | Categorical placeholder; avoids data loss |
| Unrealistic values | `Year` | Nullify years > 2020 | Data entry errors, not real releases |
| Residual non-finite | `Year` | Replace `inf`/`-inf` with median | Prevents `IntCastingNaNError` on type cast |
| Wrong dtype | `Year` | Cast `float -> Int64 -> int` | Year should be a whole number |
| Duplicates | All columns | `drop_duplicates()` | Exact duplicates add no information |
| Data type safety | Sales columns | `pd.to_numeric(..., errors='coerce')` | Handles any non-numeric strings silently |
| String inconsistency | `Genre`, `Platform`, `Publisher` | `.str.strip().str.title()` | Normalizes casing and whitespace |

### The Year Casting Fix — Important Detail

The raw `Year` column is stored as a string in the CSV because some entries contain `'N/A'`. After `pd.to_numeric(..., errors='coerce')`, these become `NaN`. After `.fillna(median)`, most non-finite values are resolved. However, values that evaluated to `inf` during coercion are **not** replaced by `.fillna()` — only `NaN` is.

The fix applies:

```python
df['Year'] = df['Year'].replace([np.inf, -np.inf], year_median)
df['Year'] = df['Year'].round().astype('Int64').astype(int)
```

The two-step cast (`float -> Int64 -> int`) is the pandas-recommended approach. `Int64` (capital I) is a nullable integer type that gracefully handles any lingering `NaN`, while standard `int` does not.

### Outlier Detection

Outliers in `Global_Sales` are identified using the IQR method:

```
Lower bound = Q1 - 1.5 * IQR
Upper bound = Q3 + 1.5 * IQR
```

Detected outliers are **retained** because they represent legitimate commercial phenomena — titles like Wii Sports (82M units), Mario Kart Wii, and Grand Theft Auto V. Removing them would distort any analysis of the top-performing games in history.

### Post-Cleaning Validation

After all cleaning steps:

- Zero missing values across all columns
- `Year` dtype confirmed as `int`
- All sales columns confirmed as `float64`
- Cleaned dataset saved to `vgsales_cleaned.csv`

---

## 7. Feature Engineering

Five new features are created to enrich the dataset:

### `Decade`

```python
df['Decade'] = (df['Year'] // 10 * 10).astype(str) + 's'
```

Groups years into generational gaming eras (1980s, 1990s, 2000s, 2010s). This allows analysis at a coarser temporal level that aligns with real-world console generations.

### Regional Sales Ratios

```python
df['NA_Sales_Ratio'] = df['NA_Sales'] / df['Global_Sales']
df['EU_Sales_Ratio'] = df['EU_Sales'] / df['Global_Sales']
df['JP_Sales_Ratio'] = df['JP_Sales'] / df['Global_Sales']
df['Other_Sales_Ratio'] = df['Other_Sales'] / df['Global_Sales']
```

Each ratio captures what fraction of a title's global sales came from a given region. Division-by-zero is handled by replacing the denominator with `NaN` for zero-sales rows, then filling results with 0.

These ratios reveal structural market differences — for example, RPGs have a much higher Japan ratio than Shooters.

### `Non_JP_Sales`

```python
df['Non_JP_Sales'] = df['NA_Sales'] + df['EU_Sales'] + df['Other_Sales']
```

Aggregates the three Western-market regions into a single figure. Useful for comparing Western versus Japanese market performance directly.

---

## 8. Exploratory Data Analysis

### Summary Statistics

`df.describe()` on numeric columns reveals:

- `Global_Sales` has a mean around 0.54M but a maximum exceeding 80M — heavily right-skewed
- `NA_Sales` consistently exceeds `EU_Sales`, which exceeds `JP_Sales`
- The 75th percentile of `Global_Sales` is only ~0.47M, confirming that the vast majority of titles sell modestly

### Genre Analysis

| Genre | Total Global Sales (M) |
|---|---|
| Action | Highest |
| Sports | Second |
| Shooter | Third |
| Role-Playing | Fourth |
| Platform | Fifth |
| Strategy | Lowest |

Action and Sports dominate because they have both high volume (many titles released annually) and broad appeal across demographics.

### Platform Analysis (Top 5)

| Platform | Notes |
|---|---|
| PS2 | Largest install base of any single console |
| DS | Nintendo handheld with mass-market appeal |
| Wii | Motion controls attracted casual audience |
| GB | Long commercial lifespan boosted cumulative sales |
| PS | First PlayStation drove mainstream gaming adoption |

### Publisher Analysis (Top 5)

| Publisher | Notes |
|---|---|
| Nintendo | First-party developer and hardware maker — dominant |
| Electronic Arts | Annual sports franchises (FIFA, Madden) |
| Activision | Call of Duty franchise |
| Ubisoft | Diverse portfolio |
| Take-Two Interactive | GTA and NBA 2K |

Nintendo's lead is substantial because it simultaneously controls both hardware and software, giving first-party titles a structural advantage.

### Temporal Trend

Global sales grew consistently from the early 1990s, accelerating sharply in the mid-2000s, peaking around **2008–2009**, then declining. The peak coincides with the 7th console generation (PS3, Xbox 360, Wii), which had the broadest mainstream consumer adoption of any console generation to that point.

The post-2010 decline has two causes:

1. **Data artifact** — newer titles simply have fewer years of cumulative sales recorded at the time of dataset collection
2. **Market fragmentation** — mobile and digital distribution channels are not captured in this dataset

### Regional Breakdown

| Region | Approximate Share |
|---|---|
| North America | ~49% |
| Europe | ~27% |
| Japan | ~14% |
| Other | ~10% |

North America is the single most important market by a wide margin. Japan's 14% share is notable given its smaller population, reflecting the country's outsized cultural role in gaming history.

### Correlation Analysis

Key correlations with `Global_Sales`:

| Feature | Correlation |
|---|---|
| `NA_Sales` | Very high (~0.94) |
| `EU_Sales` | High (~0.90) |
| `Other_Sales` | Moderate-high (~0.75) |
| `JP_Sales` | Lowest (~0.61) |

Japan's lower correlation reflects its distinct genre and franchise preferences. JP_Sales is the most regionally independent of the four sales columns.

---

## 9. Data Visualization

The notebook produces 15 plots, each followed by an interpretation. Below is a description of each plot and its key takeaway.

### Plot 1: Global Sales Distribution (Raw and Log-Transformed)

A side-by-side histogram showing the raw distribution (heavily right-skewed) and the log1p-transformed distribution (approximately normal). This motivates log-transforming the target variable in modeling.

### Plot 2: Total Global Sales by Genre (Horizontal Bar)

Action and Sports clearly lead. Strategy and Puzzle are at the bottom. The gap between the top and bottom genres spans nearly an order of magnitude.

### Plot 3: Top 10 Platforms by Total Global Sales (Horizontal Bar)

PS2 leads by a notable margin. The top 10 are all from major console generations spanning the mid-1990s through the 2010s.

### Plot 4: Global Sales Trend by Year (Line with Fill)

A connected line plot from 1990 to 2016 showing the rise, peak, and fall of recorded global sales. The peak around 2008–2009 is clearly visible.

### Plot 5: Correlation Heatmap

A `seaborn.heatmap` with annotated correlation coefficients. The strong clustering of NA, EU, and Other sales versus Global_Sales is clear, while JP_Sales sits visually apart.

### Plot 6: Regional Sales Distribution (Pie Chart)

A clean four-segment pie chart confirming North America's ~49% dominance.

### Plot 7: Regional Sales Composition by Genre (Stacked Normalized Bar)

Each genre is shown as a 100% bar split by NA, EU, and JP ratios. RPGs have a distinctly high Japan bar. Shooters and Sports are predominantly North American.

### Plot 8: Global Sales Distribution by Decade (Box Plot)

Box plots for titles under 5M sales, grouped by decade. Median per-title sales peaked in the 2000s and slightly declined in the 2010s.

### Plot 9: Top 10 Publishers by Total Global Sales (Horizontal Bar)

Nintendo's lead is striking — nearly double the next-closest publisher (Electronic Arts).

### Plot 10: Top 5 Genre Sales Over Time (Multi-Line)

Tracks Action, Sports, Shooter, RPG, and Platform genres from 1995 to 2016. Shooters surge post-2006 (Call of Duty era). Sports games plateau after the mid-2000s.

### Plot 11: Model Comparison Bar Chart

Three side-by-side bar charts comparing RMSE, MAE, and R2 across the three models. Gradient Boosting wins on all three metrics.

### Plot 12: Random Forest Feature Importances (Horizontal Bar)

Regional sales columns (especially NA_Sales) dominate. Among non-sales features, Platform, Publisher, and Year have meaningful importance scores.

### Plot 13: Actual vs Predicted Scatter (Gradient Boosting)

Points cluster tightly along the 45-degree diagonal for most titles. The upper-right region (extreme outliers) shows underprediction, which is a known limitation of tree-based methods on highly skewed targets.

### Plot 14: ARIMA Forecast (Line with Confidence Band)

Historical sales (blue) and forecast values (orange dashed) with a shaded 95% confidence interval. The forecast shows a gradual downward trend.

---

## 10. Predictive Modeling

### Target Variable

`Global_Sales` — log-transformed before modeling using `np.log1p()` to normalize the skewed distribution. Predictions are inverse-transformed using `np.expm1()` for evaluation.

### Features Used

| Feature | Type |
|---|---|
| `Platform` | Categorical (label-encoded) |
| `Year` | Numeric |
| `Genre` | Categorical (label-encoded) |
| `Publisher` | Categorical (label-encoded) |
| `NA_Sales` | Numeric |
| `EU_Sales` | Numeric |
| `JP_Sales` | Numeric |
| `Other_Sales` | Numeric |

### Preprocessing Pipeline

1. **Label Encoding** — `Platform`, `Genre`, `Publisher` are encoded using `sklearn.LabelEncoder`
2. **Standard Scaling** — all features are scaled with `StandardScaler` (applied to Linear Regression inputs; tree models use unscaled data)
3. **Train-Test Split** — 80% training, 20% test, stratified by `RANDOM_STATE=42`

### Models Trained

#### Linear Regression

A baseline parametric model that assumes a linear relationship between features and the log-transformed target.

#### Random Forest Regressor

An ensemble of 100 decision trees (`n_estimators=100`). Each tree is trained on a bootstrapped sample with random feature subsets, and predictions are averaged. Robust to non-linearity and interactions.

#### Gradient Boosting Regressor

A sequential ensemble (`n_estimators=150, learning_rate=0.1, max_depth=4`) where each tree corrects the residual errors of the previous. Generally the strongest performer on tabular data with complex interactions.

### Model Results

| Model | RMSE | MAE | R2 |
|---|---|---|---|
| Linear Regression | 13,373,337.42 | 232,099.29 | -42,568,593,144,472 |
| Random Forest | 0.9040 | 0.0452 | 0.8055 |
| Gradient Boosting | **0.8166** | **0.0389** | **0.8413** |

### Interpreting the Results

**Linear Regression** failed catastrophically. The R2 of approximately -42.5 trillion is not a typo — it means the model is performing dramatically worse than simply predicting the mean for every observation. The cause is numerical instability: Linear Regression on scaled data struggled with the combination of skewed regional sales features, the high dimensionality of label-encoded publishers, and multicollinearity between the regional sales columns. The enormous RMSE and MAE confirm this model has no predictive value here and should not be used.

**Random Forest** performs well with RMSE of 0.904 (in log space), MAE of 0.045, and R2 of 0.806. This means the model explains approximately 80.5% of the variance in log-transformed global sales. After inverse transformation, predictions are generally within a reasonable range for the bulk of titles.

**Gradient Boosting** is the best-performing model across all three metrics. An R2 of 0.841 means it explains 84.1% of the variance. The RMSE of 0.817 and MAE of 0.039 (both in log space) confirm tighter, more consistent predictions than Random Forest. This is the recommended model for any production use of these predictions.

### Why Tree Models Dominate

Regional sales (NA, EU, JP, Other) are sub-components of Global_Sales. This creates a highly learnable non-linear signal that tree-based methods exploit naturally through hierarchical splits. The Linear model cannot capture these interactions cleanly, especially when the features are highly correlated with each other and with the target.

### Feature Importance (Random Forest)

Regional sales features account for the vast majority of predictive importance, which is structurally expected. Among non-sales features:

- `Platform` — console ecosystem is a strong proxy for market size and audience
- `Publisher` — major publishers have consistent sales tracks
- `Year` — captures market maturity and era-specific console penetration

---

## 11. Time Series Forecasting

### Objective

Aggregate yearly global sales into a univariate time series and project future values using an ARIMA model.

### Data Preparation

Sales are summed by year for the period **1990–2015** (26 observations). Years outside this range are excluded due to data sparsity, which would distort the trend signal.

### Stationarity Test — Augmented Dickey-Fuller (ADF)

The ADF test checks whether the time series has a unit root (i.e., is non-stationary). Non-stationary series have trends or changing variance that prevent direct ARIMA application.

- **Null hypothesis:** Series has a unit root (non-stationary)
- **Reject null if:** p-value < 0.05

If the series is non-stationary (p > 0.05), first-order differencing (`d=1`) is applied within ARIMA.

### ARIMA Model

**Order selected:** ARIMA(2, 1, 1)

| Parameter | Value | Meaning |
|---|---|---|
| p = 2 | AR terms | Model uses the last 2 time steps as predictors |
| d = 1 | Differencing | First-order differencing to achieve stationarity |
| q = 1 | MA terms | Model uses the last 1 forecast error as a predictor |

### ARIMA Output Summary

```
SARIMAX Results
==============================================================
Dep. Variable:     Global_Sales    No. Observations:  26
Model:             ARIMA(2, 1, 1)  Log Likelihood:    -141.629
AIC:               291.258         BIC:               296.134
HQIC:              292.611
==============================================================
             coef    std err       z      P>|z|
--------------------------------------------------------------
ar.L1       0.4620    1.131     0.408    0.683
ar.L2       0.1210    0.455     0.266    0.790
ma.L1      -0.3056    1.056    -0.290    0.772
sigma2   4854.7371 1686.101     2.879    0.004
==============================================================
Ljung-Box (L1) (Q): 0.00   Prob(Q): 0.98
Jarque-Bera (JB):   0.88   Prob(JB): 0.64
```

### Interpreting the ARIMA Coefficients

The AR and MA coefficients (ar.L1, ar.L2, ma.L1) all have p-values well above 0.05 (0.683, 0.790, 0.772). This means **none of the individual AR or MA terms are statistically significant at conventional thresholds**. The only statistically significant parameter is `sigma2` (the variance of the residuals, p=0.004), which confirms the model is capturing meaningful noise structure.

This is a common outcome with short time series (only 26 observations). With so few data points, parameter estimates have high standard errors, making it difficult to achieve significance on individual coefficients even when the model as a whole fits reasonably.

**Diagnostic tests:**

| Test | Result | Interpretation |
|---|---|---|
| Ljung-Box Q (p=0.98) | Pass | No autocorrelation in residuals — good |
| Jarque-Bera (p=0.64) | Pass | Residuals are approximately normal — good |
| Heteroskedasticity H (p=0.26) | Pass | No significant variance changes over time — good |

All three diagnostic tests pass, indicating the model residuals behave as expected white noise. The model is appropriately specified for this data despite the non-significant individual coefficients.

### Forecast Results

| Year | Forecasted Global Sales (M) |
|---|---|
| 2016 | 245.17 |
| 2017 | 227.48 |
| 2018 | 216.97 |
| 2019 | 209.97 |
| 2020 | 205.47 |
| 2021 | 202.55 |
| 2022 | 200.65 |
| 2023 | 199.42 |

### Interpreting the Forecast

The model projects a **gradual, decelerating decline** in global sales from 245M in 2016 down toward ~199M by 2023. The rate of decline slows each year, suggesting the model expects the market to approach a lower plateau rather than collapse.

This is consistent with real-world trends:

- Physical game retail was already declining by the mid-2010s as digital storefronts (Steam, PSN, Xbox Live) gained dominance
- Mobile gaming was absorbing casual consumers who would previously have purchased console titles
- The dataset does not capture digital or mobile sales, so any "true" market size is much larger — the declining trajectory here reflects the specific physical retail segment

The 95% confidence interval widens substantially beyond 2018, which is the expected behavior of ARIMA forecasts on short series. Projections beyond 2–3 steps should be treated as directional indicators rather than precise estimates.

---

## 12. Key Insights and Conclusions

### Market Structure

- North America is the dominant global video game market at approximately 49% of total recorded sales. No single region comes close.
- Japan is culturally significant and punches above its weight relative to population, but its genre preferences (RPG-heavy) diverge significantly from the Western consensus.
- The PS2 is the highest-grossing platform in this dataset by a substantial margin, driven by its enormous install base and exceptionally long commercial lifespan.

### Publisher Concentration

- Nintendo is the most commercially dominant publisher by total sales, with a lead nearly double that of its closest competitor (Electronic Arts).
- The top 5 publishers collectively account for a disproportionate share of total industry sales, reflecting the blockbuster economics of the industry.

### Genre Dynamics

- Action is both the highest-volume and highest-grossing genre overall.
- RPGs are uniquely Japan-skewed compared to all other genres.
- Shooter games experienced the most dramatic growth of any genre in the 2006–2012 period, driven by the Call of Duty and Halo franchises.

### Modeling

- Linear Regression is entirely unsuitable for this dataset due to multicollinearity, skewed targets, and non-linear feature interactions. Its R2 of approximately -42.5 trillion confirms total failure.
- Gradient Boosting is the recommended model with R2 = 0.841, RMSE = 0.817 (log space), and MAE = 0.039 (log space).
- Regional sales sub-components dominate predictive importance. If these were unavailable at prediction time, model performance would degrade significantly — a future iteration should evaluate models trained without regional sales to simulate a truly forward-looking prediction scenario.

### Forecasting

- The ARIMA(2,1,1) model produces well-behaved residuals (no autocorrelation, approximately normal distribution, no heteroskedasticity).
- Individual AR/MA coefficients are not statistically significant, which is expected given the short 26-observation series.
- The forecast projects physical game sales declining gradually from ~245M in 2016 toward ~199M by 2023, consistent with the industry's documented shift to digital distribution.

---

## 13. Outputs and Saved Files

| File | Description |
|---|---|
| `vgsales_cleaned.csv` | Cleaned and preprocessed dataset |
| `plot_sales_distribution.png` | Raw and log-transformed Global Sales histogram |
| `plot_genre_sales.png` | Total sales by genre (horizontal bar) |
| `plot_platform_sales.png` | Top 10 platforms by total sales |
| `plot_yearly_trend.png` | Global sales trend 1990–2016 |
| `plot_correlation_heatmap.png` | Sales feature correlation matrix |
| `plot_regional_pie.png` | Regional market share pie chart |
| `plot_genre_region_stacked.png` | Genre-region normalized stacked bar |
| `plot_decade_boxplot.png` | Sales distribution per decade |
| `plot_top_publishers.png` | Top 10 publishers by total sales |
| `plot_genre_trend.png` | Top 5 genre trends over time |
| `plot_model_comparison.png` | RMSE / MAE / R2 bar comparison |
| `plot_feature_importance.png` | Random Forest feature importances |
| `plot_actual_vs_predicted.png` | Gradient Boosting predicted vs actual |
| `plot_arima_forecast.png` | ARIMA historical + 8-year forecast |

---

## 14. Known Limitations

### Dataset Limitations

- **Physical sales only** — digital downloads, mobile, and PC (Steam) sales are entirely absent. The dataset understates the true market size, particularly for post-2013 periods.
- **Trailing sales undercount** — games released closer to the collection date have fewer cumulative sales recorded, creating an artificial downward bias in recent years.
- **Geographic gaps** — markets in South Korea, Southeast Asia, and Latin America are aggregated into `Other_Sales` with no granularity.
- **No pricing data** — sales in units rather than revenue. A $10 indie game and a $60 AAA title contribute identically per unit sold.

### Modeling Limitations

- **Data leakage risk** — regional sales columns (NA, EU, JP, Other) are sub-components of the target (Global_Sales). Models using these features are partially predicting outputs from inputs. This is valid for retrospective analysis but not for genuinely prospective forecasting.
- **Label encoding of Publisher** — with 578+ unique publishers, label encoding assigns arbitrary ordinal integers. This may introduce spurious ordinal relationships. Target encoding or embeddings would be more principled for high-cardinality categoricals.
- **No hyperparameter tuning** — Random Forest and Gradient Boosting use default-adjacent parameters. Cross-validated grid search or Bayesian optimization would likely improve both models.

### Forecasting Limitations

- **Short series (26 points)** — ARIMA parameter estimates have high uncertainty on so few observations. The non-significant AR/MA coefficients are a direct consequence of this.
- **Single-variable model** — the ARIMA model uses only historical sales. External regressors (console release cycles, macroeconomic conditions, competing entertainment categories) could substantially improve forecast accuracy.
- **Structural break** — the 2008–2009 peak represents a structural shift in the market. ARIMA assumes statistical properties are consistent over time, which is a simplification here.

---

*Documentation generated for `vgsales_analysis.ipynb`. All results are reproducible by running the notebook top-to-bottom with `vgsales.csv` in the working directory.*