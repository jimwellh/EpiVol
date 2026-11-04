# Project Master Blueprint

- **Project Title**: _Do Epidemic State-Space Filters Improve News-Driven Volatility Forecasts? Contagious Semantics: A Mass-Conserving SIRS State-Space Filter for News-Driven Volatility_
- **GitHub Repository Name** EpiVol
- **Author Mode**: Conference-Style NeurIPS Paper + 5-Minute Video
- **Compute Environment**:
  - Local Workstation: NVIDIA RTX 5070 Ti Laptop GPU ($16\text{ GB}$ GDDR7 VRAM, NVMe SSD, PyTorch FP16/BF16, DuckDB/Polars)
  - Cloud Backup: Google Colab Pro (NVIDIA L4 / A100 GPU, High-RAM runtime)

---

## 1. Executive Summary & Core Research Question

### 1.1 The Core Research Question

> **When converting irregular financial news embeddings into daily features, does a causal state-space filter with SIR epidemic dynamics reduce $1\text{-to-}10\text{-day}$ realized volatility forecast errors compared to zero-filling, forward-filling, or monotonic exponential decay?**

---

## 2. Theoretical Motivation & Mathematical Intuition

### 2.1 Structural Mismatch in Multimodal Financial Time Series

Stock prices and realized volatility ($\text{RV}_{i,t}$) are observed on a regular discrete trading-day grid ($t = 1, 2, \dots, T$). In contrast, material corporate disclosures, earnings calls, and financial news arrive as **sparse, irregularly sampled discrete events**. Between two news clusters, individual stocks frequently experience multi-day or multi-week silent periods (**Information Blackouts**).

When unstructured text on day $t$ is encoded via a pretrained language model (e.g., `ProsusAI/finbert`) into an exogenous numerical shock vector $\mathbf{u}_{i,t} \in \mathbb{R}^d$ (where $\mathbf{u}_{i,t} = \mathbf{0}$ on silent days), any forecasting pipeline must specify a **causal temporal filter** that maps the history of sparse shocks $\{\mathbf{u}_{i,1}, \dots, \mathbf{u}_{i,t}\}$ into a current latent text state $\mathbf{z}_{i,t} \in \mathbb{R}^d$.

### 2.2 Limitations of Traditional Filling & Decay Baselines

1. **Impulse / Zero-Filling ($\mathbf{z}_{i,t}^{\text{imp}} = \tilde{m}_{i,t}\mathbf{u}_{i,t}$)**:
   - _Implicit Assumption_: Instantaneous market digestion at $t = 0$ (ultra-strong market efficiency). On day $t+1$, if no new articles arrive, the text feature drops immediately to $\mathbf{0}$.
   - _Failure Mode_: Completely ignores multi-day market underreaction and post-announcement drift.
2. **Causal Forward-Filling / Last Observation Carried Forward (LOCF)**:
   - _Implicit Assumption_: $\mathbf{z}_{i,t}^{\text{ff}} = \tilde{m}_{i,t}\mathbf{u}_{i,t}$ when news arrives, and $\mathbf{z}_{i,t}^{\text{ff}} = \mathbf{z}_{i,t-1}^{\text{ff}}$ during silent days.
   - _Failure Mode_: Treats a two-week-old news shock as $100\%$ fresh throughout a long blackout period, injecting severe stale-signal noise into medium-horizon volatility forecasts.
3. **Monotonic Exponential Decay / First-Order Causal AR(1) Filter (EWMA)**:
   - _Implicit Assumption_: $\mathbf{z}_{i,t}^{\text{exp}} = \lambda \mathbf{z}_{i,t-1}^{\text{exp}} + \tilde{m}_{i,t}\mathbf{u}_{i,t}$ with $\lambda \in (0, 1)$.
   - _Failure Mode_: Enforces a strictly **monotonic** impulse response where maximum impact occurs instantaneously at $t = 0$ and decays geometrically thereafter.

### 2.3 Why Mass-Conserving Causal SIR Epidemic Dynamics? (Hump-Shaped Memory Kernel)

When a complex disclosure (such as an earnings release or major strategic shift) arrives, market participants do not digest its semantic nuances simultaneously. Information diffuses across the investor population via **social contagion**:

- Initially, a small fraction of attentive "newswatchers" or fast algorithms are directly triggered by the news shock ($\alpha_0 \tilde{m}_{i,t} S_{i,t-1}$).
- Secondary media coverage, analyst commentary, and peer discussions transmit the existing semantic signal from active investors ($I_{i,t-1}$) to previously unengaged market participants ($S_{i,t-1}$) via the endogenous contagion flux $\beta S_{i,t-1} I_{i,t-1}$.
- Consequently, the active market attention and volatility pressure $I_{i,t}$ exhibits a **non-monotonic, hump-shaped trajectory**: when the susceptible pool is abundant ($S_{i,t-1} > \gamma / \beta$), the effective step-to-step amplification factor $(1 - \gamma + \beta S_{i,t-1})$ exceeds $1$, causing the filtered semantic vector $\mathbf{z}_{i,t}^{\text{SIR}}$ to **amplify during subsequent silent days**, reach a delayed peak at $\tau_{\text{peak}} > 0$, and **subsequently decay** once $S_{i,t-1} < \gamma / \beta$ as the market absorbs and prices in the information into the Recovered state ($R_{i,t}$).
- Because all transitions occur strictly between the three compartments ($S_{i,t} + I_{i,t} + R_{i,t} = 1$, $\Delta S_{i,t} + \Delta I_{i,t} + \Delta R_{i,t} = 0$), the system is **strictly mass-conserving and bounded in $[0, 1]^3$**, preventing numerical blow-up even under dense news clustering.

### 2.4 Terminology Discipline & Strict Causality

- **Standard Academic Terms to Use**:
  - _Mass-Conserving Causal State-Space Filter with Discrete-Time SIRS Epidemic Dynamics_
  - _Non-Monotonic / Hump-Shaped Memory Kernel_
  - _Second-Order Causal State-Space Filter (Gamma / Bi-Exponential Kernel)_
- **Definition of "Strictly Causal"**:
  A temporal filter is **strictly causal** if the latent state $\mathbf{z}_{i,t}$ computed at forecast origin $t$ ($16:00\text{ EST}$ market close) is a function solely of observations available on or before $16:00\text{ EST}$ on day $t$:
  $$
  \mathbf{z}_{i,t} = \mathcal{F}\left(\mathbf{u}_{i,1}, \mathbf{u}_{i,2}, \dots, \mathbf{u}_{i,t}\right)
  $$
  No future observations ($\mathbf{u}_{i, t+\tau}$ for $\tau \ge 1$), no retrospective smoothing (e.g., Kalman smoothing, linear interpolation across blackouts, or centered moving averages), and no after-hours news from day $t$ are ever permitted to influence $\mathbf{z}_{i,t}$.

---

## 3. Verified Literature Review (Peer-Reviewed References, see Annotated Bibliography for detail)

All citations below are real, peer-reviewed papers to be verified and cited in the BibTeX file (`references.bib`):

### Pillar 1: Gradual Information Diffusion & Textual Drift in Modern Markets

1. **Hong, H., & Stein, J. C. (1999)**. _A Unified Theory of Underreaction, Momentum Trading, and Overreaction in Asset Markets_. **Journal of Finance**, 54(6), 2143–2184.
   - Foundational theory showing that gradual diffusion of private/complex information across "newswatchers" and momentum traders creates delayed price and volatility reactions.
2. **Hong, H., Lim, T., & Stein, J. C. (2000)**. _Bad News Travels Slowly: Size, Analyst Coverage, and the Profitability of Momentum Strategies_. **Journal of Finance**, 55(1), 265–295.
   - Empirical proof that gradual information diffusion is significantly stronger in stocks with sparser coverage and smaller market capitalization (directly motivating our 3-tier coverage stratification).
3. **Meursault, V., Liang, P. J., Routledge, B. R., & Scanlon, M. (2023)**. _PEAD.txt: Post-Earnings-Announcement Drift Using Text_. **Journal of Financial and Quantitative Analysis (JFQA)**, 58(6), 2299–2326.
   - Crucial modern validation: shows that while numerical earnings surprises (SUE) are rapidly arbitraged by modern quants, **textual surprises (`SUE.txt`) extracted via NLP exhibit strong multi-day and multi-week delayed drift** because qualitative text requires time to digest.
4. **Cohen, L., Malloy, C., & Nguyen, Q. (2020)**. _Lazy Prices_. **Journal of Finance**, 75(3), 1371–1415.
   - Demonstrates that textual changes in corporate filings are digested with substantial multi-day/multi-week delays even in modern institutional markets.
5. **Jiang, G. J., Li, D., & Wang, W. (2021)**. _Pervasive Underreaction: Evidence from High-Frequency Data_. **Journal of Financial Economics (JFE)**, 141(2), 573–599.
   - Confirms using modern high-frequency market data that underreaction and gradual multi-day digestion remain pervasive after major information shocks.
6. **Lopez-Lira, A., & Tang, Y. (2023)**. _Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models_. **SSRN / Management Science Working Paper**.
   - Connects modern LLM news embeddings directly to Hong & Stein (1999) information diffusion frictions over subsequent trading days.

### Pillar 2: Epidemic (SIR) Models, Hawkes Kernels, & Hump-Shaped Diffusion

7. **Shive, S. (2010)**. _An Epidemic Model of Investor Behavior_. **Journal of Financial and Quantitative Analysis (JFQA)**, 45(1), 169–198.
   - Directly applies the Kermack-McKendrick SIR epidemic model to investor trading and information contagion in equities.
8. **Matsubara, Y., Sakurai, Y., Prakash, B. A., Li, L., & Faloutsos, C. (2012)**. _Rise and Fall Patterns of Information Diffusion: Model and Implications_. **Proceedings of the 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)**, 6–14.
   - Establishes that real-world news and online information diffusion follow a hump-shaped rise-and-fall differential equation rather than simple exponential decay.
9. **Woo, J., & Chen, H. (2016)**. _Epidemic Model for Information Diffusion in Web Forums: Experiments in Marketing Exchange and Political Dialog_. **SpringerPlus**, 5(1), 66.
   - Validates SIR state dynamics on text topic diffusion time series.
10. **Bacry, E., Mastromatteo, I., & Muzy, J. F. (2015)**. _Hawkes Processes in Finance_. **Market Microstructure and Liquidity**, 1(01), 1550005.
    - Mathematical framework for causal memory kernels $\phi(t - t_i)$ driving financial volatility clustering.
11. **Osthus, D., Hickmann, K. S., Caragea, P. C., Higdon, D., & Del Valle, S. Y. (2017)**. _Forecasting Seasonal Influenza with a State-Space SIR Model_. **Journal of the American Statistical Association (JASA)**, 112(517), 162–172.
    - Standard statistical reference for discrete-time state-space SIR formulations.

### Pillar 3: Multimodal Time-Series Alignment, Financial NLP, & Volatility Modeling

12. **Dong, Z., Fan, X., & Peng, Z. (2024)**. _FNSPID: A Comprehensive Financial News Dataset in Time Series_. **arXiv:2402.06698 / KDD Workshop**.
    - Primary public dataset providing aligned timestamps, financial news text, and daily OHLCV stock records.
13. **Liu, H., et al. (2024)**. _Time-MMD: Multi-Domain Multimodal Dataset for Time Series Analysis_. **Advances in Neural Information Processing Systems (NeurIPS 2024 Datasets and Benchmarks Track)**.
    - Establishes multimodal text-to-time-series alignment protocols and binary validity-window (forward-filling) baselines.
14. **Wang, X., et al. (2024)**. _From News to Forecast: Integrating Event Analysis in LLM-Based Time Series Forecasting with Reflection_. **Advances in Neural Information Processing Systems (NeurIPS 2024)**.
    - Addresses temporal alignment and heterogeneous duration of news impacts in time-series forecasting.
15. **Araci, D. (2019)**. _FinBERT: Financial Sentiment Analysis with Pre-trained Language Models_. **arXiv:1908.10063** (`ProsusAI/finbert`).
    - Pretrained financial language model used as our frozen feature extractor.
16. **Xu, Y., & Cohen, S. B. (2018)**. _Stock Movement Prediction from Tweets and Historical Prices_. **Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL)**, 1970–1979.
    - Standard benchmark illustrating zero-filling / masking on silent days.
17. **Audrino, F., & Tetereva, A. (2019)**. _Sentiment Spillover Effects for US and European Companies_. **Journal of Banking & Finance**, 106, 542–567.
    - Standard reference for exponential smoothing (EWMA) decay of daily news sentiment in volatility models.
18. **Corsi, F. (2009)**. _A Simple Approximate Long-Memory Model of Realized Volatility_. **Journal of Financial Econometrics**, 7(2), 174–196.
    - Foundational paper for the Heterogeneous Autoregressive Realized Volatility (HAR-RV) baseline model.
19. **Parkinson, M. (1980)**. _The Extreme Value Method for Estimating the Variance of the Rate of Return_. **Journal of Business**, 53(1), 61–65.
    - Range-based daily realized volatility estimator using daily High and Low prices.

---

## 4. Complete Data, Preprocessing, & Anti-Leakage Architecture

### 4.1 Dataset & Chronological Split

- **Source**: FNSPID (`https://github.com/Zdong104/FNSPID_Financial_News_Dataset`)(`https://huggingface.co/datasets/Zihan1004/FNSPID`).
- **Temporal Unit**: Daily (US stock market trading days according to the NYSE calendar).
- **Full Span**: January 1, 2016 to December 31, 2023 ($8\text{ years}$, $T \approx 2,012\text{ trading days}$ per stock).
- **Strict Chronological Splits**:
  1. **Training Period**: January 1, 2016 – December 31, 2020 ($5\text{ years}$, $\approx 1,258\text{ trading days}$, $\sim 62.5\%$). Used to fit `StandardScaler`, `PCA(n_components=16)`, stock coverage stratification, and Ridge regression weights $\mathbf{w}_h$.
  2. **Validation Period**: January 1, 2021 – December 31, 2022 ($2\text{ years}$, $\approx 504\text{ trading days}$, $\sim 25.0\%$). Spans both the 2021 bull market and 2022 tightening bear market. Used exclusively to select hyperparameters ($\alpha, \lambda, \beta, \gamma$) and early-stop SGD.
  3. **Final Test Period**: January 1, 2023 – December 31, 2023 ($1\text{ year}$, $\approx 252\text{ trading days}$, $\sim 12.5\%$). Untouched until Milestone 4. Completely post-dates the 2020 release of `ProsusAI/finbert`.

### 4.2 Bias-Free Stratified Stock Selection (50 Stocks)

To prevent look-ahead selection bias and test how news sparsity moderates epidemic filtering:

1. **Eligible Universe Filter**: Identify S&P 500 stocks in FNSPID that have (a) complete daily OHLC price records across all trading days in 2016–2023 (zero missing price gaps), and (b) at least $15$ news articles per year during the **2016–2020 training period** (filtering out broken scraper tickers).
2. **Training-Only Coverage Calculation**: For each eligible stock $i$, compute the **Training News Coverage Ratio** strictly over $t \in \text{Train (2016–2020)}$:
   $$
   C_i^{\text{train}} = \frac{1}{T_{\text{train}}} \sum_{t=1}^{T_{\text{train}}} \mathbb{I}\left(N_{i,t} > 0\right)
   $$
   where $N_{i,t}$ is the number of news articles mapped to stock $i$ on trading day $t$.
3. **Stratified Sampling into 3 Tiers**: Partition the eligible stocks into tertiles based on $C_i^{\text{train}}$ and draw $50$ stocks total using fixed random seed `seed = 42`:
   - **Tier 1 — High-Coverage (17 stocks)**: Frequent news arrivals, minimal silent gaps ($1\text{–}2\text{ days}$).
   - **Tier 2 — Medium-Coverage (17 stocks)**: Moderate news frequency, regular $3\text{–}6\text{ day}$ silent gaps.
   - **Tier 3 — Sparse-Coverage (16 stocks)**: Event-driven arrivals (mostly around earnings or major disclosures), exhibiting $5\text{–}15\text{ day}$ information blackouts.

### 4.3 Forecast Origin & Point-in-Time Timestamp Alignment ($16:00\text{ EST}$ Cutoff)

- **Forecast Origin Definition**: The forecast origin for trading day $t$ is exactly $16:00:00\text{ America/New\_York}$ **(market close)** on day $t$.
- **Timestamp Routing Rule**:
  1. Parse every raw news timestamp from UTC and convert to timezone `America/New_York` (automatically accounting for EST/EDT Daylight Saving Time shifts).
  2. If an article is published on trading day $t$ at or before $16:00:00\text{ ET}$, it is assigned to trading day $t$.
  3. If an article is published on trading day $t$ **after** $16:00:00\text{ ET}$ (e.g., after-hours earnings release at $16:15\text{ ET}$) or on a weekend/NYSE holiday, it is **strictly routed to the next valid NYSE trading day** $t+1$.
  4. **Unit Test Requirement**: Write an automated `pytest` assertion verifying that $0\%$ of articles incorporated into feature vector $\mathbf{u}_{i,t}$ have an ET timestamp exceeding $16:00:00$ on date $t$.

### 4.4 Target Variable: Multi-Horizon Log Parkinson Realized Volatility

For stock $i$ on trading day $t$ with daily High price $H_{i,t}$ and Low price $L_{i,t}$ ($H_{i,t} \ge L_{i,t} > 0$):

1. Compute the daily Parkinson (1980) extreme-value variance estimator:
   $$
   \text{RV}_{i,t} = \frac{1}{4 \ln 2} \left(\ln \frac{H_{i,t}}{L_{i,t}}\right)^2
   $$
2. Define the log-volatility series (adding a tiny floor $\epsilon = 10^{-8}$ to handle zero-range flat days):
   $$
   v_{i,t} = \ln\left(\sqrt{\text{RV}_{i,t} + \epsilon}\right)
   $$
3. **Target at Horizon** $h \in \{1, 2, 3, 5, 10\}$:
   - **Point Target**: $y_{i, t+h} = v_{i, t+h}$ (the log realized volatility on trading day $t+h$).

### 4.5 Exogenous Text Feature Extraction (`ProsusAI/finbert` + Training-Only PCA)

For each article $j \in \{1, \dots, N_{i,t}\}$ mapped to stock $i$ on trading day $t$:

1. Pass the concatenated headline and summary through frozen `ProsusAI/finbert` (using FP16 on RTX 5070 Ti):
   - Extract the $3$-dimensional softmax sentiment probability vector $\mathbf{p}_{i,t,j} = [P_{\text{pos}}, P_{\text{neg}}, P_{\text{neu}}]^\top \in \Delta^2$.
   - Extract the $768$-dimensional `[CLS]` pooled hidden representation $\mathbf{e}_{i,t,j} \in \mathbb{R}^{768}$.
2. **Intra-Day Aggregation** (when $N_{i,t} \ge 1$):
   - Compute mean sentiment probabilities: $\bar{\mathbf{p}}_{i,t} = \frac{1}{N_{i,t}} \sum_{j=1}^{N_{i,t}} \mathbf{p}_{i,t,j} \in \mathbb{R}^3$.
   - Compute mean $768$-D embedding: $\bar{\mathbf{e}}_{i,t} = \frac{1}{N_{i,t}} \sum_{j=1}^{N_{i,t}} \mathbf{e}_{i,t,j} \in \mathbb{R}^{768}$.
3. **Training-Only PCA Compression**:
   - Fit a `StandardScaler` and `PCA(n_components=16)` **strictly on the non-empty daily embeddings** $\{\bar{\mathbf{e}}_{i,t} : N_{i,t} > 0, t \in \text{Train (2016–2020)}\}$.
   - Transform all active days across Train, Val, and Test into $16$-dimensional semantic principal components $\mathbf{c}_{i,t} \in \mathbb{R}^{16}$.
4. **Daily Exogenous Shock Vector** $\mathbf{u}_{i,t} \in \mathbb{R}^{19}$ **& Bounded Shock Intensity** $\tilde{m}_{i,t} \in [0, 1)$:
   - When $N_{i,t} > 0$:
     $$
     \mathbf{u}_{i,t} = \begin{bmatrix} \bar{\mathbf{p}}_{i,t} \\ \mathbf{c}_{i,t} \end{bmatrix} \in \mathbb{R}^{19}, \quad m_{i,t} = \left(\bar{P}_{\text{pos}, i, t} + \bar{P}_{\text{neg}, i, t}\right) \ln\left(1 + N_{i,t}\right), \quad \tilde{m}_{i,t} = 1 - \exp(-m_{i,t})
     $$
   - When $N_{i,t} = 0$ (Silent / Blackout Day):
     $$
     \mathbf{u}_{i,t} = \mathbf{0}_{19}, \quad m_{i,t} = 0, \quad \tilde{m}_{i,t} = 0
     $$

---

## 5. Mathematical Specification of Models (Dimension-Matched $\mathbb{R}^{19}$ Core & $\mathbb{R}^{20}$ Ablation)

To ensure a strictly fair **Controlled Comparison** (Page 2 & 3 of the Guide) with zero confounding from feature dimensionality or degrees of freedom, **all core text filters map the exact same daily intensity-weighted shock input $\tilde{m}_{i,t} \mathbf{u}_{i,t} \in \mathbb{R}^{19}$ into a dimension-matched $\mathbb{R}^{19}$ filtered state vector $\mathbf{z}_{i,t} \in \mathbb{R}^{19}$** before entering the identical downstream Ridge predictor.

_(Note on Implicit Intensity Recovery: Because the first three coordinates of $\mathbf{u}_{i,t}$ are the FinBERT sentiment probabilities $\bar{P}_{\text{pos}} + \bar{P}_{\text{neg}} + \bar{P}_{\text{neu}} = 1$, the sum of the first three coordinates of $\mathbf{z}_{i,t}$ in every filter automatically equals the scalar filtered intensity, e.g., $z_{i,t,1}^{\text{SIR}} + z_{i,t,2}^{\text{SIR}} + z_{i,t,3}^{\text{SIR}} = I_{i,t}$ when normalized, avoiding collinearity while giving the linear readout full access to overall shock magnitude.)\_

### 5.1 Downstream Predictor (Fixed Across All Methods)

For each forecast horizon $h \in \{1, 2, 3, 5, 10\}$, the prediction at origin $t$ is:

$$
\hat{y}_{i, t+h} = \mu_i + \theta_{h,1} v_{i,t}^{(d)} + \theta_{h,2} v_{i,t}^{(w)} + \theta_{h,3} v_{i,t}^{(m)} + \mathbf{w}_h^\top \mathbf{z}_{i,t}
$$

where the HAR-RV (Corsi, 2009) numerical baseline features are:

$$
v_{i,t}^{(d)} = v_{i,t}, \quad v_{i,t}^{(w)} = \frac{1}{5}\sum_{\tau=0}^{4} v_{i, t-\tau}, \quad v_{i,t}^{(m)} = \frac{1}{22}\sum_{\tau=0}^{21} v_{i, t-\tau}
$$

and $\mathbf{z}_{i,t} \in \mathbb{R}^{19}$ is the causal filtered text state produced by one of the filters below. Ridge penalty $\alpha \in \{10^{-3}, 10^{-2}, 10^{-1}, 1, 10, 10^2, 10^3\}$ is tuned on the 2021–2022 validation set.

### 5.2 Dimension-Matched Baseline Filters ($\mathbf{z}_{i,t} \in \mathbb{R}^{19}$)

- **Baseline 0 (Text-Free HAR-RV)**: $\mathbf{z}_{i,t} = \mathbf{0}_{19}$ (uses only $v_{i,t}^{(d)}, v_{i,t}^{(w)}, v_{i,t}^{(m)}$).
- **Baseline 1 (Impulse / Zero-Filling, $\mathbb{R}^{19}$)**:
  $$
  \mathbf{z}_{i,t}^{\text{imp}} = \tilde{m}_{i,t} \mathbf{u}_{i,t} \in \mathbb{R}^{19}
  $$
- **Baseline 2 (Causal Forward-Filling / LOCF, $\mathbb{R}^{19}$)**:
  $$
  \mathbf{z}_{i,t}^{\text{ff}} = \begin{cases} \tilde{m}_{i,t} \mathbf{u}_{i,t} & \text{if } N_{i,t} > 0 \\ \mathbf{z}_{i,t-1}^{\text{ff}} & \text{if } N_{i,t} = 0 \end{cases} \quad \in \mathbb{R}^{19}
  $$
- **Baseline 3 (Monotonic Exponential Decay / Causal AR(1) Filter, $\mathbb{R}^{19}$)**:
  $$
  \mathbf{z}_{i,t}^{\text{exp}} = \lambda \mathbf{z}_{i,t-1}^{\text{exp}} + (1 - \lambda) \tilde{m}_{i,t} \mathbf{u}_{i,t} \in \mathbb{R}^{19}
  $$
  where $\lambda \in \{0.1, 0.2, \dots, 0.9, 0.95\}$ is tuned on the validation set.

### 5.3 Proposed Advanced Filters (Split Contagion-Spark Formulation in $\mathbb{R}^{19}$)

- **Proposed Method 1 — Mass-Conserving Causal SIRS State-Space Filter ($\mathbb{R}^{19}$)**:
  Maintains a 3-compartment probability simplex state $(S_{i,t}, I_{i,t}, R_{i,t}) \in \Delta^2$ per stock, initialized at $S_{i,0} = 1, I_{i,0} = 0, R_{i,0} = 0$. To guarantee bounded transition probabilities $p_{S \to I, t} = \beta I_{i,t-1} + \alpha_0 \tilde{m}_{i,t} \le 1$ and $p_{R \to S, t} = \delta + \eta \tilde{m}_{i,t} \le 1$ without interdependent hyperparameter constraints, we reparameterize the exogenous spark rates using independent multipliers $k_\alpha, k_\eta \in (0, 1]$:

  $$
  \begin{aligned}
  \alpha_0 &= k_\alpha \cdot (1 - \beta) \\
  \eta &= k_\eta \cdot (1 - \delta)
  \end{aligned}
  $$

  The state update equations are rigorously defined as:

  $$
  \begin{aligned}
  S_{i,t} &= S_{i,t-1} - \left(\beta I_{i,t-1} + \alpha_0 \tilde{m}_{i,t}\right) S_{i,t-1} + \left(\delta + \eta \tilde{m}_{i,t}\right) R_{i,t-1} \\
  I_{i,t} &= (1 - \gamma) I_{i,t-1} + \left(\beta I_{i,t-1} + \alpha_0 \tilde{m}_{i,t}\right) S_{i,t-1} \\
  R_{i,t} &= (1 - \delta - \eta \tilde{m}_{i,t}) R_{i,t-1} + \gamma I_{i,t-1} = 1 - S_{i,t} - I_{i,t}
  \end{aligned}
  $$

  _Exact Conservation Property_: Summing the three update equations yields $\Delta S_{i,t} + \Delta I_{i,t} + \Delta R_{i,t} = 0$ and $S_{i,t} + I_{i,t} + R_{i,t} = 1$ for all $t \ge 0$.

  _Split Semantic Direction Update (Separating Endogenous Contagion from Exogenous Spark)_:
  Because secondary infections $\beta S_{i,t-1} I_{i,t-1}$ are transmitted by existing carriers $I_{i,t-1}$, they inherit the **existing active semantic direction $\tilde{\mathbf{u}}_{i,t-1}$**, whereas newly sparked infections $\alpha_0 \tilde{m}_{i,t} S_{i,t-1}$ carry the **new incoming article direction $\mathbf{u}_{i,t}$**:

  $$
  \tilde{\mathbf{u}}_{i,t} = \frac{\left(1 - \gamma + \beta S_{i,t-1}\right) I_{i,t-1} \tilde{\mathbf{u}}_{i,t-1} + \alpha_0 \tilde{m}_{i,t} S_{i,t-1} \mathbf{u}_{i,t}}{I_{i,t} + \epsilon}
  $$

  _Division-Free Equivalent State Vector Update ($\mathbf{z}_{i,t}^{\text{SIR}} = I*{i,t} \tilde{\mathbf{u}}*{i,t} \in \mathbb{R}^{19}$)_:
  Multiplying both sides by $I_{i,t}$ (for $\epsilon \to 0$) yields the exact, division-free vector recursion:

  $$
  \mathbf{z}_{i,t}^{\text{SIR}} = \underbrace{\left(1 - \gamma + \beta S_{i,t-1}\right)}_{\text{State-dependent gain } g_{i,t}} \mathbf{z}_{i,t-1}^{\text{SIR}} + \alpha_0 \tilde{m}_{i,t} S_{i,t-1} \mathbf{u}_{i,t} \in \mathbb{R}^{19}
  $$

  _Key Theoretical Insight_:
  1. During silent blackout days ($\tilde{m}_{i,t} = 0, \mathbf{u}_{i,t} = \mathbf{0}$), the direction remains invariant ($\tilde{\mathbf{u}}_{i,t} = \tilde{\mathbf{u}}_{i,t-1}$) and $\mathbf{z}_{i,t}^{\text{SIR}} = (1 - \gamma + \beta S_{i,t-1}) \mathbf{z}_{i,t-1}^{\text{SIR}} = I_{i,t} \tilde{\mathbf{u}}_{i,t-1}$.
  2. Unlike Baseline 3 where the multiplier $\lambda < 1$ is constant and strictly decaying, the SIR multiplier $g_{i,t} = 1 - \gamma + \beta S_{i,t-1}$ is **strictly greater than $1$ whenever $S_{i,t-1} > \gamma / \beta$**, enabling endogenous hump-shaped amplification across silent days before decaying once $S_{i,t-1} < \gamma / \beta$.
     To reduce tuning complexity, we grid-search $(\beta, \gamma, k_\alpha, k_\eta, \delta)$ over $\beta \in \{0.1, 0.25, \dots, 0.85\}$, $\gamma \in \{0.1, 0.25, \dots, 0.85\}$, $k_\alpha \in \{0.2, 0.4 \dots, 0.8\}$, $k_\eta \in \{0.2, 0.4 \dots, 0.8\}$, $\delta \in \{0.05, 0.1 \dots, 0.2\}$ on the validation set.

- **Proposed Method 2 (Analytical Linear Fallback) — Second-Order Causal Gamma Filter ($\mathbb{R}^{19}$)**:
  A linear 2-state cascade system with state $\mathbf{x}_{i,t}^{(1)} \in \mathbb{R}^{19}$ (latent exposure) and $\mathbf{z}_{i,t}^{\text{gamma}} = \mathbf{x}_{i,t}^{(2)} \in \mathbb{R}^{19}$ (active market impact):

  $$
  \begin{aligned}
  \mathbf{x}_{i,t}^{(1)} &= (1 - \gamma) \mathbf{x}_{i,t-1}^{(1)} + \gamma \tilde{m}_{i,t} \mathbf{u}_{i,t} \\
  \mathbf{z}_{i,t}^{\text{gamma}} = \mathbf{x}_{i,t}^{(2)} &= (1 - \gamma) \mathbf{x}_{i,t-1}^{(2)} + \gamma \mathbf{x}_{i,t-1}^{(1)} \quad \in \mathbb{R}^{19}
  \end{aligned}
  $$

  Whose impulse response is the exact discrete Gamma / Rayleigh hump-shaped kernel $h(\tau) \propto \tau (1 - \gamma)^{\tau-1}$ peaking at $\tau_{\text{peak}} \approx 1/\gamma$, requiring only **one** scalar parameter $\gamma$ (identical degrees of freedom to Baseline 3's $\lambda$).

- **Proposed Method 3 (Stretch Extension) — Context-Dependent Learnable SIRS Filter (PyTorch SGD, $\mathbb{R}^{19}$)**:
  Parameterizes infection rate $\beta_{i,t} = (1 - \alpha_0)\sigma(\mathbf{w}_\beta^\top \mathbf{u}_{i,t} + b_\beta)$ and recovery rate $\gamma_{i,t} = \sigma(\mathbf{w}_\gamma^\top \mathbf{u}_{i,t} + b_\gamma)$ as functions of the incoming news embedding, preserving $S_{i,t} + I_{i,t} + R_{i,t} = 1$ while training end-to-end via Backpropagation Through Time (BPTT) with AdamW on the 2016–2020 training set.

### 5.4 Ablation Study: $\mathbb{R}^{19}$ Pure Semantic State vs. $\mathbb{R}^{20}$ Susceptible-Augmented State (Adding $S_{i,t}$)

In the main comparison (Sections 5.2–5.3), all models are strictly constrained to $\mathbb{R}^{19}$ so that no method receives extra feature dimensions. In a dedicated **Ablation Study**, we test whether explicitly exposing the **remaining susceptible market capacity $S_{i,t} \in [0, 1]$** as a 20th coordinate improves medium-to-long horizon ($h = 5, 10$) forecasts:

$$
\mathbf{z}_{i,t}^{\text{SIR-20}} = \begin{bmatrix} S_{i,t} \\ \mathbf{z}_{i,t}^{\text{SIR}} \end{bmatrix} = \begin{bmatrix} S_{i,t} \\ I_{i,t} \tilde{\mathbf{u}}_{i,t} \end{bmatrix} \in \mathbb{R}^{20}
$$

- **Hypothesis to Test**: Because a linear readout on $\mathbf{z}_{i,t}^{\text{SIR}} \in \mathbb{R}^{19}$ only observes the current active intensity $I_{i,t}$, it cannot distinguish whether a moderate $I_{i,t}$ is on the **rising phase** of an outbreak ($S_{i,t}$ high $\implies$ volatility will continue to rise over $t+3 \dots t+10$) or the **falling phase** ($S_{i,t}$ depleted $\implies$ volatility will subside). Appending $S_{i,t}$ (and comparing against appending the raw shock indicator $\tilde{m}_{i,t}$ to the baselines so they also reside in $\mathbb{R}^{20}$) directly tests whether latent capacity tracking adds predictive value at longer horizons.

### 5.5 Reference Python Implementation of the Split SIRS Filter (`src/filters/sir_filter.py`)

```python
import numpy as np

def run_causal_sirs_filter(
    u_seq: np.ndarray,             # Shape: (T, 19) daily exogenous text vectors
    m_tilde_seq: np.ndarray,       # Shape: (T,) bounded daily shock intensities in [0, 1)
    beta: float,                   # Endogenous contagion rate in (0, 1)
    gamma: float,                  # Recovery rate in (0, 1)
    k_alpha: float = 0.25,         # Spark multiplier in (0, 1]; alpha_0 = k_alpha * (1 - beta)
    delta: float = 0.05,           # Baseline attention reset rate in [0, 1)
    k_eta: float = 0.20,           # Reset multiplier in (0, 1]; eta = k_eta * (1 - delta)
    include_s_state: bool = False  # False -> R^19 (Core); True -> R^20 (Ablation)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Strictly causal, mass-conserving discrete-time SIRS state-space filter.
    Uses orthogonal reparameterization (k_alpha, k_eta) to guarantee bounded
    transition probabilities in [0, 1], S_t + I_t + R_t == 1.0, and exact
    division-free vector recursion without clipping.
    """
    assert 0.0 < beta < 1.0, f"beta must be in (0, 1), got {beta}"
    assert 0.0 < gamma < 1.0, f"gamma must be in (0, 1), got {gamma}"
    assert 0.0 < k_alpha <= 1.0, f"k_alpha must be in (0, 1], got {k_alpha}"
    assert 0.0 <= delta < 1.0, f"delta must be in [0, 1), got {delta}"
    assert 0.0 < k_eta <= 1.0, f"k_eta must be in (0, 1], got {k_eta}"

    # Orthogonal reparameterization: strictly guarantees beta + alpha_0 <= 1 and delta + eta <= 1
    alpha_0 = k_alpha * (1.0 - beta)
    eta = k_eta * (1.0 - delta)

    T, d = u_seq.shape
    S = np.zeros(T, dtype=np.float64)
    I = np.zeros(T, dtype=np.float64)
    R = np.zeros(T, dtype=np.float64)
    Z = np.zeros((T, d), dtype=np.float64)

    # Initial state before t=0: fully susceptible market, zero active infection
    s_prev, i_prev, r_prev = 1.0, 0.0, 0.0
    z_prev = np.zeros(d, dtype=np.float64)

    for t in range(T):
        m_t = m_tilde_seq[t]
        u_t = u_seq[t]

        # Mass-conserving compartment transitions (Delta S + Delta I + Delta R == 0)
        flux_s_to_i = (beta * i_prev + alpha_0 * m_t) * s_prev
        flux_i_to_r = gamma * i_prev
        flux_r_to_s = (delta + eta * m_t) * r_prev

        s_curr = s_prev - flux_s_to_i + flux_r_to_s
        i_curr = i_prev + flux_s_to_i - flux_i_to_r
        r_curr = r_prev + flux_i_to_r - flux_r_to_s

        # Division-free split semantic state update:
        # Endogenous contagion (beta * s_prev) amplifies existing semantic state z_prev;
        # Exogenous spark (alpha_0 * m_t * s_prev) injects new article direction u_t.
        gain_t = 1.0 - gamma + beta * s_prev
        z_curr = gain_t * z_prev + (alpha_0 * m_t * s_prev) * u_t

        S[t], I[t], R[t] = s_curr, i_curr, r_curr
        Z[t] = z_curr

        s_prev, i_prev, r_prev = s_curr, i_curr, r_curr
        z_prev = z_curr

    if include_s_state:
        # R^20 Ablation: prepend S_t as the 1st coordinate
        Z_out = np.column_stack([S, Z])
    else:
        # R^19 Core: dimension-matched to all baselines
        Z_out = Z

    return Z_out, np.column_stack([S, I, R])
```

---

## 6. Recommended Workspace Structure

```
EpiVol/
├── README.md                        # High-level project overview and setup instructions
├── .gitignore                       # gitignor include CLAUDE.md / GEMINI.md and other unnessary files
├── CLAUDE.md / GEMINI.md            # AI assistant symlinks to the Blueprint
├── requirements.txt                 # torch, transformers, duckdb, polars, scikit-learn, statsmodels, matplotlib, pytest
├── configs/
│   └── config.yaml                  # Seeds, date splits, horizons [1,2,3,5,10], grid search ranges
├── docs/                            # Central repository for all project documentation
│   ├── blueprints/
│   │   ├── Project Master Blueprint.md  # The master reference document
│   │   └── experiment_notes.md                     # Running log of ablation studies and observations
│   ├── bibliography/
│   │   └── annotated_bibliography.md               # 19-entry academic literature review
│   └── milestones/                                 # Official course deliverables
│       ├── milestone_1_proposal/
│       │   └── proposal.txt                        # Submitted proposal (Completed)
│       ├── milestone_2_baseline/
│       │   └── baseline_checkpoint_report.md
│       ├── milestone_3_early_draft/
│       │   └── early_draft_paper.pdf
│       └── milestone_6_final_paper/                # Final NeurIPS-style submission
│           ├── main.tex                            # NeurIPS 2026 style template
│           ├── references.bib                      # Verified BibTeX bibliography
│           └── figures/                            # PDF/PNG figures generated by src/plotting/make_figures.py
├── data/
│   ├── raw/                         # Raw FNSPID CSV/Parquet files (gitignored)
│   ├── processed/                   # Aligned daily panel Parquet & extracted FinBERT embeddings
│   └── metadata/                    # 50 stratified tickers list & coverage tier table
├── src/
│   ├── data_pipeline/
│   │   ├── filter_fnspid.py         # DuckDB streaming filter for S&P 500 & 2016-2023 dates
│   │   ├── align_timestamps.py      # UTC -> America/New_York 16:00:00 cutoff routing
│   │   └── stratify_tickers.py      # 2016-2020 training coverage tertile sampling (seed=42)
│   ├── features/
│   │   ├── extract_finbert.py       # Batch FP16 FinBERT inference on RTX 5070 Ti
│   │   ├── fit_pca.py               # Training-only StandardScaler & 16-D PCA
│   │   └── build_har_targets.py     # Parkinson RV & multi-horizon target construction
│   ├── filters/
│   │   ├── baselines.py             # Impulse, Forward-fill, Monotonic Exponential Decay (all R^19)
│   │   ├── sir_filter.py            # Mass-conserving split SIRS filter (R^19 core & R^20 ablation)
│   │   └── learnable_sir_torch.py   # Differentiable PyTorch Context-Dependent SIRS module
│   ├── evaluation/
│   │   ├── ridge_forecaster.py      # Direct multi-step Ridge regression & validation grid search
│   │   ├── bootstrap_ci.py          # Moving Block Bootstrap for dependent overlapping forecasts
│   │   └── blackout_stress_test.py  # Theme 12 contiguous blackout masking experiment
│   └── plotting/
│       └── make_figures.py          # Publication-quality NeurIPS figures (matplotlib/seaborn)
├── tests/
│   ├── test_no_leakage.py           # Verifies 16:00 EST cutoff, chronological split, PCA fit scope
│   └── test_filter_causality.py     # Verifies future shock perturbations at t+1 do not alter z_t and S_t+I_t+R_t == 1
└── notebooks/
    └── eda_and_diagnostics.ipynb    # Interactive inspection
```

---

## 7. Step-by-Step Execution Roadmap Across All 6 Milestones

### MILESTONE 1: Project Proposal (`proposal.txt`) — [COMPLETED]

- **Due Date**: Monday, November 2, 2026, 11:59 PM EST
- **Status**: Completed and submitted with the mass-conserving SIRS formulation ($\Delta S_t + \Delta I_t + \Delta R_t = 0$).

### MILESTONE 2: Baseline Checkpoint

- **Due Date**: Monday, November 9, 2026, 11:59 PM EST
- **Course Requirement**: _"Submit evidence of a working data and evaluation pipeline together with at least one valid baseline result. Briefly identify any data-quality, alignment, or leakage issues discovered so far."_
- **Action Plan**:
  1. Ingest FNSPID via DuckDB, stratify 50 stocks across High/Medium/Sparse 2016–2020 coverage tertiles (`seed=42`).
  2. Audit and document 4 data-quality/alignment issues: (a) After-hours news (>16:00 ET) routing to $t+1$; (b) Weekend/holiday news aggregation; (c) Flat intraday high-low days ($\epsilon = 10^{-8}$ floor); (d) Syndicated duplicate headline removal.
  3. Extract FP16 FinBERT features, fit training-only PCA(16), run unit tests (`test_no_leakage.py`), and report 2021–2022 Validation RMSE/MAE for HAR-RV, Impulse, Forward-fill, and Exponential Decay (all in $\mathbb{R}^{19}$).

### MILESTONE 3: Early Paper Draft

- **Due Date**: Monday, November 16, 2026, 11:59 PM EST
- **Course Requirement**: _"Submit an early paper draft containing the problem formulation, data description, methods, experimental setup, preliminary results, and an outline of the remaining sections."_
- **Action Plan**:
  1. Implement the mass-conserving split SIRS filter and 2nd-order Gamma filter (`src/filters/sir_filter.py`), plus unit tests verifying $S_t + I_t + R_t = 1$ and silent-day amplification when $S_{t-1} > \gamma/\beta$.
  2. Generate Figure 1 showing the hump-shaped impulse response ($S_t, I_t, R_t$ trajectories and state-dependent gain $g_t = 1 - \gamma + \beta S_{t-1}$) vs. monotonic exponential decay.
  3. Draft Sections 1–5 of the NeurIPS LaTeX paper with validation set results and Section 6 outline.

### MILESTONE 4: Analysis Complete

- **Due Date**: Monday, November 30, 2026, 11:59 PM EST
- **Course Requirement**: _"Submit a concise analysis checkpoint containing the final experiments, model comparisons, principal figures or tables, and the limitations you expect to discuss in the paper."_
- **Action Plan**:
  1. Unlock the untouched 2023 test set using frozen validation hyperparameters.
  2. Compute Moving Block Bootstrap 95% confidence intervals across rolling origins.
  3. Run the $\mathbb{R}^{19}$ vs. $\mathbb{R}^{20}$ (adding $S_{i,t}$) Ablation Study, Coverage-Tier & Earnings-Window disaggregation, Theme 12 Synthetic Blackout Stress Test, and PyTorch Context-Dependent SIRS evaluation.
  4. **Run Learnable SIRS Gate Interpretability Analysis ($\mathbf{w}_\beta, \mathbf{w}_\gamma \in \mathbb{R}^{19}$)**: Inspect the trained gate weights $(\mathbf{w}_\beta, \mathbf{w}_\gamma)$ from the Context-Dependent SIRS filter across the 3 FinBERT sentiment coordinates and 16 PCA semantic axes (interpreted via top-projection training headlines), and evaluate the implied semantic reproduction ratio $R_0(\mathbf{u}_{i,t}) = \beta_{i,t}(\mathbf{u}_{i,t}) / \gamma_{i,t}(\mathbf{u}_{i,t})$ to identify which news sentiments and topics drive multi-day hump-shaped contagion ($R_0 > 1$) versus rapid digestion ($R_0 < 1$).
  5. Compile all final figures (including the $(\mathbf{w}_\beta, \mathbf{w}_\gamma)$ contagion-versus-digestion interpretability chart), tables, and the 4 core empirical limitations.

### MILESTONE 5: Paper and Video Draft

- **Due Date**: Friday, December 4, 2026, 11:59 PM EST
- **Course Requirement**: _"Submit a complete draft of the NeurIPS-style paper and five-minute video draft."_

### MILESTONE 6: Final NeurIPS-Style Paper & Five-Minute Video

- **Due Date**: Friday, December 11, 2026, 11:59 PM EST
- **Course Requirement**: _"Submit an individual time-series project as a NeurIPS-style paper and five-minute video."_
