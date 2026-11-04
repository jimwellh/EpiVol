# Annotated Bibliography

---

## Pillar 1: Gradual Information Diffusion and Textual Drift in Modern Markets

### 1. Hong, H., & Stein, J. C. (1999)

**Citation:** Hong, H., & Stein, J. C. (1999). A Unified Theory of Underreaction, Momentum Trading, and Overreaction in Asset Markets. _Journal of Finance_, 54(6), 2143–2184.

- **Core Methodology:** Establishes a bounded-rationality, two-agent equilibrium model dividing market participants into "newswatchers" and "momentum traders." Fundamental information is parsed into $z$ independent pieces that transmit linearly among newswatchers at a constant rate of $1/z$ per period. Momentum traders trend-chase on past price changes $\Delta P_{t-j}$, generating a characteristic humped impulse-response function characterized by initial underreaction followed by subsequent overreaction across the time horizon.
- **Comparison with Present Research:**
  - **Similarities:** Provides the fundamental theoretical grounding for our work: asset markets cannot instantaneously process complex information at $t=0$. Their "initial newswatchers" correspond directly to our exogenous ignition component $\alpha_0 \tilde{m}_{i,t} S_{i,t-1}$, while "follow-up investor adoption and social word-of-mouth" parallel our endogenous contagion component $\beta S_{i,t-1} I_{i,t-1}$.
  - **Differences:** Hong & Stein assume a deterministic, uniform linear diffusion rate ($1/z$ per period) evaluated purely via theoretical derivations. We operationalize this mechanism into a non-linear, discrete SIRS state-space filter capable of ingesting empirical 19-dimensional FinBERT news vectors, allowing the diffusion rate to vary dynamically with remaining potential capacity $S_{i,t-1}$.

---

### 2. Hong, H., Lim, T., & Stein, J. C. (2000)

**Citation:** Hong, H., Lim, T., & Stein, J. C. (2000). Bad News Travels Slowly: Size, Analyst Coverage, and the Profitability of Momentum Strategies. _Journal of Finance_, 55(1), 265–295.

- **Core Methodology:** Applies cross-sectional Fama-MacBeth regressions and dual-sorted portfolios (controlling first for firm capitalization $\ln(\text{Size})$ and computing residual analyst coverage $\ln(1 + \text{Analysts})$) to empirically assess the impact of information sparsity on the persistence of price momentum.
- **Comparison with Present Research:**
  - **Similarities:** Directly supports our data sampling design: informational diffusion frictions are most pronounced in low-attention, news-sparse assets. This empirical finding provides the theoretical basis for our 3-tier stratified sampling approach across high, medium, and low market attention tiers.
  - **Differences:** The authors employ monthly-frequency panels and analyst counts as proxies to evaluate return continuation over 1- to 6-month horizons. In contrast, we sample tickers without look-ahead bias based directly on daily news coverage frequencies during the 2016–2020 training split, evaluating predictive efficacy on realized volatility across multiple short-to-medium horizons $h \in \{1, 2, 3, 5, 10\}$ trading days.

---

### 3. Meursault, V., Liang, P. J., Routledge, B. R., & Scanlon, M. (2023)

**Citation:** Meursault, V., Liang, P. J., Routledge, B. R., & Scanlon, M. (2023). PEAD.txt: Post-Earnings-Announcement Drift Using Text. _Journal of Financial and Quantitative Analysis_, 58(7), 2841–2874.

- **Core Methodology:** Vectorizes corporate earnings conference calls and press releases, training regularized logistic regression models to construct a text-based earnings surprise metric (SUE.txt). Leverages an event study framework to evaluate cumulative abnormal returns (CAR) over the post-announcement window $[t+2, t+60]$.
- **Comparison with Present Research:**
  - **Similarities:** Demonstrates that even in modern US equity markets dominated by quantitative execution, unstructured textual shocks produce protracted multi-day drift, even as purely numerical earnings surprises (numerical SUE) are rapidly arbitrated away.
  - **Differences:** PEAD.txt is defined exclusively on quarterly earnings dates, compressing text into a single scalar for static cross-sectional regressions. Our architecture operates over a continuous daily grid, simultaneously modeling irregular news bursts alongside multi-day blackout periods while retaining a 19-dimensional semantic state vector $\mathbf{z}_{i,t}$.

---

### 4. Cohen, L., Malloy, C., & Nguyen, Q. (2020)

**Citation:** Cohen, L., Malloy, C., & Nguyen, Q. (2020). Lazy Prices. _Journal of Finance_, 75(3), 1371–1415.

- **Core Methodology:** Computes cosine similarity and edit distances across sequential 10-K and 10-Q regulatory filings to measure structural language revisions. Constructs calendar-time portfolios to verify delayed market absorption of substantive reporting modifications.
- **Comparison with Present Research:**
  - **Similarities:** Confirms that institutional investors exhibit marked inattention toward complex financial prose, aligning with our modeling premise that the majority of investors reside in an unreacted/susceptible state $S_{i,t-1}$ upon initial news release.
  - **Differences:** The authors quantify low-frequency quarter-over-quarter document shifts without explicitly modeling daily shifts in market attention between reporting dates. Our SIRS filter tracks continuous state-transition paths $(S_{i,t}, I_{i,t}, R_{i,t})$ across daily intervals.

---

### 5. Jiang, G. J., Li, D., & Wang, W. (2021)

**Citation:** Jiang, G. J., Li, D., & Wang, W. (2021). Pervasive Underreaction: Evidence from High-Frequency Data. _Journal of Financial Economics_, 141(2), 572–599.

- **Core Methodology:** Merges intraday jump detections (Lee-Mykland jump tests) with precise news timestamps to isolate major information shocks. Evaluates multi-day return and volatility continuation following jumps using autoregressive impulse response analyses.
- **Comparison with Present Research:**
  - **Similarities:** Strictly segregates intraday from after-hours information boundaries (cut off at 16:00 EST), offering high-frequency empirical evidence that asset markets require multi-day horizons to fully ingest critical news shocks.
  - **Differences:** Identifies shocks ex post via price jumps and evaluates dynamics through standard linear autoregressions. Conversely, we characterize shocks purely ex ante from exogenous textual features $(\tilde{m}_{i,t}, \mathbf{u}_{i,t})$, benchmarking non-linear epidemiological filters directly against linear decay formulations.

---

### 6. Lopez-Lira, A., & Tang, Y. (2023)

**Citation:** Lopez-Lira, A., & Tang, Y. (2023). Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models. _arXiv preprint arXiv:2304.07619_.

- **Core Methodology:** Feeds daily financial headlines into a large language model (ChatGPT) to prompt discrete sentiment scores $\in \{-1, 0, +1\}$. Averages scores across headlines using a 16:00 EST cutoff to predict next-day ($t+1$) equity returns as an unpersisted impulse model (no signal generated on days without news).
- **Comparison with Present Research:**
  - **Similarities:** Enforces an identical 16:00 EST cutoff for information availability (assigning post-market announcements to $t+1$) and attributes the predictive capacity of LLM text representations to Hong & Stein (1999) diffusion frictions.
  - **Differences:**
    1. Relies on a proprietary, closed-source LLM exposed to look-ahead contamination via pre-training overlaps on historical financial news; we deploy a frozen 2020 ProsusAI/finbert baseline tested on 2021–2023 out-of-sample data, ensuring zero contamination.
    2. Treats signals purely as single-day impulses (matching our Baseline 1) without accounting for multi-day memory decay or multi-step forecasts ($h=1\dots 10$).

---

## Pillar 2: Epidemic (SIR) Models, Hawkes Memory Kernels, and Humped Diffusion

### 7. Shive, S. (2010)

**Citation:** Shive, S. (2010). An Epidemic Model of Investor Behavior. _Journal of Financial and Quantitative Analysis_, 45(1), 169–198.

- **Core Methodology:** Introduces Kermack-McKendrick mass-action epidemiology into financial markets, regressing daily retail trading activity against the interaction of unexposed market participants $S_{i,t-1}$ (proxied via float turnover balance) and recently active participants $I_{i,t-1}$:
  $$\text{Trade}_{i,t} = \alpha + \beta (S_{i,t-1} I_{i,t-1}) + \text{Controls}$$
- **Comparison with Present Research:**
  - **Similarities:** Incorporates a non-linear bilinear contagion term $\beta S_{i,t-1} I_{i,t-1}$ to capture peer influence and social contagion dynamics in financial trading.
  - **Differences:** Shive derives $S_t$ and $I_t$ directly from observable broker trading volume without text data. In our framework, $(S_{i,t}, I_{i,t}, R_{i,t})$ operate as latent state variables driven by exogenous news embedding shocks that modulate the temporal envelope of 19-dimensional semantic vectors.

---

### 8. Matsubara, Y., Sakurai, Y., Prakash, B. A., Li, L., & Faloutsos, C. (2012)

**Citation:** Matsubara, Y., Sakurai, Y., Prakash, B. A., Li, L., & Faloutsos, C. (2012). Rise and Fall Patterns of Information Diffusion: Model and Implications (SPIKEM). _Proceedings of the 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_, 6–14.

- **Core Methodology:** Formulates a discrete-time SIR difference equation incorporating an initial infection pulse and non-linear contagion term $\beta S_t I_t$ to characterize web and blog discussion cycles. Employs Levenberg-Marquardt non-linear least squares to retrospectively fit rise-and-fall attention trajectories for isolated events.
- **Comparison with Present Research:**
  - **Similarities:** Demonstrates mathematically that when $S_{t-1} > \gamma/\beta$, information diffusion exhibits a humped, rise-and-fall trajectory rather than monotonic exponential decay.
  - **Differences:** SPIKEM performs offline, retrospective curve fitting on scalar event counts for single isolated events. Our proposed SIRS filter operates as a strictly causal online state filter running continuously across multi-event sequences spanning 8 years (2,012 trading days), utilizing an attention-reset mechanism $(\delta + \eta \tilde{m}_{i,t})R_{i,t-1}$ to update 19-dimensional semantic state representations.

---

### 9. Woo, J., & Chen, H. (2016)

**Citation:** Woo, J., & Chen, H. (2016). Epidemic Model for Information Diffusion in Web Forums. _SpringerPlus_, 5(1), 609.

- **Core Methodology:** Implements an ordinary differential equation (ODE) formulation of a standard closed SIR system:
  $$\frac{dS}{dt} = -\beta S I, \quad \frac{dI}{dt} = \beta S I - \gamma I, \quad \frac{dR}{dt} = \gamma I$$
  Fits this dynamic system to daily active poster counts discussing specific text topics on web forums.
- **Comparison with Present Research:**
  - **Similarities:** Empirically demonstrates that the lifecycle of textual discussion topics conforms to three-compartment SIR state transitions.
  - **Differences:** Employs an unmodified closed SIR formulation lacking exogenous shocks or replenishment, restricting modeling capacity to single isolated outbreaks (the system permanently terminates as $S_t \to 0$). We introduce a bounded news arrival rate $\tilde{m}_{i,t}$ and a reparameterized conservative replenishment term $\eta = k_\eta(1-\delta)$, preventing zero-infection deadlocks and resource exhaustion across long multi-year financial horizons.

---

### 10. Bacry, E., Mastromatteo, I., & Muzy, J. F. (2015)

**Citation:** Bacry, E., Mastromatteo, I., & Muzy, J. F. (2015). Hawkes Processes in Finance. _Market Microstructure and Liquidity_, 1(1), 1550005.

- **Core Methodology:** Applies self-exciting point processes to model event intensity:
  $$\lambda(t) = \mu + \sum_{t_k < t} \phi(t - t_k)$$
  Compares monotonic exponential kernels $\phi_{\text{exp}}(\tau) = \alpha e^{-\gamma \tau}$ with non-monotonic Gamma/Rayleigh humped kernels $\phi_{\text{gamma}}(\tau) = \alpha \tau e^{-\gamma \tau}$ to evaluate their explanatory power on volatility clustering.
- **Comparison with Present Research:**
  - **Similarities:** Provides the formal signal-processing foundation for our benchmark hierarchy: the exponential kernel corresponds to our Baseline 3 (AR(1)/EWMA filter), while the non-monotonic Gamma kernel directly matches our Proposed Method 2 (a second-order linear Gamma state-space filter).
  - **Differences:** Standard Hawkes processes represent linear additive systems where every incoming headline injects identical memory weight regardless of prevailing saturation. Our non-linear SIRS filters (Methods 1 & 3) scale inflows by remaining market capacity $S_{i,t-1} \in [0, 1]$, naturally accounting for market attention saturation.

---

### 11. Osthus, D., Hickmann, K. S., Caragea, P. C., Higdon, D., & Del Valle, S. Y. (2017)

**Citation:** Osthus, D., Hickmann, K. S., Caragea, P. C., Higdon, D., & Del Valle, S. Y. (2017). Forecasting Seasonal Influenza with a State-Space SIR Model. _Journal of the American Statistical Association_, 112(518), 546–560.

- **Core Methodology:** Defines a discrete-time SIR transition system constrained to the 2-simplex ($\Delta^2$, such that $S_t + I_t + R_t = 1$) as a latent state-space process, coupled to an observation equation for predictive epidemiological tracking.
- **Comparison with Present Research:**
  - **Similarities:** Inspires our state-space parameterization, where latent vectors $(S_{i,t}, I_{i,t}, R_{i,t})$ are strictly bounded to the probability simplex $\Delta^2$ (enforcing $\Delta S + \Delta I + \Delta R = 0$ and $S+I+R=1$).
  - **Differences:** Osthus et al. target scalar case counts of annual, single-peak flu seasons. We merge the probability simplex state space with 19-dimensional news embeddings, deriving a division-free vector recursion:
    $$\mathbf{z}_{i,t}^{\text{SIR}} = (1 - \gamma + \beta S_{i,t-1})\mathbf{z}_{i,t-1}^{\text{SIR}} + \alpha_0 \tilde{m}_{i,t} S_{i,t-1}\mathbf{u}_{i,t}$$

---

## Pillar 3: Multimodal Time-Series Alignment, Financial NLP, and Volatility Modeling

### 12. Dong, Z., Fan, X., & Peng, Z. (2024)

**Citation:** Dong, Z., Fan, X., & Peng, Z. (2024). FNSPID: A Comprehensive Financial News Dataset in Time Series. _arXiv preprint arXiv:2402.06698_.

- **Core Methodology:** Assembles an aligned multimodal financial benchmark comprising 4,775 firms, 15.7 million UTC-timestamped news articles, and matching daily OHLCV equity prices. Benchmark experiments concatenate text sentiment/embeddings directly onto numerical price features (using zero-filling or forward-filling during non-event days) and feed them into LSTM, Transformer, and TimesNet architectures.
- **Comparison with Present Research:**
  - **Similarities:** Serves as the primary public dataset source for our empirical evaluations.
  - **Differences:** FNSPID passes filled representations directly into deep sequential architectures without isolating the impact of temporal filtering/imputation choices, and without strictly observing a 16:00 EST market cutoff or stratified sampling across market attention tiers. We implement an information-leakage-free 16:00 EST alignment and hold the downstream predictor fixed to a Ridge regression, isolating the empirical performance of competing state-space temporal filters.

---

### 13. Liu, H., et al. (2024)

**Citation:** Liu, H., et al. (2024). Time-MMD: Multi-Domain Multimodal Dataset for Time Series Analysis. _Advances in Neural Information Processing Systems (NeurIPS 2024)_.

- **Core Methodology:** Addresses irregular text arrivals against regular numerical time series by defining a "binary validity window": the frozen language model embedding of the latest report is carried forward unchanged (Last Observation Carried Forward, LOCF) until a new report arrives, merging representations via linear projections or cross-attention.
- **Comparison with Present Research:**
  - **Similarities:** Directly mirrors our Baseline 2 (Forward-Filling / LOCF baseline). Its late-fusion additive structure matches the linear formulation of our downstream Ridge forecasting model.
  - **Differences:** Time-MMD models textual relevance as an unattenuated step function throughout non-event intervals. Our findings demonstrate that during prolonged blackout intervals, static forward-filling continually injects stale noise, whereas SIRS-governed decay reduces multi-step forecast errors.

---

### 14. Wang, X., et al. (2024)

**Citation:** Wang, X., et al. (2024). From News to Forecast: Integrating Event Analysis in LLM-Based Time Series Forecasting with Reflection. _Advances in Neural Information Processing Systems (NeurIPS 2024)_.

- **Core Methodology:** Deploys an LLM reflection agent to filter and categorize news into transient shocks or enduring structural shifts, embedding formatted summaries directly into text prompts to generate predictive series trajectories.
- **Comparison with Present Research:**
  - **Similarities:** Shares the core intuition that news content displays heterogeneous diffusion dynamics and memory spans based on semantic content, motivating our Proposed Method 3 (a semantic-adaptive learnable SIRS filter).
  - **Differences:** Calls large LLM inferences at every forward forecasting step, creating high computational overhead and risks of temporal look-ahead leakage. Our Method 3 employs a lightweight (40-parameter), mathematically interpretable differentiable state-space gating formulation:
    $$\beta_{i,t} = (1-\alpha_0)\sigma(\mathbf{w}_\beta^\top \tilde{\mathbf{u}}_{i,t} + b_\beta), \quad \gamma_{i,t} = \sigma(\mathbf{w}_\gamma^\top \tilde{\mathbf{u}}_{i,t} + b_\gamma)$$
    Trained end-to-end via PyTorch backpropagation through time (BPTT) to modulate semantic propagation rates.

---

### 15. Araci, D. (2019)

**Citation:** Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. _arXiv preprint arXiv:1908.10063_.

- **Core Methodology:** Pre-trains a BERT architecture on domain-specific financial text (TRC2-financial) and fine-tunes a three-class classification head on Financial PhraseBank, yielding a 3D probability vector $\mathbf{p} = [P_{\text{pos}}, P_{\text{neg}}, P_{\text{neu}}]^\top \in \Delta^2$ and a 768-dimensional `[CLS]` hidden embedding $\mathbf{h} \in \mathbb{R}^{768}$.
- **Comparison with Present Research:**
  - **Similarities:** We utilize weights from the frozen `ProsusAI/finbert` checkpoint as our base feature extractor.
  - **Differences:** Araci evaluates static, sentence-level classification. We aggregate multi-article arrivals per firm-day, concatenating the 3D probability output with a 16-dimensional PCA projection of the semantic space (fitted exclusively on the 2016–2020 training set) to form $\mathbf{u}_{i,t} \in \mathbb{R}^{19}$. We then apply a Poisson hazard transformation:
    $$\tilde{m}_{i,t} = 1 - \exp(-(\bar{P}_{\text{pos}}+\bar{P}_{\text{neg}})\ln(1+N_{i,t}))$$
    to quantify arrival shock intensity.

---

### 16. Xu, Y., & Cohen, S. B. (2018)

**Citation:** Xu, Y., & Cohen, S. B. (2018). Stock Movement Prediction from Tweets and Historical Prices (StockNet). _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_, 1970–1979.

- **Core Methodology:** Introduces the Variational Movement Decoder (VMD), merging historical pricing with social media feeds to predict directional price movement. On non-event trading days ($N_{i,t} = 0$), the daily text representation defaults to a zero vector $\mathbf{0}$ or is masked out.
- **Comparison with Present Research:**
  - **Similarities:** Directly corresponds to our Baseline 1 (Impulse / Zero-Filling baseline), assuming exogenous text shocks dissipate immediately on non-event days ($\mathbf{z}_{i,t}^{\text{imp}} = \mathbf{0}$).
  - **Differences:** StockNet delegates temporal memory to a recurrent neural network (GRU). We decouple temporal memory into an explicit state filter, demonstrating that raw zero-filling fails to account for delayed market digestion following corporate earnings releases and high-impact disclosures.

---

### 17. Audrino, F., & Tetereva, A. (2019)

**Citation:** Audrino, F., & Tetereva, A. (2019). Sentiment Spillover Effects for US and European Companies. _Journal of Banking & Finance_, 107, 105593.

- **Core Methodology:** Smooths daily headline sentiment scores $s_{i,t}$ (or attention counts) using a first-order exponentially weighted moving average (EWMA) filter:
  $$S_{i,t}^{\text{EWMA}} = \lambda S_{i,t-1}^{\text{EWMA}} + (1 - \lambda) s_{i,t}, \quad \lambda \in (0, 1)$$
  The smoothed scalar series $S_{i,t}^{\text{EWMA}}$ is incorporated as an exogenous covariate in a linear HAR-RV volatility model (with regularization selected via LASSO/validation).
- **Comparison with Present Research (Baseline 3 & SIRS):**
  - **Similarities with Baseline 3:** Our Baseline 3 (Monotonic Exponential Decay / Causal AR(1) Filter) utilizes the exact same normalized recursive formulation:
    $$\mathbf{z}_{i,t}^{\text{exp}} = \lambda \mathbf{z}_{i,t-1}^{\text{exp}} + (1 - \lambda) \tilde{m}_{i,t} \mathbf{u}_{i,t}$$
    where weights sum to unity ($\lambda + (1-\lambda) = 1$) prior to entering a linear HAR-RV model.
  - **Differences with Present Research:**
    1. _Scalar Cancellation vs. 19-Dimensional Directional Impact:_ Audrino & Tetereva compress daily news into a 1D net-sentiment scalar $s_{i,t}$. If ten positive and ten negative articles break on an earnings day, net sentiment cancels to zero. Our formulations preserve non-neutral shock magnitude via $(\bar{P}_{\text{pos}}+\bar{P}_{\text{neg}})\ln(1+N_{i,t})$ applied to 19-dimensional vectors $\tilde{m}_{i,t}\mathbf{u}_{i,t} \in \mathbb{R}^{19}$ (retaining 3 sentiment dimensions alongside 16 PCA topics).
    2. _Monotonic Decay vs. State-Dependent Hump Amplification:_ The scalar decay factor $\lambda < 1$ in EWMA forces the impact of an event to peak strictly at release ($t=0$) and decay monotonically. Our SIRS filter generalizes this parameter into a state-dependent gain $g_{i,t} = 1 - \gamma + \beta S_{i,t-1}$. When latent attention capacity is sufficient ($S_{i,t-1} > \gamma/\beta$), $g_{i,t} > 1$, producing a humped adoption trajectory that peaks post-release before fading.

---

### 18. Corsi, F. (2009)

**Citation:** Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility (HAR-RV). _Journal of Financial Econometrics_, 7(2), 174–196.

- **Core Methodology:** Grounded in the Heterogeneous Market Hypothesis (distinguishing daily traders, weekly rebalancers, and monthly institutions), this framework regresses realized volatility against trailing lags over three temporal scales: past 1-day ($y_{i,t}$), 5-day average ($y_{i,t}^{(5)}$), and 22-day average ($y_{i,t}^{(22)}$).
- **Comparison with Present Research:**
  - **Similarities:** Serves as our Baseline 0 (pure numerical HAR-RV). The three multi-horizon autoregressive components $\mathbf{x}_{i,t}^{\text{har}} = [y_{i,t}, y_{i,t}^{(5)}, y_{i,t}^{(22)}]^\top \in \mathbb{R}^3$ are retained across every comparative model as fixed autoregressive controls.
  - **Differences:** Corsi implements unregularized OLS for individual market indices without textual features. We adapt the model to a 50-stock panel, incorporating the 19-dimensional latent state vector $\mathbf{z}_{i,t}$ and applying $L_2$-regularized Ridge regression to prevent overfitting across high-dimensional covariates.

---

### 19. Parkinson, M. (1980)

**Citation:** Parkinson, M. (1980). The Extreme Value Method for Estimating the Variance of the Rate of Return. _Journal of Business_, 53(1), 61–65.

- **Core Methodology:** Leverages extreme value properties of Brownian motion to derive an unbiased estimator of intraday return variance using daily highs ($H_{i,t}$) and lows ($L_{i,t}$):
  $$\text{RV}_{i,t}^2 = \frac{1}{4\ln 2}(\ln H_{i,t} - \ln L_{i,t})^2$$
  This estimator achieves roughly fivefold higher asymptotic efficiency relative to standard squared close-to-close returns $(\ln C_t - \ln C_{t-1})^2$.
- **Comparison with Present Research:**
  - **Similarities:** Serves as the mathematical formulation for constructing daily target realized volatility values.
  - **Differences:** Parkinson specifies volatility in the original variance scale. To stabilize variance, approximate normality, and prevent extreme outlier trading days from dominating the MSE objective, we apply a logarithmic transform:
    $$y_{i,t} = \ln(\text{RV}_{i,t} + 10^{-8})$$
    Standardized via Z-scores using exclusively the firm-level mean and variance computed over the 2016–2020 training period.
