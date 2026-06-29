# Formative 3 — Probability Distributions, Bayesian Probability & Gradient Descent
Group 22

---

## Repository Structure

```
├── GaltonFamilies.csv                          # Dataset used in Part 1
├── Formative3-Probability-Distributions-Assignment_Group22.ipynb  # Main notebook
└── README.md
```

---

## Part 1 — Probability Distributions & EM Algorithm

### Overview
We were given a dataset of heights from two populations: Mothers and Children (Galton Families dataset). The task required us to treat the data as unlabeled — a mixture of two Gaussian distributions — and use the Expectation-Maximization (EM) algorithm to recover the underlying populations.

### Dataset
- Source: GaltonFamilies dataset
- Populations modelled: Mothers and Children
- Preprocessing: Height columns were extracted and combined into a single unlabeled array to simulate a real-world mixture scenario.

### Approach

#### Why not just split at the global mean?
Splitting at the global mean assigns a hard label to every data point based on a single threshold. This ignores the fact that the two distributions overlap, meaning many heights are plausibly from either population. The EM algorithm instead uses responsibilities — every data point gets a probability of belonging to each group, which leads to far more accurate parameter estimates.

#### EM Algorithm — Step by Step

Initialization
- Two means (`μ1`, `μ2`) are randomly seeded from the data.
- Standard deviations (`σ1`, `σ2`) are both set to the global standard deviation.
- Mixing coefficients (`π1`, `π2`) are initialized equally at 0.5.

E-Step (Expectation)  
For each data point, compute the responsibility — the posterior probability that the point belongs to each Gaussian component — using Bayes' theorem:

$$r_1^{(i)} = \frac{\pi_1 \cdot \mathcal{N}(x^{(i)}; \mu_1, \sigma_1)}{\pi_1 \cdot \mathcal{N}(x^{(i)}; \mu_1, \sigma_1) + \pi_2 \cdot \mathcal{N}(x^{(i)}; \mu_2, \sigma_2)}$$

M-Step (Maximization)  
Update all parameters using the responsibilities computed in the E-step:
- Means: Weighted average of data points, where weights are the responsibilities.
- Variances: Weighted variance around the updated mean.
- Mixing coefficients: Proportion of total responsibility assigned to each component.

Convergence  
Iteration stops when the change in means between steps falls below a tolerance of `1e-6`, indicating the log-likelihood has effectively converged.

### Optimization Tracking Table

| Iteration | μ1 (Children) | μ2 (Mothers) | σ1² | σ2² | π1 | π2 | Log-Likelihood |
|-----------|--------------|-------------|-----|-----|----|----|----------------|
| 0 (Init)  | —            | —           | —   | —   | —  | —  | —              |
| 1         | —            | —           | —   | —   | —  | —  | —              |
| 2         | —            | —           | —   | —   | —  | —  | —              |

> Values are printed by running the notebook. The table above will be completed during the presentation.

### Classification
Given a test height, the model outputs:
- `P(Child | height)` — posterior probability the height belongs to a Child
- `P(Mother | height)` — posterior probability the height belongs to a Mother

### Visualizations
1. Mixed height histogram — shows the combined unlabeled data with the global mean marked.
2. Fitted Gaussians — overlays the two learned distributions on the histogram.
3. Log-likelihood convergence — shows how the model improves over iterations until convergence.

---

## Part 2 — Bayesian Probability
> To be completed by Thomas

---

## Part 3 — Gradient Descent (Manual Calculations)
> To be completed by the group

---

## Part 4 — Gradient Descent in Code
> To be completed by Arnold

---

## How to Run

1. Open the notebook in Google Colab.
2. Upload `GaltonFamilies.csv` to the Colab session or place it in `/content/`.
3. Run all cells from top to bottom.
4. To classify a test height, call:
   ```python
   classify_height(64, mu1, mu2, sigma1, sigma2, pi1, pi2)
   ```
   Replace `64` with any height given during the presentation.

---

## Contributors

| Name | Contribution |
|------|-------------|
| Racheal Resty Akello | Part 1 — EM Algorithm implementation, visualizations |
| [Teammate 2] | Part 2 — Bayesian Probability |
| [Teammate 3] | Part 3 & 4 — Gradient Descent |
| [Teammate 4] | Part 3 & 4 — Gradient Descent |