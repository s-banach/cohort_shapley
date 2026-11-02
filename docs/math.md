# Decomposing a change in mean prediction

Let $f:X\rightarrow\mathbb{R}$ be a prediction function, where $X$ is a feature space with $d$ coordinates.

Let $C_1,C_2$ be two finite, nonempty cohorts of data in $X$.
Our goal is to decompose the difference

$$
\Delta=\text{Mean}_{C_2}(f)-\text{Mean}_{C_1}(f)
$$

into the contribution of each of the $d$ features.

A familiar trick is to compute the difference in mean SHAP vectors between the cohorts:

$$
\text{Mean}_{x\in C_2} S_B(f,x)
-\text{Mean}_{x\in C_1} S_B(f,x),
$$

where $S_B(f,x)\in\mathbb{R}^d$ is the SHAP vector of $f$ at $x$, relative to a fixed background distribution $B$.
For example, $B$ might be the empirical distribution of the training dataset.

## Questions for the reader

1. Why do you compute SHAP over a background dataset?
   Can this ever lead you astray?

2. Have you noticed how indirect the process is?
   We want to explain a single number $\Delta$, but we explain millions of predictions, average their explanations, and take the difference of the averages.
   If we look at the math, is there a more direct way to compute the thing we actually want?

3. Have you ever tried to compute SHAP vectors at millions of points, realized it was too slow, and fallen back to TreeSHAP?
   Does this work properly when the model applies a nonlinear link between the tree scores and the prediction?

## 1. Compare not thyself unto thy neighbor, but unto thyself of yesterday

Suppose $C_1$ and $C_2$ contain last year's and this year's records for the same car insurance policyholders.

One driver lives in Michigan in both years, but her predicted premium rises as she gets older.
Yet the `state` entry in her SHAP vector may change, because the effect of replacing Michigan with a _background state_ may depend on her age.

Applying this principle to the dataset as a whole, each policyholder in $C_1$ should be compared to herself in $C_2$ to explain the change in premium for the cohort as a whole, not to a background dataset.


## 2. Apply the Shapley formula directly to $\Delta$

[SHAP](https://arxiv.org/abs/1705.07874) applies the general Shapley formula to an individual prediction relative to a background distribution.
We can apply the same formula directly to $\Delta$, without first decomposing the individual predictions.

For any subset of features, replace those features in each driver's $C_1$ record with the corresponding values from their $C_2$ record.
Average the resulting predictions and subtract the original mean prediction.
Replacing no features gives zero; replacing all features gives $\Delta$.
The Shapley formula decomposes $\Delta$ by averaging each feature's increments across replacement orders.

For example, suppose $f$ uses two features: age and annual mileage.
For each row below, predict every driver's premium using her own age and mileage from the specified years, then average the predictions.
Suppose we obtain:

| Inputs for each driver | Mean predicted premium (dollars) |
|---|---:|
| Age and mileage from last year | 1,000 |
| Age from this year, mileage from last year | 1,030 |
| Age from last year, mileage from this year | 1,050 |
| Age and mileage from this year | 1,100 |

Here $\Delta=1100-1000=100$ dollars.
Replacing age before mileage increases the mean by $1030-1000=30$ dollars; replacing age after mileage increases it by $1100-1050=50$ dollars.
The Shapley contribution of age to $\Delta$ is therefore $(30+50)/2=40$ dollars, leaving 60 dollars for mileage.

When exact computation is too costly, we can sample drivers and feature subsets to estimate this decomposition directly.

## 3. Can we convert TreeSHAP's answer to dollars?

TreeSHAP's fast exact calculation explains the additive tree score, before the nonlinear link converts it to a prediction.
Exponentiating, linearizing, or rescaling those contributions does not generally recover exact Shapley contributions to the prediction.
No conversion based only on the score contributions and endpoint scores can do so for all models; [Appendix C](#appendix-c-converting-score-contributions) gives the proof.
We must apply the Shapley formula to the actual prediction function $f$.

## 4. What if there is cohort turnover?

Cohort turnover means entry into and exit from the cohort.
Removing policyholders with below-average predicted premiums raises the book's mean prediction without changing the remaining policyholders' predictions.
When at least one policyholder remains in both cohorts, we split $\Delta$ into three comparisons:

| Component | Comparison |
|---|---|
| Matched | Each continuing policyholder's record in $C_1$ to her own record in $C_2$ |
| Entry | Continuing policyholders to entrants, using their records in $C_2$ |
| Exit | Departing policyholders to continuing policyholders, using their records in $C_1$ |

We apply the Shapley formula to each comparison, scaling entry by the entrants' share of $C_2$ and exit by the departing policyholders' share of $C_1$.
The three sets of feature contributions together sum to $\Delta$.
The [code guide](usage.md#include-cohort-turnover) gives an example and the accounting formulas.

## Appendix A. The Shapley formula for a change in mean

Let $D=\{1,\ldots,d\}$.
For $A\subseteq D$, write $h_A(p,q)$ for the **hybrid observation** with coordinates in $A$ taken from $q$ and all other coordinates from $p$.
Assume $f$ can evaluate these observations.

Let $(p,q)$ be a random pair, with $p$ uniform over $C_1$ and $q$ uniform over $C_2$.
When both cohorts contain the same policyholders, choose one uniformly and take her records from the two years.
Without such a pairing, we can draw the two observations independently.
These choices have the same cohort means but can give different feature contributions.

Define

$$
v(A)=\mathbb{E}\bigl[f(h_A(p,q))-f(p)\bigr].
$$

Thus $v(\varnothing)=0$ and $v(D)=\Delta$.
The **Shapley contribution** of feature $j$ is

$$
c_j=\sum_{A\subseteq D\setminus\{j\}}
\frac{|A|!\,(d-|A|-1)!}{d!}
\bigl[v(A\cup\{j\})-v(A)\bigr].
$$

The coefficient is the fraction of feature replacement orders with exactly $A$ before $j$.
Consequently, $c_j$ is the average prediction increment from replacing feature $j$, averaged over both orders and pairs.
Along every complete order, the increments sum to $f(q)-f(p)$, so

$$
\sum_{j=1}^d c_j=\Delta.
$$

For a pair of the form $(p_1,p_2)\rightarrow(q_1,p_2)$, replacing feature 2 leaves every hybrid observation unchanged.
More generally, if feature $j$ agrees within every pair, then $v(A\cup\{j\})=v(A)$ whenever $j\notin A$, so $c_j=0$.

## Appendix B. A background comparison can credit an unchanged feature

<a id="an-unchanged-feature-receives-zero-contribution-within-a-pair"></a>

Consider $f(x_1,x_2)=x_1x_2$ and two entities:

| Entity | $C_1$ | $C_2$ | Prediction change |
|---|---|---|---:|
| A | (1, 10) | (2, 10) | 10 |
| B | (2, 1) | (1, 1) | −1 |

The mean prediction increases from 6 to 10.5, so $\Delta=4.5$.
Feature 2 stays fixed within each entity; comparing each entity with itself gives contributions $(4.5,0)$.

Take $B$ to be the empirical distribution of $C_1$.
Use **interventional SHAP**: for each subset of features, keep those features from the observation being explained and take the remaining features from a randomly drawn background observation.
The mean SHAP vector is $(0,0)$ in $C_1$ and $(2.25,2.25)$ in $C_2$, assigning half the change to feature 2.
The interaction $x_1x_2$ makes the feature-2 entry depend on feature 1, even when feature 2 stays fixed within an entity.

With a fixed background and exact SHAP vectors on the prediction scale, the difference in mean SHAP vectors still sums to $\Delta$, because the common baseline cancels.
That identity does not require unchanged features to receive zero contribution.

We can also apply the Shapley formula directly to the original background comparison.
Define

$$
v_B(A)=
\mathbb{E}_{x\sim C_2,\,r\sim B}f(h_A(r,x))
-\mathbb{E}_{x\sim C_1,\,r\sim B}f(h_A(r,x)),
$$

where $x$ and $r$ are drawn independently and each cohort is sampled uniformly.
By linearity, the Shapley vector of $v_B$ equals the difference in mean SHAP vectors from the introduction.
We can therefore compute the aggregate explanation directly whether we use the supplied pairs or the background comparison.

## Appendix C. Converting score contributions

Write a tree ensemble's score as $g(x)=b+\sum_t T_t(x)$, where $T_t(x)$ is the terminal-node value from tree $t$, and its prediction as $f(x)=\sigma(g(x))$.
The **inverse link** $\sigma$ converts scores to predictions: for a log link, $\sigma(g)=e^g$; for a logit link, $\sigma(g)=1/(1+e^{-g})$.

For a fixed comparison, write $c_j(u)$ for the Shapley contribution computed using output function $u$.
Linearity gives $c_j(g)=\sum_t c_j(T_t)$.
The [TreeSHAP derivation](https://arxiv.org/html/1802.03888v3#S3) uses this additivity to combine explanations of separate trees.
A nonlinear $\sigma$ generally prevents the same calculation for $f$.

### Common conversions

For a comparison between two records, we could try:

- **Exponentiate the contributions** for a log link, since exponentiating the score gives the prediction.
  This gives multiplicative factors whose product is the prediction ratio, not additive contributions to the prediction change.
- **Multiply by the derivative of the inverse link at the starting score**, treating the conversion to prediction units as locally linear.
  This is a first-order approximation; the prediction increment depends on the starting score.
- **Rescale all contributions to sum to the prediction change**, preserving their relative sizes.
  For a nonzero score change, this gives the correct total, but the nonlinear transformation can change the relative sizes of the Shapley contributions.

For supported nonlinear output transforms, SHAP version 0.52.0's interventional C++ implementation uses the third conversion separately for each background observation before averaging; see the [implementation](https://github.com/shap/shap/blob/v0.52.0/shap/cext/tree_shap.h#L1136-L1148).

### Two models with identical score contributions

Let $x_1$ indicate a claim and $x_2$ a speeding conviction.
Consider scores $g(x_1,x_2)=x_1+2x_1x_2$ and $\widetilde g(x_1,x_2)=2x_1+x_2$ on $\{0,1\}^2$.
Use the inverse link $\sigma(g)=100\times 2^g$ for both models, giving these premiums in whole dollars:

| Claim | Speeding conviction | Premium from $g$ | Premium from $\widetilde g$ |
|---|---|---:|---:|
| No | No | \$100 | \$100 |
| Yes | No | \$200 | \$400 |
| No | Yes | \$100 | \$200 |
| Yes | Yes | \$800 | \$800 |

Compare the original record $(0,0)$ with $(1,1)$, after a claim and a conviction.
For $g$, adding the claim first increases the score by 1, while adding it after the conviction increases the score by 3.
The claim's average score increment is therefore 2, leaving 1 for the conviction.
For $\widetilde g$, the claim always adds 2 score units and the conviction always adds 1.
Both models thus have score contributions $(2,1)$ and endpoint scores 0 and 3.

Now apply the same calculation to the premiums.
For $g$, the claim adds $200-100=100$ dollars before the conviction and $800-100=700$ dollars after it, averaging to 400 dollars.
For $\widetilde g$, the claim adds $400-100=300$ dollars before the conviction and $800-200=600$ dollars after it, averaging to 450 dollars.
The remaining contributions to the 700-dollar premium increase are 300 and 250 dollars respectively.

A conversion given only the score contributions and endpoint scores receives identical inputs for the two models, so it cannot produce the correct answer for both.
Both score functions can be represented by tree ensembles on this binary feature space.

### What the conversions produce

For this inverse link, the three conversions give the following results for either model:

| Conversion | Claim | Conviction |
|---|---:|---:|
| Exponentiate to obtain premium factors | $2^2=4$ | $2^1=2$ |
| Multiply by the derivative at the starting score, $100\ln 2$ | About 139 dollars | About 69 dollars |
| Multiply by the ratio of premium change to score change, $700/3$ | About 467 dollars | About 233 dollars |

The factors multiply to the correct premium ratio of 8.
The rescaled contributions sum to the correct premium increase of 700 dollars.
Neither property gives the Shapley contributions: $(400,300)$ for the first model and $(450,250)$ for the second.

### Transforming the trees themselves

Applying $\sigma$ separately to each tree's terminal-node values produces a sum of transformed outputs, which generally differs from transforming their sum.
This changes the prediction function being explained.
For a single tree, we can transform its terminal-node predictions and explain the resulting tree exactly.
For an ensemble, combining all trees into one tree can require a separate region for each attainable combination of terminal nodes, making the combined tree much larger.
The efficient calculation on the separate trees therefore does not extend to arbitrary nonlinear inverse links by converting its output.
