## Method

For one query with candidate passages $d_1,\dots,d_n$:

- $r_j$: cross-encoder logit (InfoMax) or z-score fusion of cross-encoder logit and dense cosine (InfoMax+F).
- $e_j$: all-MiniLM-L6-v2 embedding, centred on the pool mean and renormalised; $q$ the query embedding, treated the same way.
- $w_j = \mathrm{minmax}(r)_j^{\gamma}$: relevance weight.

**Target.** A latent query-information vector $z \in \mathbb{R}^r$, with prior $z \sim \mathcal N(0,\Lambda)$, on the aspect basis
$U = [\,q,\ u_2,\dots,u_r\,]$. Here $u_2,\dots,u_r$ are the top right-singular vectors of the relevance-weighted pool embeddings
after removing the query direction, and $\Lambda = \mathrm{diag}(1, \rho\, s_k^2/s_1^2)$.

**Observations.** Each passage is a noisy linear measurement of the target:

$$y_j = \sqrt{w_j}\,(U^\top e_j)^\top z + g_j^\top u + \varepsilon_j,\qquad g_j = (I-UU^\top)e_j .$$

The shared nuisance $u \sim \mathcal N(0,\beta I)$ correlates the noise of passages with similar off-query content (duplicates,
same source), and $\varepsilon_j \sim \mathcal N(0,\sigma^2)$.

**Selection.** Greedy maximisation of the conditional mutual information

$$I(z; y_j \mid y_S) = \tfrac12 \log \frac{\mathrm{Var}(y_j \mid y_S)}{\mathrm{Var}(y_j \mid y_S, z)},$$

computed exactly by Schur complements. No probabilities from a language model are needed: in the linear-Gaussian model the
posterior entropy depends only on which passages are selected.

**Anchoring.** The first pick is the top-1 of the base ranking $r$. At each later step, the base ranking's next item $b$ is replaced by
the max-gain item $j^\star$ only if $\mathrm{gain}(j^\star) > (1+\delta)\,\mathrm{gain}(b)$. With $\delta=\infty$ the selector returns
the base ranking exactly.

**Guarantee.** None is claimed. Correlated noise breaks submodularity of the objective, so the greedy rule has no worst-case
approximation bound.
