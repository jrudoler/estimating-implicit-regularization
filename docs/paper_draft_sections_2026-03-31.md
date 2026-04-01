# Draft Paper Sections (2026-03-31)

These are draft sections/edits for `paper/main.tex`. Review and integrate as needed.

---

## 1. Fix contributions list (replace line 53 TODO)

Replace:
```latex
\item TODO: something something deep learning yay
```

With:
```latex
\item Empirical estimation of the implicit regularization induced by dropout in deep nonlinear networks, finding that the effective $\ell_2$ penalty increases monotonically with dropout rate.
\item A diagnostic analysis showing when individual regularizer components are identifiable from gradient information and when they are not, based on the collinearity of candidate regularizer gradients.
```

---

## 2. Flesh out dropout section (replace lines 318-620)

Replace the skeletal dropout subsection with:

```latex
\subsection{Dropout regularization}
\label{sec:dropout}

Dropout \citep{srivastava_dropout_2014} randomly zeroes out hidden units during training and is broadly understood to act as an adaptive regularizer that penalizes over-reliance on specific connections. Several theoretical analyses have shown that, for linear models, dropout is equivalent to a form of adaptive $\ell_2$ regularization whose strength depends on the dropout rate and the second moments of the activations \citep{wager_dropout_2013, wei_implicit_2020}.

We test this prediction empirically using our gradient-matching framework. We train deep ReLU classifiers on MNIST with varying dropout rates ($p \in \{0, 0.05, 0.1, \ldots, 0.5\}$), depths ($d \in \{1, 3, 5\}$), and widths ($w \in \{128, 256, 512\}$). No explicit weight decay is used ($\lambda = 0$). After training each model, we estimate the best effective $\ell_2$ regularizer (scalar ridge penalty $\hat\lambda \|\theta\|_2^2$) by fitting a single parameter to the gradient-matching objective. We repeat each configuration across 10 random seeds.

Figure~\ref{fig:dropout-l2} shows the estimated ridge penalty $\hat{\lambda}$ as a function of dropout rate, faceted by depth and width. The estimated $\ell_2$ penalty increases monotonically with dropout rate across all architectures, consistent with the theoretical prediction that dropout acts as adaptive $\ell_2$ regularization. The effect is most pronounced for narrower networks (width 128), where the per-weight regularization pressure is highest. For wider networks, the absolute magnitude of the estimated penalty decreases but the monotonic trend persists. The color encodes the gradient-matching loss (darker = better fit), confirming that the $\ell_2$ model fits well across all configurations.

\begin{figure}[h]
    \centering
    \includegraphics[width=\linewidth]{figures/dropout_bias_ridge_panel.png}
    \caption{Estimated $\ell_2$ regularization strength ($\hat{\lambda}$) as a function of dropout rate for MNIST classifiers. Each point is one seed; columns vary width, rows vary depth. Color indicates gradient-matching loss (darker = better fit). The monotonic increase with dropout rate is consistent with the theoretical prediction that dropout acts as adaptive weight decay.}
    \label{fig:dropout-l2}
\end{figure}
```

---

## 3. New section: Early stopping in nonlinear networks (after dropout, before "Norms vs rank")

```latex
\subsection{Early stopping in nonlinear networks}
\label{sec:nonlinear-early-stopping}

Having verified in Section~\ref{sec:gd-ols} that our framework correctly recovers the theoretically predicted $\ell_2$ regularization of early stopping in linear regression, we now ask whether the same qualitative behavior extends to nonlinear models.

We train a depth-2 ReLU network (width 32, input dimension 12) on a synthetic teacher-student regression task \emph{without any explicit regularization}, using Adam optimization. At multiple training checkpoints (epochs 10, 25, 50, 100, 200, 500, 1000, 2000), we estimate two models of implicit regularization:
\begin{enumerate}
    \item A scalar $\ell_2$ penalty $\hat\lambda \|\theta\|_2^2$, fit in closed form via least-squares projection of the target gradient onto $2\theta$.
    \item A smoothed power-family penalty $\hat\lambda \sum_j (\theta_j^2 + \epsilon)^{p/2}$ with jointly estimated $(\hat\lambda, \hat p)$, fit via gradient descent.
\end{enumerate}

Figure~\ref{fig:early-stopping-nonlinear} shows the results. Consistent with the OLS theory, the estimated $\ell_2$ penalty strength rises during early training (peaking around epoch 50 when the model is still underfitting) and then decays toward zero as the model converges. The power-family estimator settles on an exponent $\hat p \approx 3.2$, suggesting that the implicit bias of early stopping in this nonlinear setting is not purely $\ell_2$ but involves a penalty that is sharper on large weights.

The gradient cosine similarity (fit quality) is modest ($\approx 0.2$--$0.3$), indicating that neither $\ell_2$ nor the power family captures the full structure of the implicit regularization. This is expected: the regularization-as-stationarity assumption is only approximate for a model trained with Adam and early stopping. Nevertheless, the qualitative agreement with the theoretical prediction---regularization strength that peaks early and decays---provides evidence that the general phenomenon extends beyond the linear setting.

\begin{figure}[h]
    \centering
    \includegraphics[width=\linewidth]{figures/nonlinear_early_stopping_probe/early_stopping_probe.pdf}
    \caption{Implicit regularization of early stopping in a nonlinear ReLU network. \textbf{Left:} Estimated regularization strength $\hat\lambda$ decreases with training, matching the OLS theory prediction. \textbf{Center:} The power-family exponent stabilizes near $p \approx 3.2$, above the $\ell_2$ value of $p = 2$. \textbf{Right:} Gradient cosine (fit quality) peaks in the intermediate training regime where the stationarity approximation is best.}
    \label{fig:early-stopping-nonlinear}
\end{figure}
```

---

## 4. Model evaluation section (replace red-text lines 414-448)

Replace the red `\color{red}` block with:

```latex
\subsection{Evaluating learned regularization parameters}
\label{sec:model-eval}

Once we fit the BER, how do we assess its quality? We consider three complementary criteria:

\paragraph{Parameter recovery.} When the true regularization is known (e.g., in our explicit-regularizer sanity checks), we compare estimated parameters directly: $\|\hat\Lambda - \Lambda_{\text{true}}\|$.

\paragraph{Gradient alignment.} Since our framework estimates regularization by matching gradients, a natural quality metric is the cosine similarity between the predicted regularizer gradient $\nabla_\theta \mathcal{R}(\theta, \hat\Lambda)$ and the target gradient $-\nabla_\theta \mathcal{L}(\theta, S)$. This measures how well the candidate regularizer explains the direction of the residual loss gradient. A cosine of 1 indicates perfect directional agreement.

\paragraph{Retrain validation.} If the estimated regularizer accurately captures the effective bias of the original training procedure, then retraining a model with the estimated regularizer as an explicit penalty (and no other implicit bias) should recover similar weights or predictions. Formally, if $\hat\theta_{\text{retrain}} = \arg\min_\theta \mathcal{L}(\theta, S) + \mathcal{R}(\theta, \hat\Lambda)$, we expect $\hat\theta_{\text{retrain}} \approx \hat\theta$ when the regularizer model is correct (see Theorem~1 for the linear case).

\paragraph{Regression in gradient space.} Our framework can also be viewed as fitting a regression model where the target variable is $-\nabla_\theta \mathcal{L}(\hat\theta, S)$ and the predictors are the gradients of candidate regularizers at $\hat\theta$. Standard regression diagnostics---residual analysis, variance explained, and collinearity between candidate regularizer gradients---provide principled tools for model selection and identifiability assessment.
```

---

## 5. Resolve `\jhr{}` notes

- **Line 467-468** (`\jhr{Why are the data generated from a linear model?}`): Delete or replace with: "We use a linear data-generating process for this initial validation because it isolates the regularizer recovery problem from function-approximation error. Section~\ref{sec:nonlinear-early-stopping} extends to nonlinear data."
- **Line 505** (`\jhr{Figure caption...}`): Already addressed in new caption text above.
- **Line 593-601** (multiple `\jhr{}` about noise and convergence): Delete -- these are resolved by the nonlinear experiments.
- **Line 601** (Yotam's idea about varying SGD noise): Delete or move to future work.
- **Line 634-637** (Konrad's ViT idea): Replace with: "Extending this framework to large-scale models (e.g., Vision Transformers or language models) is a natural direction for future work."

---

## 6. Remove placeholder sections

- **Line 604** (`\subsection{Neural nets stuff}`): Remove entirely. The dropout and early-stopping sections now cover this.
- **Line 633-637** (`\subsection{Identifying regularization in complex models}`): Replace with a brief future-work paragraph in the discussion, or remove.

---

## 7. Rename figure file

The elastic net figure uses a W&B artifact name:
```
figures/sweep_krcszr6z_elasticnet_estimation_heatmap_detailed.pdf
```
Rename to something readable like `elasticnet_estimation_heatmap.pdf` and update the reference on line 500.
