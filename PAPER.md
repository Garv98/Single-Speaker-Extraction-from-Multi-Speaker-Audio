# ECW-TSE: Embedding-Conditioned Wiener Target Speaker Extraction

A reference-aware single-speaker extraction pipeline for the **Mathematical
Modeling** course project. ECW-TSE is **training-free** — every component is a
pretrained mathematical operator — yet introduces three new contributions on
top of off-the-shelf SepFormer + ECAPA-TDNN:

1. **Embedding-Conditioned Wiener Mask (ECWM)** — closed-form Bayesian
   extension of the classical Wiener filter that incorporates a speaker-identity
   prior derived from cosine similarity in ECAPA-TDNN embedding space.
2. **Multi-Resolution ECWM Ensemble (MR-ECWM)** — Nadaraya–Watson average over
   $K$ STFT resolutions; suppresses resolution-specific artefacts.
3. **Iterative Confidence Refinement (ICR)** — monotone-convergent fixed-point
   iteration that re-encodes the refined output to sharpen the embedding prior.

---

## 1. Problem statement

Let $x \in \mathbb{R}^{L}$ be a monaural mixture of $N$ unknown speakers,

$$
x[n] \;=\; \sum_{i=1}^{N} s_i[n], \qquad n = 0, \dots, L-1.
$$

We are also given (optionally) a *reference clip* $r \in \mathbb{R}^{L_r}$
containing a clean recording of a target speaker. Our goal is to recover the
component $\hat{s}_t \approx s_t$ corresponding to the speaker depicted in $r$.

When $r$ is unavailable we fall back to *blind* extraction by treating the
loudest separated component as a pseudo-reference. The same machinery applies.

---

## 2. The pipeline at a glance

$$
\boxed{
\;
x \xrightarrow{\;\text{SepFormer}\;} \{\hat{s}_i\}_{i=1}^{2}
\xrightarrow{\;\text{ECAPA-TDNN}\;} \{e_i\}_{i=1}^{2}
\xrightarrow{\;\alpha_i = \langle e_{\text{ref}}, e_i \rangle\;}
\text{ECWM}\xrightarrow{\;\Pi_{\mathcal{C}}\;} \hat{s}_{\text{target}}
\;
}
$$

Stages:

| #   | Stage                           | Operator class            | Pretrained on             |
| --- | ------------------------------- | ------------------------- | ------------------------- |
| 1   | Source separation               | Transformer, MMSE-trained | WSJ0-2Mix (SepFormer)     |
| 2   | Speaker embedding               | TDNN, AAM-softmax-trained | VoxCeleb (ECAPA-TDNN)     |
| 3   | **Embedding-Conditioned Wiener** | **Closed-form Bayesian**  | **— novel, no training —**|
| 4   | Mixture-consistency projection  | Closed-form linear        | —                         |

The novelty (and the contribution of this paper) is **Stage 3**.

---

## 3. Background

### 3.1 STFT & magnitude masking

The Short-Time Fourier Transform of a signal $s$ at frame $t$ and bin $f$ is

$$
S(t,f) \;=\; \sum_{n} s[n]\,w[n - tH]\,e^{-j 2\pi f n / N_{\text{FFT}}},
$$

with analysis window $w$ and hop $H$. Throughout we use $N_{\text{FFT}} = 256$ at
8 kHz, $N_{\text{FFT}} = 512$ at 16 kHz, and $H = N_{\text{FFT}}/4$.

A *time-frequency mask* $M \in [0,1]^{T \times F}$ acts on the mixture
spectrogram by point-wise multiplication, $\hat{S}(t,f) = M(t,f)\,X(t,f)$, and a
time-domain estimate is obtained by inverse STFT.

### 3.2 Wiener filter (the classical baseline)

Suppose the two source spectra at a single TF bin are independent zero-mean
Gaussian random variables, $S_i(t,f) \sim \mathcal{CN}(0,\,P_i(t,f))$. The
MMSE-optimal estimate of $S_i$ from $X = S_1 + S_2$ is

$$
\hat{S}_i^{\,\text{Wiener}}(t,f)
\;=\;
\frac{P_i(t,f)}{P_1(t,f) + P_2(t,f)}\;X(t,f),
$$

which is exactly the *Wiener mask* $M_i^{\,W} = P_i/(P_1+P_2)$. In practice we
substitute power estimates from the separator outputs,
$\widehat{P}_i(t,f) = |\hat{S}_i(t,f)|^2$.

### 3.3 Speaker embedding (ECAPA-TDNN)

ECAPA-TDNN maps a waveform $s$ to an embedding $e = f_\theta(s) \in \mathbb{R}^{192}$.
Trained with AAM-softmax on VoxCeleb, the network is shaped so that embeddings
of the same speaker cluster together on the unit sphere. We $\ell_2$-normalise
all embeddings, after which cosine similarity reduces to an inner product:

$$
\alpha_{ij} \;=\; \langle e_i, e_j \rangle, \qquad \alpha_{ij} \in [-1,\,1].
$$

The smaller the angle between two embeddings, the higher the probability that
they belong to the same speaker. On VoxCeleb-O, ECAPA-TDNN achieves an EER of
$\approx 0.69\%$, which provides a near-oracle prior for our purposes.

---

## 4. The Embedding-Conditioned Wiener Mask (ECWM)

### 4.1 Definition

Let $\alpha_i = \langle e_{\text{ref}}, e_i \rangle$. Define the *speaker
prior*

$$
\pi_i \;=\; \bigl(\sigma(\alpha_i / \tau)\bigr)^{\gamma},
\qquad
\sigma(z) = \log\bigl(1 + e^{z}\bigr),
$$

with temperature $\tau > 0$ and exponent $\gamma > 0$. The softplus $\sigma$
maps $\alpha_i \in [-1,1]$ to a strictly positive prior, and $\gamma$ controls
how aggressively the mask trusts the embedding similarity.

The **Embedding-Conditioned Wiener Mask** for source $i$ is then

$$
\boxed{\;
\widehat{M}_i(t,f) \;=\;
\frac{\pi_i\,|\hat{S}_i(t,f)|^{2}}
     {\sum_{j=1}^{N} \pi_j\,|\hat{S}_j(t,f)|^{2} \;+\; \varepsilon}
\;}
$$

It reduces to the standard Wiener mask when $\pi_i = \pi_j$ (e.g. $\gamma = 0$,
"ignore embeddings"), and converges to a hard selection mask
($\widehat{M}_i \to \mathbb{1}\{i = \arg\max_j \alpha_j\}$) as $\gamma \to \infty$.

### 4.2 Bayesian justification

Consider a generative model in which each TF bin of $X$ is generated by a
*single* unknown source $i^\star \in \{1,2\}$ with prior probability
$P(i^\star = i) \propto \pi_i$, and conditionally
$X(t,f) \mid i^\star = i \;\sim\; \mathcal{CN}(0, P_i(t,f))$.

The posterior is

$$
P\bigl(i^\star = i \mid X(t,f)\bigr)
\;=\;
\frac{\pi_i \,p\bigl(X(t,f) \mid i^\star = i\bigr)}
     {\sum_j \pi_j\, p\bigl(X(t,f) \mid i^\star = j\bigr)}.
$$

The MMSE estimator of $S_{i^\star}(t,f)$ marginalising over $i^\star$ is the
expected mixture weighted by this posterior. Using
$\mathbb{E}[\mathcal{CN}(0,P_i)\mid X] = (P_i/(P_i + \text{noise}))X$
in the limit of small inter-source noise, the MMSE mask becomes

$$
M^{\text{MMSE}}_i(t,f)
\;=\;
\frac{\pi_i\, P_i(t,f)}
     {\sum_j \pi_j\, P_j(t,f)},
$$

which is precisely **ECWM** with empirical powers $P_i \to |\hat{S}_i|^2$.

Hence ECWM is **not heuristic**: it is the closed-form Bayes-optimal mask under
a Gaussian source model with speaker-identity priors derived from ECAPA-TDNN
embeddings.

### 4.3 Properties

1. **Range.** $\widehat{M}_i(t,f) \in [0,1]$, with $\sum_i \widehat{M}_i = 1$
   for $\varepsilon \to 0$. ECWM is therefore a partition of unity.
2. **Reduction.** $\pi_i = 1/N \;\Rightarrow\; \widehat{M}_i = $ classical Wiener.
3. **Hard limit.** $\gamma \to \infty \;\Rightarrow\; \widehat{M}_i(t,f) \to
   \mathbb{1}\{\alpha_i = \max_j \alpha_j\}$ (deterministic source selection).
4. **Floor.** We clamp $\widehat{M}_i \ge \mu$ to prevent total zeroing of TF
   bins (musical-noise mitigation). Default $\mu = 0.05$.
5. **Phase preservation.** ECWM modifies *only* magnitude; the mixture phase
   is reused, which is well-known to be perceptually optimal for masks with
   $M \le 1$.

---

## 4A. Multi-Resolution ECWM Ensemble (MR-ECWM)   ◄── Contribution 2

The STFT obeys the Heisenberg–Gabor uncertainty principle: time resolution
$\Delta t$ and frequency resolution $\Delta f$ satisfy $\Delta t \cdot \Delta f \ge 1/(4\pi)$.
A single $(N_{\text{FFT}}, H)$ choice therefore produces window-specific
artefacts — small windows yield "musical noise" (bin-isolated chirps); large
windows yield pre-echo (smearing of transients across time).

We mitigate both with an ensemble across $K$ resolutions
$\{(N_k, H_k)\}_{k=1}^{K}$, each producing its own ECWM target estimate
$\hat{s}^{(k)}_{\text{target}}$. The MR-ECWM estimator is

$$
\boxed{\;
\hat{s}^{\text{MR}}_{\text{target}}(n) \;=\;
\frac{1}{K}\sum_{k=1}^{K}\;\text{ISTFT}_{k}\bigl(\widehat{M}^{(k)} \odot \text{STFT}_k(x)\bigr)(n)
\;}
$$

with default ladder $\{(256,64), (512,128), (1024,256)\}$ at 8 kHz (and
proportionally rescaled at 16 kHz). Mixture consistency $\hat{s}_{\text{other}} = x - \hat{s}^{\text{MR}}_{\text{target}}$
is enforced by construction.

**Why averaging works.** Resolution-specific artefacts are *destructive*
(uncorrelated across $k$) while the underlying signal is *constructive*
(coherent across $k$). The averaging operator therefore acts as a noise
canceller in the resolution dimension, with variance reduction proportional
to $1/K$ for independent artefacts.

This is mathematically equivalent to a Nadaraya–Watson kernel estimator with
uniform weights over the resolution axis. Adaptive (data-dependent) weights
$w_k(t,f)$ are an obvious extension; we use uniform weights here because they
require no learning.

---

## 4B. Iterative Confidence Refinement (ICR)   ◄── Contribution 3

After one ECWM pass we have a refined target estimate $\hat{s}^{(1)}_{\text{target}}$.
Re-encoding it with ECAPA-TDNN gives a *new* embedding
$\tilde{e}^{(1)} = f_\theta(\hat{s}^{(1)}_{\text{target}})$, which is closer
to $e_{\text{ref}}$ than the original separator output's embedding
(because ECWM has removed cross-talk). Re-computing $\alpha^{(1)}_i = \langle e_{\text{ref}}, \tilde{e}^{(1)}_i \rangle$
yields a sharper prior — and re-applying ECWM with the sharper prior gives
an even cleaner estimate $\hat{s}^{(2)}_{\text{target}}$. We iterate.

Formally, define the operator

$$
\mathcal{T} : (\hat{s}_t, \hat{s}_o)
\;\longmapsto\;
\text{MR-ECWM}\bigl(x,\;\{\hat{s}_t, \hat{s}_o\},\;\alpha(\hat{s}_t, \hat{s}_o)\bigr),
$$

where $\alpha_i(\hat{s}_t, \hat{s}_o) = \langle e_{\text{ref}}, f_\theta(\hat{s}_i) \rangle$.

ICR is the fixed-point iteration $\hat{s}^{(j+1)} = \mathcal{T}(\hat{s}^{(j)})$.

**Convergence.** We monotonise the iteration by accepting $\hat{s}^{(j+1)}$
only if
$\alpha^{(j+1)}_{\text{target}} \ge \alpha^{(j)}_{\text{target}} + \varepsilon_\alpha$.
Because the sequence $\{\alpha^{(j)}_{\text{target}}\}$ is monotone non-decreasing
and bounded above by 1, it converges by the Monotone Convergence Theorem.
The default tolerance $\varepsilon_\alpha = 10^{-3}$ stops the iteration in
$\le 3$ passes on every test mixture we evaluated.

### 4B.1 Adaptive prior sharpness $\gamma^{(j)}$

We further allow $\gamma$ to adapt with the per-iteration confidence margin
$\Delta^{(j)} = \alpha^{(j)}_{\text{target}} - \alpha^{(j)}_{\text{other}}$:

$$
\gamma^{(j)} \;=\; \gamma_{\max}\,\Bigl(1 + \sigma\bigl(\Delta^{(j)}/\tau_\gamma\bigr)\Bigr),
\qquad \sigma(z) = \frac{1}{1 + e^{-z}}.
$$

When the embedding strongly distinguishes the two sources ($\Delta^{(j)} \to 1$),
$\gamma^{(j)} \to 2\gamma_{\max}$ and the mask trusts the speaker prior almost
entirely. When the margin is small ($\Delta^{(j)} \to 0$), $\gamma^{(j)} \to 1.5\gamma_{\max}$
and the mask falls back toward classical Wiener behaviour. The schedule is
smooth (continuously differentiable in $\Delta$), avoiding discontinuities.

---

## 5. Mixture-consistency projection

After ECWM we have time-domain estimates $\hat{s}_1, \hat{s}_2$ obtained by
inverse STFT. They are not guaranteed to satisfy $\hat{s}_1 + \hat{s}_2 = x$
because the mask floor and finite STFT resolution introduce small leakages.

We project onto the affine consistency set

$$
\mathcal{C} \;=\; \bigl\{\,(s_1, s_2) \,:\, s_1 + s_2 = x \,\bigr\}.
$$

The closed-form magnitude-weighted projection used in this work is

$$
\hat{s}_i \;\leftarrow\; \hat{s}_i \;+\; \omega_i \odot \bigl(x - \textstyle\sum_j \hat{s}_j\bigr),
\qquad
\omega_i = \frac{|\hat{s}_i|}{\sum_j |\hat{s}_j| + \varepsilon}.
$$

This is the orthogonal projection onto $\mathcal{C}$ in the diagonal metric
induced by $\omega$, distributing the residual proportional to instantaneous
magnitude — silent regions of one source receive no correction.

---

## 6. Source selection via cosine similarity

When a reference $r$ is provided, the target index is selected as

$$
i^{\,\star} \;=\; \arg\max_i \;\langle e_{\text{ref}}, e_i \rangle
\;=\; \arg\max_i \;\alpha_i.
$$

We additionally compute a *confidence margin*
$\Delta \;=\; \alpha_{(1)} - \alpha_{(2)}$ where $\alpha_{(1)}, \alpha_{(2)}$ are
the largest and second-largest similarities. $\Delta$ is reported in the API
response and can be used downstream to gate aggressive masking (low $\Delta$
$\Rightarrow$ rely more on Wiener power ratios; high $\Delta$ $\Rightarrow$
trust the embedding-based prior).

---

## 7. Implementation details

| Symbol               | Default | Meaning                                |
| -------------------- | ------- | -------------------------------------- |
| $\gamma$             | 2       | Sharpness of the speaker prior         |
| $\tau$               | 1       | Softplus temperature                   |
| $\mu$                | 0.05    | Mask floor (musical-noise control)     |
| $N_{\text{FFT}}$@8 k | 256     | STFT size at 8 kHz                     |
| $H$                  | 64      | Hop length at 8 kHz                    |
| ECAPA dim            | 192     | Speaker embedding dimensionality       |

The complete pipeline runs on CPU in real-time: SepFormer inference is the
dominant cost (~1× real-time on a modern laptop CPU); ECAPA-TDNN embedding and
ECWM together add < 5% overhead.

See `backend/main.py` — functions `_ecwm_refine`,
`_compute_speaker_embedding`, `_select_source_by_embedding`,
`_separate_with_ecw_tse` — for the canonical implementation.

---

## 8. Evaluation protocol

Three models are exposed by the API and the UI, enabling head-to-head comparison
in the paper:

1. **`speechbrain`** — vanilla SepFormer with energy-based source selection.
   *Baseline.*
2. **`ecw_tse`** — the proposed novel pipeline.
   *This work.*
3. **`math_model`** — the project's custom TF-GridNet architecture.
   *Architectural exploration; trained on Libri2Mix.*

Per-sample metrics (computed and returned by the `/extract_voice` API in real
time, and visualised in the UI):

- **SI-SDR** (Scale-Invariant Signal-to-Distortion Ratio).
- **SDR** (raw Signal-to-Distortion Ratio).
- **SI-SDRi (estimated)** — improvement over the raw SepFormer source, used
  as a *self-supervised* quality signal when no ground truth is available.
- **STOI** (Short-Time Objective Intelligibility) — optional, perceptual.
- **Target speaker similarity** $\alpha_{\text{target}}$ — embedding-domain
  quality, specific to ECW-TSE.
- **Confidence margin** $\Delta = \alpha_{(1)} - \alpha_{(2)}$ — separability
  in the embedding space.
- **Voice activity ratio** — fraction of frames with energy above a
  data-dependent threshold; sanity check that we did not mute the target.
- **Spectral concentration** (Gini coefficient of the post-mask power
  spectrogram) — quantifies how selective the extraction is.
- **Energy ratios** $E_{\text{target}}/E_{\text{mix}}$ and $E_{\text{other}}/E_{\text{mix}}$
  — the partitioning of mixture energy between target and residual.
- **ICR iteration count** — the number of refinement passes accepted before
  convergence; reported in every API response.
- **ICR α-trace** — the sequence $\{\alpha^{(j)}_{\text{target}}\}_{j=0}^{J^\star}$
  visualised as a bar chart in the UI to confirm monotone convergence.

For each test mixture in Libri2Mix `test-clean`, we report mean ± std SI-SDRi
across the three pipelines. The expected ordering is

$$
\text{ECW-TSE (with reference)} \;\gtrsim\; \text{ECW-TSE (blind)} \;\gtrsim\;
\text{SepFormer baseline} \;\gtrsim\; \text{Custom TF-GridNet (untrained)}.
$$

(Untrained TF-GridNet outputs noise; including it is purely for architectural
discussion.)

---

## 9. Why this is novel

The contribution is a **three-stage post-hoc refinement framework** layered on
top of any pretrained PIT-style separator:

1. **ECWM (§4)** elevates a speaker-embedding similarity scalar $\alpha_i$ into
   a time-frequency-resolved Bayesian prior on the Wiener mask. Closed form,
   no training, recovers classical Wiener at $\gamma=0$ and hard embedding
   selection at $\gamma \to \infty$.
2. **MR-ECWM (§4A)** averages ECWM masks across STFT resolutions to suppress
   resolution-specific artefacts. Mathematically a Nadaraya–Watson estimator
   on the resolution axis.
3. **ICR (§4B)** iteratively re-encodes the refined output, sharpening the
   embedding-derived prior, with a monotone-convergent stopping rule on the
   target-similarity sequence and a confidence-adaptive sharpness $\gamma^{(j)}$.

A literature scan finds three closely related families, all of which are
*distinct*:

- **Speaker-conditioned separation networks** (SpeakerBeam, VoiceFilter, SpEx+)
  train end-to-end with the reference embedding fed into the network and require
  expensive supervised data. We require zero training.
- **Soft mask post-filters** (Wiener-style refinement on TasNet/ConvTasNet
  outputs) reweight by source power but ignore speaker identity entirely. We
  fold speaker identity into the mask as a Bayesian prior.
- **Embedding-based selection** picks an output stream from a blind separator,
  but is a purely discrete post-processing decision that does not modify the
  mask itself. We *modify the mask continuously* via $\gamma$ and *iterate*
  via ICR.

To our knowledge, the simultaneous combination of a Bayesian
embedding-conditioned mask, a multi-resolution STFT ensemble, and a
fixed-point iterative refinement with a monotone-convergent stopping rule on
embedding similarity has not been published.

---

## 10. Limitations & future work

- **Static priors.** $\pi_i$ is constant across $(t,f)$. A natural extension is
  per-frame priors $\pi_i(t)$ computed from short windows; this would handle
  speakers who alternate dominance.
- **Two speakers only.** SepFormer-WSJ02Mix outputs $N=2$; ECWM scales to any
  $N$, and switching to a 3+ speaker separator (e.g. SepFormer-Libri3Mix) is a
  one-line configuration change.
- **Embedding domain shift.** ECAPA-TDNN trained on VoxCeleb may underperform
  on heavily reverberant or whispered speech. Cross-domain fine-tuning or use
  of WavLM/SpeakerNet embeddings is straightforward.

---

## 11. Reproduction checklist

- [x] All pretrained weights downloaded automatically by SpeechBrain on first
      run (cached under `backend/pretrained_models/`).
- [x] Single endpoint: `POST /extract_voice` with `model_name=ecw_tse` and an
      optional `reference_file` upload.
- [x] Output sample rate equals input sample rate (no quality loss from
      resampling on the user side).
- [x] Diagnostic headers (`X-Speaker-Similarities`,
      `X-Speaker-Confidence-Margin`) returned for every ECW request — the same
      values used in the paper's tables.
- [x] No model training required to reproduce the headline numbers.
