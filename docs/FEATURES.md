# ForenSightX — Feature Extraction Specification & Heuristic Formulas

**Document:** `docs/FEATURES.md`  
**Purpose:** Formal mathematical definitions, decision thresholds, and assumption audit for all non-deep-learning and heuristic forensic indicators.  
**System Reference:** Authoritative Spec `ForenSightX_Project_Architecture.docx` Sections 4, 5.3, 6, and 8.

---

## 1. Feature Vector Overview (7-Dimensional Authoritative Specification)

Per Section 5.3 of `ForenSightX_Project_Architecture.docx`, the Multimodal Feature Vector $\mathbf{x} \in [0, 1]^7$ consumed by the ANN is defined as:

$$\mathbf{x} = \begin{bmatrix} s_{\text{cnn}} \\ s_{\text{bilstm}} \\ s_{\text{url}} \\ s_{\text{kw}} \\ s_{\text{att}} \\ s_{\text{time}} \\ s_{\text{email}} \end{bmatrix} = \begin{bmatrix} \text{Visual Evidence Suspicion Probability (CNN)} \\ \text{Sequence Context Suspicion Probability (BiLSTM)} \\ \text{Embedded URL Suspicion Score} \\ \text{Forensic Keyword Suspicion Score} \\ \text{High-Risk Attachment Indicator} \\ \text{Timeline Sequence Anomaly Score} \\ \text{Email Header / Origin Anomaly Score} \end{bmatrix}$$

---

## 2. Mathematical Formulations & Thresholds

### 2.1. Feature 3: `url_score` ($s_{\text{url}}$)

*Function:* `extract_url_score(text_or_urls: Union[str, List[str]]) -> float`

#### Formulation:
Given an evidence string or extracted URL set $\mathcal{U} = \{u_1, u_2, \dots, u_m\}$:
If $|\mathcal{U}| = 0$, then $s_{\text{url}} = 0.0$.
For each URL $u \in \mathcal{U}$, an additive penalty $p(u)$ is computed:
$$p(u) = 0.20 \quad (\text{baseline per link})$$
$$+ \; 0.80 \times \mathbb{I}(\text{IPv4 literal host in } u)$$
$$+ \; 0.60 \times \mathbb{I}(\text{TLD} \in \{.xyz, .top, .tk, .ru, .buzz, .work\})$$
$$+ \; 0.70 \times \mathbb{I}(\text{'@' embedded in authority string})$$
$$+ \; 0.50 \times \mathbb{I}(\text{path contains } \{\text{'login'}, \text{'verify'}, \text{'secure'}, \text{'update'}, \text{'banking'}\})$$

The final feature value is bounded:
$$s_{\text{url}} = \min\left(1.0, \; \frac{1}{\max(1, |\mathcal{U}|)} \sum_{u \in \mathcal{U}} p(u)\right)$$

- `[ASSUMPTION-FEAT-URL-01]` Weight of 0.80 for IPv4 literals assumes benign corporate services utilize DNS hostnames rather than raw decimal IP addresses.
- `[ASSUMPTION-FEAT-URL-02]` Specified TLD list reflects prevalent phishing abuse registars based on academic threat feeds.

---

### 2.2. Feature 4: `keyword_score` ($s_{\text{kw}}$)

*Function:* `extract_keyword_score(text: str) -> float`

#### Formulation:
Given text sequence $T$, normalized to lowercase $T_{\text{lower}}$. A predefined forensic lexicon $\mathcal{K} = \{(k_i, w_i)\}$ contains weighted indicator terms:

| Keyword Category | Target Tokens | Weight ($w_i$) |
|------------------|---------------|----------------|
| Authentication & Credentials | `otp`, `credentials`, `password`, `verify` | 1.00, 0.95, 0.90, 0.60 |
| Financial Urgency | `wire`, `transfer`, `giftcard`, `bitcoin`, `crypto`, `invoice`, `payment` | 0.90, 0.70, 0.95, 0.85, 0.80, 0.60, 0.60 |
| Coercion & Urgency | `urgent`, `urgency`, `immediately`, `suspended`, `unauthorized`, `breach` | 0.80, 0.80, 0.70, 0.85, 0.85, 0.80 |
| Confidentiality / Authority | `confidential`, `compliance`, `bank` | 0.65, 0.50, 0.60 |

The score is accumulated over regex word boundaries $\text{Matches}(T) = \{w_i \mid \text{regex}(\backslash b k_i \backslash b) \in T_{\text{lower}}\}$:
$$s_{\text{kw}} = \begin{cases} 0.0 & \text{if } |\text{Matches}(T)| = 0 \\ \min\left(1.0, \; \frac{\sum_{w \in \text{Matches}(T)} w}{3.0}\right) & \text{otherwise} \end{cases}$$

- `[ASSUMPTION-FEAT-KW-01]` Saturation divisor of $3.0$ assumes the presence of three moderate-to-high risk terms establishes definitive linguistic suspicion.
- `[ASSUMPTION-FEAT-KW-02]` Exact word boundary matching ($\backslash b$) is assumed to prevent false triggers (e.g., "hotplate" triggering "otp").

---

### 2.3. Feature 5: `attachment_score` ($s_{\text{att}}$)

*Function:* `extract_attachment_score(attachments: List[str]) -> float`

#### Formulation:
Given attachment filename list $\mathcal{A} = \{a_1, a_2, \dots, a_n\}$:
If $|\mathcal{A}| = 0$, then $s_{\text{att}} = 0.0$.
For each filename $a \in \mathcal{A}$:
- If $a$ contains multiple extension delimiters (e.g. `document.pdf.exe`), base risk is assigned $0.95$.
- Dangerous extension penalties:
  $$\text{weight}(\text{ext}) = \begin{cases} 
  1.00 & \text{if } \text{ext} \in \{.exe, .scr\} \\
  0.95 & \text{if } \text{ext} \in \{.vbs\} \\
  0.90 & \text{if } \text{ext} \in \{.bat, .cmd, .ps1\} \\
  0.85 & \text{if } \text{ext} \in \{.iso\} \\
  0.80 & \text{if } \text{ext} \in \{.js, .jar\} \\
  0.40 & \text{if } \text{ext} \in \{.zip\} \\
  0.00 & \text{otherwise}
  \end{cases}$$

The final feature value is the maximum risk observed across all attachments:
$$s_{\text{att}} = \max_{a \in \mathcal{A}} \text{risk}(a)$$

- `[ASSUMPTION-FEAT-ATT-01]` Double extensions (e.g. `*.pdf.exe`) are assumed to represent intentional social engineering camouflage.
- `[ASSUMPTION-FEAT-ATT-02]` Archive files (`.zip`) carry a moderate baseline of $0.40$ due to dual-use utility (routine compression vs payload evasion).

---

### 2.4. Feature 6: `timeline_anomaly` ($s_{\text{time}}$)

*Function:* `extract_timeline_anomaly(timestamps: List[datetime]) -> float`

#### Formulation:
Given a sequence of parsed event timestamps $\mathcal{T} = [t_1, t_2, \dots, t_N]$:
If $N < 2$, then $s_{\text{time}} = 0.0$.
Let $\mathcal{T}_{\text{sorted}} = \text{sort}(\mathcal{T})$ with inter-event deltas $\Delta_i = (t_{i+1} - t_i)$ in seconds.

$$\text{rapid\_count} = \sum_{i=1}^{N-1} \mathbb{I}(0 \le \Delta_i < 3.0)$$
$$\text{clock\_skew} = \sum_{i=1}^{N-1} \mathbb{I}(\Delta_i^{\text{raw}} < 0)$$

$$s_{\text{time}} = \min\left(1.0, \; \left(0.50 + 0.10 \times \text{rapid\_count}\right) \times \mathbb{I}(\text{rapid\_count} \ge 2) + 0.80 \times \mathbb{I}(\text{clock\_skew} > 0)\right)$$

- `[ASSUMPTION-FEAT-TIME-01]` Consecutive distinct actions occurring $< 3.0$ seconds apart are assumed to reflect automated script-driven exploitation or bot activity.
- `[ASSUMPTION-FEAT-TIME-02]` Negative time intervals in raw event logs assume client-side clock tampering, log forgery, or out-of-order relaying.

---

### 2.5. Feature 7: `email_anomaly` ($s_{\text{email}}$)

*Function:* `extract_email_anomaly(email_dict: Dict[str, Any]) -> float`

#### Formulation:
Given an email metadata structure with fields $\text{sender}$, $\text{reply\_to}$, $\text{display\_name}$, and security headers $\text{headers}$:

$$s_{\text{email}} = \min(1.0, \; a_{\text{domain}} + a_{\text{spf}} + a_{\text{dkim}} + a_{\text{spoof}})$$

Where:
- $a_{\text{domain}} = 0.60$ if $\text{domain}(\text{sender}) \neq \text{domain}(\text{reply\_to})$ (and both non-empty).
- $a_{\text{spf}} = 0.50$ if $\text{headers}[\text{'spf'}] == \text{'fail'}$.
- $a_{\text{dkim}} = 0.50$ if $\text{headers}[\text{'dkim'}] == \text{'fail'}$.
- $a_{\text{spoof}} = 0.70$ if $\text{display\_name}$ contains executive titles (`ceo`, `chief executive`, `admin`, `it support`, `security`) while $\text{sender}$ originates from public webmail (`gmail`, `yahoo`, `hotmail`, `proton`).

- `[ASSUMPTION-FEAT-EMAIL-01]` Sender and Reply-To domain divergence is assumed to indicate credential harvesting bounce redirection.
- `[ASSUMPTION-FEAT-EMAIL-02]` High-authority display names paired with free webmail providers are assumed to indicate Business Email Compromise (BEC) impersonation.

---

## 3. Label Provenance & Circularity Notice

```
WARNING & METHODOLOGICAL DISCLOSURE:
In synthetic SMOKE mode, ground-truth labels y in {0, 1} are generated alongside synthetic records.
The heuristic extractors defined above evaluate indicators (keywords, URLs, extensions) that are
naturally correlated with the synthetic attack templates. To prevent circular reasoning from
distorting academic conclusions, strict circularity controls (shuffled-label controls, logistic
regression heuristic baselines, and ablation studies) are mandated and evaluated in Part 3.
```

