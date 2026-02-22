# ⚖️ Critical Discussion

## Model Limitations

While the Player Selection System utilizes robust Rolling Form Features mapped via a Random Forest, it possesses several inherent limitations:

- **Exclusion of Extraneous Context:** Machine Learning models evaluated purely on mathematical ball-by-ball outputs fail to account for player psychology, pressure management, injury history, and fitness levels—factors human selectors consider implicitly.
- **Pitch and Environmental Factors:** The current feature set does not adequately isolate the match venue's pitch deterioration or weather conditions. A bowler having an "Average" economy on a flat batting paradise pitch might actually be performing "Excellently" relative to expected par scores.

## Data Quality Issues

The model is entirely dependent on the integrity of Cricsheet's data entry.

- **Missing Historical Context:** T20 Internationals have a relatively low match frequency compared to franchise leagues. To solve the sparsity issue, we integrated Lanka Premier League (LPL) data. However, standardising LPL performance against T20I performance risks skewed scaling, as facing domestic bowlers is statistically less demanding than facing international ones.
- **Small Squad Sizes:** Filtering for active Sri Lankan players creates a naturally constrained dataset. Machine Learning models generally require tens of thousands of rows for deep patterns; we are working with high-dimensional but low-volume historical aggregates per player.

## Risks of Bias or Unfairness

When relying heavily on past aggregates, the model risks generating **Confirmation Bias loops**.

- If a struggling veteran is given consistently easier batting positions (e.g., lower down the order against spinners rather than opening against fresh pace), their "Form Runs" may artificially inflate compared to a rookie forced to bat in high-pressure slots.
- The system heavily biases towards **strike rate and boundaries**. Technically sound, defensive anchor batsmen might be perpetually penalized and categorized as `Poor` despite being tactically necessary after early top-order collapses.

## Real-World Impact and Ethical Considerations

Should an AI dictate team selection?

- **Positive Impact:** It removes the nepotism, emotional bias, and instinct-driven errors that frequently plague sports management. It identifies who is mathematically contributing to winning right now.
- **Ethical Dilemma (Job Security):** Professional cricket is a livelihood. Relegating a player based purely on automated thresholds strips them of human grace periods during temporary slumps.
- **The Verdict:** The system should not act as an autonomous dictator but as a **Decision Support System (DSS)**. It empowers selectors with transparent, explainable insights (via SHAP) to challenge their own intuitions and justify dropped players objectively.
