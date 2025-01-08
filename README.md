# Cross-Impact-of-Order-Flow-Imbalance-in-Equity-Markets

This repository contains the implementation of a cross-impact analysis of Order Flow Imbalance (OFI) in equity markets. The goal is to compute OFI metrics, analyze cross-asset impacts on short-term price changes, and evaluate the predictive power of lagged OFI. This project follows the methodologies presented in the research paper "Cross-Impact of Order Flow Imbalance in Equity Markets."

This project involves:

1. Computing multi-level OFI metrics for several stocks and integrating them using Principal Component Analysis (PCA).
2. Analyzing contemporaneous and predictive cross-impact of OFI on price changes.
3. Using regression models to compare self-impact (within the same stock) and cross-impact (between stocks).
4. Visualizing key relationships and trends.


The analysis showed that integrated OFI metrics have moderate explanatory power for contemporaneous price changes but offer limited predictive power for future returns, as evidenced by low R-squared values. Self-impact dominates cross-impact in influencing price dynamics.

# Run Analysis

1. Install packages using in requirements.txt
2. Run notebook in notebook/ folder
3. Data can be obtained from databento. mbp-10 schema nasdaq data was used for this analysis