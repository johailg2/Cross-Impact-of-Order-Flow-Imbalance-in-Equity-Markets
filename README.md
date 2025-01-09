# Cross-Impact-of-Order-Flow-Imbalance-in-Equity-Markets

This document summarizes the methodology, findings, and insights from the analysis on Order Flow Imbalance (OFI). The task replicates several results from the paper \textit{Cross-Impact of Order Flow Imbalance in Equity Markets} by Rama Cont, Mihai Cucuringu, and Chao Zhang, published in \textit{Quantitative Finance} (2023). The OFI metric was calculated on high-frequency equity market data, 5 highly liquid NASDAQ stocks in particular, across multiple levels of the Level Order Book. The OFI metric was used to evaluate cross-asset impacts on short term price changes. OFIs across different levels were condensed into a singular metric using Principal Component Analysis. Trends observed only over a day's worth of trading data showcased moderate explanatory power over contemporaneous price changes and log returns. Models integrating cross impact had significantly higher explanatory power than those considering only self-impact. Data was acquired via a batch download using the Databento API. More information can be found here: https://databento.com/docs/api-reference-historical/batch?historical=python&live=python&reference=python

# Run Analysis

1. Install packages using in requirements.txt
2. Run notebook in notebook/ folder
3. Data can be obtained from databento. mbp-10 schema nasdaq data was used for this analysis