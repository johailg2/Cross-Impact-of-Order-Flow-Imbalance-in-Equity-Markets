# Cross-Impact-of-Order-Flow-Imbalance-in-Equity-Markets

This repository contains the implementation of a cross-impact analysis of Order Flow Imbalance (OFI) in equity markets. The goal is to compute OFI metrics, analyze cross-asset impacts on short-term price changes. This project follows the methodologies presented in the research paper "Cross-Impact of Order Flow Imbalance in Equity Markets." The OFI metric was used to evaluate cross-asset impacts on short term price changes. OFIs across different levels were condensed into a singular metric using Principal Component Analysis. Trends observed only over a day's worth of trading data showcased moderate explanatory power over contemporaneous price changes and log returns. Models integrating cross impact had significantly higher explanatory power than those considering only self-impact. Data was acquired via a batch download using the Databento API. More information can be found here: https://databento.com/docs/api-reference-historical/batch?historical=python&live=python&reference=python

# Run Analysis

1. Install packages using in requirements.txt
2. Run notebook in notebook/ folder
3. Data can be obtained from databento. mbp-10 schema nasdaq data was used for this analysis