# Presentation & Viva Outline (5 Marks)

## Suggested Slide Structure (10–12 slides)

1. **Title Slide**  
   Bitcoin Price Prediction using Machine Learning  
   Your Name | Roll No | Guide Name | College

2. **Problem Statement**  
   Why Bitcoin prediction is hard + objectives

3. **Dataset**  
   Source, period, columns, size, why this dataset

4. **Preprocessing & Feature Engineering**  
   List of technical indicators + time-series split

5. **Exploratory Data Analysis**  
   Key charts (price trend, volume, returns distribution, correlation)

6. **Machine Learning Algorithms**  
   Brief description of Linear, RF, XGBoost, LSTM

7. **Model Evaluation**  
   Table of RMSE / MAE / MAPE / R² / Dir.Acc  
   Best model highlight

8. **Model Improvement**  
   Hyperparameter tuning, early stopping, feature selection

9. **Streamlit Application Demo**  
   Screenshots of the dashboard + live prediction

10. **Deployment**  
    GitHub + Streamlit Cloud link

11. **Conclusion & Future Work**  
    Limitations + possible extensions (sentiment, on-chain, ensemble)

12. **References & Q&A**

---

## Expected Viva Questions & Short Answers

**Q1. Why time-series split instead of random split?**  
A: Random split would leak future information into training, causing overly optimistic metrics. Chronological split respects temporal order.

**Q2. Why is MAPE useful here?**  
A: It is scale-independent and easy to interpret as percentage error, which is intuitive for price prediction.

**Q3. What is the lookback window in LSTM?**  
A: Number of past days the model sees to predict the next day (we used 60).

**Q4. How did you avoid data leakage?**  
A: Scaler fitted only on train set; features that use future information were carefully excluded; target is next-day close.

**Q5. Why include both classical ML and LSTM?**  
A: Classical models are faster, interpretable and often strong on tabular features; LSTM can capture sequential patterns.

**Q6. Is the model production-ready for trading?**  
A: No. It is educational. Real trading needs transaction costs, slippage, risk management, regime detection, etc.

**Q7. How would you improve further?**  
A: Add sentiment (Twitter/News), on-chain metrics, multi-timeframe features, ensemble stacking, online learning.

**Q8. What is Directional Accuracy?**  
A: Percentage of times the model correctly predicts whether price will go up or down the next day.
