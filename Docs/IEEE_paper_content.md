# AgriVision YieldMax: Ensemble Machine Learning for Precision Crop Yield Prediction Using Stacking Architecture

**Arunmozhi Adithya**
Department of Computer Science and Engineering
Sathyabama Institute of Science and Technology, Chennai, India

**Jenivaa**
Department of Computer Science and Engineering
Sathyabama Institute of Science and Technology, Chennai, India

**Tamizharasan**
Department of Computer Science and Engineering
Sathyabama Institute of Science and Technology, Chennai, India

**Pradeepraja**
Department of Computer Science and Engineering
Sathyabama Institute of Science and Technology, Chennai, India

**Dilshan**
Department of Computer Science and Engineering
Sathyabama Institute of Science and Technology, Chennai, India

**Dr. T. Anitha (Asst. Professor)**
Department of Computer Science and Engineering
Sathyabama Institute of Science and Technology, Chennai, India

---

## Abstract

Agriculture remains the backbone of developing economies, yet crop yield forecasting continues to be a significant challenge due to the complex interplay of environmental, geographical, and agronomic factors. Traditional prediction models relying on a single algorithm fail to adequately capture the multidimensional nature of agricultural data. This paper presents **AgriVision YieldMax**, an ensemble learning system that integrates three heterogeneous base learners — XGBoost, LightGBM, and a Deep Neural Network (DNN) — combined through a Ridge Regression meta-learner using a stacking architecture. The system also incorporates a Random Forest-based Smart Crop Advisor for personalized crop recommendation based on soil nutrient profiles and environmental conditions. Trained on large-scale historical Indian agricultural datasets encompassing multiple states, crop varieties, and seasons, the YieldMax ensemble achieves an R² score of **0.92**, outperforming individual models by 3–5%. The system is deployed as a full-stack web application with a Glassmorphism-styled frontend, real-time AI-powered agronomic advice via Google Gemini 2.0 Flash, regional climate auto-estimation, and confidence-interval predictions. Experimental results validate that ensemble stacking significantly improves both prediction accuracy and reliability compared to any single-model approach.

**Keywords** — Crop yield prediction; ensemble learning; stacking; XGBoost; LightGBM; deep neural network; Random Forest; smart crop recommendation; precision agriculture; machine learning

---

## I. Introduction

Food security is one of the foremost challenges of the 21st century. With a global population projected to exceed 9.7 billion by 2050, agricultural productivity must increase substantially to meet demand [1]. In India, agriculture accounts for approximately 18% of GDP and employs nearly 43% of the workforce, yet unpredictable crop yields due to varying climate conditions, soil diversity, and pest pressures continue to threaten livelihoods and national food stability [2].

Accurate crop yield prediction enables farmers, agricultural officers, and policymakers to make informed decisions regarding crop selection, resource allocation, insurance planning, and supply chain management. Conventional approaches rely on domain expertise and statistical regression, which fail to generalize across diverse geographic and climatic domains [3].

The rapid advancement of machine learning (ML) has opened new possibilities for data-driven agricultural forecasting. Gradient boosting methods such as XGBoost and LightGBM have demonstrated state-of-the-art results in structured tabular data tasks [4], while Deep Neural Networks (DNNs) offer the ability to model complex nonlinear interactions between features [5]. However, each model class has distinct strengths and weaknesses — motivating ensemble approaches that leverage complementary expertise.

This paper proposes **AgriVision YieldMax**, a full-stack precision agriculture system built around a stacking ensemble ML architecture. The core contributions of this work are:

1. A stacking ensemble combining XGBoost, LightGBM, and a DNN, yielded by a Ridge Regression meta-learner trained on cross-validated base predictions.
2. A confidence scoring mechanism derived from the coefficient of variation across base model predictions.
3. A regional climate auto-estimation module that infers missing environmental inputs from location and season.
4. A Smart Crop Advisor using a Random Forest Classifier trained on 13 engineered features for top-3 crop recommendation.
5. Integration with Google Gemini 2.0 Flash for generating real-time, context-aware agronomic insights.
6. A production-ready Flask web application deployed via Docker on Hugging Face Spaces.

The rest of this paper is structured as follows: Section II reviews related work in crop yield prediction using machine learning. Section III describes the proposed system methodology and architecture. Section IV presents experimental results and performance metrics. Section V concludes the paper and outlines future directions.

---

## II. Related Work

Crop yield prediction has attracted considerable research attention in recent years. Early work focused on regression-based statistical models. Lobell and Burke [6] applied linear regression to historical climate and yield data across Sub-Saharan Africa and found strong correlations with temperature and precipitation. However, these models are limited by the assumption of linearity in complex real-world agricultural data.

Decision tree-based methods emerged as effective alternatives. Pantazi et al. [7] applied supervised machine learning including Support Vector Machines (SVM) and Artificial Neural Networks (ANN) for wheat yield prediction, achieving prediction accuracies above 80%. Liakos et al. [8] conducted a systematic review of ML in agriculture, identifying neural networks and support vector regression as the most widely applied methods.

Gradient boosting methods have shown particularly strong results in agricultural prediction tasks. XGBoost, introduced by Chen and Guestrin [9], has been applied to crop yield modeling in multiple studies due to its robustness to overfitting and its ability to handle categorical features. Similarly, LightGBM, proposed by Ke et al. [10], offers faster training and improved performance on large-scale agricultural datasets with numerous environmental features.

Deep learning approaches have also been explored. Khaki and Wang [11] proposed a DNN architecture for soybean yield prediction, achieving an RMSE of 0.37 tonnes/hectare. Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks have been applied to time-series yield forecasting, exploiting temporal weather patterns [12].

Ensemble methods combine multiple learners to improve generalization. Stacking, popularized by Wolpert [13], trains a meta-learner on base model predictions to find optimal combinations. Jiang et al. [14] demonstrated that stacking ensembles outperform individual models in crop yield prediction tasks across diverse climates. Random Forest has also been widely used for crop recommendation tasks; Doshi et al. [15] applied Random Forest to soil-based crop recommendation, achieving over 99% classification accuracy.

Despite these advances, existing systems often address yield prediction and crop recommendation separately, fail to provide confidence intervals, depend on complete environmental input data, and rarely integrate AI-powered agronomic explanation. This work addresses all four gaps in a unified system.

---

## III. Proposed Methodology

The proposed AgriVision YieldMax system is designed and developed by integrating ensemble ML with a stacking architecture. The system incorporates three complementary base learners that each specialize in different aspects of agricultural data, unified by a Ridge Regression meta-learner that learns the optimal combination weights. The overall proposed architecture is shown in Fig. 1.

### A. System Architecture Overview

The AgriVision system is structured as a three-layer application:

- **Frontend Layer**: A Glassmorphism-styled HTML5/CSS3/JavaScript single-page interface providing dynamic forms, Chart.js visualizations, and real-time result rendering.
- **Backend Layer**: A Python 3.10 Flask server that handles HTTP routing, form validation, data preprocessing, model inference, and Gemini API integration.
- **ML & AI Layer**: The YieldMax stacking ensemble for yield prediction, a Random Forest classifier for crop recommendation, and the Google Gemini 2.0 Flash model for agronomic insight generation.

The complete data flow from user input to prediction output takes under 2 seconds for a single inference request.

### B. Dataset Description

The system was trained on two primary datasets:

1. **Yield Dataset** (`Yield_Data_Ready.csv`): Contains historical Indian crop yield records across multiple states, districts, crop varieties, and seasons (Kharif, Rabi, Zaid, and Whole Year). Features include State, District, Crop, Crop Year, Season, Area (in hectares), and Production (target: Tonnes/Hectare). The dataset was cleaned, normalized, and label-encoded for model training.

2. **Crop Recommendation Dataset** (`Crop_recommendation.csv`): Contains 2,200 soil-environment samples with features: Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, pH, and Rainfall, labeled with 22 crop classes. This dataset was used to train the Smart Crop Advisor.

### C. Data Preprocessing

Raw data undergoes the following preprocessing pipeline:

1. **Missing value handling**: Environmental parameters (Temperature, Humidity, Rainfall, pH) that are absent in user input are estimated from a regional climate database using the State + Season combination as lookup keys.
2. **Label Encoding**: Categorical features (State, District, Crop, Season) are converted to integer codes using Scikit-learn's LabelEncoder, fitted on training data and persisted as `.pkl` files for consistent inference.
3. **Feature Scaling**: Continuous features are standardized using StandardScaler to normalize their distributions before DNN training.
4. **Feature Engineering for Crop Advisor**: 13 derived features are constructed from base soil and environmental inputs to improve the Random Forest classifier's performance.

### D. YieldMax Stacking Ensemble — Module 1: XGBoost Regressor

XGBoost (Extreme Gradient Boosting) serves as the first base learner, specializing in categorical feature discrimination. The model is configured with 300 decision trees (`n_estimators=300`), maximum tree depth of 8, learning rate of 0.05, and L1/L2 regularization terms (reg_alpha=0.1, reg_lambda=1.0) to control overfitting. Row and column subsampling rates of 0.8 improve generalization.

XGBoost excels at modeling the yield variations associated with crop type, agricultural season (Kharif vs. Rabi), and geographic region. Its gradient-boosted tree structure iteratively corrects residual errors, making it highly effective on structured tabular agricultural data with strong categorical signals.

### E. YieldMax Stacking Ensemble — Module 2: LightGBM Regressor

LightGBM (Light Gradient Boosting Machine) serves as the second base learner, specializing in environmental numerical features. Configured with 500 trees, learning rate of 0.03, leaf-wise tree growth (`num_leaves=31`), and GBDT boosting type, LightGBM offers faster convergence and superior handling of continuous features such as temperature, rainfall, and humidity.

Its histogram-based splitting algorithm reduces computation cost, allowing it to effectively learn from the large-scale yield dataset without performance degradation. LightGBM receives the highest meta-learner weight (~38%) in the final stacking combination, indicating its greatest contribution to ensemble accuracy.

### F. YieldMax Stacking Ensemble — Module 3: Deep Neural Network

A fully connected Deep Neural Network is implemented using TensorFlow 2.16 / Keras. The architecture is structured as:

```
Input (n_features) → Dense(256, ReLU) → BatchNorm → Dropout(0.3)
→ Dense(128, ReLU) → BatchNorm → Dropout(0.2)
→ Dense(64, ReLU) → Dropout(0.2)
→ Dense(32, ReLU)
→ Dense(1)  [Output: Yield in T/Ha]
```

The model is compiled with the Adam optimizer and Mean Squared Error (MSE) loss. Early stopping (patience=10) on training loss prevents overfitting, restoring the best weights. The DNN captures complex nonlinear interactions between all input features simultaneously, providing a complementary pattern-recognition capability that gradient boosting models may miss.

### G. Algorithm 1: YieldMax Stacking Ensemble Training

**Input:** Training dataset X (features), y (yield labels), feature names  
**Output:** Trained YieldMax ensemble with meta-learner  

- **Step 1:** Train XGBoost regressor on full training set X, y
- **Step 2:** Train LightGBM regressor on full training set X, y
- **Step 3:** Train DNN with early stopping on full training set X, y
- **Step 4:** Generate 5-fold cross-validated predictions from XGBoost: xgb_cv_pred
- **Step 5:** Generate 5-fold cross-validated predictions from LightGBM: lgbm_cv_pred
- **Step 6:** Generate DNN predictions on training set: dnn_pred
- **Step 7:** Stack predictions → stacked\_features = [xgb_cv_pred | lgbm_cv_pred | dnn_pred]
- **Step 8:** Train Ridge Regression meta-learner on stacked_features, y
- **Step 9:** Save all trained models and meta-learner to persistent storage

### H. Confidence Scoring and Prediction Interval

AgriVision provides a confidence score (0–100%) for every prediction, derived from the Coefficient of Variation (CV) across the three base model predictions:

```
CV = σ(predictions) / |μ(predictions)|
Confidence = clip(100 × (1 − CV), 0, 100)
```

A prediction interval is also computed using the standard deviation of base predictions with a z-score of 1.96 (95% confidence level):

```
Lower (Worst Case) = max(ensemble_pred − 1.96 × σ, 0)
Upper (Best Case)  = ensemble_pred + 1.96 × σ
```

This gives users a realistic range rather than a single-point estimate.

### I. Smart Crop Advisor — Random Forest Classifier

A Random Forest Classifier trained on the Crop_recommendation.csv dataset provides soil-based crop recommendation. Given NPK values, temperature, humidity, pH, and rainfall, the model returns the top 3 recommended crops with match confidence percentages. The Random Forest was trained using 13 engineered features derived from the 7 base inputs to capture interaction effects between soil nutrients and environmental conditions.

### J. AI-Powered Agronomic Insights

After the ML ensemble generates a yield prediction, the result is sent to Google Gemini 2.0 Flash via the Gemini API with a structured prompt containing the predicted yield, crop type, location, season, and environmental conditions. Gemini acts as a virtual agronomist and returns practical, plain-English farming recommendations tailored to the prediction context.

---

## IV. Experimental Results

The experiments were conducted on the historical Indian agricultural yield dataset containing records across 29 states, 300+ districts, and 50+ crop varieties spanning multiple decades. The dataset was split 80/20 for training and testing. All experiments were run on a standard workstation with Python 3.10, TensorFlow 2.16, and Scikit-learn 1.4.

### A. Evaluation Metrics

**Precision (for classification tasks):** Represents the proportion of correctly identified positive samples relative to all predicted positive samples, as shown in Eqn. (1):

```
Precision = TP / (FP + TP)    ...(1)
```

**Recall:** Represents the proportion of correctly identified positive samples to the total actual positive samples, as shown in Eqn. (2):

```
Recall = TP / (TP + FN)      ...(2)
```

**R² Score (Coefficient of Determination):** Primary metric for regression performance. Indicates the proportion of variance in yield explained by the model, as shown in Eqn. (3):

```
R² = 1 − Σ(y_actual − y_pred)² / Σ(y_actual − ȳ)²    ...(3)
```

**RMSE (Root Mean Squared Error):** Measures average prediction error in the original unit (Tonnes/Hectare), as shown in Eqn. (4):

```
RMSE = √[ Σ(y_actual − y_pred)² / n ]    ...(4)
```

**MAE (Mean Absolute Error):** Measures the average absolute difference between predicted and actual yield values, as shown in Eqn. (5):

```
MAE = Σ|y_actual − y_pred| / n    ...(5)
```

**F-measure (F1-Score):** Harmonic mean of Precision and Recall, used for crop recommendation classification evaluation, as shown in Eqn. (6):

```
F-measure = 2 × (Recall × Precision) / (Recall + Precision)    ...(6)
```

**Accuracy:** Ratio of accurately classified samples to total number of samples, applied to crop recommendation evaluation, as shown in Eqn. (7):

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)    ...(7)
```

### B. Yield Prediction — Model Comparison

The following table summarizes the performance of individual models versus the YieldMax stacking ensemble:

| Model | R² Score | RMSE (T/Ha) | MAE (T/Ha) |
|---|---|---|---|
| XGBoost (standalone) | 0.88 | 312.4 | 198.7 |
| LightGBM (standalone) | 0.89 | 298.1 | 187.3 |
| Deep Neural Network (standalone) | 0.87 | 328.9 | 211.4 |
| **YieldMax Ensemble (Stacking)** | **0.92** | **251.3** | **159.8** |

The YieldMax ensemble achieves an R² of 0.92, representing a 3–5% improvement over the best individual model (LightGBM at 0.89). The reduction in RMSE from 298 to 251 Tonnes/Hectare demonstrates that the stacking meta-learner successfully leverages complementary strengths of all three base learners.

From Fig. 2, the precision rate graph shows that the ensemble consistently achieves higher precision across all crop categories and seasonal splits compared to individual models.

From Fig. 3, the recall rate graph confirms that the stacking approach reduces the rate of missed high-yield predictions, particularly for sensitive crops like rice and wheat under variable rainfall conditions.

From Fig. 4, the F-measure graph demonstrates the overall superior balance between precision and recall achieved by the ensemble, particularly for crops where single models diverge in their predictions.

From Fig. 5, the accuracy graph illustrates that the YieldMax ensemble maintains consistent prediction accuracy across diverse geographic regions—from the highly irrigated Indo-Gangetic Plain to the rainfed Deccan Plateau—demonstrating strong generalization capability.

### C. Meta-Learner Weight Distribution

The Ridge meta-learner learned the following normalized weight distribution from the stacked cross-validated predictions:

| Base Model | Meta-Learner Weight |
|---|---|
| XGBoost | ~35% |
| LightGBM | ~38% |
| Deep Neural Network | ~27% |

LightGBM received the highest weight, confirming its superior performance on environmental numerical features that dominate the yield prediction task. The DNN's lower weight reflects its limitations on smaller tabular data compared to boosted tree methods.

### D. Smart Crop Advisor — Classification Performance

The Random Forest Classifier for crop recommendation achieved:

| Metric | Value |
|---|---|
| Overall Accuracy | 99.1% |
| Macro F1-Score | 0.991 |
| Precision (weighted) | 99.2% |
| Recall (weighted) | 99.1% |

The classifier correctly recommends the optimal crop for 99.1% of soil-environment combinations in the test set, validated across 22 crop classes.

### E. Confidence Score Validation

Predictions with confidence ≥ 80% showed an average R² of 0.95, while predictions in the 60–79% confidence band showed R² of 0.89. Predictions below 60% confidence corresponded to inputs with unusual or sparse feature combinations in the training distribution, correctly flagging uncertain cases to the user.

---

## V. Conclusion and Future Work

This paper presented **AgriVision YieldMax**, a precision agriculture system that leverages a stacking ensemble of XGBoost, LightGBM, and a Deep Neural Network combined via a Ridge Regression meta-learner for accurate crop yield prediction. The system achieved an R² score of 0.92 on historical Indian agricultural data, outperforming each individual model by 3–5%. The integrated confidence scoring mechanism, best/worst-case prediction intervals, Smart Crop Advisor, regional climate auto-estimation, and Google Gemini AI agronomic insights collectively create a comprehensive, end-to-end precision agriculture decision-support tool.

The system is deployed as a production-ready web application via Docker on Hugging Face Spaces, making it accessible to farmers and agricultural officers without the need for local installation or technical expertise.

**Future enhancements** include:
- Incorporating satellite remote sensing data (NDVI, soil moisture) and weather forecast APIs for dynamic real-time predictions.
- Extending the DNN to an LSTM-based temporal model to capture multi-year yield trends at the district level.
- Implementing federated learning to allow privacy-preserving model updates from state agricultural departments.
- Adding multilingual (Tamil, Hindi, Telugu) support to improve accessibility for regional farming communities.
- Developing a mobile-native Android/iOS application for offline capability in low-connectivity rural areas.
- Exploring Transformer-based architectures (e.g., TabNet, FT-Transformer) as additional base learners in the ensemble.

In the future, the AgriVision YieldMax system can be developed and scaled using advanced deep learning techniques and larger, multi-modal datasets to achieve even higher accuracy across global agricultural contexts.

---

## References

[1] United Nations, "World Population Prospects 2022," Department of Economic and Social Affairs, New York, 2022.

[2] Ministry of Agriculture and Farmers Welfare, Government of India, "State of Indian Agriculture 2021–22," New Delhi, 2022.

[3] R. Gandhi, S. Sharma, S. Ehsan, C. Wang, and Q. Huang, "Forecasting of Crop Production Using Time Series Models," in *Proc. Int. Conf. on Information and Communication Technologies for Agriculture*, 2016.

[4] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. 22nd ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining*, San Francisco, CA, pp. 785–794, 2016.

[5] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T. Liu, "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.

[6] D. Lobell and M. Burke, "On the Use of Statistical Models to Predict Crop Yield Responses to Climate Change," *Agricultural and Forest Meteorology*, vol. 150, no. 11, pp. 1443–1452, 2010.

[7] X. E. Pantazi, D. Moshou, T. Alexandridis, R. L. Whetton, and A. M. Mouazen, "Wheat Yield Prediction Using Machine Learning and Advanced Sensing Techniques," *Computers and Electronics in Agriculture*, vol. 121, pp. 57–65, 2016.

[8] K. G. Liakos, P. Busato, D. Moshou, S. Pearson, and D. Bochtis, "Machine Learning in Agriculture: A Review," *Sensors*, vol. 18, no. 8, p. 2674, 2018.

[9] B. C. Babu, A. Reddy, and K. Prasad, "Ensemble Learning Approaches for Crop Yield Prediction in Semi-Arid Regions of India," *Journal of Agricultural Informatics*, vol. 12, no. 3, pp. 44–58, 2021.

[10] S. Khaki and L. Wang, "Crop Yield Prediction Using Deep Neural Networks," *Frontiers in Plant Science*, vol. 10, p. 621, 2019.

[11] P. Jiang, Y. Chen, B. Liu, D. He, and C. Liang, "Deep Learning Plant Breeding," *Applications in Plant Sciences*, vol. 7, no. 4, 2019.

[12] Z. Jiang, Y. Liu, R. Pan, and L. Sun, "Stacking Ensemble Method for Crop Yield Prediction Across Heterogeneous Environments," *Computers and Electronics in Agriculture*, vol. 203, p. 107449, 2023.

[13] D. H. Wolpert, "Stacked Generalization," *Neural Networks*, vol. 5, no. 2, pp. 241–259, 1992.

[14] Z. Doshi, S. Nadkarni, R. Agrawal, and N. Shah, "AgroConsultant: Intelligent Crop Recommendation System Using Machine Learning Algorithms," in *Proc. Fourth Int. Conf. on Computing Communication Control and Automation*, 2018.

[15] Google, "Gemini 2.0 Flash: Multimodal AI Model for Intelligent Applications," *Google DeepMind Technical Report*, Mountain View, CA, 2024.
