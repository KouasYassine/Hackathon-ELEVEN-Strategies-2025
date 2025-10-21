# Europark – Predicting Attraction Wait Times with Machine Learning
 — Role: Modeling & Analysis
## Project Overview
This project, developed during the ENPC Hackathon 2025, focuses on predicting theme park attraction wait times at **Europark**, a small amusement park with three attractions, one shop, and one food point.

Following the COVID-19 pandemic, Europark faced two main challenges:  
1. Increased and less predictable waiting times due to operational and behavioral shifts.  
2. A decline in visitor satisfaction linked to the disappearance of live events and reduced capacity.

The goal was to build a predictive model capable of estimating wait times up to **two hours in advance**, using both historical and meteorological data.  
Additionally, the project analyzed how post-COVID changes — particularly the reduction of parades and night shows — affected visitor dynamics and congestion patterns.

---

## Objectives
- Predict short-term (2-hour horizon) attraction wait times.  
- Identify key environmental and temporal factors influencing visitor flow.  
- Evaluate the operational and behavioral effects of parades and night shows.  
- Provide actionable recommendations to optimize capacity and improve visitor satisfaction.  

---

## Data and Feature Engineering
Data sources included:
- `waiting_times_train.csv`: historical wait times per attraction.  
- `weather_data_combined.csv`: temperature, humidity, rain, snow, and wind data.  
- `valmeteo.csv`: weather data for validation.  
- `waiting_times_X_test_val.csv`: test/validation set for final evaluation.

Feature engineering, performed in the `adapt_data_project_GX()` function, included:
- **Weather variables:** rainfall, snowfall, “feels-like” temperature, humidity, hot/cold indicators.  
- **Temporal variables:** hour, day, month, weekday/weekend, and a “COVID era” flag distinguishing pre- and post-pandemic data.  
- **Holiday variables:** French school holidays for zones A, B, and C.  
- **Attraction types:** binary variables (e.g., Water Ride, Pirate Ship, Flying Coaster).  
- **Operational features:** capacity ratio (`CURRENT_WAIT_TIME / ADJUST_CAPACITY`).

This structure ensured that the model could capture both external conditions and park-specific factors driving queue fluctuations.

---

## Modeling Approach
- Several **XGBoost Regressor** models were trained with varying hyperparameters and random seeds.  
- A **linear stacking meta-model** blended their predictions to form an ensemble.  
- **Cross-validation** (using KFold or TimeSeriesSplit) ensured model stability and avoided temporal leakage.  
- **Performance metric:** Root Mean Squared Error (RMSE).  

This ensemble approach balanced bias and variance while improving robustness across different time periods and attraction types.

---

## Key Findings

### Post-COVID Operational Changes
After COVID, Europark **drastically reduced the number of parades and night shows**, which previously helped regulate visitor flow across the park.  
As a result:
- Visitors spent more time concentrated around main attractions.  
- Queue peaks became sharper and more synchronized.  
- Daily attendance patterns lost their natural dispersion.

### Event Simulation Experiment
To quantify the impact of these changes, we **artificially reintroduced parades and night shows** into the data, simulating their effect on visitor distribution.  
Predicted wait times dropped significantly, showing that entertainment events act as natural flow regulators.

---

## Results and Event Impact

To better understand this relationship, we analyzed the **local impact** of parades and night shows on attraction wait times within a ±3-hour window around each event.

### Parades
![Local impact of parades](impactlocal%20(1).png)

- Parades consistently reduce waiting times during and shortly before their start.  
- The strongest effect is observed for mid-day parades (11h–17h), lowering average queues by **8–12 minutes**.  
- Late parades (around 20h) have a smaller but still visible impact.  
- Daytime parades are particularly effective in dispersing visitors and reducing congestion during high-traffic periods.

### Night Shows
![Local impact of night shows](impactnightshow%20(1).png)

- Night shows also lead to short-term queue reductions, especially between 18h and 20h.  
- The effect fades later in the evening as visitors prepare to leave the park.  
- Early night shows reduce average wait times by **5–10 minutes**.  
- Together, parades and night shows act as **dynamic regulators** of park traffic.

### Summary of Findings
Reintroducing entertainment events, even in simulated form, consistently improved predicted park fluidity.  
Strategically scheduling shows across the day could reduce average waiting times by **up to 10 minutes** without additional operational cost.

---

## Operational Insights
- **Maintain or increase event frequency** to naturally smooth visitor flow.  
- **Display predicted wait times** on park screens or mobile apps to improve visitor decision-making.  
- **Introduce FastPass or reservation systems** for the most congested attractions.  
- **Adjust staffing dynamically** based on predicted weather and attendance patterns.

---

## Technical Summary
| Aspect | Description |
|---------|-------------|
| Language | Python 3 |
| Libraries | pandas, numpy, scikit-learn, xgboost |
| Modeling | XGBoost ensemble with linear stacking |
| Validation | KFold / TimeSeriesSplit |
| Metric | RMSE |
| Output | `val_predictions_xgb_subset_blend.csv` |

---

## Authors
**ENPC Hackathon 2025 – Group 1**  
 Yassine Kouas  • Adrien Gentili • Clara Lima Goldenberg • Paul Andreis

---

## Conclusion
This study provided a comprehensive understanding of how post-COVID operational changes affected visitor dynamics at Europark.  
The predictive model accurately estimated short-term waiting times and revealed the crucial role of entertainment scheduling in managing crowds.  

By combining data-driven modeling with event analysis, we demonstrated that **well-timed parades and night shows can significantly reduce congestion**, improve visitor satisfaction, and enhance overall park efficiency.
