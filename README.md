# Tehran House Price Prediction

A machine learning project that predicts apartment prices in Tehran (in USD) from basic property features: neighborhood, area, number of rooms, and amenities. It covers the full workflow: data cleaning, outlier removal, preprocessing, model comparison with hyperparameter tuning, and evaluation.

> **Status:** model training is complete. A web app (see [Roadmap](#roadmap)) is coming soon so anyone can try the model in the browser.

## Dataset

The dataset (`TehranHouse.csv`) contains about 3,450 listings with these columns:

| Column | Description |
|---|---|
| `Area` | Property size in square meters |
| `Room` | Number of rooms |
| `Parking` | Has parking (True/False) |
| `Warehouse` | Has storage room (True/False) |
| `Elevator` | Has elevator (True/False) |
| `Address` | Neighborhood name |
| `Price` | Price in Iranian Toman |
| `Price(USD)` | Price in US dollars (**target**) |

## Approach

1. **Missing values:** rows with a missing `Address` are dropped.
2. **Dirty data:** `Area` was stored as text and contained invalid entries (for example, prices typed into the area column). It is converted to numeric and unparseable rows are removed.
3. **Outlier removal:**
   - Rows with less than 28 m² per room are removed as unrealistic.
   - Price per square meter (`price_per_m`) is computed **only for cleaning**. Within each neighborhood, rows more than `k` standard deviations from that neighborhood's mean are dropped (`k=2`). This column is *not* used as a model feature, since it is derived from the target and would cause data leakage.
4. **Features and target:** features are `Address`, `Area`, `Room`, `Parking`, `Warehouse`, `Elevator`; the target is `Price(USD)`.
5. **Preprocessing:** one-hot encoding for `Address`, standard scaling for `Area` and `Room`, boolean columns passed through.
6. **Target transform:** the target is log-transformed (`log1p`) to handle the strongly skewed price distribution, and predictions are converted back with `expm1`.
7. **Model selection:** four regressors are compared using `GridSearchCV` (5-fold cross-validation) inside a pipeline:
   - Support Vector Regression (SVR)
   - Random Forest
   - Ridge Regression
   - Decision Tree
8. **Evaluation:** the best configuration of each model is evaluated on a held-out 20% test set using R².

## Results

| Model | Best CV R² | Test R² | Best parameters |
|---|---|---|---|
| SVR | _TBD_ | _TBD_ | _TBD_ |
| Random Forest | _TBD_ | _TBD_ | _TBD_ |
| Ridge | _TBD_ | _TBD_ | _TBD_ |
| Decision Tree | _TBD_ | _TBD_ | _TBD_ |

_(Fill in the table with the results of the final notebook run.)_

**Key takeaway:** the neighborhood (`Address`) is the most important feature. Removing it dropped the Random Forest cross-validation R² noticeably in earlier experiments.

## Project Structure

```
tehran-house-price/
├── README.md
├── requirements.txt
├── data/
│   └── TehranHouse.csv
├── notebooks/
│   └── tehran_housing_price.ipynb
├── models/
│   └── model.joblib          # saved full pipeline (planned)
└── app/                      # web app (planned)
```

## Getting Started

```bash
git clone https://github.com/amintnt2000/tehran-house-price.git
cd tehran-house-price

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
jupyter notebook notebooks/tehran_housing_price.ipynb
```

`requirements.txt` should include:

```
pandas
numpy
scikit-learn
joblib
jupyter
```

## Roadmap

- [x] Data cleaning and outlier removal
- [x] Model comparison with hyperparameter tuning
- [ ] Re-run and record final results without data leakage
- [ ] Save the full pipeline with `joblib`
- [ ] Build a web app (Streamlit) where users enter neighborhood, area, rooms, and amenities and get a price estimate
- [ ] Deploy the app and add a live demo link and screenshot here

## Limitations

- The dataset is a snapshot of listings, not official sale prices, so predictions are estimates.
- Neighborhoods that do not appear in the training data cannot be predicted well.
- Features such as building age, floor, and exact location are not available in the data.

## Author

**Mohammad Amin Nochiyan**
GitHub: [@amintnt2000](https://github.com/amintnt2000)
