import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV


MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attribs, cat_attribs):
    # Numerical pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

        # Categorical pipeline
    cat_pipeline = Pipeline([
            # ("ordinal", OrdinalEncoder())  # Use this if you prefer ordinal encoding
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        # Full pipeline
    full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", cat_pipeline, cat_attribs),
        ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # Lets Train the model
    # 1. Load the data
    housing = pd.read_csv("housing.csv")

    # 2. Create a stratified test set based on income category
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        housing.loc[test_index].drop("income_cat", axis=1).to_csv("input.csv", index=False)  # Save the test set for inference
        housing = housing.loc[train_index].drop("income_cat", axis=1) 
        
    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)     
    
    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]
    
    pipeline = build_pipeline(num_attribs, cat_attribs) 
    housing_prepared = pipeline.fit_transform(housing_features)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)
    
    # ── Evaluation ──────────────────────────────────────────
    # 1. Cross-validation RMSE (more reliable than single split)
    cv_scores = cross_val_score(model, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=5)
    cv_rmse = np.sqrt(-cv_scores)

    # 2. Training set metrics (to check overfitting)
    train_preds = model.predict(housing_prepared)
    train_rmse  = root_mean_squared_error(housing_labels, train_preds)
    train_r2    = r2_score(housing_labels, train_preds)

    # 3. Print results
    print("\n📊 Model Evaluation — RandomForestRegressor")
    print("=" * 45)
    print(f"  Train RMSE : ${train_rmse:,.0f}")
    print(f"  Train R²   : {train_r2:.4f}")
    print(f"\n  CV RMSE    : ${cv_rmse.mean():,.0f}  (±{cv_rmse.std():,.0f})")
    print(f"  CV Scores  : {[f'${s:,.0f}' for s in cv_rmse]}")
    print("=" * 45)
    
    
    # ── Hyperparameter Tuning ────────────────────────────────
    print("\n🔍 Running RandomizedSearchCV... (this takes 2–4 mins)")

    param_grid = {
        "n_estimators"      : [100, 200, 300, 500],
        "max_features"      : ["sqrt", "log2", 0.3, 0.5],
        "max_depth"         : [None, 10, 20, 30, 40],
        "min_samples_split" : [2, 5, 10],
        "min_samples_leaf"  : [1, 2, 4],
        "bootstrap"         : [True, False],
    }

    rscv = RandomizedSearchCV(
        estimator           = RandomForestRegressor(random_state=42),
        param_distributions = param_grid,
        n_iter              = 20,          # tries 20 random combos
        scoring             = "neg_mean_squared_error",
        cv                  = 5,
        verbose             = 1,           # shows progress in terminal
        random_state        = 42,
        n_jobs              = -1,          # uses all CPU cores → faster
    )

    rscv.fit(housing_prepared, housing_labels)

    # ── Best model results ───────────────────────────────────
    best_model    = rscv.best_estimator_
    best_cv_rmse  = np.sqrt(-rscv.best_score_)
    best_params   = rscv.best_params_

    print("\n✅ Tuning Complete!")
    print("=" * 45)
    print(f"  Best CV RMSE  : ${best_cv_rmse:,.0f}")
    print(f"  vs Before     : $49,942  (your baseline)")
    print(f"  Improvement   : ${49942 - best_cv_rmse:,.0f}")
    print("\n  Best Params:")
    for k, v in best_params.items():
        print(f"    {k:22s}: {v}")
    print("=" * 45)

    # ── Save the BEST model 
    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(pipeline,   PIPELINE_FILE)
    print("\n💾 Best model saved!")
else:
    # Lets do inference
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    
    input_data = pd.read_csv('input.csv')
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data['predicted_house_value'] = predictions.round(2)   

    print(f"\n✅ Predictions done — {len(predictions)} rows")
    print(f"   Avg predicted value : ${predictions.mean():,.0f}")
    print(f"   Min → Max           : ${predictions.min():,.0f} → ${predictions.max():,.0f}")

    input_data.to_csv('output.csv', index=False)
    print("   Results saved to output.csv")