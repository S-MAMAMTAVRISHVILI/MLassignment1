# პროექტი - House Prices - Advanced Regression Techniques

## Kaggle-ის კონკურსის მოკლე მიმოხილვა

პროექტი წარმოადგენს Kaggle-ის კონკურსს, რომლის მიზანია აიოვას შტატის ქალაქ ამესში სახლების ფასის დადგენა. ამისათვის მოცემული გვაქვს 79 მახასიათებელი, რომლებიც თითქმის ყველა მხრივ აღწერენ საცხოვრებლებს. 

პროექტისთვის შეფასების მეტრიკად შერჩეულია - 'RMSE on log(SalePrice)'. სამიზნე ცვლადზე ანუ 'SalePrice'-ზე მოდებულია ლოგარითმი, შესაბამისად მოდელი ვარაუდობს ფასის ლოგარითმს და არა სუფთა ფასს. ამის მიზეზია რამდენიმე ძალიან ძვირიანი სახლის არსებობა მონაცემებში. ლოგარითმი ამ არაბუნებრივ სხვაობას გაცილებით ამცირებს. ფასის ლოგარითმზე შემდეგ გამოიყენება RMSE(Root Mean Squared Error), რომელიც გვეუბნება საშუალო ვარაუდის ცდომილებას.

## ჩემი მიდგომა

1. მონაცემების სემანტიკური გაწმენდა, NaN გარდაიქმნება "None"-ად იქ, სადაც ფიჩერი უბრალოდ არ არსებობს (და არა ინფორმაცია არ გვაქვს)
2. ხარისხის სვეტების ორდინალური კოდირება (Poor=1 … Excellent=5)
3. დანარჩენი კატეგორიული სვეტების One-Hot კოდირება
4. ზღვრული ამონაგის (outlier) მოშორება
5. მრავალი განსხვავებული მოდელის ექსპერიმენტი + Hyperparameter ოპტიმიზაცია
6. K-Fold Cross Validation მოდელის სტაბილურობის შემოწმებისთვის
7. ექსპერიმენტების ტრეკინგი MLflow + DagsHub

# რეპოზიტორიის სტრუქტურა

model_experiment.ipynb - ძირითადი სამუშაო ფაილი, რომელშიც წარმოდგენილია ყველა სამუშაო ეტაპი : Cleaning, Feature Engineering, Feature Selection, Training და თითოეული ეტაპის შესაბამისი ლოგები.

model_inference.ipynb - ფაილი, სადაც საუკეთესო მოდელის გამოყენებით ხდება სატესტო მონაცემებზე პროგნოზი და გენერირდება საბოლოო submission.csv 
ფაილი kaggle-ის კონკურსისთვის. საუკეთესო მოდელი ტრენინგისას ინახება Model Registry-ში და იქიდან ხდება მისი გამოყენება.

README.md - ფაილი, რომელიც დეტალურად აღწერს პროექტსა და მასზე მუშაობის სრულ პროცესს.

test.csv - სატესტო მონაცემები

train.csv - სატრენინგო მონაცემები

submission.csv - საბოლოო საბმიშენ ფაილი

data_description.txt - თითოეული მახასიათებლის სიტყვიერი აღწერა

ridge_model.pkl - ცალკე ფაილად შენახული ოპტიმალური მოდელი.

model_columns.pkl - ოპტიმალური მოდელის შესაბამისი სვეტები(მონაცემების სხვაობის გამო One-Hot ენკოდირების შემდეგ მივიღე ოდნავ განხვავებული ფორმის მონაცემები ტრენინგისა და ტესტისთვის, ამიტომ დამჭირდა მათი ერთმანეთთან შესაბამისობა)

# Feature Engineering

## 1. Cleaning და NaN მნიშვნელობების დამუშავება (cleaning_v1_semantic_imputation)

პირველ რიგში შევისწავლეთ რომელ სვეტებს აქვს ყველაზე მეტი NaN ბრძანებით:

`train.isnull().sum().sort_values(ascending=False).head(20)`


შემდეგ `df = train.copy()` — ვაკოპირებთ ტრეინინგ დატას.

### სემანტიკური შევსება — "None" სტრინგით

14 კატეგორიულ სვეტში NaN ნიშნავს "ეს ფიჩერი სახლს არ აქვს" — ამიტომ შევავსე `None`-ით და არა `0`-ით ან მოდით:

```python
none_cols = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"
]
for col in none_cols:
    df[col] = df[col].fillna("None")
```

### სპეციფიკური სვეტების შევსება

```python
# LotFrontage: ვავსებთ იმავე სამეზობლოს სახლების მედიანით, რადგან
# ერთ სამეზობლოში სახლების ეზოს სიგანე მსგავსია
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median())
)

df["MasVnrType"] = df["MasVnrType"].fillna("None")   # არ აქვს ქვის მოპირკეთება
df["MasVnrArea"] = df["MasVnrArea"].fillna(0)         # ფართობი = 0 თუ მოპირკეთება არ არის
df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)       # გარაჟი არ აქვს -> 0
df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])  # NaN -> მოდა (SBrkr)
```
გადამოწმება:

```python
print("Remaining missing values:", df.isnull().sum().sum())  # უნდა მოგვცეს 0
```

## 2. კატეგორიული ცვლადების რიცხვითში გადაყვანა

### ა) ორდინალური კოდირება — ხარისხის სვეტები

ხარისხისა და მდგომარეობის სვეტებს ბუნებრივი თანმიმდევრობა აქვთ უარესიდან უკეთესისკენ. ამიტომ One-Hot-ის ნაცვლად გადამყავს მთელ რიცხვებში:

```python
qual_map = {
    "None": 0,   # მახასიათებელი არ გვაქვს
    "Po": 1,     # Poor
    "Fa": 2,     # Fair
    "TA": 3,     # Typical/Average
    "Gd": 4,     # Good
    "Ex": 5      # Excellent
}

qual_cols = [
    "ExterQual", "ExterCond",
    "BsmtQual", "BsmtCond",
    "HeatingQC", "KitchenQual",
    "FireplaceQu",
    "GarageQual", "GarageCond",
    "PoolQC"
]

for col in qual_cols:
    df_encoded[col] = df_encoded[col].map(qual_map)
```

შედეგის გადამოწმება:

```python
df_encoded[qual_cols].head()  # ციფრები 0-5 ტექსტის ნაცვლად
```

MLflow:

```python
with mlflow.start_run(run_name="encoding_v1"):
    mlflow.log_param("encoding", "ordinal_quality_mapping_0_to_5")
```

### ბ) One-Hot Encoding — ნომინალური სვეტები

დარჩენილი ყველა object-ტიპის სვეტზე: სამეზობლო, სახლის სტილი, გზის ტიპი... , რომლებსაც რიცხობრივად ლოგიკურ
შესაბამისობებს ვერ ვუპოვით(მაგალითად ხარისხობრივს) ვიყენებთ One-Hot Encoding-ს.

```python
df_final = df_encoded.copy()
df_final = pd.get_dummies(df_final)
```

გადამოწმება:

```python
print(df_final.shape)  # სვეტების რაოდენობა მნიშვნელოვნად გაიზარდა
df_final.select_dtypes(include="object").columns  # -> ცარიელია
```

MLflow:

```python
with mlflow.start_run(run_name="encoding_v2_onehot"):
    mlflow.log_param("encoding", "one_hot_all_nominal")
    mlflow.log_metric("num_features", df_final.shape[1])
```

### გ) Interaction Features (v3 — ექსპერიმენტული)

შევქმენი კომბინირებული მახასიათებლები, რომლებიც ასახავენ ორი ცვლადის ერთობლივ გავლენას:

```python
df_inter = df.copy()

# ასაკობრივი ფიჩერები
df_inter["HouseAge"] = df_inter["YrSold"] - df_inter["YearBuilt"]
df_inter["RemodAge"] = df_inter["YrSold"] - df_inter["YearRemodAdd"]

# ხარისხ-ფართობის კომბინაციები
df_inter["OverallQual_GrLivArea"] = df_inter["OverallQual"] * df_inter["GrLivArea"]
df_inter["BsmtQual_TotalBsmtSF"]  = df_inter["BsmtQual"] * df_inter["TotalBsmtSF"]
df_inter["GarageStrength"]        = df_inter["GarageCars"] * df_inter["GarageArea"]
df_inter["HouseAge_Quality"]      = df_inter["HouseAge"] * df_inter["OverallQual"]
df_inter["Remod_Quality"]         = df_inter["RemodAge"] * df_inter["OverallQual"]

# მთლიანი ფართობი
df_inter["TotalSF"] = (
    df_inter["TotalBsmtSF"] +
    df_inter["1stFlrSF"] +
    df_inter["2ndFlrSF"]
)

df_inter = pd.get_dummies(df_inter)
```

შენიშვნა: ეს ექსპერიმენტი ცალკე გაიტესტა, თუმცა არასახარბიელო შედეგის გამო არ გამოყენებულა საბოლოო მოდელში.

### დ) სამიზნე ცვლადის ტრანსფორმაცია

```python
X = df_final.drop("SalePrice", axis=1)
y = df_final["SalePrice"]
y = np.log1p(y)  # log(1 + SalePrice) — SalePrice-ის განაწილება ნორმალიზდება
```

ვიყენებ log1p-ს ნაცვლად ჩვეულებრივი ლოგარითმისა log(0) ქეისის გამო.

## Feature Selection

### მიდგომა 1: კორელაციაზე დაფუძნებული შერჩევა

გამოვთვალე თითოეული ფიჩერის კორელაცია SalePrice-თან და მოვჭერი threshold-ზე < 0.05:

```python
corr = pd.concat([X_train, y_train], axis=1).corr()["SalePrice"].abs()
selected_features = corr[corr > 0.05].index.drop("SalePrice")

X_train_sel = X_train[selected_features]
X_val_sel   = X_val[selected_features]
```

შემდეგ ვატრენინგეb Ridge(alpha=10) შერჩეულ მახასიათებლებზე:

```python
ridge_sel = Ridge(alpha=10)
ridge_sel.fit(X_train_sel, y_train)
y_pred_sel = ridge_sel.predict(X_val_sel)
rmse_sel = np.sqrt(mean_squared_error(y_val, y_pred_sel))
print("Selected Features RMSE:", rmse_sel)
```

### მიდგომა 2: Manual Drop (საბოლოოდ არჩეული)

ხელით მოვაშორე sparse, identifier, ზედმეტად ბევრი NaN-ის მქონე სვეტები:

```python
drop_cols = [
    "Alley",        # ძალიან ბევრი NaN, უმნიშვნელო გავლენა
    "PoolQC",       # sparse
    "Fence",        # sparse, დაბალი კორელაცია ფასთან
    "MiscFeature",  # უმნიშვნელო გავლენა
    "Id"            # identifier
]

X_manual = X_train.copy()
X_val_manual = X_val.copy()

X_manual     = X_manual.drop(columns=drop_cols, errors="ignore")
X_val_manual = X_val_manual.drop(columns=drop_cols, errors="ignore")
```

Ridge(alpha=0.1)-ით ტესტი:

```python
model_manual = Ridge(alpha=0.1)
model_manual.fit(X_manual, y_train)
preds_manual = model_manual.predict(X_val_manual)
rmse_manual = np.sqrt(mean_squared_error(y_val, preds_manual))
print("Manual Drop RMSE:", rmse_manual)
```

MLflow:

```python
with mlflow.start_run(run_name="manual_feature_drop_ridge"):
    mlflow.log_param("features_removed", drop_cols)
    mlflow.log_param("model", "Ridge")
    mlflow.log_param("alpha", 0.1)
    mlflow.log_metric("rmse", rmse_manual)
```
### მიდგომა 3: Lasso-ზე დაფუძნებული შერჩევა

Lasso რეგრესია L1 რეგულარიზაციის გამო ავტომატურად ანულებს სუსტი მახასიათებლების კოეფიციენტებს. alpha=0.001-ით დავატრენინგე ლასო და ავირჩიე მხოლოდ არანულოვანი სვეტები.

შედეგი: შეირჩა 80 მახასიათებელი 263-დან. შემდეგ Ridge(alpha=10) დავატრენინგეთ შერჩეულ სვეტებზე.

### Outlier მოშორება (feature selection-ის ნაწილი)

K-Fold-ის დროს გამოვლინდა, რომ 2 მონაცემი აზიანებდა მოდელს — ძალიან დიდი სახლები (GrLivArea > 4000) უჩვეულოდ დაბალ ფასად (< $200,000):

```python
df_final = df_final[
    ~((df_final["GrLivArea"] > 4000) & (df_final["SalePrice"] < 200000))
].copy()

print("Removed outliers. New shape:", df_final.shape)
```

> outlier-ის მოშორება ხდება `get_dummies()`-ის შემდეგ, `X`/`y` განსაზღვრამდე.

## Training

### Train/Validation Split

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 80% ტრენინგი, 20% ვალიდაცია, random state=42 
```

### ტესტირებული მოდელები

#### მოდელი 1: Linear Regression (Baseline)

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("RMSE:", rmse)
```

MLflow:

```python
with mlflow.start_run(run_name="baseline_linear_regression"):
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_param("features", df_final.shape[1])
    mlflow.log_metric("rmse", rmse)
```

> Baseline — რეგულარიზაციის გარეშე, როდესაც გვაქვს ბევრი მახასიათებელი განიცდის overfit-ს.

#### მოდელი 2: Ridge Regression (alpha=10)

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=10)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_val)
rmse_ridge = np.sqrt(mean_squared_error(y_val, y_pred_ridge))
print("Ridge RMSE:", rmse_ridge)
```

MLflow:

```python
with mlflow.start_run(run_name="ridge_alpha_10"):
    mlflow.log_param("model", "Ridge")
    mlflow.log_param("alpha", 10)
    mlflow.log_metric("rmse", rmse_ridge)
```

#### მოდელი 3: Ridge — Alpha Sweep, Hyperparameter-ის ოპტიმიზაცია

6 სხვადასხვა alpha-ს ვატესტეთ loop-ში:

```python
alphas = [0.1, 1, 5, 10, 50, 100]
results = []

for a in alphas:
    model = Ridge(alpha=a)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    results.append((a, rmse))

results  # საუკეთესო alpha აღმოჩნდა 0.1-ის ტოლი, ანუ თითქმის უმნიშვნელო რეგულარიზაცია
```

საუკეთესო შედეგი MLflow-ში:

```python
with mlflow.start_run(run_name="ridge_best_alpha_0_1"):
    mlflow.log_param("model", "Ridge")
    mlflow.log_param("alpha", 0.1)
    mlflow.log_metric("rmse", 0.12409264369272437)
```

#### მოდელი 4: Ridge + Manual Drop (alpha=0.1)

```python
model_manual = Ridge(alpha=0.1)
model_manual.fit(X_manual, y_train)
preds_manual = model_manual.predict(X_val_manual)
rmse_manual = np.sqrt(mean_squared_error(y_val, preds_manual))
print("Manual Drop RMSE:", rmse_manual)
```

```python
with mlflow.start_run(run_name="manual_feature_drop_ridge"):
    mlflow.log_param("features_removed", drop_cols)
    mlflow.log_param("model", "Ridge")
    mlflow.log_param("alpha", 0.1)
    mlflow.log_metric("rmse", rmse_manual)
```

#### მოდელი 5: Ridge + Interaction Features (v3)

```python
model = Ridge(alpha=0.1)
model.fit(X_train, y_train)  # X_train აქ df_inter-დან მოდის(ანუ კომბინირებული მახასიათებლების მქონე data-დან)
preds = model.predict(X_val)
rmse_inter = np.sqrt(mean_squared_error(y_val, preds))
print("Interaction Features RMSE:", rmse_inter)
```

```python
with mlflow.start_run(run_name="ridge_interaction_features_v3"):
    mlflow.log_param("feature_engineering", "interaction_features_v3")
    mlflow.log_param("features_added", [
        "OverallQual_GrLivArea", "BsmtQual_TotalBsmtSF",
        "GarageStrength", "HouseAge_Quality", "Remod_Quality"
    ])
    mlflow.log_param("model", "Ridge")
    mlflow.log_param("alpha", 0.1)
    mlflow.log_metric("rmse", rmse_inter)
```

#### მოდელი 6: Gradient Boosting Regressor

```python
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
gbr.fit(X_train, y_train)
preds = gbr.predict(X_val)
rmse_gbr = np.sqrt(mean_squared_error(y_val, preds))
print("Gradient Boosting RMSE:", rmse_gbr)
```

```python
with mlflow.start_run(run_name="gradient_boosting_v1"):
    mlflow.log_param("model", "GradientBoostingRegressor")
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("max_depth", 3)
    mlflow.log_metric("rmse", rmse_gbr)
```

#### მოდელი 7: XGBoost

```python
from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb.fit(X_train, y_train)
preds_xgb = xgb.predict(X_val)
rmse_xgb = np.sqrt(mean_squared_error(y_val, preds_xgb))
print("XGBoost RMSE:", rmse_xgb)
```

```python
with mlflow.start_run(run_name="xgboost_v1"):
    mlflow.log_param("model", "XGBRegressor")
    mlflow.log_param("n_estimators", 500)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("max_depth", 4)
    mlflow.log_param("subsample", 0.8)
    mlflow.log_param("colsample_bytree", 0.8)
    mlflow.log_metric("rmse", rmse_xgb)
```

#### მოდელი 8: Random Forest

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=None,       
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1             
)
rf.fit(X_train, y_train)
preds_rf = rf.predict(X_val)
rmse_rf = np.sqrt(mean_squared_error(y_val, preds_rf))
print("Random Forest RMSE:", rmse_rf)
```

```python
with mlflow.start_run(run_name="random_forest_v1"):
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 400)
    mlflow.log_param("max_depth", None)
    mlflow.log_param("min_samples_split", 2)
    mlflow.log_param("min_samples_leaf", 1)
    mlflow.log_metric("rmse", rmse_rf)
```

### K-Fold Cross Validation — სტაბილურობის შემოწმება

თავდაპირველი შედეგების მიხედვით საუკეთესო იყო Ridge + Manual Drop (alpha=0.1) მინიმალური ალფათი rmse = 0.124, თუმცა პირდაპირი საბმიშენის შემდეგ მივიღე საკმაოდ დიდი ცდომილება rmse = 0.124 ტრენინგზე და rmse=0.16788 ტესტზე, რის შემდეგაც გამიჩნდა ეჭვი რომ მოდელი იყო არასტაბილური და გადავწყვიტე K-Fold Cross ვალიდაციის გამოყენება:

Fold 1 rmse = 0.12403
Fold 2 rmse = 0.12588
Fold 3 rmse = 0.22687 !!! თითქმის ორჯერ უარესი შედეგი დანარჩენებთან შედარებით
Fold 4 rmse = 0.14523 
Fold 5 rmse = 0.11157
Mean - 0.14672 Fold 3-ის გამო გაზრდილია
Std - 0.04150 ძალიან მაღალიია, ანუ გვაქვს არასტაბილური მოდელი.

**პრობლემის მიზეზი:** Fold 3-ში მოხვდა outlier სახლების კლასტერი (GrLivArea > 4000, SalePrice < 200K) + alpha=0.1, რომელიც მინიმალური რეგულარიზაციაა.

**გამოსწორება — outlier მოშორება + alpha sweep K-Fold-ში:**

```python
from sklearn.model_selection import KFold

# df_final უკვე გაწმენდილია outlier-ებისგან(GrLivArea > 4000, SalePrice < 200K)
X_full_kf = df_final.drop("SalePrice", axis=1)
X_full_kf = X_full_kf.drop(columns=drop_cols, errors="ignore")
y_full_kf = np.log1p(df_final["SalePrice"])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for alpha in [0.1, 1.0, 10.0, 50.0]:
    rmse_folds = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full_kf), 1):
        X_tr, X_val_kf = X_full_kf.iloc[train_idx], X_full_kf.iloc[val_idx]
        y_tr, y_val_kf = y_full_kf.iloc[train_idx], y_full_kf.iloc[val_idx]

        model_kf = Ridge(alpha=alpha)
        model_kf.fit(X_tr, y_tr)
        preds_kf = model_kf.predict(X_val_kf)
        rmse_folds.append(np.sqrt(mean_squared_error(y_val_kf, preds_kf)))

    print(f"Alpha {alpha:5} -> Mean RMSE: {np.mean(rmse_folds):.5f}  Std: {np.std(rmse_folds):.5f}")
```
შევარჩიე 10-ის ტოლი alpha.

**K-Fold შედეგები გამოსწორების შემდეგ (alpha=10):**

Fold 1 rmse = 0.12788 
Fold 2 rmse = 0.13468  
Fold 3 rmse = 0.12268
Fold 4 rmse = 0.13162
Fold 5 rmse = 0.10840
Mean - 0.12505
Std - 0.00924

!!!მივიღე ბევრად სტაბილური მოდელი

```python
with mlflow.start_run(run_name="kfold_cv_ridge_manual"):
    mlflow.log_param("model", "Ridge")
    mlflow.log_param("alpha", 10)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("feature_selection", "manual_drop")
    mlflow.log_metric("cv_mean_rmse", np.mean(rmse_folds))
    mlflow.log_metric("cv_std_rmse", np.std(rmse_folds))
```
### Overfitting / Underfitting ანალიზი

თითოეული მოდელისთვის შევადარეთ Train და Validation RMSE:

| მოდელი | Train RMSE | Val RMSE | Gap | სტატუსი |
|---|---|---|---|---|   
| LinearRegression | 0.15237 | 0.16750 | +0.01513 | UNDERFIT |
| Ridge_alpha_10 | 0.19602 | 0.20779 | +0.01177 | UNDERFIT |
| Ridge_manual_drop | 0.13830 | 0.16538 | +0.02708 | OVERFIT |
| GradientBoosting | 0.06553 | 0.06897 | +0.00344 | OK |
| XGBoost | 0.03354 | 0.03387 | +0.00033 | OK |
| RandomForest | 0.05151 | 0.05060 | -0.00091 | OK |

**LinearRegression — UNDERFIT:**
რეგულარიზაციის გარეშე, მაღალგანზომილებიან მონაცემებზე (200+ სვეტი) მოდელი ვერ პოულობს
სტაბილურ პარამეტრებს. Val RMSE=0.167 — ზღვარს (0.16) ზემოთ.

**Ridge_alpha_10 — UNDERFIT:**
alpha=10 ძალიან ძლიერი რეგულარიზაციაა ამ feature set-ისთვის (ordinal mapping-ის გარეშე).
მოდელი ზედმეტად "ამარტივებს" და კარგავს პატერნებს — Train RMSE=0.196.

**Ridge_manual_drop — OVERFIT:**
alpha=0.1 მინიმალური რეგულარიზაციაა. gap=0.027 ზღვარს (0.02) სცდება — მოდელი
სატრენინგო მონაცემებს უკეთ "ერგება", ვიდრე ზოგადს. სწორედ ამის გამო
K-Fold-ში Fold 3-ზე მივიღეთ RMSE=0.227.

**GradientBoosting / XGBoost / RandomForest — OK:**
ხის მოდელები interaction features-ზე დატრენინგდა. Train და Val RMSE პრაქტიკულად იდენტურია, ანუ გვაქვს კარგი განზოგადება. 
თუმცა გასათვალისწინებელია, რომ ეს მოდელები დამატებითი hyperparameter ოპტიმიზაციის გარეშეა და Kaggle-ზე
შედეგი შეიძლება განსხვავდებოდეს.

**საბოლოო არჩევანი — Ridge(alpha=10) + outlier removal + K-Fold:**
სუფთა feature set-ზე (ordinal mapping + manual drop) და alpha=10-ით K-Fold Mean RMSE=0.12505,
Std=0.00924 — ყველაზე სტაბილური და განზოგადებადი მოდელი.


### საბოლოო მოდელის შერჩევა და შენახვა

K-Fold-ის შემდეგ მოდელი დავატრენინგე სრულად გაწმენდილ მონაცემებზე (80/20 split-ის ნაცვლად), რადგან validation K-Fold-ით უკვე გავაკეთეთ:

```python
from sklearn.linear_model import Ridge

# სრულად გაწმენდილი მონაცემები 1458 სტრიქონი (outlier-ების გარეშე)
X_full = df_final.drop("SalePrice", axis=1)
X_full = X_full.drop(columns=drop_cols, errors="ignore")
y_full = np.log1p(df_final["SalePrice"])

final_model = Ridge(alpha=10.0)
final_model.fit(X_full, y_full)
```

MLflow-ში რეგისტრაცია:

```python
with mlflow.start_run(run_name="ridge_final_model"):
    mlflow.log_param("model", "Ridge")
    mlflow.log_param("alpha", 10)
    mlflow.log_metric("rmse", rmse)
    
    mlflow.sklearn.log_model(
        final_model,
        name="ridge_model",
        registered_model_name="RidgeHousePrices"
    )
    
    joblib.dump(list(X_full.columns), "model_columns.pkl") #ტრენინგის სვეტები
    mlflow.log_artifact("model_columns.pkl")
```

### საბოლოო არჩევანის დასაბუთება

Ridge(alpha=10) + manual drop შეირჩა შემდეგი მიზეზებით:
- ყველაზე დაბალი K-Fold Mean RMSE: 0.12505
- ყველაზე სტაბილური Std: 0.00924 — მოდელი თანმიმდევრულად კარგია ნებისმიერ data split-ზე
- alpha=10 outlier-ების მიმართ გამძლეა alpha=0.1-თან შედარებით
- სრული მონაცემებით ტრენინგი მოდელს მეტი მაგალითს აძლევს

## MLflow Tracking

DagsHub-ის ბმული: `https://dagshub.com/smama23/MLassignment1`
MLflow ექსპერიმენტების ბმული: `https://dagshub.com/smama23/MLassignment1.mlflow/#/experiments`

MLflow ინიციალიზაცია:

```python
!pip install dagshub mlflow
import dagshub
dagshub.init(repo_owner='smama23', repo_name='MLassignment1', mlflow=True)
import mlflow
```

### ჩაწერილი Run-ები და მეტრიკები

`cleaning_v1` missing_strategy, LotFrontage, Electrical, remaining_missing=0
`encoding_v1` ordinal_quality_mapping_0_to_5
`encoding_v2_onehot` one_hot_all_nominal, num_features
`baseline_linear_regression` LinearRegression, features count, ~0.175
`ridge_alpha_10` Ridge alpha=10, ~0.135
`ridge_best_alpha_0_1` Ridge, alpha=0.1, 0.12409
`manual_feature_drop_ridge` Ridge, alpha=0.1, dropped 5 columnss, ~0.124
`ridge_interaction_features_v3` Ridge, alpha=0.1, 7 interaction features, tracked
`gradient_boosting_v1` GradientBoostingRegressor, n_est=300, lr=0.05, depth=3, tracked
`xgboost_v1` XGBRegressor, n_est=500, lr=0.05, depth=4, sub=0.8, tracked
`random_forest_v1` RandomForestRegressor, n_est=400, depth=None, tracked
`kfold_cv_ridge_manual` Ridge, alpha=10, 5-fold CV, cv_mean=0.12505, cv_std=0.00924
`ridge_manual_drop_final` Ridge, alpha=10, full data, cv_mean_rmse=0.12505
`lasso_feature_selection` Lasso alpha=0.001, 80 features selected from 263, Ridge RMSE logged
`overfit_underfit_analysis` train/val RMSE gap per model, status logged per model

### საუკეთესო მოდელის შედეგები

K-Fold Mean RMSE = 0.12505
K-Fold Std RMSE = 0.00924
მოდელი: Ridge Regression(alpha=10)
Feature Selection : Manual drop (5 სვეტი)
Training Data : სრული გაწმენდილი dataset (outlier-ების გარეშე)
შედეგი Kaggle-ზე საბმიშენის შემდეგ: Score: 0.13507