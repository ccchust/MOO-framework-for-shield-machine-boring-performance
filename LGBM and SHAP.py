import pandas as pd
import lightgbm as lgb
import shap 
import matplotlib.pyplot as plt

X = pd.read_excel('path')
y = pd.read_excel('path')

lgb_model = lgb.LGBMRegressor(
    learning_rate=0.05,
    max_depth=2,
    reg_alpha=0.5,          
    reg_lambda=5,           
    n_estimators=200,
    min_child_samples=10,   
    boosting_type='gbdt'    
)


lgb_model.fit(X, y)
importance = lgb_model.feature_importances_
feature_names = X.columns.tolist()
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
importance_df = importance_df.sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Feature Importance Score')
plt.title('LightGBM Feature Importance')
plt.tight_layout()
plt.savefig('')
plt.close()

explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, plot_type="dot", show=False)
plt.title('SHAP Feature Impact (Dot Plot)')
plt.tight_layout()
plt.savefig('')
plt.close()

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title('SHAP Feature Importance (Bar Plot)')
plt.tight_layout()
plt.savefig('')
plt.close()
