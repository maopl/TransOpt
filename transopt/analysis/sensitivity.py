import shap
import numpy as np
import xgboost as xgb
from lime import lime_tabular
from typing import Dict

def sensitivity_analysis(X, y, nodes, method='LIME') -> Dict[str, float]:

    if method.upper() == 'SHAP':
        model = xgb.XGBRegressor()
        model.fit(X, y)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        importance_scores = np.abs(shap_values).mean(axis=0)
        
        importance_dict = {}
        for node, score in zip(nodes, importance_scores):
            importance_dict[node] = float(score)
            
        return importance_dict
    
    elif method.upper() == 'LIME':
        # 训练一个基础模型
        model = xgb.XGBRegressor()
        model.fit(X, y)
        
        # 创建LIME解释器
        explainer = lime_tabular.LimeTabularExplainer(
            X,
            feature_names=nodes,
            mode='regression',
            training_labels=y,
            random_state=42
        )
        
        # 计算所有样本的LIME重要性并取平均
        importance_dict = {node: 0.0 for node in nodes}
        n_samples = min(100, len(X))  # 使用前100个样本或全部样本
        
        for i in range(n_samples):
            explanation = explainer.explain_instance(
                X[i], 
                model.predict,
                num_features=len(nodes)
            )
            # 累加每个特征的重要性
            for feature, importance in explanation.local_exp[1]:
                importance_dict[nodes[feature]] += abs(importance)
        
        # 计算平均重要性
        for node in importance_dict:
            importance_dict[node] = float(importance_dict[node] / n_samples)
            
        return importance_dict
    
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'SHAP' or 'LIME'.")



