import sys
import numpy as np

def test():
    from catfish_core import load_dataset, engineer_features, prepare_features, KMeansClassifier
    import warnings
    warnings.filterwarnings('ignore')
    
    df = load_dataset()
    df = engineer_features(df)
    X, y = prepare_features(df)
    
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_sc, y_train)
    
    km = KMeansClassifier(n_clusters=30, random_state=42)
    km.fit(X_train_bal, y_train_bal)
    
    fracs = {k: round(v, 3) for k, v in km.cluster_catfish_frac_.items()}
    print("Max fraction:", max(fracs.values()))
    
if __name__ == '__main__':
    test()
