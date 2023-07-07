from typing import List, Any
from dataclasses import dataclass
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, precision_score, f1_score



@dataclass
class AssigPipeline:
    id:      str
    
    # Dades de cada assignatura en particular
    y_train: List[Any]
    y_test:  List[Any]
    
    # Preprocessem les dades categòriques
    numerical_features:   List[str]
    categorical_features: List[str]
    
    # Model (Estimador) i Pipeline d'entrenament
    clf: BaseEstimator
    pl:  Pipeline   = None
    is_fitted: bool = False
    
    def __post_init__(self):        
        preprocessing = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown='ignore'), self.categorical_features),
                ("num", SimpleImputer(strategy='mean'), self.numerical_features),
            ]
        )
        
        self.pl = Pipeline([
            ("preprocessor", preprocessing),
            ("clf", self.clf),
        ])
    
    def __repr__(self) -> str:
        return f'Assig( {self.id}, {self.clf.__class__.__name__} )'
    
    def fit(self, X_train: List[Any]):
        self.pl.fit(X_train, self.y_train)
        self.is_fitted = True

    def get_feature_names(self):
        transformer  = self.pl.named_steps['preprocessor']
        num_features = self.numerical_features
        
        if self.categorical_features:
            cat_features = transformer.named_transformers_['cat'].get_feature_names_out(self.categorical_features).tolist()
        else:
            cat_features = []
        
        # NOTE: CRÍTIC!!!! Important mantenir el mateix ordre segons el prerpocessing (columnes categoriques sqguides de numeriques)
        # return num_features + cat_features
        return cat_features + num_features
    

    def score(self, X_test: List[Any]):
        y_pred = self.pl.predict(X_test)
        # return self.pl.score(X_test, self.y_test)
        return ( 
            recall_score(self.y_test, y_pred), 
            precision_score(self.y_test, y_pred),
            f1_score(self.y_test, y_pred),
            )