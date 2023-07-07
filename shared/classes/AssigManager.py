from typing import List, Dict
from tqdm import tqdm
from shared.classes.AssigPipeline import AssigPipeline
from shared.utils.timer import Timer
from dataclasses import dataclass, field

@dataclass
class AssigManager(object):
    models: Dict[str, AssigPipeline] = field(default_factory=dict)
    fitted_acr_models: List[str]     = field(default_factory=list)
    fit_time: float = None

    def __getitem__(self, acr):
        return self.models[acr]
    
    def __iter__(self):
        for model in self.models.values():
            yield model
    
    def add_model(self, acr, model):
        self.models[acr] = model
    
    def fit(self, acrlst, X_train):
        timer = Timer()
        timer.start()
        for acr in tqdm(acrlst, desc="Entrenant els models"):
            self.models[acr].fit(X_train)
            self.fitted_acr_models.append(acr)
        self.fit_time = timer.elapsed_time()
        timer.reset()