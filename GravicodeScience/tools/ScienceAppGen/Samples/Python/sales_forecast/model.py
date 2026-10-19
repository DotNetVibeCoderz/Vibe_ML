"""The demand model: a random forest against a linear baseline."""
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


class DemandModel:
    def __init__(self, n_trees=200, seed=42):
        self.forest = RandomForestRegressor(n_estimators=n_trees, random_state=seed)
        self.baseline = LinearRegression()

    def fit(self, x, y):
        self.forest.fit(x, y)
        self.baseline.fit(x, y)
        return self

    def evaluate(self, x, y):
        results = {}
        for name, model in [("forest", self.forest), ("linear", self.baseline)]:
            predicted = model.predict(x)
            results[name] = {
                "mae": mean_absolute_error(y, predicted),
                "r2": r2_score(y, predicted),
            }
        return results

    def importances(self, names):
        return sorted(zip(names, self.forest.feature_importances_), key=lambda p: p[1], reverse=True)
