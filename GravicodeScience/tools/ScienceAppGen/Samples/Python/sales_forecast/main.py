"""Forecast daily store sales from price, promotions, season and weather."""
from sklearn.model_selection import train_test_split

from data_loader import load_sales, summarize
from features import FEATURES, build_features, scale, weekend_share
from model import DemandModel
from plots import plot_forecast, plot_importances


def main():
    df = load_sales()
    print(f"rows: {len(df)}, stores: {df['store'].nunique()}")
    print(f"weekend share: {weekend_share(df):.2f}")
    print("mean sales by store:")
    for store, mean in summarize(df).items():
        print(f"  {store}: {mean:.1f}")

    x, y = build_features(df)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=42)
    train_x, test_x = scale(train_x, test_x)

    model = DemandModel(n_trees=200).fit(train_x, train_y)
    for name, scores in model.evaluate(test_x, test_y).items():
        print(f"{name:7s} MAE {scores['mae']:.2f}  R2 {scores['r2']:.3f}")

    print("feature importance:")
    pairs = model.importances(FEATURES)
    for feature, weight in pairs:
        print(f"  {feature:12s} {weight:.3f}")

    print("saved", plot_forecast(test_y[:60], model.forest.predict(test_x)[:60]))
    print("saved", plot_importances(pairs))


if __name__ == "__main__":
    main()
