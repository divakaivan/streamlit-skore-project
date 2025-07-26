from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from skore import EstimatorReport
from skore import Project
from pathlib import Path


def make_classifier_report():
    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = LogisticRegression(max_iter=10)
    return EstimatorReport(model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


def make_regressor_report():
    X, y = make_regression(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = LinearRegression()
    return EstimatorReport(model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


def get_project():
    project = Project("skore-streamlit", workspace=Path("."))
    project.put("simple-classifier", make_classifier_report())
    project.put("simple-regressor", make_regressor_report())
    return project
