import mlflow.sklearn
import pickle
import numpy as np
import warnings
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from flask import Flask, current_app, render_template, render_template_string, request
from wtforms import Form, TextAreaField, validators
from sklearn.model_selection import GridSearchCV
from _preprocess import *
from _loaders import *

app = Flask(__name__, template_folder='templates', static_folder='static')

# load the model
final_model = pickle.load(open('best_pipe.pkl', 'rb'))
transformer = pickle.load(open('tfidf.pkl', 'rb'))


def classify(tweet):
    label = {0: 'Seems fine!', 1: 'Toxic!'}
    X = transformer.transform([preprocessor(t) for t in [tweet]])
    y = final_model.predict(X)[0]
    proba = np.max(final_model.predict_proba(X))
    return label[y], proba


class ReviewForm(Form):
    tweet_classify = TextAreaField('', [validators.DataRequired(), validators.length(min=5)])


@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template_string("{% extends 'index.html' %}",
                                  form=form)


@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['tweet_classify']
        y, proba = classify(review)
        return render_template_string("{% extends 'results.html' %}",
                               content=review,
                               prediction=y,
                               probability=round(proba * 100, 2))
    return render_template_string("{% extends 'index.html' %}", form=form)


if __name__ == '__main__':

    warnings.filterwarnings("ignore")
    np.random.seed(np.random.randint(1, 50000))
    MLFLOW_SERVER_URI = 'http://web:5000'

    client = mlflow.tracking.MlflowClient(MLFLOW_SERVER_URI)
    mlflow.set_tracking_uri(MLFLOW_SERVER_URI)

    EXP_NAME = "Experiment-" + str(np.random.randint(1, 1000))
    EXP_ID = mlflow.create_experiment(EXP_NAME)
    print(f'Experiment created: {EXP_NAME} with ID: {EXP_ID}')
    mlflow.set_experiment(EXP_NAME)
    print(f'Training started...')
    all_classifiers = {'lr': lr_clf,
                       'sgd': sgd_clf,
                       'rf': rf_clf,
                       'xgb': xgb_clf,
                       'tree': tree_clf,
                       }

    best_models = {}
    accuracy_dict = {}
    for clf_name, clf in all_classifiers.items():
        with mlflow.start_run():
            # assert run.info.experiment_id == EXP_ID
            tfidf_clf_pipe = Pipeline([('vect', tfidf), ('clf', clf)])
            tfidf_clf_pipe_gs = GridSearchCV(tfidf_clf_pipe,
                                             param_grid,
                                             scoring='accuracy',
                                             verbose=0,
                                             cv=3,
                                             n_jobs=-1)
            tfidf_clf_pipe_gs.fit(texts_train, y_train)
            best_model = tfidf_clf_pipe_gs.best_estimator_
            best_models[clf] = best_model
            y_pred = best_model.predict(texts_test)
            acc = accuracy_score(y_test, y_pred)
            accuracy_dict[clf_name] = acc
            print('classifier algorithm = %s' % clf_name)
            print("Number of mislabeled points out of a total %d points : %d" % (
                len(texts_test), (y_test != y_pred).sum()))
            print('Test Accuracy: %.3f' % acc)

            # mlflow stuff
            mlflow.log_param("Classifier", clf_name)
            mlflow.log_metric("Accuracy", acc)
            # mlflow.sklearn.log_model(final_model, "model")

    experiment = client.get_experiment_by_name(EXP_NAME)
    model_name = "Twitter-Classify" + str(np.random.randint(1, 1000))
    client.create_registered_model(model_name)

    # staging model
    run_info = client.search_runs(experiment.experiment_id)[0]
    result = client.create_model_version(
        name=model_name,
        source=f"{run_info.info.artifact_uri}/model",
        run_id=run_info.info.run_id
    )
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Staging"
    )
    # prod model
    run_info = client.search_runs(experiment.experiment_id)[-1]
    result = client.create_model_version(
        name=model_name,
        source=f"{run_info.info.artifact_uri}/model",
        run_id=run_info.info.run_id
    )
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Production"
    )

    current_staging = [v for v in client.search_model_versions(f"name='{model_name}'") if v.current_stage == 'Staging'][-1]
    current_staging_metrics = client.get_run(current_staging.run_id).data.metrics
    current_prod = [v for v in client.search_model_versions(f"name='{model_name}'") if v.current_stage == 'Production'][-1]
    prod_metrics = client.get_run(current_prod.run_id).data.metrics

    # Task 1
    for run_info in client.search_runs(experiment.experiment_id):
        run = mlflow.get_run(run_info.info.run_id)

        # Compare each experiment with prod
        current_metrics = client.get_run(run_info.info.run_id).data.metrics
        current_tags = client.get_run(run_info.info.run_id).data.tags

        client.set_tag(run_info.info.run_id, "compared_with", current_prod.version)

        if all(current_metrics[k] > v for k, v in prod_metrics.items() if k ==
                                                                          'Accuracy'):
            print(f'CURR: {current_metrics}')
            print(f'PROD: {prod_metrics}')
            client.set_tag(run_info.info.run_id, "staging", "rc")
        else:
            client.set_tag(run_info.info.run_id, "staging", "rejected")

    # find production id and metrics
    def get_production(client_):
        for mv in client_.search_model_versions(f"name='{model_name}'"):
            if dict(mv)['current_stage'] == 'Production':
                return mv

    prod_mv = get_production(client)
    print(f'Production ID: {prod_mv.run_id}')
    print(f'Production Version: {prod_mv.version}')

    # find staging id and metrics
    def get_staging(client_):
        for mv in client_.search_model_versions(f"name='{model_name}'"):
            if dict(mv)['current_stage'] == 'Staging':
                return mv

    staging_mv = get_staging(client)
    print(f'Staging ID: {staging_mv.run_id}')
    print(f'Staging Version: {staging_mv.version}')

    # Task 2
    metrics = {}

    for run_info in client.search_runs(experiment.experiment_id):
        run = mlflow.get_run(run_info.info.run_id)
        # for all models that pass the initial selection (if experiment is a release candidate)
        if 'rc' in run.data.tags['staging']:
            test_metrics = client.get_run(run_info.info.run_id).data.metrics
            print(f'TEST: {test_metrics}')
            print(f'PROD: {prod_metrics}')
            if all(test_metrics[k] > v for k, v in prod_metrics.items() if k ==
                                                                           'Accuracy'):
                client.set_tag(run_info.info.run_id, "staging", "rc")
                prod = client.create_model_version(
                    name=model_name,
                    source=f"{run_info.info.artifact_uri}/model",
                    run_id=run_info.info.run_id
                )
                client.transition_model_version_stage(
                    name=model_name,
                    version=prod.version,
                    stage="Production"
                )
                prod_metrics = client.get_run(prod.run_id).data.metrics
            metrics[run_info.info.run_id] = test_metrics.get('Accuracy')

    print(f'Production ID: {get_production(client).run_id}')
    print(f'Production Version: {get_production(client).version}', end="\n\n")

    print(f'Staging ID: {get_staging(client).run_id}')
    print(f'Staging Version: {get_staging(client).version}')

    print(f'Production Metrics: {prod_metrics}')
    print(f'Production Metrics: {mlflow.get_run(get_production(client).run_id).data.metrics}')

    for run_info in client.search_runs(experiment.experiment_id):
        for tag in ['staging', 'compared_with']:
            client.delete_tag(run_info.info.run_id, tag)

    print(f'Training finished...')
    print(f'Opening the web app...')
    app.run(debug=True, host="0.0.0.0", port=5001)
