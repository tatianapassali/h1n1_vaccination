import numpy as np
from alipy import ToolBox
import copy
from sklearn.ensemble import RandomForestClassifier
from data_loader import preprocess_dataframe
from alipy.experiment import ExperimentAnalyser


def create_dataset_splits(data, labels):
    # Define X and y target
    X = data.values
    y = np.asarray(labels)
    print(X.shape, y.shape)
    # Create alibox tool box
    toolbox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path='.')
    # Split data to train and test and keep only 0.01 of the original data as labeled
    toolbox.split_AL(test_ratio=0.15, initial_label_rate=0.1)
    return X,y,toolbox  # Return toolbox with the splits


def create_and_implement_strategy(strategy_name, X, y, alibox):
    # Create Logistic Regression model ( Default Setting)
    model = alibox.get_default_model()
    # Create Random Forest Classifier
    # model = RandomForestClassifier(n_estimators=750, random_state=1, max_depth=5)
    # Use uncertainty sampling as query strategy
    uncertainty_strategy = alibox.get_query_strategy(strategy_name=strategy_name)
    # random_strategy = alibox.get_query_strategy(strategy_name='QueryInstanceRandom')
    # qbc_strategy = alibox.get_query_strategy(strategy_name='QueryInstanceQBC')
    # Create array to save the uncertainty sampling results
    examples = []
    # Set stopping criterion, we will stop in 1000 labeled examples
    stopping_criterion = alibox.get_stopping_criterion('num_of_queries', 500)
    # Split the data in each round
    train_idx, test_idx, labeled_idx, unlabeled_idx = alibox.get_split(0)
    # Save the result of the round
    saver = alibox.get_stateio(0)
    print(train_idx.shape, test_idx.shape)
    # Starting with some labeled examples
    model.fit(X=X[labeled_idx.index, :], y=y[labeled_idx.index])
    y_pred = model.predict(X[test_idx, :])  # and try to predict some others
    # Calculate the accuracy of the prediction
    accuracy = alibox.calc_performance_metric(y_true=y[test_idx], y_pred=y_pred, performance_metric='accuracy_score')
    # Save accuracy of the prediction
    saver.set_initial_point(accuracy)

    while not stopping_criterion.is_stop():
        # Select example of the unlabeled dataset
        example = uncertainty_strategy.select(labeled_idx, unlabeled_idx, model=model, batch_size=1)
        # Update the label idxs
        labeled_idx.update(example)
        unlabeled_idx.difference_update(example)
        # Train model for the added example
        model.fit(X=X[labeled_idx.index, :], y=y[labeled_idx.index])
        y_pred = model.predict(X[test_idx, :])
        accuracy = alibox.calc_performance_metric(y_true=y[test_idx], y_pred=y_pred, performance_metric='accuracy_score')
        # f1 = alibox.calc_performance_metric(y_true=y[test_idx], y_pred=y_pred, performance_metric='f1_score')

        # Save update results
        state = alibox.State(select_index=example, performance=accuracy)
        saver.add_state(state)
        saver.save()

        # Update prgress for stopping criterion
        stopping_criterion.update_information(saver)

    stopping_criterion.reset()
    examples.append(copy.deepcopy(saver))

    # # Save selected x_train examples
    # X_train = X[labeled_idx, :]
    # # Save labels for the examples
    # y_train = y[labeled_idx, :]
    # # Reshape target
    # y_train = np.array(y_train).reshape(-1)
    return examples


def plot_learning_curves(strategy1, strategy2,strategy3):
    # Plot learning curves
    experiment_analyser = ExperimentAnalyser(x_axis='num_of_queries')
    experiment_analyser.add_method(method_name='uncertainty', method_results=strategy1)
    experiment_analyser.add_method(method_name='random', method_results=strategy2)
    experiment_analyser.add_method(method_name='qbc',method_results=strategy3)
    print(experiment_analyser)
    experiment_analyser.plot_learning_curves(title='Learning Curves', std_area=True)





