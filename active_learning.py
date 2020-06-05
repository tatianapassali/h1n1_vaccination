import pickle
import numpy as np
from alipy import ToolBox
import copy
from alipy.experiment import ExperimentAnalyser, StateIO


def create_dataset_splits(data, labels):
    # Define X and y target
    X = data.values
    y = np.asarray(labels)

    # Create alibox tool box
    toolbox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path='.')

    # Split data to train and test and keep only 0.01 of the original data as labeled
    toolbox.split_AL(test_ratio=0.15, initial_label_rate=0.1)
    train_idx, test_idx, labeled_idx, unlabeled_idx = toolbox.get_split(0)

    X_train = X[train_idx]
    y_train = y[train_idx]
    y_train = np.array(y_train).reshape(-1)
    X_test = X[test_idx]
    y_test = y[test_idx]
    y_test = np.array(y_test).reshape(-1)

    # Save dataset splits
    with open('dataset','wb') as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)

    # Save dataset splits indexes for active learning
    with open('dataset_al', 'wb') as f:
        pickle.dump((train_idx, test_idx, labeled_idx, unlabeled_idx), f)


def create_and_implement_strategy(strategy_name, data, labels, queries):

    # Keep only the values of data and labels dataframe (Later, we use the global split based on idxs)
    X = data.values
    y = np.asarray(labels)
    toolbox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path='.')

    # Create Logistic Regression model ( Default Setting with liblinear solver)
    model = toolbox.get_default_model()

    # Implement query strategy
    uncertainty_strategy = toolbox.get_query_strategy(strategy_name=strategy_name)

    # Create array to save the results
    examples = []

    # Set stopping criterion, we will stop in 1000 labeled examples
    stopping_criterion = toolbox.get_stopping_criterion('num_of_queries', queries)

    # Get the indexes of the global split
    with open("dataset_al", "rb") as f:
        train_idx, test_idx, labeled_idx, unlabeled_idx = pickle.load(f)

    # Create saver to save the results
    saver = StateIO(round=0, train_idx=train_idx,
                    test_idx=test_idx, init_L=labeled_idx,
                    init_U=unlabeled_idx, saving_path='.')

    # print(train_idx.shape, test_idx.shape)

    # Starting with some labeled examples
    model.fit(X=X[labeled_idx.index, :], y=y[labeled_idx.index])
    y_pred = model.predict(X[test_idx, :])

    # Calculate the accuracy of the prediction
    accuracy = toolbox.calc_performance_metric(y_true=y[test_idx], y_pred=y_pred, performance_metric='accuracy_score')

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
        # Calculate accuracy
        accuracy = toolbox.calc_performance_metric(y_true=y[test_idx], y_pred=y_pred,
                                                   performance_metric='accuracy_score')
        # f1 = alibox.calc_performance_metric(y_true=y[test_idx], y_pred=y_pred, performance_metric='f1_score')

        # Save update results
        state = toolbox.State(select_index=example, performance=accuracy)
        saver.add_state(state)
        saver.save()

        # Update progress for stopping criterion
        stopping_criterion.update_information(saver)

    stopping_criterion.reset()
    examples.append(copy.deepcopy(saver))

    # Uncomment and return in ordet to save the new active learning dataset
    # Save selected x_train examples
    # X_train = X[labeled_idx, :]
    # Save labels for the examples
    # y_train = y[labeled_idx, :]
    # Reshape target
    # y_train = np.array(y_train).reshape(-1)

    return examples


# Plot learning curves
def plot_learning_curves(strategy1, strategy2, strategy3):
    # Create Experiment Analyser to plot learning curves for the implemented strategies
    experiment_analyser = ExperimentAnalyser(x_axis='num_of_queries')
    # Add methods to the experiment analyser
    experiment_analyser.add_method(method_name='uncertainty', method_results=strategy1)
    experiment_analyser.add_method(method_name='random', method_results=strategy2)
    experiment_analyser.add_method(method_name='qbc', method_results=strategy3)
    # print(experiment_analyser)
    # Plot the learning curves
    experiment_analyser.plot_learning_curves(title='Learning Curves', std_area=True)
