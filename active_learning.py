import numpy as np
from alipy import ToolBox
import copy


def create_active_learning_dataset(data, labels):
    # Define X and y target
    X = data.values
    y = np.asarray(labels)
    print(X.shape, y.shape)

    # Create alibox tool box
    alibox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path='.')
    # Define number of rounds/splits
    rounds = 1
    # Split data to train and test and keep only 0.01 of the original data as labeled
    alibox.split_AL(test_ratio=0.3, initial_label_rate=0.1, split_count=rounds)  # Define 5 split counts

    # Create Logistic Regression model ( Default Setting)
    model = alibox.get_default_model()

    # Use uncertainty sampling as query strategy
    strategy = alibox.get_query_strategy(strategy_name='QueryInstanceUncertainty')
    # Create array to save the results
    total_examples = []
    # Set stopping criterion, we will stop in 1000 labeled examples
    stopping_criterion = alibox.get_stopping_criterion('num_of_queries', 10)

    for i in range(rounds):
        # Split the data in each round
        train_idx, test_idx, labeled_idx, unlabeled_idx = alibox.get_split(i)
        # Save the result of the round
        saver = alibox.get_stateio(i)
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
            example = strategy.select(labeled_idx, unlabeled_idx, model=model, batch_size=1)
            # Update the label idxs
            labeled_idx.update(example)
            unlabeled_idx.difference_update(example)
            # Train model for the added example
            model.fit(X=X[labeled_idx.index, :], y=y[labeled_idx.index])
            y_pred = model.predict(X[test_idx, :])
            accuracy = alibox.calc_performance_metric(y_true=y[test_idx], y_pred=y_pred, performance_metric='accuracy_score')

            # Save update results
            state = alibox.State(select_index=example, performance=accuracy)
            saver.add_state(state)
            saver.save()

            # Update prgress for stopping criterion
            stopping_criterion.update_information(saver)
            print(y[labeled_idx.index])

        stopping_criterion.reset()
        total_examples.append(copy.deepcopy(saver))
        # Save selected x_train examples
        X_train = X[labeled_idx, :]
        # Save labels for the examples
        y_train = y[labeled_idx, :]
        # Reshape target
        y_train = np.array(y_train).reshape(-1)

        # print(X_train.shape, y_train.shape)

    # Plot learning curves
    # experiment_analyser = alibox.get_experiment_analyser(x_axis='num_of_queries')
    # experiment_analyser.add_method(method_name='uncertainty', method_results=total_examples)
    # print(experiment_analyser)
    # experiment_analyser.plot_learning_curves(title='Active Learning Uncertainty Sampling Learning Curves', std_area=True)
    return X_train, y_train