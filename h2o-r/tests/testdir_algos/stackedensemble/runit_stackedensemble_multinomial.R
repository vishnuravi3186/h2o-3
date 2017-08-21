setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../../scripts/h2o-r-test-setup.R")

stackedensemble.multinomial.test <- function() {

    # This test checks the following (for multinomial regression):
    #
    # 1) That h2o.stackedEnsemble executes w/o errors
    #    on a 3-model "manually constucted" ensemble.
    # 2) That h2o.predict works on a stack
    # 3) That h2o.performance works on a stack
    # 4) That the training and test performance is
    #    better on a ensemble vs the base learners.
    # 5) That the validation_frame arg on
    #    h2o.stackedEnsemble works correctly


    train_file <- locate("bigdata/laptop/mnist/train.csv.gz")
    test_file <- locate("bigdata/laptop/mnist/test.csv.gz")
    train <- h2o.importFile(train_file)
    test <- h2o.importFile(test_file)
    y <- "C785"
    x <- setdiff(names(train), y)
    train[,y] <- as.factor(train[,y])
    test[,y] <- as.factor(test[,y])
    train <- train[1:5000,]
    test <- test[1:5000,]
    # Number of CV folds (to generate level-one data for stacking)
    nfolds <- 2


    # Train & Cross-validate a GBM
    my_gbm <- h2o.gbm(x = x,
                        y = y,
                        training_frame = train,
                        distribution = "multinomial",
                        max_depth = 3,
                        learn_rate = 0.2,
                        nfolds = nfolds,
                        fold_assignment = "Modulo",
                        keep_cross_validation_predictions = TRUE,
                        seed = 1)
                        # Eval perf
                        perf_gbm_train <- h2o.performance(my_gbm)
                        perf_gbm_test <- h2o.performance(my_gbm, newdata = test)
                        print("GBM training performance: ")
                        print(perf_gbm_train)
                        print("GBM test performance: ")
                        print(perf_gbm_test)


    # Train & Cross-validate a RF
    my_rf <- h2o.randomForest(x = x,
                                y = y,
                                training_frame = train,
                                ntrees = 30,
                                nfolds = nfolds,
                                fold_assignment = "Modulo",
                                keep_cross_validation_predictions = TRUE,
                                seed = 1)
                                # Eval perf
                                perf_rf_train <- h2o.performance(my_rf)
                                perf_rf_test <- h2o.performance(my_rf, newdata = test)
                                print("RF training performance: ")
                                print(perf_rf_train)
                                print("RF test performance: ")
                                print(perf_rf_test)


    # Train & Cross-validate a extremely-randomized RF
    my_xrf <- h2o.randomForest(x = x,
                                y = y,
                                training_frame = train,
                                ntrees = 50,
                                histogram_type = "Random",
                                nfolds = nfolds,
                                fold_assignment = "Modulo",
                                keep_cross_validation_predictions = TRUE,
                                seed = 1)
                                # Eval perf
                                perf_xrf_train <- h2o.performance(my_xrf)
                                perf_xrf_test <- h2o.performance(my_xrf, newdata = test)
                                print("XRF training performance: ")
                                print(perf_xrf_train)
                                print("XRF test performance: ")
                                print(perf_xrf_test)


    print("StackedEnsemble 1 of 2")
    # Train a stacked ensemble using the GBM and GLM above
    stack <- h2o.stackedEnsemble(x = x,
                                y = y,
                                training_frame = train,
                                validation_frame = test,  #also test that validation_frame is working
                                model_id = "my_ensemble_multinomial",
                                base_models = list(my_gbm@model_id, my_rf@model_id, my_xrf@model_id))

    # Check that prediction works
    pred <- h2o.predict(stack, newdata = test)
    expect_equal(nrow(pred), nrow(test))
    expect_equal(ncol(pred), 11)

    # And again
    pred <- h2o.predict(stack, newdata = test)
    expect_equal(nrow(pred), nrow(test))
    expect_equal(ncol(pred), 11)

    # Eval ensemble perf
    perf_stack_train <- h2o.performance(stack)
    perf_stack_test <- h2o.performance(stack, newdata = test)

    # And again
    perf_stack_train <- h2o.performance(stack)
    perf_stack_test <- h2o.performance(stack, newdata = test)

    print("Perf on Train for 1st Stacked Ensemble")
    print(perf_stack_train)
    print("Perf on Test for 1st Stacked Ensemble")
    print(perf_stack_test)

    # And again:
    print("StackedEnsemble 2 of 2")
    stack <- h2o.stackedEnsemble(x = x,
                                y = y,
                                training_frame = train,
                                validation_frame = test,  #also test that validation_frame is working
                                model_id = "my_ensemble_multinomial",
                                base_models = list(my_gbm@model_id, my_rf@model_id, my_xrf@model_id))

    pred <- h2o.predict(stack, newdata = test)
    expect_equal(nrow(pred), nrow(test))
    expect_equal(ncol(pred), 11)

    perf_stack_train <- h2o.performance(stack)
    perf_stack_test <- h2o.performance(stack, newdata = test)
    print("Perf on Train for 2nd Stacked Ensemble")
    print(perf_stack_train)
    print("Perf on Test for 2nd Stacked Ensemble")
    print(perf_stack_test)

    # Training mean_per_class_error for each base learner
    baselearner_best_mean_per_class_error_train <- min(h2o.mean_per_class_error(perf_gbm_train), h2o.mean_per_class_error(perf_rf_train), h2o.mean_per_class_error(perf_xrf_train))
    stack_mean_per_class_error_train <- h2o.mean_per_class_error(perf_stack_train)
    print(sprintf("Best Base-learner Training mean_per_class_error:  %s", baselearner_best_mean_per_class_error_train))
    print(sprintf("Ensemble Training mean_per_class_error:  %s", stack_mean_per_class_error_train))
    expect_equal(TRUE,stack_mean_per_class_error_train <= baselearner_best_mean_per_class_error_train)  #Does not pass in this example, but this is ok

    # Check that stack perf is better (smaller) than the best (smaller) base learner perf:
    # Test mean_per_class_error for each base learner
    baselearner_best_mean_per_class_error_test <- min(h2o.mean_per_class_error(perf_gbm_test), h2o.mean_per_class_error(perf_rf_test), h2o.mean_per_class_error(perf_xrf_test))
    stack_mean_per_class_error_test <- h2o.mean_per_class_error(perf_stack_test)
    print(sprintf("Best Base-learner Test mean_per_class_error:  %s", baselearner_best_mean_per_class_error_test))
    print(sprintf("Ensemble Test mean_per_class_error:  %s", stack_mean_per_class_error_test))
    expect_equal(TRUE, stack_mean_per_class_error_test <= baselearner_best_mean_per_class_error_test)

    # Check that passing `test` as a validation_frame
    # produces the same metrics as h2o.performance(stack, test)
    # Since the metrics object is not exactly the same, we can just test that mean_per_class_error is the same
    perf_stack_validation_frame <- h2o.performance(stack, valid = TRUE)
    expect_identical(stack_mean_per_class_error_test, h2o.mean_per_class_error(perf_stack_validation_frame))

}

doTest("Stacked Ensemble Multinomial Test", stackedensemble.multinomial.test)
