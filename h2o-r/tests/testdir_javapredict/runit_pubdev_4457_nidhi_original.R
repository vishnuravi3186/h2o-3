setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../scripts/h2o-r-test-setup.R")

#----------------------------------------------------------------------
# Test and make sure h2o predict is working properly as specified
# in PUBDEV-4457.  Runit test derived from Nidhi template.
#----------------------------------------------------------------------
test <-
  function() {
    h2o.removeAll()
    set.seed(12345)
    n = 100000

    true_prob_train = runif(n)
    true_prob_test = runif(n)

    # More complex data, setting up myvar to break scoring
    df_train <- data.frame(response = rbinom(n, size = 1, true_prob_train),
    true_prob = true_prob_train,
    myvar =c(rep(-0.0935, n/2), rep(-0.094, n/2)),
    matrix(runif(n*100), n, 100),
    matrix(rnorm(n*100), n, 100))
    df_test <- data.frame(response = rbinom(n, size = 1, true_prob_test),
    true_prob = true_prob_test,
    myvar =c(rep(-0.09375, n)),
    matrix(runif(n*100), n, 100),
    matrix(rnorm(n*100), n, 100))

    df_test_v2 <- data.frame(response = rbinom(n, size = 1, true_prob_test),
    true_prob = true_prob_test,
    myvar =c(rep(-0.09374999, n)),
    matrix(runif(n*100), n, 100),
    matrix(rnorm(n*100), n, 100))

    write.csv(df_train, "./df_train.csv", row.names = F)
    write.csv(df_test, "./df_test.csv", row.names = F)
    write.csv(df_test_v2, "./df_test_v2.csv", row.names = F)

    #### Read data in H2O ####
    train.hex <- h2o.importFile(path = paste0(getwd(), "/df_train.csv"),
    destination_frame = "train.hex")
    test.hex <- h2o.importFile(path = paste0(getwd(), "/df_test.csv"),
    destination_frame = "test.hex")
    test_v2.hex <- h2o.importFile(path = paste0(getwd(), "/df_test_v2.csv"),
    destination_frame = "test.hex")

    #### Train h2o model ####
    predictors <- setdiff(colnames(train.hex), "response")

    # Building grid, although it will be a single model...
    ntrees_opts <- 50
    max_depth_opts <- 50
    min_rows_opts <- 100
    learn_rate_opts <- 0.01
    sample_rate_opts <- 1
    col_sample_rate_opts <- 1
    col_sample_rate_per_tree_opts <- 1

    hyper_params_experiment  = list( ntrees = ntrees_opts,
    max_depth = max_depth_opts,
    min_rows = min_rows_opts,
    learn_rate = learn_rate_opts,
    sample_rate = sample_rate_opts,
    col_sample_rate = col_sample_rate_opts,
    col_sample_rate_per_tree = col_sample_rate_per_tree_opts
    )

    GRID_ID = "pojo_mismatch_3"
    # LOG:

    grid_pojo_experiment <- h2o.grid(
    algorithm = "gbm",
    grid_id = GRID_ID,
    x = predictors,
    y = "response",
    training_frame = train.hex,
    seed = 12345,
    hyper_params = hyper_params_experiment
    )

    grid_pojo_experiment <- h2o.getGrid(grid_id = GRID_ID)
    model_pojo_experiment <- lapply(grid_pojo_experiment@model_ids, h2o.getModel)[[1]]
    h2o.saveModel(model_pojo_experiment, getwd())
    h2o.download_pojo(model_pojo_experiment, get_jar = T, path = getwd())

    #### Compare predictions for test data ####
    # Prediction from the model
    pred_test_fromModel <- h2o.predict(model_pojo_experiment, test.hex)  # test data 
    pred_train_fromModel <- h2o.predict(model_pojo_experiment, train.hex)  # train data
    pred_test_fromModel <- as.data.frame(pred_test_fromModel)
    pred_train_fromModel <- as.data.frame(pred_train_fromModel)
    write.csv(pred_test_fromModel, "./h2oPredictTest.csv", quote=FALSE, row.names=FALSE)
    write.csv(pred_train_fromModel, "./h2oPredictTrain.csv", quote=FALSE, row.names=FALSE)
    head(pred_test_fromModel)

    browser()
    
    system("javac -cp h2o-genmodel.jar -J-Xmx32g PredictCsv.java pojo_mismatch_3_model_0.java")
    system("java -ea -cp :h2o-genmodel.jar hex.genmodel.tools.PredictCsv --header --model pojo_mismatch_3_model_0 --input ./df_test.csv --output ./pred_test_viaPojo.csv --decimal") # test data
    system("java -ea -cp :h2o-genmodel.jar hex.genmodel.tools.PredictCsv --header --model pojo_mismatch_3_model_0 --input ./df_train.csv --output ./pred_train_viaPojo.csv --decimal")  # train data
    
    pred_test_viaPojo <- read.csv("pred_test_viaPojo.csv")
    pred_train_viaPojo <- read.csv("pred_train_viaPojo.csv")

    compareFrames(pred_train_viaPojo, pred_train_fromModel, 1e-6)
    compareFrames(pred_test_viaPojo, pred_test_fromModel, 1e-6)
  }

compareFrames<-function(f1, f2, tol) {
  sameNum = sum(abs(f1$predict-f2$predict) <tol)
  diffNum = length(f1)-sameNum
  print(paste("Number of elements that differ more than tolerance is ", diffNum, sep=' '))
  summary(f1$predict-f2$predict)
}

doTest("pubdev-4457: PredictCsv test", test)