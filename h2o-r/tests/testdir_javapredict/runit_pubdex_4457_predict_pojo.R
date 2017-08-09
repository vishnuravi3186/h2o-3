setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../scripts/h2o-r-test-setup.R")

#----------------------------------------------------------------------
# Test and make sure h2o predict is working properly as specified
# in PUBDEV-4457.  Runit test derived from Nidhi template.
#----------------------------------------------------------------------
test <-
  function() {
    h2o.removeAll()
    #### Generate data, variable myvar which should break the model ####
    set.seed(12345)
    # n = 100000
    #
    # true_prob_train = runif(n)
    # true_prob_test = runif(n)
    #
    # # More complex data, setting up myvar to break scoring
    # df_train <- data.frame(response = rbinom(n, size = 1, true_prob_train),
    # true_prob = true_prob_train,
    # myvar =c(rep(-0.0935, n/2), rep(-0.094, n/2)),
    # matrix(runif(n*100), n, 100),
    # matrix(rnorm(n*100), n, 100))
    # df_test <- data.frame(response = rbinom(n, size = 1, true_prob_test),
    # true_prob = true_prob_test,
    # myvar =c(rep(-0.09375, n)),
    # matrix(runif(n*100), n, 100),
    # matrix(rnorm(n*100), n, 100))
    #
    # df_test_v2 <- data.frame(response = rbinom(n, size = 1, true_prob_test),
    # true_prob = true_prob_test,
    # myvar =c(rep(-0.09374999, n)),
    # matrix(runif(n*100), n, 100),
    # matrix(rnorm(n*100), n, 100))
    #
    #
    # write.csv(df_train, "./df_train.csv", row.names = F)
    # write.csv(df_test, "./df_test.csv", row.names = F)
    # write.csv(df_test_v2, "./df_test_v2.csv", row.names = F)

    #### Read data in H2O ####
    # train.hex <- h2o.importFile(locate("bigdata/laptop/jira/df_train.csv.zip"),
    # destination_frame = "train.hex")
    # test.hex <- h2o.importFile(locate("bigdata/laptop/jira/df_test.csv.zip"),
    # destination_frame = "test.hex")
    # test_v2.hex <- h2o.importFile(locate("bigdata/laptop/jira/df_test_v2.csv.zip"),
    # destination_frame = "test_v2.hex")

    train.hex <- h2o.importFile(path = paste0(getwd(), "/df_train.csv"),
    destination_frame = "train.hex")
    test.hex <- h2o.importFile(path = paste0(getwd(), "/df_test.csv"),
    destination_frame = "test.hex")
    test_v2.hex <- h2o.importFile(path = paste0(getwd(), "/df_test_v2.csv"),
    destination_frame = "test.hex")

    #### Train h2o model ####
    predictors <- setdiff(colnames(train.hex), "response")

    params <- list()
    params$ntrees <- 50
    params$max_depth <- 50
    params$x <- predictors
    params$y <- "response"
    params$training_frame <-train.hex
    params$learn_rate <- 0.01
    params$sample_rate <- 1
    params$col_sample_rate <- 1
    params$col_sample_rate_per_tree <- 1
    params$seed <- 12345
    
    browser()
    doJavapredictTest("gbm", locate("bigdata/laptop/jira/df_test.csv"), test.hex, params)
#    doJavapredictTest("gbm", locate("bigdata/laptop/jira/df_test_v2.csv"), test_v2.hex, params)   # pass with no error
    



# 
# 
#     # Building grid, although it will be a single model...
#     ntrees_opts <- 50
#     max_depth_opts <- 50
#     min_rows_opts <- 100
#     learn_rate_opts <- 0.01
#     sample_rate_opts <- 1
#     col_sample_rate_opts <- 1
#     col_sample_rate_per_tree_opts <- 1
# 
#     hyper_params_experiment  = list( ntrees = ntrees_opts,
#     max_depth = max_depth_opts,
#     min_rows = min_rows_opts,
#     learn_rate = learn_rate_opts,
#     sample_rate = sample_rate_opts,
#     col_sample_rate = col_sample_rate_opts,
#     col_sample_rate_per_tree = col_sample_rate_per_tree_opts
#     )
# 
#     GRID_ID = "pojo_mismatch_3"
#     # LOG:
# 
#     grid_pojo_experiment <- h2o.grid(
#     algorithm = "gbm",
#     grid_id = GRID_ID,
#     x = predictors,
#     y = "response",
#     training_frame = train.hex,
#     seed = 12345,
#     hyper_params = hyper_params_experiment
#     )
# 
#     grid_pojo_experiment <- h2o.getGrid(grid_id = GRID_ID)
#     model_pojo_experiment <- lapply(grid_pojo_experiment@model_ids, h2o.getModel)[[1]]
#     h2o.saveModel(model_pojo_experiment, getwd())
#     
#     browser()
#     
#     h2o.download_pojo(model_pojo_experiment, get_jar = T, path = getwd())
# 
#     #### Compare predictions for test data ####
#     # Prediction from the model
#     pred_test_fromModel <- h2o.predict(model_pojo_experiment, test.hex)
#     pred_test_fromModel <- as.data.frame(pred_test_fromModel)
#     head(pred_test_fromModel)
# 
#     system("javac -cp h2o-genmodel.jar -J-Xmx32g PredictCsv.java pojo_mismatch_3_model_0.java")
#     system("java -ea -cp :h2o-genmodel.jar hex.genmodel.tools.PredictCsv --header --model pojo_mismatch_3_model_0 --input ./df_test.csv --output ./pred_test_viaPojo.csv")
# 
#     pred_test_viaPojo <- read.csv("pred_test_viaPojo.csv")
# 
#     all(pred_test_viaPojo$predict == pred_test_fromModel$predict)
#     sum(pred_test_viaPojo$predict == pred_test_fromModel$predict) #0
#     sum(pred_test_viaPojo$predict != pred_test_fromModel$predict) #100000
#     # See differences:
#     cbind(pred_test_viaPojo$predict, pred_test_fromModel$predict)[1:100,]
#     summary(pred_test_viaPojo$predict - pred_test_fromModel$predict)
#     #      Min.    1st Qu.     Median       Mean    3rd Qu.       Max.
#     #-9.987e-02 -1.330e-02 -9.234e-05 -7.335e-05  1.307e-02  1.159e-01
# 
#     #### Version 2 does not have any differences ####
#     pred_test_v2_fromModel <- h2o.predict(model_pojo_experiment, test_v2.hex)
#     pred_test_v2_fromModel <- as.data.frame(pred_test_v2_fromModel)
# 
# 
#     system("java -ea -cp :h2o-genmodel.jar hex.genmodel.tools.PredictCsv --header --model pojo_mismatch_3_model_0 --input ./df_test_v2.csv --output ./pred_test_v2_viaPojo.csv")
# 
#     pred_test_v2_viaPojo <- read.csv("pred_test_v2_viaPojo.csv")
#     head(pred_test_v2_viaPojo)
# 
#     all(pred_test_v2_viaPojo$predict == pred_test_v2_fromModel$predict)
#     sum(pred_test_v2_viaPojo$predict == pred_test_v2_fromModel$predict) #100000
#     sum(pred_test_v2_viaPojo$predict != pred_test_v2_fromModel$predict) #0
#     # See no differences:
#     cbind(pred_test_v2_viaPojo$predict, pred_test_v2_fromModel$predict)[1:100,]  
    }


doTest("pubdev-4457: PredictCsv test", test)