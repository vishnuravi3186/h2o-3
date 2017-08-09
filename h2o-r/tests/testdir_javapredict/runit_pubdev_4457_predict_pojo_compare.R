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
    
    
    # call pojo to make prediction
    system("java -ea -cp /Users/wendycwong/h2o-3/h2o-r/tests/testdir_javapredict/Rsandbox_RTest/tmp_model_14057/h2o-genmodel.jar:/Users/wendycwong/h2o-3/h2o-r/tests/testdir_javapredict/Rsandbox_RTest/tmp_model_14057 hex.genmodel.tools.PredictCsv --header --model GBM_model_R_1502293352878_1 --input /Users/wendycwong/h2o-3/h2o-r/tests/testdir_javapredict/Rsandbox_RTest/tmp_model_14057/in.csv --output /Users/wendycwong/h2o-3/h2o-r/tests/testdir_javapredict/Rsandbox_RTest/tmp_model_14057/pred_test_viaPojo.csv --decimal")
 
    browser()
    
    pojoPrediction = read.csv("/Users/wendycwong/h2o-3/h2o-r/tests/testdir_javapredict/Rsandbox_RTest/tmp_model_14057/pred_test_viaPojo.csv")
    # contains h2o predict
    h2oPrediction = read.csv("/Users/wendycwong/h2o-3/h2o-r/tests/testdir_javapredict/Rsandbox_RTest/tmp_model_14057/out_h2o.csv")
    
    sum(h2oPrediction$predict==pojoPrediction$predict)
    sum(h2oPrediction$predict!=pojoPrediction$predict)
    summary(h2oPrediction$predict-pojoPrediction$predict)

    }


doTest("pubdev-4457: PredictCsv test", test)