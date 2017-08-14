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

    ### Read data in H2O ####
    test.hex <- h2o.importFile(locate("temp.csv"), destination_frame="test.hex")

    #### Compare predictions for test data ####
    # Prediction from the model trained from before
    model_pojo_experiment = h2o.loadModel("/Users/wendycwong/h2o-3/h2o-r/tests/testdir_javapredict/pojo_mismatch_3_model_0")
    pred_test_fromModel <- h2o.predict(model_pojo_experiment, test.hex) # use test data
    pred_test_fromModel <- as.data.frame(pred_test_fromModel)
    write.csv(pred_test_fromModel, "./h2oPredict_test.csv", quote=FALSE, row.names=FALSE)
    head(pred_test_fromModel)
    
    system("javac -cp h2o-genmodel.jar -J-Xmx32g PredictCsv.java pojo_mismatch_3_model_0.java")
    system("java -ea -cp :h2o-genmodel.jar hex.genmodel.tools.PredictCsv --header --model pojo_mismatch_3_model_0 --input ./temp.csv --output ./pred_test_viaPojo.csv --decimal")
    
   
    pred_test_viaPojo <- read.csv("pred_test_viaPojo.csv")
    compareFrames(pred_test_viaPojo, pred_test_fromModel, 1e-6)
  }

compareFrames <- function(f1, f2, tol) {
  sameNum = sum(abs(f1$predict - f2$predict) < tol)
  browser()
  diffNum = length(f1) - sameNum
  print(paste(
    "Number of elements that differ more than tolerance is ",
    diffNum,
    sep = ' '
  ))
  summary(f1$predict - f2$predict)
  expect_true(sameNum==1)
}

doTest("pubdev-4457: PredictCsv test", test)