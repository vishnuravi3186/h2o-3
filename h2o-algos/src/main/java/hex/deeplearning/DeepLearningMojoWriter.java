package hex.deeplearning;

import hex.ModelMojoWriter;

import java.io.IOException;

import static water.H2O.technote;

public class DeepLearningMojoWriter extends ModelMojoWriter<DeepLearningModel,
        DeepLearningModel.DeepLearningParameters, DeepLearningModel.DeepLearningModelOutput> {

  @SuppressWarnings("unused")
  public DeepLearningMojoWriter() {}
  private DeepLearningModel.DeepLearningParameters _parms;
  private DeepLearningModelInfo _model_info;
  private DeepLearningModel.DeepLearningModelOutput _output;

  public DeepLearningMojoWriter(DeepLearningModel model) {
    super(model);
    _parms = model.get_params();
    _model_info = model.model_info();
    _output = model._output;
    if (_model_info.isUnstable()) { // do not generate mojo for unstable model
      throw new UnsupportedOperationException(technote(4, "Refusing to create a MOJO for an unstable model."));
    }
  }

  @Override
  public String mojoVersion() {
    return "1.00";
  }

  @Override
  protected void writeModelData() throws IOException {
    writekv("mini_batch_size", _parms._mini_batch_size);
    writekv("nums", _model_info.data_info._nums);
    writekv("cats", _model_info.data_info._cats);
    writekv("cat_offsets", _output.catoffsets);
    writekv("norm_mul", _output.normmul);
    writekv("norm_sub", _output.normsub);
    writekv("norm_resp_mul", _output.normrespmul);
    writekv("norm_resp_sub", _output.normrespsub);
    writekv("use_all_factor_levels", _parms._use_all_factor_levels);
    writekv("standardize", _parms._standardize);  // check if need to standardize input
    writekv("hidden", _parms._hidden);
    writekv("activation", _parms._activation);
    writekv("input_dropout_ratio", _parms._input_dropout_ratio);
    writekv("hidden_dropout_ratio", _parms._hidden_dropout_ratios);
    boolean imputeMeans=_parms._missing_values_handling.equals(DeepLearningModel.DeepLearningParameters.MissingValuesHandling.MeanImputation);
    writekv("mean_imputation", imputeMeans);
    if (imputeMeans && _model_info.data_info._cats>0) { // only add this if there are categorical columns
      writekv("cat_modes", _model_info.data_info.catNAFill());
    }

//    writeblob("model_params", _model_info);

  }



}
