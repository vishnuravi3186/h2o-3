package hex.genmodel.algos.deeplearning;

import hex.genmodel.MojoModel;

public class DeeplearningMojoModel extends MojoModel {

  /***
   * Should set up the neuron network frame work here
   * @param columns
   * @param domains
   */
  DeeplearningMojoModel(String[] columns, String[][] domains) {
    super(columns, domains);
  }

  /***
   * This method will be derived from the scoring/prediction function of deeplearning model itself.
   * @param doubles
   * @param offset
   * @param preds
   * @return
   */
  @Override
  public final double[] score0(double[] doubles, double offset, double[] preds) {
    assert(doubles != null) : "doubles are null";
    float[] floats;

    return preds;
  }

  @Override
  public double[] score0(double[] row, double[] preds) {
    return score0(row, 0.0, preds);
  }
}
