#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Copyright 2016 H2O.ai;  Apache License Version 2.0 (see LICENSE for details)
#
from __future__ import absolute_import, division, print_function, unicode_literals

import inspect
import types
import warnings

import h2o
from h2o.exceptions import H2OValueError
from h2o.frame import H2OFrame
from h2o.job import H2OJob
from h2o.utils.compatibility import *  # NOQA
from h2o.utils.shared_utils import quoted
from h2o.utils.typechecks import assert_is_type, is_type, numeric
from ..model.autoencoder import H2OAutoEncoderModel
from ..model.binomial import H2OBinomialModel
from ..model.clustering import H2OClusteringModel
from ..model.dim_reduction import H2ODimReductionModel
from ..model.metrics_base import (H2OBinomialModelMetrics, H2OClusteringModelMetrics, H2ORegressionModelMetrics,
                                  H2OMultinomialModelMetrics, H2OAutoEncoderModelMetrics, H2ODimReductionModelMetrics,
                                  H2OWordEmbeddingModelMetrics)
from ..model.model_base import ModelBase
from ..model.multinomial import H2OMultinomialModel
from ..model.regression import H2ORegressionModel
from ..model.word_embedding import H2OWordEmbeddingModel


class EstimatorAttributeError(AttributeError):
    def __init__(self, obj, method):
        super(AttributeError, self).__init__("No {} method for {}".format(method, obj.__class__.__name__))


class H2OEstimator(ModelBase):
    """
    Base class for H2O Estimators.

    H2O Estimators implement the following methods for model construction:

        - ``start()`` - Top-level user-facing API for asynchronous model build
        - ``join()``  - Top-level user-facing API for blocking on async model build
        - ``train()`` - Top-level user-facing API for model building.
        - ``fit()`` - Used by scikit-learn.

    Because H2OEstimator instances are instances of ModelBase, these objects can use the H2O model API.
    """

    def start(self, x, y=None, training_frame=None, offset_column=None, fold_column=None,
              weights_column=None, validation_frame=None, **params):
        """
        Train the model asynchronously (to block for results call :meth:`join`).

        :param x: A list of column names or indices indicating the predictor columns.
        :param y: An index or a column name indicating the response column.
        :param H2OFrame training_frame: The H2OFrame having the columns indicated by x and y (as well as any
            additional columns specified by fold, offset, and weights).
        :param offset_column: The name or index of the column in training_frame that holds the offsets.
        :param fold_column: The name or index of the column in training_frame that holds the per-row fold
            assignments.
        :param weights_column: The name or index of the column in training_frame that holds the per-row weights.
        :param validation_frame: H2OFrame with validation data to be scored on while training.
        """
        self._future = True
        self.train(x=x,
                   y=y,
                   training_frame=training_frame,
                   offset_column=offset_column,
                   fold_column=fold_column,
                   weights_column=weights_column,
                   validation_frame=validation_frame,
                   **params)


    def join(self):
        """Wait until job's completion."""
        self._future = False
        self._job.poll()
        model_key = self._job.dest_key
        self._job = None
        model_json = h2o.api("GET /%d/Models/%s" % (self._rest_version, model_key))["models"][0]
        self._resolve_model(model_key, model_json)


    def train(self, x=None, y=None, training_frame=None, offset_column=None, fold_column=None,
              weights_column=None, validation_frame=None, max_runtime_secs=None, ignored_columns=None,
              model_id=None, verbose=False):
        """
        Train the H2O model.

        :param x: A list of column names or indices indicating the predictor columns.
        :param y: An index or a column name indicating the response column.
        :param H2OFrame training_frame: The H2OFrame having the columns indicated by x and y (as well as any
            additional columns specified by fold, offset, and weights).
        :param offset_column: The name or index of the column in training_frame that holds the offsets.
        :param fold_column: The name or index of the column in training_frame that holds the per-row fold
            assignments.
        :param weights_column: The name or index of the column in training_frame that holds the per-row weights.
        :param validation_frame: H2OFrame with validation data to be scored on while training.
        :param float max_runtime_secs: Maximum allowed runtime in seconds for model training. Use 0 to disable.
        :param bool verbose: Print scoring history to stdout. Defaults to False.
        """
        assert_is_type(training_frame, H2OFrame)
        assert_is_type(validation_frame, None, H2OFrame)
        assert_is_type(y, None, int, str)
        assert_is_type(x, None, int, str, [str, int], {str, int})
        assert_is_type(ignored_columns, None, [str, int], {str, int})
        assert_is_type(offset_column, None, int, str)
        assert_is_type(fold_column, None, int, str)
        assert_is_type(weights_column, None, int, str)
        assert_is_type(max_runtime_secs, None, numeric)
        assert_is_type(model_id, None, str)
        assert_is_type(verbose,bool)
        algo = self.algo
        if verbose and algo not in ["drf","gbm","deeplearning","xgboost"]:
            raise H2OValueError("Verbose should only be set to True for drf, gbm, deeplearning, and xgboost models")
        parms = self._parms.copy()
        if "__class__" in parms:  # FIXME: hackt for PY3
            del parms["__class__"]
        is_auto_encoder = bool(parms.get("autoencoder"))
        is_supervised = not(is_auto_encoder or algo in {"pca", "svd", "kmeans", "glrm", "word2vec"})
        ncols = training_frame.ncols
        names = training_frame.names
        if is_supervised:
            if y is None: y = "response"
            if is_type(y, int):
                if not (-ncols <= y < ncols):
                    raise H2OValueError("Column %d does not exist in the training frame" % y)
                y = names[y]
            else:
                if y not in names:
                    raise H2OValueError("Column %s does not exist in the training frame" % y)
            self._estimator_type = "classifier" if training_frame.types[y] == "enum" else "regressor"
        elif y is not None:
            raise H2OValueError("y should not be provided for an unsupervised model")
        assert_is_type(y, str, None)
        ignored_columns_set = set()
        if ignored_columns is not None:
            if x is not None:
                raise H2OValueError("Properties x and ignored_columns cannot be specified simultaneously")
            for ic in ignored_columns:
                if is_type(ic, int):
                    if not (-ncols <= ic < ncols):
                        raise H2OValueError("Column %d does not exist in the training frame" % ic)
                    ignored_columns_set.add(names[ic])
                else:
                    if ic not in names:
                        raise H2OValueError("Column %s not in the training frame" % ic)
                    ignored_columns_set.add(ic)
        if x is None:
            xset = set(names) - {y} - ignored_columns_set
        else:
            xset = set()
            if is_type(x, int, str): x = [x]
            for xi in x:
                if is_type(xi, int):
                    if not (-ncols <= xi < ncols):
                        raise H2OValueError("Column %d does not exist in the training frame" % xi)
                    xset.add(names[xi])
                else:
                    if xi not in names:
                        raise H2OValueError("Column %s not in the training frame" % xi)
                    xset.add(xi)
        x = list(xset)

        parms["offset_column"] = offset_column
        parms["fold_column"] = fold_column
        parms["weights_column"] = weights_column
        parms["max_runtime_secs"] = max_runtime_secs
        # Overwrites the model_id parameter only if model_id is passed
        if model_id is not None:
            parms["model_id"] = model_id

        # Step 2
        is_auto_encoder = "autoencoder" in parms and parms["autoencoder"]
        is_unsupervised = is_auto_encoder or self.algo in {"pca", "svd", "kmeans", "glrm", "word2vec"}
        if is_auto_encoder and y is not None: raise ValueError("y should not be specified for autoencoder.")
        if not is_unsupervised and y is None: raise ValueError("Missing response")

        # Step 3
        parms["training_frame"] = training_frame
        if validation_frame is not None: parms["validation_frame"] = validation_frame
        if is_type(y, int): y = training_frame.names[y]
        if y is not None: parms["response_column"] = y
        if not isinstance(x, (list, tuple)): x = [x]
        if is_type(x[0], int):
            x = [training_frame.names[i] for i in x]
        offset = parms["offset_column"]
        folds = parms["fold_column"]
        weights = parms["weights_column"]
        ignored_columns = list(set(training_frame.names) - set(x + [y, offset, folds, weights]))
        parms["ignored_columns"] = None if ignored_columns == [] else [quoted(col) for col in ignored_columns]
        parms["interactions"] = (None if "interactions" not in parms or parms["interactions"] is None else
                                 [quoted(col) for col in parms["interactions"]])
        parms = {k: H2OEstimator._keyify_if_h2oframe(parms[k]) for k in parms}
        rest_ver = parms.pop("_rest_version") if "_rest_version" in parms else 3

        model = H2OJob(h2o.api("POST /%d/ModelBuilders/%s" % (rest_ver, self.algo), data=parms),
                       job_type=(self.algo + " Model Build"))

        if self._future:
            self._job = model
            self._rest_version = rest_ver
            return

        model.poll(verbose_model_scoring_history=verbose)
        model_json = h2o.api("GET /%d/Models/%s" % (rest_ver, model.dest_key))["models"][0]
        self._resolve_model(model.dest_key, model_json)


    @staticmethod
    def _keyify_if_h2oframe(item):
        if isinstance(item, H2OFrame):
            return item.frame_id
        elif isinstance(item, list) and all(i is None or isinstance(i, H2OFrame) for i in item):
            return [quoted(i) if i is None else quoted(i.frame_id) for i in item]
        else:
            return item


    def _resolve_model(self, model_id, model_json):
        metrics_class, model_class = H2OEstimator._metrics_class(model_json)
        m = model_class()
        m._id = model_id
        m._model_json = model_json
        m._have_pojo = model_json.get('have_pojo', True)
        m._have_mojo = model_json.get('have_mojo', True)
        m._metrics_class = metrics_class
        m._parms = self._parms
        m._estimator_type = self._estimator_type

        if model_id is not None and model_json is not None and metrics_class is not None:
            # build Metric objects out of each metrics
            for metric in ["training_metrics", "validation_metrics", "cross_validation_metrics"]:
                if metric in model_json["output"]:
                    if model_json["output"][metric] is not None:
                        if metric == "cross_validation_metrics":
                            m._is_xvalidated = True
                        model_json["output"][metric] = \
                            metrics_class(model_json["output"][metric], metric, model_json["algo"])

            if m._is_xvalidated:
                m._xval_keys = [i["name"] for i in model_json["output"]["cross_validation_models"]]

            # build a useful dict of the params
            for p in m._model_json["parameters"]:
                m.parms[p["name"]] = p
        H2OEstimator.mixin(self, model_class)
        self.__dict__.update(m.__dict__.copy())


    # TODO: replace with a property which is overriden in subclasses
    def _compute_algo(self):
        name = self.__class__.__name__
        if name == "H2ODeepLearningEstimator": return "deeplearning"
        if name == "H2ODeepWaterEstimator": return "deepwater"
        if name == "H2OAutoEncoderEstimator": return "deeplearning"
        if name == "H2OGradientBoostingEstimator": return "gbm"
        if name == "H2OGeneralizedLinearEstimator": return "glm"
        if name == "H2OGeneralizedLowRankEstimator": return "glrm"
        if name == "H2OKMeansEstimator": return "kmeans"
        if name == "H2ONaiveBayesEstimator": return "naivebayes"
        if name == "H2ORandomForestEstimator": return "drf"
        if name == "H2OXGBoostEstimator": return "xgboost"
        if name == "H2OPCA": return "pca"
        if name == "H2OSVD": return "svd"


    @staticmethod
    def mixin(obj, cls):
        for name in cls.__dict__:
            if name.startswith("__") and name.endswith("__"): continue
            if not isinstance(cls.__dict__[name], types.FunctionType): continue
            obj.__dict__[name] = cls.__dict__[name].__get__(obj)


    #------ Scikit-learn Interface Methods -------
    def fit(self, x, y=None, **params):
        """
        Fit an H2O model as part of a scikit-learn pipeline or grid search.

        A warning will be issued if a caller other than sklearn attempts to use this method.

        :param H2OFrame x: An H2OFrame consisting of the predictor variables.
        :param H2OFrame y: An H2OFrame consisting of the response variable.
        :param params: Extra arguments.
        :returns: The current instance of H2OEstimator for method chaining.
        """
        stk = inspect.stack()[1:]
        warn = True
        for s in stk:
            mod = inspect.getmodule(s[0])
            if mod:
                warn = "sklearn" not in mod.__name__
                if not warn: break
        if warn:
            warnings.warn("\n\n\t`fit` is not recommended outside of the sklearn framework. Use `train` instead.",
                          UserWarning, stacklevel=2)
        training_frame = x.cbind(y) if y is not None else x
        x = x.names
        y = y.names[0] if y is not None else None
        self.train(x, y, training_frame, **params)
        return self


    def get_params(self, deep=True):
        """
        Obtain parameters for this estimator.

        Used primarily for sklearn Pipelines and sklearn grid search.

        :param deep: If True, return parameters of all sub-objects that are estimators.

        :returns: A dict of parameters
        """
        out = dict()
        for key, value in self.parms.items():
            if deep and isinstance(value, H2OEstimator):
                deep_items = list(value.get_params().items())
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out


    def set_params(self, **parms):
        """
        Used by sklearn for updating parameters during grid search.

        :param parms: A dictionary of parameters that will be set on this model.
        :returns: self, the current estimator object with the parameters all set as desired.
        """
        self._parms.update(parms)
        return self


    @staticmethod
    def _metrics_class(model_json):
        model_type = model_json["output"]["model_category"]
        if model_type == "Binomial":
            metrics_class = H2OBinomialModelMetrics
            model_class = H2OBinomialModel
        elif model_type == "Clustering":
            metrics_class = H2OClusteringModelMetrics
            model_class = H2OClusteringModel
        elif model_type == "Regression":
            metrics_class = H2ORegressionModelMetrics
            model_class = H2ORegressionModel
        elif model_type == "Multinomial":
            metrics_class = H2OMultinomialModelMetrics
            model_class = H2OMultinomialModel
        elif model_type == "AutoEncoder":
            metrics_class = H2OAutoEncoderModelMetrics
            model_class = H2OAutoEncoderModel
        elif model_type == "DimReduction":
            metrics_class = H2ODimReductionModelMetrics
            model_class = H2ODimReductionModel
        elif model_type == "WordEmbedding":
            metrics_class = H2OWordEmbeddingModelMetrics
            model_class = H2OWordEmbeddingModel
        else:
            raise NotImplementedError(model_type)
        return [metrics_class, model_class]
