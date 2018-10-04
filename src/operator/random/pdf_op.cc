/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2018 by Contributors
 * \file pdf_op.cc
 * \brief CPU-operators for computing the pdf of random distributions. 
 */

#include "./pdf_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(PdfParam);

#define MXNET_OPERATOR_REGISTER_PDF(distr, pdffunc, num_parms, \
                                    parm_name_1, parm_name_2, \
                                    parm_desc_1, parm_desc_2, \
                                    description) \
  NNVM_REGISTER_OP(_random_pdf_##distr) \
  .add_alias("random_pdf_" #distr) \
  .describe(description()+std::string(ADD_FILELINE)) \
  .set_num_inputs(num_parms+1) \
  .set_num_outputs(1) \
  .set_attr_parser(ParamParser<PdfParam>) \
  .set_attr<nnvm::FListInputNames>("FListInputNames", \
    [](const NodeAttrs& attrs) { \
      std::vector<std::string> v = {"sample", parm_name_1, parm_name_2}; v.resize(num_parms+1); return v; \
    }) \
  .set_attr<nnvm::FInferShape>("FInferShape", PdfOpShape) \
  .set_attr<nnvm::FInferType>("FInferType", PdfOpType) \
  .set_attr<FCompute>("FCompute<cpu>", PdfOpForward<cpu, pdffunc, num_parms>) \
  .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{"_backward_pdf_" #distr}) \
  .add_argument("sample", "NDArray-or-Symbol", "Samples from the distributions.") \
  .add_argument(parm_name_1, "NDArray-or-Symbol", parm_desc_1) \
  .add_arguments(PdfParam::__FIELDS__())

#define MXNET_OPERATOR_REGISTER_PDF_GRAD(distr, pdffunc, num_parms) \
  NNVM_REGISTER_OP(_backward_pdf_##distr) \
  .set_num_inputs(num_parms+3) \
  .set_num_outputs(num_parms+1) \
  .set_attr_parser(ParamParser<PdfParam>) \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs) \
    { return std::vector<std::pair<int, int> >{{1, 0}, {2, 1}, {3, 2}}; }) \
  .set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs) \
    { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; }) \
  .set_attr<nnvm::TIsBackward>("TIsBackward", true) \
  .set_attr<FCompute>("FCompute<cpu>", PdfOpBackward<cpu, pdffunc##_Grad, num_parms>);


#define MXNET_OPERATOR_REGISTER_PDF1(distr, pdffunc, parm_name, parm_desc, description) \
    MXNET_OPERATOR_REGISTER_PDF(distr, pdffunc, 1, parm_name, parm_name, \
                                parm_desc, parm_desc, description); \
    MXNET_OPERATOR_REGISTER_PDF_GRAD(distr, pdffunc, 1)

#define MXNET_OPERATOR_REGISTER_PDF2(distr, pdffunc, parm_name_1, parm_name_2, \
                                     parm_desc_1, parm_desc_2, description) \
  MXNET_OPERATOR_REGISTER_PDF(distr, pdffunc, 2, parm_name_1, parm_name_2, \
                                   parm_desc_1, parm_desc_2, description) \
  .add_argument(parm_name_2, "NDArray-or-Symbol", parm_desc_2); \
  MXNET_OPERATOR_REGISTER_PDF_GRAD(distr, pdffunc, 2)

inline std::string uniform_desc() {
  return std::string(R"code(Computes the value of the PDF of samples of
uniform distributions on the intervals given by *[low,high)*.
)code");
}

inline std::string normal_desc() {
  return std::string(R"code(Computes the value of the PDF of samples of
normal distributions with parameters *mu* (mean) and *sigma* (standard deviation).
)code");
}

inline std::string gamma_desc() {
  return std::string(R"code(Computes the value of the PDF of samples of
gamma distributions with parameters *alpha* (shape) and *beta* (scale).
)code");
}

inline std::string exponential_desc() {
  return std::string(R"code(Computes the value of the PDF of samples of
exponential distributions with parameters lambda (rate).
)code");
}

inline std::string poisson_desc() {
  return std::string(R"code(Computes the value of the PDF of samples of
Poisson distributions with parameters lambda (rate).
)code");
}

inline std::string negative_binomial_desc() {
  return std::string(R"code(Computes the value of the PDF of samples of
negative binomial distributions with parameters *k* (failure limit) and *p* (failure probability).
)code");
}

inline std::string generalized_negative_binomial_desc() {
  return std::string(R"code(Computes the value of the PDF of samples of
)code");
}

MXNET_OPERATOR_REGISTER_PDF2(uniform, PDF_Uniform, "low", "high",
  "Lower bounds of the distributions.", "Upper bounds of the distributions.", uniform_desc)
MXNET_OPERATOR_REGISTER_PDF2(normal, PDF_Normal, "mu", "sigma",
  "Means of the distributions.", "Standard deviations of the distributions.", normal_desc)
MXNET_OPERATOR_REGISTER_PDF2(gamma, PDF_Gamma, "alpha", "beta",
  "Alpha (shape) parameters of the distributions.", "Beta (scale) parameters of the distributions.",
  gamma_desc)
MXNET_OPERATOR_REGISTER_PDF1(exponential, PDF_Exponential, "lam",
  "Lambda (rate) parameters of the distributions.", exponential_desc)
MXNET_OPERATOR_REGISTER_PDF1(poisson, PDF_Poisson, "lam",
  "Lambda (rate) parameters of the distributions.", poisson_desc)
MXNET_OPERATOR_REGISTER_PDF2(negative_binomial, PDF_NegativeBinomial, "k", "p",
  "Limits of unsuccessful experiments.", "Failure probabilities in each experiment.",
  negative_binomial_desc)
MXNET_OPERATOR_REGISTER_PDF2(generalized_negative_binomial,
  PDF_GeneralizedNegativeBinomial, "mu", "alpha",
  "Means of the distributions.", "Alpha (dispersion) parameters of the distributions.",
  generalized_negative_binomial_desc)

}  // namespace op
}  // namespace mxnet
