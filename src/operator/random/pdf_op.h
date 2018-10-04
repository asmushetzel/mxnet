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
 * \file pdf_op.h
 * \brief Operators for computing the pdf of random distributions.
 */
#ifndef MXNET_OPERATOR_RANDOM_PDF_OP_H_
#define MXNET_OPERATOR_RANDOM_PDF_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {

template<bool logpdf>
struct PDF_Uniform {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(int i, int sample_size, DType *out, IType1 *sample, IType2 *lower, IType2 *upper) {
    const int index(i / sample_size);
    const DType l(lower[index]), h(upper[index]);
    // No check whether sample is in the support.
    out[i] = logpdf ? -DType(log(h-l)) : DType(1.0)/(h-l);
  }
};

template<bool logpdf>
struct PDF_Uniform_Grad {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(int i, int sample_size,
                  DType *out, IType1 *sample, IType2 *lower, IType2 *upper,
                  DType *grad_out, IType1 *grad_sample, IType2 *grad_lower, IType2 *grad_upper) {
    const int index(i / sample_size);
    const DType l(lower[index]), h(upper[index]);
    const DType scaling(grad_out[i]*(logpdf ? DType(1) : out[i]));
    grad_lower[i]  = scaling/(h-l);
    grad_upper[i]  = scaling/(l-h);
    grad_sample[i] = 0;
  }
};

template<bool logpdf>
struct PDF_Normal {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(int i, int sample_size, DType *out, IType1 *sample, IType2 *loc, IType2 *scale) {
    const int index(i / sample_size);
    const DType u(loc[index]), s(scale[index]), x(sample[i]);
    const DType exponent((DType(-0.5)*(x-u)*(x-u))/(s*s));
    const DType normalizer(sqrt(2.0*mxnet_op::PI));
    out[i] = logpdf ? exponent-log(normalizer*s) : exp(exponent)/(normalizer*s);
  }
};

template<bool logpdf>
struct PDF_Normal_Grad {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(int i, int sample_size,
                  DType *out, IType1 *sample, IType2 *loc, IType2 *scale,
                  DType *grad_out, IType1 *grad_sample, IType2 *grad_loc, IType2 *grad_scale) {
    const int index(i / sample_size);
    const DType u(loc[index]), s(scale[index]), sq(s*s), x(sample[i]);
    const DType scaling(grad_out[i]*(logpdf ? DType(1) : out[i]));
    grad_loc[i]    = scaling*(x-u)/sq;
    grad_scale[i]  = scaling*((x-u)*(x-u)-sq)/(sq*s);
    grad_sample[i] = scaling*(u-x)/sq;
  }
};

template<bool logpdf>
struct PDF_Gamma {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(int i, int sample_size, DType *out, IType1 *sample, IType2 *alpha, IType2 *beta) {
    const int index(i / sample_size);
    const DType a(alpha[index]), b(beta[index]), x(sample[i]);
    const DType lpdf(a*log(b)+(a-1)*log(x)-b*x-lgamma(a));
    out[i] = logpdf ? lpdf : DType(exp(lpdf));
  }
};

template<bool logpdf>
struct PDF_Gamma_Grad {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(int i, int sample_size,
                  DType *out, IType1 *sample, IType2 *alpha, IType2 *beta,
                  DType *grad_out, IType1 *grad_sample, IType2 *grad_alpha, IType2 *grad_beta) {
    const int index(i / sample_size);
    const DType a(alpha[index]), b(beta[index]), x(sample[i]);
    const DType scaling(grad_out[i]*(logpdf ? DType(1) : out[i]));
    grad_alpha[i]  = scaling*(log(b)+log(x)-mshadow_op::gamma_grad::Map(a)/tgamma(a));
    grad_beta[i]   = scaling*(a/b-x);
    grad_sample[i] = scaling*((a-1)/x-b);
  }
};



template<bool logpdf>
struct PDF_Exponential {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(int i, int sample_size, DType *out, IType1 *sample, IType2 *lambda) {
    const int index(i / sample_size);
    const DType l(lambda[index]), x(sample[i]);
    out[i] = logpdf ? log(l)-l*x : l*exp(-l*x);
  }
};

template<bool logpdf>
struct PDF_Exponential_Grad {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(int i, int sample_size,
                  DType *out, IType1 *sample, IType2 *lambda,
                  DType *grad_out, IType1 *grad_sample, IType2 *grad_lambda) {
    const int index(i / sample_size);
    const DType l(lambda[index]), x(sample[i]);
    const DType scaling(grad_out[i]*(logpdf ? DType(1) : out[i]));
    grad_lambda[i] = scaling*(DType(1)/l-x);
    grad_sample[i] = -scaling*l;
  }
};


template<bool logpdf>
struct PDF_Poisson {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(int i, int sample_size, DType *out, IType1 *sample, IType2 *lambda) {
    const int index(i / sample_size);
    const DType l(lambda[index]), x(sample[i]);
    const DType lpdf(x*log(l)-l-lgamma(x+1));
    out[i] = logpdf ? lpdf  : DType(exp(lpdf));
  }
};

template<bool logpdf>
struct PDF_Poisson_Grad {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(int i, int sample_size,
                  DType *out, IType1 *sample, IType2 *lambda,
                  DType *grad_out, IType1 *grad_sample, IType2 *grad_lambda) {
    const int index(i / sample_size);
    const DType l(lambda[index]), x(sample[i]);
    const DType scaling(grad_out[i]*(logpdf ? DType(1) : out[i]));
    grad_lambda[i] = scaling*(x/l-DType(1));
    grad_sample[i] = 0;
  }
};


template<bool logpdf>
struct PDF_NegativeBinomial {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(int i, int sample_size, DType *out, IType1 *sample, IType2 *limit, IType2 *prob) {
    const int index(i / sample_size);
    const DType lpdf(LPDF(DType(limit[index]), DType(prob[index]), DType(sample[i])));
    out[i] = logpdf ? lpdf : DType(exp(lpdf));
  }
  template<typename DType>
  MSHADOW_XINLINE static DType LPDF(DType l, DType p, DType x) {
    // Note that "p" is the failure and not the success probability.
    return lgamma(x+l)-lgamma(x+1)-lgamma(l)+l*log(p)+x*log(1-p);
  }
};

template<bool logpdf>
struct PDF_NegativeBinomial_Grad {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(int i, int sample_size,
                  DType *out, IType1 *sample, IType2 *limit, IType2 *prob,
                  DType *grad_out, IType1 *grad_sample, IType2 *grad_limit, IType2 *grad_prob) {
    const int index(i / sample_size);
    std::pair<DType, DType> grads(LPDF_GRAD(DType(limit[index]), DType(prob[index]), DType(sample[i]), out[i], grad_out[i]));
    grad_limit[i]  = grads.first;
    grad_prob[i]   = grads.second;
    grad_sample[i] = 0;
  }
  template<typename DType>
  MSHADOW_XINLINE static std::pair<DType, DType> LPDF_GRAD(DType l, DType p, DType x, DType o, DType grad_o) {
    const DType scaling(grad_o*(logpdf ? DType(1) : o));
    return std::pair<DType, DType>(scaling*(mshadow_op::gammaln_grad::Map(x+l)
                                            -mshadow_op::gammaln_grad::Map(l)+log(p)),
                                   scaling*(l/p-x/(1-p)));
  }
};


template<bool logpdf>
struct PDF_GeneralizedNegativeBinomial {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(int i, int sample_size, DType *out, IType1 *sample, IType2 *mu, IType2 *alpha) {
    const int index(i / sample_size);
    // Reparametrize with limit = 1/alpha, prob = 1/(mu*alpha+1)
    const DType limit(1.0/alpha[index]), prob(1.0/(mu[index]*alpha[index]+1.0));
    const DType lpdf(PDF_NegativeBinomial<logpdf>::LPDF(limit, prob, DType(sample[i])));
    out[i] = logpdf ? lpdf : DType(exp(lpdf));
  }
};

template<bool logpdf>
struct PDF_GeneralizedNegativeBinomial_Grad {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(int i, int sample_size,
                  DType *out, IType1 *sample, IType2 *mu, IType2 *alpha,
                  DType *grad_out, IType1 *grad_sample, IType2 *grad_mu, IType2 *grad_alpha) {
    const int index(i / sample_size);
    const DType fmu(mu[index]), falpha(alpha[index]), den(fmu*falpha+1.0);
    // Reparametrize with limit = 1/alpha, prob = 1/(mu*alpha+1)
    const DType limit(1.0/falpha), prob(1.0/(fmu*falpha+1.0));
    // Grad returned as d_limit, d_prob
    std::pair<DType, DType> lpdf_grad(PDF_NegativeBinomial_Grad<logpdf>::LPDF_GRAD(limit, prob, DType(sample[i]), out[i], grad_out[i]));
    grad_mu[i]     = -lpdf_grad.second*falpha/(den*den);
    grad_alpha[i]  = -lpdf_grad.first/(falpha*falpha)-lpdf_grad.second*fmu/(den*den);
    grad_sample[i] = 0;
  }
};

struct PdfParam : public dmlc::Parameter<PdfParam> {
  bool is_log;
  int dtype;
  DMLC_DECLARE_PARAMETER(PdfParam) {
    DMLC_DECLARE_FIELD(is_log).set_default(false)
    .describe("Whether to compute the log PDF or not.");
    DMLC_DECLARE_FIELD(dtype)
    .add_enum("None", -1)
    .add_enum("float16", mshadow::kFloat16)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .set_default(-1)
    .describe("DType of the output in case this can't be inferred. "
              "Defaults to float32 if not defined (dtype=None).");
  }
};

inline bool PdfOpShape(const nnvm::NodeAttrs& attrs,
                               std::vector<TShape>* in_attrs,
                               std::vector<TShape>* out_attrs) {
  CHECK_GT(in_attrs->size(), 1)
    << "pdf operator takes at least 2 arguments (" << in_attrs->size() << " given)";
  CHECK_EQ(out_attrs->size(), 1);
  // All inputs must be defined in order to infer output shape.
  if( std::all_of((*in_attrs).begin(), (*in_attrs).end(), [](const TShape& s){ return s.ndim() > 0; }) ) {
    // Tensors of distribution parameters must have same size.
    for (size_t i = 2; i < in_attrs->size(); ++i) {
      SHAPE_ASSIGN_CHECK(*in_attrs, i, (*in_attrs)[i-1]);
    }
    // Tensors of distribution parameters must match leftmost subshape of samples.
    CHECK_LE((*in_attrs)[1].ndim(), (*in_attrs)[0].ndim())
      << "dimension of input samples (" << (*in_attrs)[0].ndim()
      << ") must be at least dimension of distribution parameters ("<< (*in_attrs)[1].ndim() << ")";
    TShape tshape((*in_attrs)[0].begin(), (*in_attrs)[0].begin()+(*in_attrs)[1].ndim());
    for (size_t i = 1; i < in_attrs->size(); ++i) {
      SHAPE_ASSIGN_CHECK(*in_attrs, i, tshape);
     } 
    // Output shape must equal to input tensor of samples.
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
    return true;
  }
  return false;
}

inline bool PdfOpType(const nnvm::NodeAttrs& attrs,
                      std::vector<int>* in_attrs,
                      std::vector<int>* out_attrs) {
  CHECK_GT(in_attrs->size(), 0);
  CHECK_EQ(out_attrs->size(), 1);
  // Distribution parameters must all be of same type.
  if( std::all_of((*in_attrs).begin()+1, (*in_attrs).end(), [](const int& dtype){ return dtype != -1; }) ) {
    for (size_t i = 2; i < in_attrs->size(); ++i) {
      TYPE_ASSIGN_CHECK(*in_attrs, i, (*in_attrs)[i-1]);
    }
  }
  // The type of the output can't be inferred from inputs.
  const PdfParam& param = nnvm::get<PdfParam>(attrs.parsed);
  int dtype = (*out_attrs)[0];
  if (dtype != -1) {
    if (param.dtype != -1) {
      // dtype given in args, check that it matches the output type
      CHECK_EQ(dtype, param.dtype) << "Inferred output type does not match requested type: "
        << dtype << " vs " << param.dtype;
    }
  } else {
    // Output type can't be inferred. Use type in args or default.
    dtype = (param.dtype == -1 ? mshadow::kFloat32 : param.dtype);
  }
  bool dtype_ok = (dtype == mshadow::kFloat16) || (dtype == mshadow::kFloat32) ||
    (dtype == mshadow::kFloat64);
  CHECK_EQ(dtype_ok, true) << "Output type must be float16, float32, or float64: dtype is "
    << dtype<< " vs " << mshadow::kFloat16 << " or " << mshadow::kFloat32 << " or "
    << mshadow::kFloat64;
  TYPE_ASSIGN_CHECK(*out_attrs, 0, dtype);
  return true;
}

template<typename OP>
struct LaunchExWrapper {
  template<typename ...Args>
  MSHADOW_XINLINE static void Map(const int start, const int length, Args... args) {
    for (int i = 0; i < length; ++i) {
      OP::Map(start+i, args...);
    }
  }
};

template<typename xpu, typename OType, typename IType1, typename IType2, typename pdf, int pnum>
struct PdfCaller;

template<typename xpu, typename OType, typename IType1, typename IType2, typename pdf>
struct PdfCaller<xpu, OType, IType1, IType2, pdf, 1> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 mshadow::Stream<xpu> *s) {
    index_t num_samples(1);
    for (int i = inputs[1].ndim(); i < inputs[0].ndim(); ++i) {
      num_samples *= inputs[0].shape_[i];
    }
    mxnet_op::Kernel<LaunchExWrapper<pdf>, xpu>::LaunchEx(s, inputs[0].Size(), num_samples,
                outputs[0].dptr<OType>(), inputs[0].dptr<IType1>(), inputs[1].dptr<IType2>());
  }
};

template<typename xpu, typename OType, typename IType1, typename IType2, typename pdf>
struct PdfCaller<xpu, OType, IType1, IType2, pdf, 2> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 mshadow::Stream<xpu> *s) {
    index_t num_samples(1);
    for (int i = inputs[1].ndim(); i < inputs[0].ndim(); ++i) {
      num_samples *= inputs[0].shape_[i];
    }
    mxnet_op::Kernel<LaunchExWrapper<pdf>, xpu>::LaunchEx(s, inputs[0].Size(), num_samples,
                outputs[0].dptr<OType>(), inputs[0].dptr<IType1>(), 
                inputs[1].dptr<IType2>(), inputs[2].dptr<IType2>());
  }
};

template<typename xpu, template<bool> class pdf, int pnum>
void PdfOpForward(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  CHECK_NE(req[0], kAddTo);
  CHECK_EQ(inputs.size(), pnum+1);
  CHECK_EQ(outputs.size(), 1);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const PdfParam& param = nnvm::get<PdfParam>(attrs.parsed);
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, IType1, {
      MSHADOW_TYPE_SWITCH(inputs[1].type_flag_, IType2, {
        if( param.is_log ) {
          PdfCaller<xpu, OType, IType1, IType2, pdf<true>, pnum>::op(inputs, outputs, s);
        } else {
          PdfCaller<xpu, OType, IType1, IType2, pdf<false>, pnum>::op(inputs, outputs, s);
        }
      });
    });
  });
}


template<typename xpu, typename OType, typename IType1, typename IType2, typename pdfgrad, int pnum>
struct PdfGradCaller;

template<typename xpu, typename OType, typename IType1, typename IType2, typename pdfgrad>
struct PdfGradCaller<xpu, OType, IType1, IType2, pdfgrad, 1> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& grads,
                 mshadow::Stream<xpu> *s) {
    index_t num_samples(1);
    for (int i = inputs[2].ndim(); i < inputs[1].ndim(); ++i) {
      num_samples *= inputs[1].shape_[i];
    }
    mxnet_op::Kernel<LaunchExWrapper<pdfgrad>, xpu>::LaunchEx(s, inputs[0].Size(), num_samples,
                inputs[3].dptr<OType>(), inputs[1].dptr<IType1>(), inputs[2].dptr<IType2>(),
                inputs[0].dptr<OType>(), grads[0].dptr<IType1>(), grads[1].dptr<IType2>());
  }
};

template<typename xpu, typename OType, typename IType1, typename IType2, typename pdfgrad>
struct PdfGradCaller<xpu, OType, IType1, IType2, pdfgrad, 2> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& grads,
                 mshadow::Stream<xpu> *s) {
    index_t num_samples(1);
    for (int i = inputs[2].ndim(); i < inputs[1].ndim(); ++i) {
      num_samples *= inputs[1].shape_[i];
    }
    mxnet_op::Kernel<LaunchExWrapper<pdfgrad>, xpu>::LaunchEx(s, inputs[0].Size(), num_samples,
                inputs[4].dptr<OType>(), inputs[1].dptr<IType1>(), inputs[2].dptr<IType2>(), inputs[3].dptr<IType2>(),
                inputs[0].dptr<OType>(), grads[0].dptr<IType1>(), grads[1].dptr<IType2>(), grads[2].dptr<IType2>());
  }
};

template<typename xpu, template<bool> class pdfgrad, int pnum>
void PdfOpBackward(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {


  // FIXME: req[0] not used (gradient of samples not added)

  using namespace mshadow;
  CHECK_EQ(inputs.size(), pnum+3);
  CHECK_EQ(outputs.size(), pnum+1);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const PdfParam& param = nnvm::get<PdfParam>(attrs.parsed);
  const size_t N(outputs[1].Size());
  const TShape src_shape(Shape2(N, outputs[0].Size()/N)), dst_shape(Shape2(N, 1));
  // Inputs to PdfOpBackward: grad, samples, parm1, parm2, pdf.
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, OType, {
    MSHADOW_TYPE_SWITCH(inputs[1].type_flag_, IType1, {
      MSHADOW_TYPE_SWITCH(inputs[2].type_flag_, IType2, {
        const size_t red_work_size(broadcast::ReduceWorkspaceSize<2, IType2>(s, dst_shape, kAddTo, src_shape));
        const size_t tmp_size(outputs[0].Size()*(pnum*sizeof(IType2))+red_work_size);
        Tensor<xpu, 1, char> tmp_space = ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(tmp_size), s);
        std::vector<TBlob> grads = {outputs[0]};
        grads.push_back(TBlob(tmp_space.dptr_, outputs[0].shape_,
                              outputs[1].dev_mask(), outputs[1].type_flag_, outputs[1].dev_id()));
        if (pnum == 2) {
          grads.push_back(TBlob(tmp_space.dptr_+outputs[0].Size()*sizeof(IType2), outputs[0].shape_,
                                outputs[2].dev_mask(), outputs[2].type_flag_, outputs[2].dev_id()));
        }
        if (param.is_log) {
          PdfGradCaller<xpu, OType, IType1, IType2, pdfgrad<true>, pnum>::op(inputs, grads, s);
        } else {
          PdfGradCaller<xpu, OType, IType1, IType2, pdfgrad<false>, pnum>::op(inputs, grads, s);
        }
        Tensor<xpu, 1, char> red_work(tmp_space.dptr_+pnum*outputs[0].Size()*sizeof(IType2), Shape1(red_work_size), s);
        broadcast::Reduce<red::sum, 2, IType2, op::mshadow_op::identity>(
           s, outputs[1].reshape(dst_shape), req[1], red_work, grads[1].reshape(src_shape));
        if (pnum == 2) {
          broadcast::Reduce<red::sum, 2, IType2, op::mshadow_op::identity>(
           s, outputs[2].reshape(dst_shape), req[2], red_work, grads[2].reshape(src_shape));
        }
      });
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_RANDOM_PDF_OP_H_
