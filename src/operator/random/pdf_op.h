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
#include "../special_functions-inl.h"
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {

template<typename DType>
DType psi(DType val) { return special_functions::cephes::psi(val); }
template<>
mshadow::half::half_t psi(mshadow::half::half_t val) { return special_functions::cephes::psi<float>(val); }


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
  MSHADOW_XINLINE static void Map(int i, OpReqType req, int sample_size,
                  DType *out, IType1 *sample, IType2 *lower, IType2 *upper,
                  DType *grad_out, IType1 *grad_sample, IType2 *grad_lower, IType2 *grad_upper) {
    const int index(i / sample_size);
    const DType l(lower[index]), h(upper[index]);
    const DType scaling(grad_out[i]*(logpdf ? DType(1) : out[i]));
    grad_lower[i]  = scaling/(h-l);
    grad_upper[i]  = scaling/(l-h);
    KERNEL_ASSIGN(grad_sample[i], req, 0);
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
  MSHADOW_XINLINE static void Map(int i, OpReqType req, int sample_size,
                  DType *out, IType1 *sample, IType2 *loc, IType2 *scale,
                  DType *grad_out, IType1 *grad_sample, IType2 *grad_loc, IType2 *grad_scale) {
    const int index(i / sample_size);
    const DType u(loc[index]), s(scale[index]), sq(s*s), x(sample[i]);
    const DType scaling(grad_out[i]*(logpdf ? DType(1) : out[i]));
    grad_loc[i]    = scaling*(x-u)/sq;
    grad_scale[i]  = scaling*((x-u)*(x-u)-sq)/(sq*s);
    KERNEL_ASSIGN(grad_sample[i], req, scaling*(u-x)/sq);
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
  MSHADOW_XINLINE static void Map(int i, OpReqType req, int sample_size,
                  DType *out, IType1 *sample, IType2 *alpha, IType2 *beta,
                  DType *grad_out, IType1 *grad_sample, IType2 *grad_alpha, IType2 *grad_beta) {
    const int index(i / sample_size);
    const DType a(alpha[index]), b(beta[index]), x(sample[i]);
    const DType scaling(grad_out[i]*(logpdf ? DType(1) : out[i]));
    grad_alpha[i]  = scaling*(log(b)+log(x)-psi(a));
    grad_beta[i]   = scaling*(a/b-x);
    KERNEL_ASSIGN(grad_sample[i], req, scaling*((a-1)/x-b));
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
  MSHADOW_XINLINE static void Map(int i, OpReqType req, int sample_size,
                  DType *out, IType1 *sample, IType2 *lambda,
                  DType *grad_out, IType1 *grad_sample, IType2 *grad_lambda) {
    const int index(i / sample_size);
    const DType l(lambda[index]), x(sample[i]);
    const DType scaling(grad_out[i]*(logpdf ? DType(1) : out[i]));
    grad_lambda[i] = scaling*(DType(1)/l-x);
    KERNEL_ASSIGN(grad_sample[i], req, -scaling*l);
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
  MSHADOW_XINLINE static void Map(int i, OpReqType req, int sample_size,
                  DType *out, IType1 *sample, IType2 *lambda,
                  DType *grad_out, IType1 *grad_sample, IType2 *grad_lambda) {
    const int index(i / sample_size);
    const DType l(lambda[index]), x(sample[i]);
    const DType scaling(grad_out[i]*(logpdf ? DType(1) : out[i]));
    grad_lambda[i] = scaling*(x/l-DType(1));
    KERNEL_ASSIGN(grad_sample[i], req, 0);
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
  MSHADOW_XINLINE static void Map(int i, OpReqType req, int sample_size,
                  DType *out, IType1 *sample, IType2 *limit, IType2 *prob,
                  DType *grad_out, IType1 *grad_sample, IType2 *grad_limit, IType2 *grad_prob) {
    const int index(i / sample_size);
    std::pair<DType, DType> grads(LPDF_GRAD(DType(limit[index]), DType(prob[index]), DType(sample[i]), out[i], grad_out[i]));
    grad_limit[i]  = grads.first;
    grad_prob[i]   = grads.second;
    KERNEL_ASSIGN(grad_sample[i], req, 0);
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
  MSHADOW_XINLINE static void Map(int i, OpReqType req, int sample_size,
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
    KERNEL_ASSIGN(grad_sample[i], req, 0);
  }
};



template<bool logpdf>
struct PDF_Dirichlet {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(int i, int sample_size, int k, DType *out, IType1 *sample, IType2 *alpha) {
    const IType1 *cur_sample = sample+i*k;
    const IType2 *cur_alpha  = alpha+(i/sample_size)*k;
    DType sum_alpha(0), sum_lgamma(0), sum_sample(0);
    for( int j = 0; j < k; ++j ) {
      sum_alpha  += cur_alpha[j];
      sum_lgamma += lgamma(cur_alpha[j]);
      sum_sample += (cur_alpha[j]-1)*log(cur_sample[j]);
    }
    DType lpdf(sum_sample-sum_lgamma+lgamma(sum_alpha));
    out[i] = logpdf ? lpdf : DType(exp(lpdf));
  }
};


template<bool logpdf>
struct PDF_Dirichlet_Grad {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(int i, OpReqType req, int sample_size, int k, 
                  DType *out, IType1 *sample, IType2 *alpha,
                  DType *grad_out, IType1 *grad_sample, IType2 *grad_alpha) {
    // Digamma function
    const IType1 *cur_sample = sample+i*k;
    const IType2 *cur_alpha  = alpha+(i/sample_size)*k;
    const DType scaling(grad_out[i]*(logpdf ? DType(1) : out[i]));
    DType sum_alpha(0);
    for( int j = 0; j < k; ++j ) {
      sum_alpha += cur_alpha[j];
    }
    const DType psi_sum(psi(sum_alpha));
    for( int j = 0; j < k; ++j ) {
      // order grad_alpha differently to allow efficient reduction at the end. 
      grad_alpha[i%sample_size+sample_size*(j+k*(i/sample_size))] = scaling * (log(cur_sample[j])-psi(cur_alpha[j])+psi_sum);
      KERNEL_ASSIGN(grad_sample[i*k+j], req, scaling*(cur_alpha[j]-1)/cur_sample[j]);
    }
  }
};

struct PdfParam : public dmlc::Parameter<PdfParam> {
  bool is_log;
  DMLC_DECLARE_PARAMETER(PdfParam) {
    DMLC_DECLARE_FIELD(is_log).set_default(false)
    .describe("Whether to compute the log PDF or not.");
  }
};

template<bool vparm = false> 
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
    if (vparm) {
      *(tshape.end()-1) = *((*in_attrs)[0].end()-1);
    }
    for (size_t i = 1; i < in_attrs->size(); ++i) {
      SHAPE_ASSIGN_CHECK(*in_attrs, i, tshape);
    } 
    // Output shape must equal input tensor of samples except for last dimension if we are 
    // dealing with samples that are itself vectors. 
    TShape oshape((*in_attrs)[0].begin(), (*in_attrs)[0].end()-(vparm ? 1 : 0));
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
    return true;
  }
  return false;
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

template<typename xpu, typename DType, typename pdf, int pnum, bool vparm = false>
struct PdfCaller;

template<typename xpu, typename DType, typename pdf>
struct PdfCaller<xpu, DType, pdf, 1, false> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 mshadow::Stream<xpu> *s) {
    CHECK_EQ(inputs[0].Size()%inputs[1].Size(), 0);
    CHECK_EQ(inputs[0].Size()%outputs[0].Size(), 0);
    index_t num_samples(inputs[0].Size()/inputs[1].Size());
    mxnet_op::Kernel<LaunchExWrapper<pdf>, xpu>::LaunchEx(s, outputs[0].Size(), num_samples,
                outputs[0].dptr<DType>(), inputs[0].dptr<DType>(), inputs[1].dptr<DType>());
  }
};

template<typename xpu, typename DType, typename pdf>
struct PdfCaller<xpu, DType, pdf, 1, true> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 mshadow::Stream<xpu> *s) {
    CHECK_EQ(inputs[0].Size()%inputs[1].Size(), 0);
    CHECK_EQ(inputs[0].Size()%outputs[0].Size(), 0);
    index_t num_samples(inputs[0].Size()/inputs[1].Size());
    index_t sample_size(inputs[0].Size()/outputs[0].Size());
    // Covers distributons parametrized by a vector of parameters (Dirichlet distribution).
    mxnet_op::Kernel<LaunchExWrapper<pdf>, xpu>::LaunchEx(s, outputs[0].Size(), num_samples, sample_size,
                outputs[0].dptr<DType>(), inputs[0].dptr<DType>(), inputs[1].dptr<DType>());
  }
};

template<typename xpu, typename DType, typename pdf>
struct PdfCaller<xpu, DType, pdf, 2, false> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 mshadow::Stream<xpu> *s) {
    CHECK_EQ(inputs[0].Size()%inputs[1].Size(), 0);
    CHECK_EQ(inputs[0].Size(), outputs[0].Size());
    index_t num_samples(inputs[0].Size()/inputs[1].Size());
    mxnet_op::Kernel<LaunchExWrapper<pdf>, xpu>::LaunchEx(s, outputs[0].Size(), num_samples,
                outputs[0].dptr<DType>(), inputs[0].dptr<DType>(), 
                inputs[1].dptr<DType>(), inputs[2].dptr<DType>());
  }
};

template<typename xpu, template<bool> class pdf, int pnum, bool vparm>
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
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    if( param.is_log ) {
      PdfCaller<xpu, DType, pdf<true>, pnum, vparm>::op(inputs, outputs, s);
    } else {
      PdfCaller<xpu, DType, pdf<false>, pnum, vparm>::op(inputs, outputs, s);
    }
  });
}


template<typename xpu, typename DType, typename pdfgrad, int pnum, int vparm = false>
struct PdfGradCaller;

template<typename xpu, typename DType, typename pdfgrad>
struct PdfGradCaller<xpu, DType, pdfgrad, 1, false> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<TBlob>& grads,
                 mshadow::Stream<xpu> *s) {
    index_t num_samples(inputs[1].Size()/inputs[2].Size());
    mxnet_op::Kernel<LaunchExWrapper<pdfgrad>, xpu>::LaunchEx(s, inputs[0].Size(), req[0], num_samples,
                inputs[3].dptr<DType>(), inputs[1].dptr<DType>(), inputs[2].dptr<DType>(),
                inputs[0].dptr<DType>(), grads[0].dptr<DType>(), grads[1].dptr<DType>());
  }
};

template<typename xpu, typename DType, typename pdfgrad>
struct PdfGradCaller<xpu, DType, pdfgrad, 1, true> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<TBlob>& grads,
                 mshadow::Stream<xpu> *s) {
    index_t num_samples(inputs[1].Size()/inputs[2].Size());
    index_t sample_size(inputs[1].Size()/inputs[0].Size());
    mxnet_op::Kernel<LaunchExWrapper<pdfgrad>, xpu>::LaunchEx(s, inputs[0].Size(), req[0], num_samples, sample_size,
                inputs[3].dptr<DType>(), inputs[1].dptr<DType>(), inputs[2].dptr<DType>(),
                inputs[0].dptr<DType>(), grads[0].dptr<DType>(), grads[1].dptr<DType>());
  }
};

template<typename xpu, typename DType, typename pdfgrad>
struct PdfGradCaller<xpu, DType, pdfgrad, 2, false> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<TBlob>& grads,
                 mshadow::Stream<xpu> *s) {
    index_t num_samples(inputs[1].Size()/inputs[2].Size());
    mxnet_op::Kernel<LaunchExWrapper<pdfgrad>, xpu>::LaunchEx(s, inputs[0].Size(), req[0], num_samples,
                inputs[4].dptr<DType>(), inputs[1].dptr<DType>(), inputs[2].dptr<DType>(), inputs[3].dptr<DType>(),
                inputs[0].dptr<DType>(), grads[0].dptr<DType>(), grads[1].dptr<DType>(), grads[2].dptr<DType>());
  }
};

template<typename xpu, template<bool> class pdfgrad, int pnum, bool vparm>
void PdfOpBackward(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), pnum+3);
  CHECK_EQ(outputs.size(), pnum+1);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const PdfParam& param = nnvm::get<PdfParam>(attrs.parsed);
  const size_t N(outputs[1].Size());
  const TShape src_shape(Shape2(N, outputs[0].Size()/N)), dst_shape(Shape2(N, 1));
  // Inputs to PdfOpBackward: grad, samples, parm1, parm2, pdf.
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    const size_t red_work_size(broadcast::ReduceWorkspaceSize<2, DType>(s, dst_shape, kAddTo, src_shape));
    const size_t tmp_size(outputs[0].Size()*pnum*sizeof(DType)+red_work_size);
    Tensor<xpu, 1, char> tmp_space = ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(tmp_size), s);
    std::vector<TBlob> grads = {outputs[0]};
    grads.push_back(TBlob(tmp_space.dptr_, outputs[0].shape_,
                          outputs[1].dev_mask(), outputs[1].type_flag_, outputs[1].dev_id()));
    if (pnum == 2) {
      grads.push_back(TBlob(tmp_space.dptr_+outputs[0].Size()*sizeof(DType), outputs[0].shape_,
                            outputs[2].dev_mask(), outputs[2].type_flag_, outputs[2].dev_id()));
    }
    if (param.is_log) {
      PdfGradCaller<xpu, DType, pdfgrad<true>, pnum, vparm>::op(inputs, req, grads, s);
    } else {
      PdfGradCaller<xpu, DType, pdfgrad<false>, pnum, vparm>::op(inputs, req, grads, s);
    }
    Tensor<xpu, 1, char> red_work(tmp_space.dptr_+pnum*outputs[0].Size()*sizeof(DType), Shape1(red_work_size), s);
    broadcast::Reduce<red::sum, 2, DType, op::mshadow_op::identity>(
       s, outputs[1].reshape(dst_shape), req[1], red_work, grads[1].reshape(src_shape));
    if (pnum == 2) {
      broadcast::Reduce<red::sum, 2, DType, op::mshadow_op::identity>(
       s, outputs[2].reshape(dst_shape), req[2], red_work, grads[2].reshape(src_shape));
    }
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_RANDOM_PDF_OP_H_
