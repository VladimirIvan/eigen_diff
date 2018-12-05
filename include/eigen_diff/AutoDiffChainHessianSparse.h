// Copyright (C) 2018 Vladimir Ivan <v.ivan@ed.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Modified from unsupported/Eigen/src/AutoDiff/AutoDiffJacobian.h

#ifndef EIGEN_AUTODIFF_CHAIN_HESSIAN_SPARSE_H
#define EIGEN_AUTODIFF_CHAIN_HESSIAN_SPARSE_H

#include <eigen_diff/AutoDiffScalar.h>
#include <eigen_diff/AutoDiffChainJacobianSparse.h>

namespace Eigen
{

template <typename Functor>
class AutoDiffChainHessianSparse : public Functor
{
  public:
    AutoDiffChainHessianSparse() : Functor() {}
    AutoDiffChainHessianSparse(const Functor &f) : Functor(f) {}

    // forward constructors
#if EIGEN_HAS_VARIADIC_TEMPLATES
    template <typename... T>
    AutoDiffChainHessianSparse(const T &... Values) : Functor(Values...)
    {
    }
#else
    template <typename T0>
    AutoDiffChainHessianSparse(const T0 &a0) : Functor(a0)
    {
    }
    template <typename T0, typename T1>
    AutoDiffChainHessianSparse(const T0 &a0, const T1 &a1) : Functor(a0, a1) {}
    template <typename T0, typename T1, typename T2>
    AutoDiffChainHessianSparse(const T0 &a0, const T1 &a1, const T2 &a2) : Functor(a0, a1, a2) {}
#endif

    typedef typename Functor::InputType InputType;
    typedef typename Functor::ValueType ValueType;
    typedef typename Functor::JacobianType JacobianType; // New definition of JacobianType
    typedef typename ValueType::Scalar Scalar;

    enum
    {
        InputsAtCompileTime = InputType::RowsAtCompileTime,
        ValuesAtCompileTime = ValueType::RowsAtCompileTime,
        JacobianInputsAtCompileTime = JacobianType::ColsAtCompileTime // Jacobian.cols() no longer have to match Input.rows()
    };

    typedef SparseMatrix<Scalar> InputJacobianType; // Jacobian.cols() matches InputJacobian.cols()
    typedef Array<SparseMatrix<Scalar>, ValuesAtCompileTime, 1> HessianType;
    typedef Array<SparseMatrix<Scalar>, InputsAtCompileTime, 1> InputHessianType;
    typedef typename JacobianType::Index Index;

    typedef SparseVector<Scalar> InnerDerivativeType; // Derivative rows() matches InputJacobian.cols()
    typedef AutoDiffScalar<InnerDerivativeType> InnerActiveScalar;
    typedef SparseVector<InnerActiveScalar> OuterDerivativeType;
    typedef AutoDiffScalar<OuterDerivativeType> OuterActiveScalar;

    typedef typename OuterDerivativeType::InnerIterator JacobianInnerIteratorType;
    typedef typename InnerDerivativeType::InnerIterator HessianInnerIteratorType;

    typedef Matrix<OuterActiveScalar, InputsAtCompileTime, 1> ActiveInput;
    typedef Matrix<OuterActiveScalar, ValuesAtCompileTime, 1> ActiveValue;

#if EIGEN_HAS_VARIADIC_TEMPLATES
    // Some compilers don't accept variadic parameters after a default parameter,
    // i.e., we can't just write _jac=0 but we need to overload operator():
    EIGEN_STRONG_INLINE
    void operator()(const InputType &x, ValueType *v) const
    {
        this->operator()(x, v, 0);
    }

    // Optional parameter InputJacobian (_ijac)
    template <typename... ParamsType>
    void operator()(const InputType &x, ValueType *v, JacobianType *_jac, HessianType *_hess, const InputJacobianType *_ijac = 0, const InputHessianType *_ihess = 0,
                    const ParamsType &... Params) const
#else
    void operator()(const InputType &x, ValueType *v, JacobianType *_jac = 0, HessianType *_hess, const InputJacobianType *_ijac = 0, const InputHessianType *_ihess = 0) const
#endif
    {
        eigen_assert(v != 0);

        if (!_jac)
        {
#if EIGEN_HAS_VARIADIC_TEMPLATES
            Functor::operator()(x, v, Params...);
#else
            Functor::operator()(x, v);
#endif
            return;
        }

        if (!_hess)
        {
            AutoDiffChainJacobianSparse<Functor> autoj(*static_cast<const Functor *>(this));
#if EIGEN_HAS_VARIADIC_TEMPLATES
            autoj(x, v, _jac, _ijac, Params...);
#else
            autoj(x, v, _jac, _ijac);
#endif
            return;
        }

        JacobianType &jac = *_jac;

        ActiveInput ax = x.template cast<OuterActiveScalar>();
        ActiveValue av(jac.rows());

        // Provide either both input jacobian and hessian, or none
        eigen_assert((_ijac && _ihess) || (!_ijac && !_ihess));

        if (!_ijac)
        {
            eigen_assert(x.rows() == jac.cols());

            for (Index j = 0; j < jac.rows(); j++)
            {
                av[j].derivatives().resize(x.rows());
                av[j].derivatives().reserve(x.rows());
                for (Index k = 0; k < x.rows(); k++)
                    av[j].derivatives().insert(k).derivatives().resize(x.rows());
            }

            for (Index i = 0; i < x.rows(); i++)
            {
                ax[i].derivatives().resize(x.rows());
                ax[i].derivatives().insert(i) = 1.0;
                ax[i].value().derivatives().resize(x.rows());
                ax[i].value().derivatives().insert(i) = 1.0;
                ax[i].derivatives().coeffRef(i).derivatives().resize(x.rows());
            }
        }
        else
        {
            // If specified, copy derivatives from InputJacobian
            const InputJacobianType &ijac = *_ijac;
            const InputHessianType &ihess = *_ihess;

            eigen_assert(x.rows() == ihess.rows());
            eigen_assert(ijac.cols() == ihess[0].rows() && ijac.cols() == ihess[0].cols());

            for (Index j = 0; j < jac.rows(); j++)
            {
                av[j].derivatives().resize(ijac.cols());
                av[j].derivatives().reserve(ijac.cols());
                for (Index k = 0; k < ijac.cols(); k++)
                    av[j].derivatives().insert(k).derivatives().resize(ijac.cols());
            }

            for (Index i = 0; i < x.rows(); i++)
            {
                ax[i].derivatives().resize(ijac.cols());
                ax[i].derivatives() = ijac.row(i);
                ax[i].value().derivatives().resize(ijac.cols());
                ax[i].value().derivatives() = ijac.row(i);
                for (Index k = 0; k < ijac.cols(); k++)
                {
                    ax[i].derivatives().coeffRef(k).derivatives() = ihess[i].row(k);
                }
            }
        }

#if EIGEN_HAS_VARIADIC_TEMPLATES
        Functor::operator()(ax, &av, Params...);
#else
        Functor::operator()(ax, &av);
#endif

        HessianType &hess = *_hess;
        Index cols = _ijac ? _ijac->cols() : x.rows();
        {
            hess.resize(jac.rows());
            for (Index i = 0; i < jac.rows(); i++)
                hess[i].resize(cols, cols);
        }

        for (int i = 0; i < jac.rows(); ++i)
        {
            (*v)[i] = av[i].value().value();
            for (JacobianInnerIteratorType it(av[i].derivatives(), 0); it; ++it)
            {
                jac.insert(i, it.row()) = av[i].value().derivatives().coeffRef(it.row());
                for (HessianInnerIteratorType ith(av[i].derivatives().coeffRef(it.row()).derivatives(), 0); ith; ++ith)
                {
                    hess[i].insert(it.row(), ith.row()) = av[i].derivatives().coeffRef(it.row()).derivatives().coeffRef(ith.row());
                }
            }
        }
    }
};

} // namespace Eigen

#endif
