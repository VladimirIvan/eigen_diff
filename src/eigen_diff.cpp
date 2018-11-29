#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

#include <iostream>

// Modified from unsupported/Eigen/src/AutoDiff/AutoDiffJacobian.h
namespace Eigen
{

template <typename Functor>
class AutoDiffChainJacobian : public Functor
{
  public:
    AutoDiffChainJacobian() : Functor() {}
    AutoDiffChainJacobian(const Functor &f) : Functor(f) {}

    // forward constructors
#if EIGEN_HAS_VARIADIC_TEMPLATES
    template <typename... T>
    AutoDiffChainJacobian(const T &... Values) : Functor(Values...)
    {
    }
#else
    template <typename T0>
    AutoDiffChainJacobian(const T0 &a0) : Functor(a0)
    {
    }
    template <typename T0, typename T1>
    AutoDiffChainJacobian(const T0 &a0, const T1 &a1) : Functor(a0, a1) {}
    template <typename T0, typename T1, typename T2>
    AutoDiffChainJacobian(const T0 &a0, const T1 &a1, const T2 &a2) : Functor(a0, a1, a2) {}
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

    typedef Matrix<Scalar, InputsAtCompileTime, JacobianInputsAtCompileTime> InputJacobianType; // Jacobian.cols() matches InputJacobian.cols()
    typedef typename JacobianType::Index Index;

    typedef Matrix<Scalar, JacobianInputsAtCompileTime, 1> DerivativeType; // Derivative rows() matches InputJacobian.cols()
    typedef AutoDiffScalar<DerivativeType> ActiveScalar;

    typedef Matrix<ActiveScalar, InputsAtCompileTime, 1> ActiveInput;
    typedef Matrix<ActiveScalar, ValuesAtCompileTime, 1> ActiveValue;

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
    void operator()(const InputType &x, ValueType *v, JacobianType *_jac, InputJacobianType *_ijac = 0,
                    const ParamsType &... Params) const
#else
    void operator()(const InputType &x, ValueType *v, JacobianType *_jac = 0, InputJacobianType *_ijac = 0) const
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

        JacobianType &jac = *_jac;

        ActiveInput ax = x.template cast<ActiveScalar>();
        ActiveValue av(jac.rows());

        if (!_ijac)
        {
            eigen_assert(InputsAtCompileTime == JacobianInputsAtCompileTime);

            if (InputsAtCompileTime == Dynamic)
                for (Index j = 0; j < jac.rows(); j++)
                    av[j].derivatives().resize(x.rows());

            for (Index i = 0; i < jac.cols(); i++)
                ax[i].derivatives() = DerivativeType::Unit(x.rows(), i);
        }
        else
        {
            // If specified, copy derivatives from InputJacobian
            InputJacobianType &ijac = *_ijac;

            if (InputsAtCompileTime == Dynamic)
                for (Index j = 0; j < jac.rows(); j++)
                    av[j].derivatives().resize(ijac.cols());

            for (Index i = 0; i < jac.cols(); i++)
                ax[i].derivatives() = ijac.row(i);
        }

#if EIGEN_HAS_VARIADIC_TEMPLATES
        Functor::operator()(ax, &av, Params...);
#else
        Functor::operator()(ax, &av);
#endif

        for (Index i = 0; i < jac.rows(); i++)
        {
            (*v)[i] = av[i].value();
            jac.row(i) = av[i].derivatives();
        }
    }
};

} // namespace Eigen

// This function combines Function3(Function2(x)).
// The result will be used for comparison with passing the derivative of Function2 into the AutoDiff.
template <typename Scalar, int NX = 1, int NY = 1>
struct Function1
{
    typedef Eigen::Matrix<Scalar, NX, 1> InputType;    // Rotation angle
    typedef Eigen::Matrix<Scalar, NY, 1> ValueType;    // Vector dot product
    typedef Eigen::Matrix<Scalar, NY, 1> JacobianType; // Derivative

    Function1() {}

    template <typename T1, typename T2>
    void operator()(const Eigen::Matrix<T1, NX, 1> &x, Eigen::Matrix<T2, NY, 1> *_y) const
    {
        Eigen::Matrix<T2, NY, 1> &y = *_y;
        y(0, 0) = (Eigen::AngleAxis<T1>(x(0, 0), Eigen::Vector3d::UnitZ()).toRotationMatrix() * Eigen::Vector3d::UnitX()).dot(Eigen::Vector3d::UnitX());
    }
};

// Function2 rotates a UnitX vector around Z axis.
// This is a helper function that will provide input for Function3.
template <typename Scalar, int NX = 1, int NY = 3>
struct Function2
{
    typedef Eigen::Matrix<Scalar, NX, 1> InputType;    // Rotation angle
    typedef Eigen::Matrix<double, NY, 1> ValueType;    // 3D vector
    typedef Eigen::Matrix<double, NY, 1> JacobianType; // Derivative

    Function2() {}

    template <typename T1, typename T2>
    void operator()(const Eigen::Matrix<T1, NX, 1> &x, Eigen::Matrix<T2, NY, 1> *_y) const
    {
        Eigen::Matrix<T2, NY, 1> &y = *_y;
        y = Eigen::AngleAxis<T1>(x(0, 0), Eigen::Vector3d::UnitZ()).toRotationMatrix() * Eigen::Vector3d::UnitX();
    }
};

// This function computes dot product between the input vector and UnitX.
// The input vector will be taken from the output of Function2 (including the derivatives).
template <typename Scalar, int NX = 3, int NY = 1>
struct Function3
{
    typedef Eigen::Matrix<Scalar, NX, 1> InputType;    // 3D vector
    typedef Eigen::Matrix<double, NY, 1> ValueType;    // Dot product
    typedef Eigen::Matrix<double, NY, 1> JacobianType; // Derivative

    Function3() {}

    template <typename T1, typename T2>
    void operator()(const Eigen::Matrix<T1, NX, 1> &x, Eigen::Matrix<T2, NY, 1> *_y) const
    {
        Eigen::Matrix<T2, NY, 1> &y = *_y;
        y(0, 0) = Eigen::Vector3d::UnitX().dot(x);
    }
};

int main(int argc, char **argv)
{
    typedef double Scalar;
    // Note that Eigen::AutoDiffChainJacobian<Function3<Scalar>>::InputJacobianType == Function2<Scalar>::JacobianType
    Eigen::AutoDiffChainJacobian<Function3<Scalar>>::InputJacobianType ij;
    Function3<Scalar>::InputType x3;
    Function1<Scalar>::InputType x;
    x(0) = 0.5;

    {
        // Compute full derivative
        Function1<Scalar>::ValueType y;
        Function1<Scalar> f;
        Eigen::AutoDiffChainJacobian<Function1<Scalar>> autoj(f);
        Function1<Scalar>::JacobianType j;
        autoj(x, &y, &j);
        std::cout << "Real value function...\n";
        std::cout << "x: " << x.transpose() << "\n";
        std::cout << "y: " << y.transpose() << "\n";
        std::cout << "J: " << j << "\n";
    }
    {
        // Compute 3D vector and its derivative
        Function2<Scalar>::ValueType y;
        Function2<Scalar> f;
        Eigen::AutoDiffChainJacobian<Function2<Scalar>> autoj(f);
        Function2<Scalar>::JacobianType j;
        autoj(x, &y, &j);
        // Use the output of this function (including the derivative)
        // to compute the derivative of the compound function.
        ij = j;
        x3 = y;
    }
    {
        // Compute the derivative of the compound function.
        Function3<Scalar>::ValueType y;
        Function3<Scalar> f;
        Eigen::AutoDiffChainJacobian<Function3<Scalar>> autoj(f);
        Function3<Scalar>::JacobianType j;
        autoj(x3, &y, &j, &ij); // Pass in the
        std::cout << "Compund function...\n";
        std::cout << "x: " << x.transpose() << "\n";
        std::cout << "y: " << y.transpose() << "\n";
        std::cout << "J: "<< j.sum() << "\n";
    }
    return 0;
}