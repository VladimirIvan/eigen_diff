#include <Eigen/Dense>
#include <eigen_diff/AutoDiffChainJacobian.h>
#include <eigen_diff/AutoDiffChainHessian.h>

#include <iostream>

template <typename Scalar, int CompileTimeInputSize, int CompileTimeValueSize, int CompileTimeJacobianCols>
struct FunctorBase
{
    typedef Eigen::Matrix<Scalar, CompileTimeInputSize, 1> InputType;
    typedef Eigen::Matrix<Scalar, CompileTimeValueSize, 1> ValueType;
    enum {JacobianColsAtCompileTime = CompileTimeJacobianCols};
    FunctorBase() = default;
};

typedef FunctorBase<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic> Functor;

// This function combines Function3(Function2(x)).
// The result will be used for comparison with passing the derivative of Function2 into the AutoDiff.
struct Function1 : public Functor
{
    template <typename T>
    void operator()(const Eigen::Matrix<T, InputType::RowsAtCompileTime, 1> &x, Eigen::Matrix<T, ValueType::RowsAtCompileTime, 1> &y) const
    {
        Eigen::Matrix<T, ValueType::RowsAtCompileTime, 1> tmp(y.rows());
        // Always cast known scalar type matrices/vectors into the templated type <T>.
        // This is required for AutoDiff to work properly.
        for(int i=0; i<4; i++)
        {
            y(i, 0) = (Eigen::AngleAxis<T>(x(i, 0), Eigen::Vector3d::UnitZ().cast<T>()).toRotationMatrix() * Eigen::Vector3d::UnitX().cast<T>()).dot(Eigen::Vector3d::UnitX().cast<T>());
            tmp(i, 0) = y(i, 0);
            if(i>0) y(i, 0) += tmp(i-1, 0);
        }
    }
};

// Function2 rotates a UnitX vector around Z axis.
// This is a helper function that will provide input for Function3.
struct Function2 : public Functor
{
    template <typename T>
    void operator()(const Eigen::Matrix<T, InputType::RowsAtCompileTime, 1> &x, Eigen::Matrix<T, ValueType::RowsAtCompileTime, 1> &y) const
    {
        Eigen::Matrix<T, ValueType::RowsAtCompileTime, 1> tmp(y.rows());
        for(int i=0; i<4; i++)
        {
            y.block(i*3,0,3,1) = Eigen::AngleAxis<T>(x(i, 0), Eigen::Vector3d::UnitZ().cast<T>()).toRotationMatrix() * Eigen::Vector3d::UnitX().cast<T>();
            tmp.block(i*3,0,3,1) = y.block(i*3,0,3,1);
            if(i>0) y.block(i*3,0,3,1) += tmp.block((i-1)*3,0,3,1);
        }
    }
};

// This function computes dot product between the input vector and UnitX.
// The input vector will be taken from the output of Function2 (including the derivatives).
struct Function3 : public Functor
{
    template <typename T>
    void operator()(const Eigen::Matrix<T, InputType::RowsAtCompileTime, 1> &x, Eigen::Matrix<T, ValueType::RowsAtCompileTime, 1> &y) const
    {
        Eigen::Matrix<T, ValueType::RowsAtCompileTime, 1> tmp(y.rows());
        for(int i=0; i<4; i++)
        {
            y(i, 0) = Eigen::Vector3d::UnitX().cast<T>().dot(x.block(i*3,0,3,1));
            tmp(i, 0) = y(i, 0);
            if(i>0) y(i, 0) += tmp(i-1, 0);
        }
    }
};

typedef double Scalar;

typedef Eigen::AutoDiffChainHessian<Function1>::InputType InputType1;
typedef Eigen::AutoDiffChainHessian<Function1>::ValueType ValueType1;
typedef Eigen::AutoDiffChainHessian<Function1>::InputJacobianType InputJacobianType1;
typedef Eigen::AutoDiffChainHessian<Function1>::JacobianType JacobianType1;
typedef Eigen::AutoDiffChainHessian<Function1>::InputHessianType InputHessianType1;
typedef Eigen::AutoDiffChainHessian<Function1>::HessianType HessianType1;

typedef Eigen::AutoDiffChainHessian<Function2>::InputType InputType2;
typedef Eigen::AutoDiffChainHessian<Function2>::ValueType ValueType2;
typedef Eigen::AutoDiffChainHessian<Function2>::InputJacobianType InputJacobianType2;
typedef Eigen::AutoDiffChainHessian<Function2>::JacobianType JacobianType2;
typedef Eigen::AutoDiffChainHessian<Function2>::InputHessianType InputHessianType2;
typedef Eigen::AutoDiffChainHessian<Function2>::HessianType HessianType2;

typedef Eigen::AutoDiffChainHessian<Function3>::InputType InputType3;
typedef Eigen::AutoDiffChainHessian<Function3>::ValueType ValueType3;
typedef Eigen::AutoDiffChainHessian<Function3>::InputJacobianType InputJacobianType3;
typedef Eigen::AutoDiffChainHessian<Function3>::JacobianType JacobianType3;
typedef Eigen::AutoDiffChainHessian<Function3>::InputHessianType InputHessianType3;
typedef Eigen::AutoDiffChainHessian<Function3>::HessianType HessianType3;

void JacobianFull(const InputType1& x)
{
    Function1 f;
    Eigen::AutoDiffChainJacobian<Function1> autoj(f);
    ValueType1 y(4, 1);
    JacobianType1 j(4, 4);
    
    // Compute full Jacobian
    autoj(x, y, j);

    std::cout << "Real value function...\n";
    std::cout << "x: " << x.transpose() << "\n";
    std::cout << "y: " << y.transpose() << "\n";
    std::cout << "J:\n"<< j << "\n";
}

void JacobianIntermediate(const InputType2& x, ValueType2& y, JacobianType2& j)
{
    Function2 f;
    Eigen::AutoDiffChainJacobian<Function2> autoj(f);

    // Compute 3D vector Jacobian (only used as input into Function3)
    autoj(x, y, j);

    std::cout << "Intermediate function...\n";
    std::cout << "x: " << x.transpose() << "\n";
    std::cout << "y: " << y.transpose() << "\n";
    std::cout << "J:\n"<< j << "\n";
}

void JacobianCompound(const InputType3& x, const InputJacobianType3& ij)
{
    Function3 f;
    Eigen::AutoDiffChainJacobian<Function3> autoj(f);
    ValueType3 y(4, 1);
    JacobianType3 j(4, 4);

    // Compute the Jacobian of the compound function.
    autoj(x, y, j, ij);

    std::cout << "Compund function...\n";
    std::cout << "x: " << x.transpose() << "\n";
    std::cout << "y: " << y.transpose() << "\n";
    std::cout << "J:\n"<< j << "\n";
}

void HessianFull(const InputType1& x)
{
    Function1 f;
    Eigen::AutoDiffChainHessian<Function1> autoj(f);
    ValueType1 y(4, 1);
    JacobianType1 j(4, 4);
    HessianType1 hess;
    
    // Compute full Jacobian and Hessian
    autoj(x, y, j, hess);

    std::cout << "Real value function...\n";
    std::cout << "x: " << x.transpose() << "\n";
    std::cout << "y: " << y.transpose() << "\n";
    std::cout << "J:\n"<< j << "\n";
    std::cout << "H:\n";
    for(int i=0;i<hess.rows();i++) std::cout << "---------\n" << hess(i) << "\n";
}

void HessianIntermediate(const InputType2& x, ValueType2& y, JacobianType2& j, HessianType2& hess)
{
    Function2 f;
    Eigen::AutoDiffChainHessian<Function2> autoj(f);

    // Compute 3D vector Jacobian and Hessian (only used as input into Function3)
    autoj(x, y, j, hess);

    std::cout << "Intermediate function...\n";
    std::cout << "x: " << x.transpose() << "\n";
    std::cout << "y: " << y.transpose() << "\n";
    std::cout << "J:\n"<< j << "\n";
    std::cout << "H:\n";
    for(int i=0;i<hess.rows();i++) std::cout << "---------\n" << hess(i) << "\n";
}

void HessianCompound(const InputType3& x, const InputJacobianType3& ij, const InputHessianType3& ihess)
{
    Function3 f;
    Eigen::AutoDiffChainHessian<Function3> autoj(f);
    ValueType3 y(4, 1);
    JacobianType3 j(4, 4);
    HessianType3 hess;

    // Compute the Jacobian and Hessian of the compound function.
    autoj(x, y, j, hess, ij, ihess);

    std::cout << "Compund function...\n";
    std::cout << "x: " << x.transpose() << "\n";
    std::cout << "y: " << y.transpose() << "\n";
    std::cout << "J:\n"<< j << "\n";
    std::cout << "H:\n";
    for(int i=0;i<hess.rows();i++) std::cout << "---------\n" << hess(i) << "\n";
}

int main(int argc, char **argv)
{
    InputType1 x(4, 1);
    x(0) = 0.5;
    x(1) = 0.6;
    x(2) = 0.7;
    x(3) = 0.8;

    ValueType2 y(3*4, 1);
    JacobianType2 j(3*4, 4);
    HessianType2 hess;

    std::cout << "\nJacobian example\n";
    JacobianFull(x);
    JacobianIntermediate(x, y, j);
    JacobianCompound(y, j);

    std::cout << "\nHessian example\n";
    HessianFull(x);
    HessianIntermediate(x, y, j, hess);
    HessianCompound(y, j, hess);

    return 0;
}