
#include <Eigen/Dense>
#include <eigen_diff/AutoDiffChainJacobian.h>
#include <eigen_diff/AutoDiffChainHessian.h>

#include <iostream>

// This function combines Function3(Function2(x)).
// The result will be used for comparison with passing the derivative of Function2 into the AutoDiff.
template <typename Scalar>
struct Function1
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> InputType;    // Rotation angle
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> ValueType;    // Vector dot product
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> JacobianType; // Derivative

    Function1() {}

    template <typename T>
    void operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1> &x, Eigen::Matrix<T, Eigen::Dynamic, 1> *_y) const
    {
        Eigen::Matrix<T, Eigen::Dynamic, 1> &y = *_y;
        // Always cast known scalar type matrices/vectors into the templated type <T>.
        // This is required for AutoDiff to work properly.
        y(0, 0) = (Eigen::AngleAxis<T>(x(0, 0), Eigen::Vector3d::UnitZ().cast<T>()).toRotationMatrix() * Eigen::Vector3d::UnitX().cast<T>()).dot(Eigen::Vector3d::UnitX().cast<T>());
    }
};

// Function2 rotates a UnitX vector around Z axis.
// This is a helper function that will provide input for Function3.
template <typename Scalar>
struct Function2
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> InputType;    // Rotation angle
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> ValueType;    // 3D vector
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> JacobianType; // Derivative

    Function2() {}

    template <typename T>
    void operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1> &x, Eigen::Matrix<T, Eigen::Dynamic, 1> *_y) const
    {
        Eigen::Matrix<T, Eigen::Dynamic, 1> &y = *_y;
        y = Eigen::AngleAxis<T>(x(0, 0), Eigen::Vector3d::UnitZ().cast<T>()).toRotationMatrix() * Eigen::Vector3d::UnitX().cast<T>();
    }
};

// This function computes dot product between the input vector and UnitX.
// The input vector will be taken from the output of Function2 (including the derivatives).
template <typename Scalar>
struct Function3
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> InputType;    // 3D vector
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> ValueType;    // Dot product
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> JacobianType; // Derivative

    Function3() {}

    template <typename T>
    void operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1> &x, Eigen::Matrix<T, Eigen::Dynamic, 1> *_y) const
    {
        Eigen::Matrix<T, Eigen::Dynamic, 1> &y = *_y;
        y(0, 0) = Eigen::Vector3d::UnitX().cast<T>().dot(x);
    }
};

typedef double Scalar;

typedef Eigen::AutoDiffChainHessian<Function1<Scalar>>::InputType InputType1;
typedef Eigen::AutoDiffChainHessian<Function1<Scalar>>::ValueType ValueType1;
typedef Eigen::AutoDiffChainHessian<Function1<Scalar>>::InputJacobianType InputJacobianType1;
typedef Eigen::AutoDiffChainHessian<Function1<Scalar>>::JacobianType JacobianType1;
typedef Eigen::AutoDiffChainHessian<Function1<Scalar>>::InputHessianType InputHessianType1;
typedef Eigen::AutoDiffChainHessian<Function1<Scalar>>::HessianType HessianType1;

typedef Eigen::AutoDiffChainHessian<Function2<Scalar>>::InputType InputType2;
typedef Eigen::AutoDiffChainHessian<Function2<Scalar>>::ValueType ValueType2;
typedef Eigen::AutoDiffChainHessian<Function2<Scalar>>::InputJacobianType InputJacobianType2;
typedef Eigen::AutoDiffChainHessian<Function2<Scalar>>::JacobianType JacobianType2;
typedef Eigen::AutoDiffChainHessian<Function2<Scalar>>::InputHessianType InputHessianType2;
typedef Eigen::AutoDiffChainHessian<Function2<Scalar>>::HessianType HessianType2;

typedef Eigen::AutoDiffChainHessian<Function3<Scalar>>::InputType InputType3;
typedef Eigen::AutoDiffChainHessian<Function3<Scalar>>::ValueType ValueType3;
typedef Eigen::AutoDiffChainHessian<Function3<Scalar>>::InputJacobianType InputJacobianType3;
typedef Eigen::AutoDiffChainHessian<Function3<Scalar>>::JacobianType JacobianType3;
typedef Eigen::AutoDiffChainHessian<Function3<Scalar>>::InputHessianType InputHessianType3;
typedef Eigen::AutoDiffChainHessian<Function3<Scalar>>::HessianType HessianType3;

void JacobianFull(const InputType1& x)
{
    Function1<Scalar> f;
    Eigen::AutoDiffChainJacobian<Function1<Scalar>> autoj(f);
    ValueType1 y(1, 1);
    JacobianType1 j(1, 1);
    
    // Compute full Jacobian
    autoj(x, &y, &j);

    std::cout << "Real value function...\n";
    std::cout << "x: " << x.transpose() << "\n";
    std::cout << "y: " << y.transpose() << "\n";
    std::cout << "J: " << j << "\n";
}

void JacobianIntermediate(const InputType2& x, ValueType2& y, JacobianType2& j)
{
    Function2<Scalar> f;
    Eigen::AutoDiffChainJacobian<Function2<Scalar>> autoj(f);

    // Compute 3D vector Jacobian (only used as input into Function3)
    autoj(x, &y, &j);

    std::cout << "Intermediate function...\n";
    std::cout << "x: " << x.transpose() << "\n";
    std::cout << "y: " << y.transpose() << "\n";
    std::cout << "J: " << j.transpose() << "\n";
}

void JacobianCompound(const InputType3& x, const InputJacobianType3& ij)
{
    Function3<Scalar> f;
    Eigen::AutoDiffChainJacobian<Function3<Scalar>> autoj(f);
    ValueType3 y(1, 1);
    JacobianType3 j(1, 1);

    // Compute the Jacobian of the compound function.
    autoj(x, &y, &j, &ij);

    std::cout << "Compund function...\n";
    std::cout << "x: " << x.transpose() << "\n";
    std::cout << "y: " << y.transpose() << "\n";
    std::cout << "J: "<< j << "\n";
}

void HessianFull(const InputType1& x)
{
    Function1<Scalar> f;
    Eigen::AutoDiffChainHessian<Function1<Scalar>> autoj(f);
    ValueType1 y(1, 1);
    JacobianType1 j(1, 1);
    HessianType1 hess;
    
    // Compute full Jacobian and Hessian
    autoj(x, &y, &j, &hess);

    std::cout << "Real value function...\n";
    std::cout << "x: " << x.transpose() << "\n";
    std::cout << "y: " << y.transpose() << "\n";
    std::cout << "J: " << j << "\n";
    std::cout << "H: " << hess(0) << "\n";
}

void HessianIntermediate(const InputType2& x, ValueType2& y, JacobianType2& j, HessianType2& hess)
{
    Function2<Scalar> f;
    Eigen::AutoDiffChainHessian<Function2<Scalar>> autoj(f);

    // Compute 3D vector Jacobian and Hessian (only used as input into Function3)
    autoj(x, &y, &j, &hess);

    std::cout << "Intermediate function...\n";
    std::cout << "x: " << x.transpose() << "\n";
    std::cout << "y: " << y.transpose() << "\n";
    std::cout << "J: " << j.transpose() << "\n";
    std::cout << "H: " << hess(0) << " " << hess(1) << " " << hess(2) << "\n";
}

void HessianCompound(const InputType3& x, const InputJacobianType3& ij, const InputHessianType3& ihess)
{
    Function3<Scalar> f;
    Eigen::AutoDiffChainHessian<Function3<Scalar>> autoj(f);
    ValueType3 y(1, 1);
    JacobianType3 j(1, 1);
    HessianType3 hess;

    // Compute the Jacobian and Hessian of the compound function.
    autoj(x, &y, &j, &hess, &ij, &ihess);

    std::cout << "Compund function...\n";
    std::cout << "x: " << x.transpose() << "\n";
    std::cout << "y: " << y.transpose() << "\n";
    std::cout << "J: "<< j << "\n";
    std::cout << "H: " << hess(0) << "\n";
}

int main(int argc, char **argv)
{
    InputType1 x(1, 1);
    x(0) = 0.5;

    ValueType2 y(3, 1);
    JacobianType2 j(3, 1);
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