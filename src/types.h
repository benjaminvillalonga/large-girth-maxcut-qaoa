#include <Eigen/Dense>

namespace types
{
  typedef double Real;
  typedef std::complex<Real> Complex;
  typedef Eigen::Vector<Real, Eigen::Dynamic> VectorReal;
  typedef Eigen::Vector<Complex, Eigen::Dynamic> VectorComplex;
  typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> MatrixReal;
  typedef Eigen::Matrix<Complex,
                        Eigen::Dynamic, Eigen::Dynamic> MatrixComplex;
  typedef unsigned long long int bitstring;
}
