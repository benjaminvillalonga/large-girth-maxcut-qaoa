#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <LBFGS.h>

#include "matrix_builder.h"
#include "types.h"

int main(int argc, char** argv)
{

  size_t precision = 8;

  std::string output_filename = argv[1];
  size_t q = std::stoi(argv[2]);
  size_t p = std::stoi(argv[3]);
  types::Real delta = std::stod(argv[4]);
  types::VectorReal gammas = types::VectorReal::Zero(p);
  types::VectorReal betas = types::VectorReal::Zero(p);
  if (argc == 5 + 2 * p)
  {
    for (size_t i = 0; i < p; i++)
    {
      gammas(i) = std::stod(argv[5 + i]);
      betas(i) = std::stod(argv[5 + p + i]);
    }
  }

  std::ofstream output_file;
  output_file.open(output_filename);

  MatrixBuilder mb(q, p);
  Params params(gammas, betas);

  // Set up parameters
  LBFGSpp::LBFGSParam<types::Real> param;
  param.epsilon = 1e-7;
  param.max_iterations = 1000;

  // Create solver and function object
  LBFGSpp::LBFGSSolver<types::Real> solver(param);

  // Initial guess
  types::VectorReal x = types::VectorReal(2 * p);
  for (size_t i = 0; i < p; i++)
  {
    x(i) = gammas(i);
    x(p + i) = betas(i);
  }
  output_file << "Optimize\n";
  output_file << q << "\n" << p << "\n";
  // Print initial parameters
  output_file << std::setprecision(precision) << x.transpose() << "\n";
  // Print delta for finite differences gradient
  output_file << std::setprecision(precision) << delta << "\n";
  // x will be overwritten to be the best point found
  types::Real fx;
  mb.set_delta(delta);
  size_t niter = solver.minimize(mb, x, fx);

  output_file << std::setprecision(precision) << x.transpose() << "\n";
  // Flipping sign because of the minimization of LBFGS++
  output_file << std::setprecision(precision) << -fx << "\n";
  types::MatrixComplex M = mb.build_matrix(params);
  output_file << std::setprecision(precision) << M.real() << "\n";
  output_file << std::setprecision(precision) << M.imag() << "\n";

  output_file.close();

  return 0;
}
