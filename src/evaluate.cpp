#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "matrix_builder.h"
#include "types.h"

int main(int argc, char** argv)
{

  size_t precision = 8;

  std::string output_filename = argv[1];
  size_t q = std::stoi(argv[2]);
  size_t p = std::stoi(argv[3]);
  types::VectorReal gammas = types::VectorReal::Zero(p);
  types::VectorReal betas = types::VectorReal::Zero(p);
  if (argc == 4 + 2 * p)
  {
    for (size_t i = 0; i < p; i++)
    {
      gammas(i) = std::stod(argv[4 + i]);
      betas(i) = std::stod(argv[4 + p + i]);
    }
  }

  std::ofstream output_file;
  output_file.open(output_filename);

  MatrixBuilder mb(q, p);
  Params params(gammas, betas);

  types::MatrixComplex M = mb.build_matrix(params);

  types::VectorReal x = types::VectorReal(2 * p);
  for (size_t i = 0; i < p; i++)
  {
    x(i) = gammas(i);
    x(p + i) = betas(i);
  }
  output_file << "Evaluate\n";
  output_file << q << "\n" << p << "\n";
  output_file << std::setprecision(precision) << x.transpose() << "\n";
  output_file << std::setprecision(precision) << mb.evaluate_matrix(M, params) << "\n";
  output_file << std::setprecision(precision) << M.real() << "\n";
  output_file << std::setprecision(precision) << M.imag() << "\n";

  output_file.close();

  return 0;
}
