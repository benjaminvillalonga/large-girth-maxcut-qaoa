#include <iomanip>
#include <iostream>
#include <omp.h>
#include <vector>

#include "matrix_builder.h"
#include "types.h"


Params::Params(
    const types::VectorReal & gammas_,
    const types::VectorReal & betas_
)
{
  p = gammas_.size();
  gammas = gammas_;
  betas = betas_;
  Gammas = types::VectorReal::Zero(2 * p);
  Betas = types::VectorReal::Zero(2 * p);
  trigs = types::MatrixComplex::Zero(2, 2 * p);
  ratios = types::VectorComplex::Zero(2 * p);
  for (size_t i = 0; i < p; i++)
  {
    Gammas(i) = gammas(i);
    Gammas(2 * p - i - 1) = -gammas(i);
    Betas(i) = betas(i);
    Betas(2 * p - i - 1) = -betas(i);
    trigs(0, i) = std::cos(betas(i));
    trigs(1, i) = types::Complex({0., 1.}) * std::sin(betas(i));
    trigs(0, 2 * p - i - 1) = std::cos(-betas(i));
    trigs(1, 2 * p - i - 1) = types::Complex({0., 1.}) * std::sin(-betas(i));
    ratios(i) = trigs(1, i) / trigs(0, i);
    ratios(2 * p - i - 1) = trigs(1, 2 * p - i - 1) / trigs(0, 2 * p - i - 1);
  }
}


MatrixBuilder::MatrixBuilder(size_t q_, size_t p_)
{
  _q = q_;
  _p = p_;
  _delta = 1e-7;
}


void MatrixBuilder::set_delta(types::Real delta)
{
  _delta = delta;
}


void MatrixBuilder::broadcast_corner(types::MatrixComplex & M)
{
  size_t l = 2 * _p;
  for (size_t r = 0; r <= _p; r++) for (size_t s = r; s <= _p; s++)
  {
    M(s, r) = M(r, s);
    M(r, l - s) = M(r, s);
    M(l - s, r) = M(r, s);
    M(s, l - r) = std::conj(M(r, s));
    M(l - s, l - r) = std::conj(M(r, s));
    M(l - r, l - s) = std::conj(M(r, s));
    M(l - r, s) = std::conj(M(r, s));
  }
}


types::MatrixComplex MatrixBuilder::build_matrix(
    const Params & params
)
{
  types::MatrixComplex M(2 * _p + 1, 2 * _p + 1);
  build_matrix(M, params);
  return M;
}


void MatrixBuilder::build_matrix(
    types::MatrixComplex & M,
    const Params & params
)
{
  for (size_t col = 0; col <= _p; col++)
  {
    build_column(M, params, col);
    broadcast_corner(M);
  }
}


types::Real MatrixBuilder::evaluate_matrix(
    const types::MatrixComplex & M,
    const Params & params
)
{
  types::Real res = 0.;
  for (size_t i = 0; i < _p; i++)
  {
    res += -std::imag(std::pow(M(i, _p), _q)) * params.Gammas(i);
  }
  res *= std::sqrt(2. / _q);
  return res;
}


types::VectorReal MatrixBuilder::gradient(
    const Params & params,
    types::Real delta,
    types::Real & nu
)
{
  types::MatrixComplex M = build_matrix(params);
  nu = evaluate_matrix(M, params);
  types::VectorReal derivatives(2 * _p);
  for (size_t i = 0; i < _p; i++)
  {
    // gammas are incremented
    types::VectorReal gammas_pd(params.gammas);
    gammas_pd(i) += delta;
    Params params_pd_gammas(gammas_pd, params.betas);
    types::MatrixComplex M_pd_gammas = build_matrix(params_pd_gammas);
    derivatives(i) = evaluate_matrix(M_pd_gammas, params_pd_gammas) - nu;

    // betas are incremented
    types::VectorReal betas_pd(params.betas);
    betas_pd(i) += delta;
    Params params_pd_betas(params.gammas, betas_pd);
    types::MatrixComplex M_pd_betas = build_matrix(params_pd_betas);
    derivatives(_p + i) = evaluate_matrix(M_pd_betas, params_pd_betas) - nu;
  }
  
  derivatives *= 1. / delta;
  return derivatives;
}


// For LBFGS++ to use
types::Real MatrixBuilder::operator()(
    const types::VectorReal & x,
    types::VectorReal & grad
)
{
  types::Real fx;
  types::VectorReal gammas(_p);
  types::VectorReal betas(_p);
  for (size_t i = 0; i < _p; i++)
  {
    gammas(i) = x(i);
    betas(i) = x(_p + i);
  }
  // Flipping sign because of the minimization function of LBFGS++
  grad = -gradient(Params(gammas, betas), _delta, fx);
  fx *= -1.;
  return fx;
}


/////////////////////////////// PRIVATE METHODS ///////////////////////////////


void MatrixBuilder::build_column(
    types::MatrixComplex & M,
    const Params & params,
    size_t col
)
{
  size_t l = 2 * col;
  size_t dim = std::pow(2, l);

  std::vector<size_t> reduced_indices(l);
  std::vector<size_t> reduced_indices_M(l);
  for (size_t i = 0; i < col; i++)
  {
    reduced_indices[i] = i;
    reduced_indices[l - i - 1] = 2 * _p - i - 1;
    reduced_indices_M[i] = i;
    reduced_indices_M[l - i - 1] = 2 * _p - i;
  }
  types::MatrixComplex reduced_M_qm1 = M(reduced_indices_M, reduced_indices_M);
  for (size_t i = 0; i < l; i++) for (size_t j = 0; j < l; j++)
  { reduced_M_qm1(i, j) = std::pow(reduced_M_qm1(i, j), _q - 1); }
  types::VectorComplex reduced_Gammas = params.Gammas(reduced_indices);
  types::MatrixComplex reduced_trigs = params.trigs(Eigen::all,
                                                    reduced_indices);
  types::MatrixComplex reduced_ratios = params.ratios(reduced_indices);
  types::Complex reduced_product = 1.;
  for (size_t i = 0; i < l; i++)
  {
    reduced_product *= reduced_trigs(0, i);
  }
  reduced_product *= 0.5;

  size_t max_threads;
  #pragma omp parallel
  { if (omp_get_thread_num() == 0) max_threads = omp_get_max_threads(); }
  std::vector<types::VectorComplex> columns(max_threads, types::VectorComplex::Zero(col + 1));
  
  #pragma omp parallel for schedule(static, 8)
  for (types::bitstring a = 0; a < dim; a++)
  {
    size_t thread_num = omp_get_thread_num();

    // Compute H(a)
    types::Complex Ha = std::exp(exponent(reduced_M_qm1, reduced_Gammas, a));

    // Compute function f(a)
    types::bitstring a_left = (a >> col) << col;
    types::bitstring a_right = a - a_left;
    // Assuming a(col) = 0. Later multiplying by 2 to account for the equal a(col) = 1 case.
    types::bitstring a_extended = (a_left << 1) + a_right;
    types::bitstring a_diff = a_extended ^ (a_extended >> 1);
    types::Complex fa =  reduced_product * f_ratio(reduced_ratios, a_diff);

    // Add term for this a with the right sign coming from aj * ak
    bool ak = (1 << col) & a_extended;
    for (size_t j = 0; j <= col; j++) 
    {
      bool aj = (1 << j) & a_extended;
      columns[thread_num](j) += ((ak == aj)? fa * Ha : -fa * Ha);
    }
  }

  for (size_t j = 0; j <= col; j++)
  {
    M(j, col) = 0.;
    for (size_t thread_num = 0; thread_num < max_threads; thread_num++)
    { M(j, col) += columns[thread_num](j); }
    M(j, col) *= 2.;
  }
}


types::Complex MatrixBuilder::exponent(
    const types::MatrixComplex & M,
    const types::VectorComplex & Gammas,
    types::bitstring a
)
{
    bool aj, ak;
    types::Complex res = 0.;
    types::Complex element;
    types::Real minus_one_half = -0.5;
    for (size_t j = 0; j < Gammas.size(); j++)
    {
      aj = (1 << j) & a;
      element = minus_one_half * M(j, j) * Gammas(j);

      for (size_t k = j + 1; k < Gammas.size(); k++)
      {
        ak = (1 << k) & a;
        element += ((aj == ak)? -M(j, k) * Gammas(k) : M(j, k) * Gammas(k));
      }
      res += element * Gammas(j);
    }

    return res;
}


types::Complex MatrixBuilder::f_ratio(
    const types::VectorComplex & ratios,
    types::bitstring a_diff
)
{
  types::Complex ratio = 1.;
  for (size_t i = 0; i < ratios.size(); i++)
  {
    if ((1 << i) & a_diff) ratio *= ratios(i);
  }

  return ratio;
}
