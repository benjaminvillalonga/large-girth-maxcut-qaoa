#ifndef MATRIX_BUILDER_H
#define MATRIX_BUILDER_H

#include <Eigen/Dense>

#include "types.h"

class Params
{
  public:
    size_t p;
    types::VectorReal gammas, betas, Gammas, Betas;
    types::MatrixComplex trigs;
    types::VectorComplex ratios;

    Params(
        const types::VectorReal & gammas_,
        const types::VectorReal & betas_
    );
};

class MatrixBuilder
{
  public:
    MatrixBuilder(size_t q_, size_t p_);

    void set_delta(types::Real delta);

    types::MatrixComplex build_matrix(
        const Params & params
    );

    types::Real evaluate_matrix(
        const types::MatrixComplex & M,
        const Params & params
    );

    types::VectorReal gradient(
        const Params & params,
        types::Real delta,
        types::Real & nu
    );

    // For LBFGS++ to use
    types::Real operator()(
        const types::VectorReal & x,
        types::VectorReal & grad
    );
  
  private:
    size_t _q;
    size_t _p;
    types::Real _delta;

    void broadcast_corner(types::MatrixComplex & M);

    void build_matrix(
        types::MatrixComplex & M,
        const Params & params
    );

    void build_column(
        types::MatrixComplex & M,
        const Params & params,
        size_t col
    );

    types::Complex exponent(
        const types::MatrixComplex & M,
        const types::VectorComplex & Gammas,
        types::bitstring a
    );

    types::Complex f_ratio(
        const types::VectorComplex & ratios,
        types::bitstring a_diff
    );
};

#endif
