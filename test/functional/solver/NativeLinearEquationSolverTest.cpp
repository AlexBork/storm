#include "gtest/gtest.h"
#include "storm-config.h"

#include "src/solver/NativeLinearEquationSolver.h"
#include "src/settings/SettingsManager.h"

TEST(NativeLinearEquationSolver, SolveWithStandardOptions) {
    ASSERT_NO_THROW(storm::storage::SparseMatrixBuilder<double> builder);
    storm::storage::SparseMatrixBuilder<double> builder;
    ASSERT_NO_THROW(builder.addNextValue(0, 0, 4));
    ASSERT_NO_THROW(builder.addNextValue(0, 1, 2));
    ASSERT_NO_THROW(builder.addNextValue(0, 2, -1));
    ASSERT_NO_THROW(builder.addNextValue(1, 0, 1));
    ASSERT_NO_THROW(builder.addNextValue(1, 1, -5));
    ASSERT_NO_THROW(builder.addNextValue(1, 2, 2));
    ASSERT_NO_THROW(builder.addNextValue(2, 0, -1));
    ASSERT_NO_THROW(builder.addNextValue(2, 1, 2));
    ASSERT_NO_THROW(builder.addNextValue(2, 2, 4));
    
    storm::storage::SparseMatrix<double> A;
    ASSERT_NO_THROW(A = builder.build());
    
    std::vector<double> x(3);
    std::vector<double> b = {11, -16, 1};
    
    ASSERT_NO_THROW(storm::solver::NativeLinearEquationSolver<double> solver);
    
    storm::solver::NativeLinearEquationSolver<double> solver;
    ASSERT_NO_THROW(solver.solveEquationSystem(A, x, b));
    ASSERT_LT(std::abs(x[0] - 1), storm::settings::nativeEquationSolverSettings().getPrecision());
    ASSERT_LT(std::abs(x[1] - 3), storm::settings::nativeEquationSolverSettings().getPrecision());
    ASSERT_LT(std::abs(x[2] - (-1)), storm::settings::nativeEquationSolverSettings().getPrecision());
}

TEST(NativeLinearEquationSolver, MatrixVectorMultplication) {
    ASSERT_NO_THROW(storm::storage::SparseMatrixBuilder<double> builder);
    storm::storage::SparseMatrixBuilder<double> builder;
    ASSERT_NO_THROW(builder.addNextValue(0, 1, 0.5));
    ASSERT_NO_THROW(builder.addNextValue(0, 4, 0.5));
    ASSERT_NO_THROW(builder.addNextValue(1, 2, 0.5));
    ASSERT_NO_THROW(builder.addNextValue(1, 4, 0.5));
    ASSERT_NO_THROW(builder.addNextValue(2, 3, 0.5));
    ASSERT_NO_THROW(builder.addNextValue(2, 4, 0.5));
    ASSERT_NO_THROW(builder.addNextValue(3, 4, 1));
    ASSERT_NO_THROW(builder.addNextValue(4, 4, 1));
    
    storm::storage::SparseMatrix<double> A;
    ASSERT_NO_THROW(A = builder.build());
    
    std::vector<double> x(5);
    x[4] = 1;
    
    storm::solver::NativeLinearEquationSolver<double> solver;
    ASSERT_NO_THROW(solver.performMatrixVectorMultiplication(A, x, nullptr, 4));
    ASSERT_LT(std::abs(x[0] - 1), storm::settings::nativeEquationSolverSettings().getPrecision());
}