#include "gtest/gtest.h"
#include "storm-config.h"

#include "src/storm/solver/StandardMinMaxLinearEquationSolver.h"
#include "src/storm/settings/SettingsManager.h"

#include "src/storm/settings/modules/NativeEquationSolverSettings.h"
#include "src/storm/storage/SparseMatrix.h"

TEST(NativeMinMaxLinearEquationSolver, SolveWithStandardOptions) {
    storm::storage::SparseMatrixBuilder<double> builder(0, 0, 0, false, true);
    ASSERT_NO_THROW(builder.newRowGroup(0));
    ASSERT_NO_THROW(builder.addNextValue(0, 0, 0.9));
    
    storm::storage::SparseMatrix<double> A;
    ASSERT_NO_THROW(A = builder.build(2));
    
    std::vector<double> x(1);
    std::vector<double> b = {0.099, 0.5};
    
    auto factory = storm::solver::NativeMinMaxLinearEquationSolverFactory<double>();
    auto solver = factory.create(A);

    ASSERT_NO_THROW(solver->solveEquations(storm::OptimizationDirection::Minimize, x, b));
    ASSERT_LT(std::abs(x[0] - 0.5), storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());
    
    ASSERT_NO_THROW(solver->solveEquations(storm::OptimizationDirection::Maximize, x, b));
    ASSERT_LT(std::abs(x[0] - 0.989991), storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());
}

TEST(NativeMinMaxLinearEquationSolver, MatrixVectorMultiplication) {
    storm::storage::SparseMatrixBuilder<double> builder(0, 0, 0, false, true);
    ASSERT_NO_THROW(builder.newRowGroup(0));
    ASSERT_NO_THROW(builder.addNextValue(0, 0, 0.9));
    ASSERT_NO_THROW(builder.addNextValue(0, 1, 0.099));
    ASSERT_NO_THROW(builder.addNextValue(0, 2, 0.001));
    ASSERT_NO_THROW(builder.addNextValue(1, 1, 0.5));
    ASSERT_NO_THROW(builder.addNextValue(1, 2, 0.5));
    ASSERT_NO_THROW(builder.newRowGroup(2));
    ASSERT_NO_THROW(builder.addNextValue(2, 1, 1));
    ASSERT_NO_THROW(builder.newRowGroup(3));
    ASSERT_NO_THROW(builder.addNextValue(3, 2, 1));
    
    storm::storage::SparseMatrix<double> A;
    ASSERT_NO_THROW(A = builder.build());
    
    std::vector<double> x = {0, 1, 0};
    
    auto factory = storm::solver::NativeMinMaxLinearEquationSolverFactory<double>();
    auto solver = factory.create(A);
    
    ASSERT_NO_THROW(solver->repeatedMultiply(storm::OptimizationDirection::Minimize, x, nullptr, 1));
    ASSERT_LT(std::abs(x[0] - 0.099), storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());
    
    x = {0, 1, 0};
    ASSERT_NO_THROW(solver->repeatedMultiply(storm::OptimizationDirection::Minimize, x, nullptr, 2));
    ASSERT_LT(std::abs(x[0] - 0.1881), storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());
    
    x = {0, 1, 0};
    ASSERT_NO_THROW(solver->repeatedMultiply(storm::OptimizationDirection::Minimize, x, nullptr, 20));
    ASSERT_LT(std::abs(x[0] - 0.5), storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());
    
    x = {0, 1, 0};
    ASSERT_NO_THROW(solver->repeatedMultiply(storm::OptimizationDirection::Maximize, x, nullptr, 1));
    ASSERT_LT(std::abs(x[0] - 0.5), storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());
    
    x = {0, 1, 0};
    ASSERT_NO_THROW(solver->repeatedMultiply(storm::OptimizationDirection::Maximize, x, nullptr, 20));
    ASSERT_LT(std::abs(x[0] - 0.9238082658), storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());
}

TEST(NativeMinMaxLinearEquationSolver, SolveWithPolicyIteration) {
	storm::storage::SparseMatrixBuilder<double> builder(0, 0, 0, false, true);
	ASSERT_NO_THROW(builder.newRowGroup(0));
	ASSERT_NO_THROW(builder.addNextValue(0, 0, 0.9));

	storm::storage::SparseMatrix<double> A;
	ASSERT_NO_THROW(A = builder.build(2));

	std::vector<double> x(1);
	std::vector<double> b = { 0.099, 0.5 };

    auto factory = storm::solver::NativeMinMaxLinearEquationSolverFactory<double>();
    factory.getSettings().setSolutionMethod(storm::solver::StandardMinMaxLinearEquationSolverSettings<double>::SolutionMethod::PolicyIteration);
    auto solver = factory.create(A);
    
	ASSERT_NO_THROW(solver->solveEquations(storm::OptimizationDirection::Minimize, x, b));
	ASSERT_LT(std::abs(x[0] - 0.5), storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());

	ASSERT_NO_THROW(solver->solveEquations(storm::OptimizationDirection::Maximize, x, b));
	ASSERT_LT(std::abs(x[0] - 0.99), storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());
}
