#include "gtest/gtest.h"
#include "storm-config.h"

#ifdef STORM_HAVE_GUROBI
#include "src/solver/GurobiLpSolver.h"
#include "src/exceptions/InvalidStateException.h"
#include "src/exceptions/InvalidAccessException.h"
#include "src/settings/SettingsManager.h"
#include "src/storage/expressions/Variable.h"
#include "src/storage/expressions/Expressions.h"

TEST(GurobiLpSolver, LPOptimizeMax) {
    storm::solver::GurobiLpSolver solver(storm::solver::LpSolver::ModelSense::Maximize);
    storm::expressions::Variable x;
    storm::expressions::Variable y;
    storm::expressions::Variable z;
    ASSERT_NO_THROW(x = solver.addBoundedContinuousVariable("x", 0, 1, -1));
    ASSERT_NO_THROW(y = solver.addLowerBoundedContinuousVariable("y", 0, 2));
    ASSERT_NO_THROW(z = solver.addLowerBoundedContinuousVariable("z", 0, 1));
    ASSERT_NO_THROW(solver.update());
    
    ASSERT_NO_THROW(solver.addConstraint("", x + y + z <= solver.getConstant(12)));
    ASSERT_NO_THROW(solver.addConstraint("", solver.getConstant(0.5) * y + z - x == solver.getConstant(5)));
    ASSERT_NO_THROW(solver.addConstraint("", y - x <= solver.getConstant(5.5)));
    ASSERT_NO_THROW(solver.update());
    
    ASSERT_NO_THROW(solver.optimize());
    ASSERT_TRUE(solver.isOptimal());
    ASSERT_FALSE(solver.isUnbounded());
    ASSERT_FALSE(solver.isInfeasible());
    double xValue = 0;
    ASSERT_NO_THROW(xValue = solver.getContinuousValue(x));
    ASSERT_LT(std::abs(xValue - 1), storm::settings::generalSettings().getPrecision());
    double yValue = 0;
    ASSERT_NO_THROW(yValue = solver.getContinuousValue(y));
    ASSERT_LT(std::abs(yValue - 6.5), storm::settings::generalSettings().getPrecision());
    double zValue = 0;
    ASSERT_NO_THROW(zValue = solver.getContinuousValue(z));
    ASSERT_LT(std::abs(zValue - 2.75), storm::settings::generalSettings().getPrecision());
    double objectiveValue = 0;
    ASSERT_NO_THROW(objectiveValue = solver.getObjectiveValue());
    ASSERT_LT(std::abs(objectiveValue - 14.75), storm::settings::generalSettings().getPrecision());
}

TEST(GurobiLpSolver, LPOptimizeMin) {
    storm::solver::GurobiLpSolver solver(storm::solver::LpSolver::ModelSense::Minimize);
    storm::expressions::Variable x;
    storm::expressions::Variable y;
    storm::expressions::Variable z;
    ASSERT_NO_THROW(x = solver.addBoundedContinuousVariable("x", 0, 1, -1));
    ASSERT_NO_THROW(y = solver.addLowerBoundedContinuousVariable("y", 0, 2));
    ASSERT_NO_THROW(z = solver.addBoundedContinuousVariable("z", 1, 5.7, -1));
    ASSERT_NO_THROW(solver.update());
    
    ASSERT_NO_THROW(solver.addConstraint("", x + y + z <= solver.getConstant(12)));
    ASSERT_NO_THROW(solver.addConstraint("", solver.getConstant(0.5) * y + z - x <= solver.getConstant(5)));
    ASSERT_NO_THROW(solver.addConstraint("", y - x <= solver.getConstant(5.5)));
    ASSERT_NO_THROW(solver.update());
    
    ASSERT_NO_THROW(solver.optimize());
    ASSERT_TRUE(solver.isOptimal());
    ASSERT_FALSE(solver.isUnbounded());
    ASSERT_FALSE(solver.isInfeasible());
    double xValue = 0;
    ASSERT_NO_THROW(xValue = solver.getContinuousValue(x));
    ASSERT_LT(std::abs(xValue - 1), storm::settings::generalSettings().getPrecision());
    double yValue = 0;
    ASSERT_NO_THROW(yValue = solver.getContinuousValue(y));
    ASSERT_LT(std::abs(yValue - 0), storm::settings::generalSettings().getPrecision());
    double zValue = 0;
    ASSERT_NO_THROW(zValue = solver.getContinuousValue(z));
    ASSERT_LT(std::abs(zValue - 5.7), storm::settings::generalSettings().getPrecision());
    double objectiveValue = 0;
    ASSERT_NO_THROW(objectiveValue = solver.getObjectiveValue());
    ASSERT_LT(std::abs(objectiveValue - (-6.7)), storm::settings::generalSettings().getPrecision());
}

TEST(GurobiLpSolver, MILPOptimizeMax) {
    storm::solver::GurobiLpSolver solver(storm::solver::LpSolver::ModelSense::Maximize);
    storm::expressions::Variable x;
    storm::expressions::Variable y;
    storm::expressions::Variable z;
    ASSERT_NO_THROW(x = solver.addBinaryVariable("x", -1));
    ASSERT_NO_THROW(y = solver.addLowerBoundedIntegerVariable("y", 0, 2));
    ASSERT_NO_THROW(z = solver.addLowerBoundedContinuousVariable("z", 0, 1));
    ASSERT_NO_THROW(solver.update());
    
    ASSERT_NO_THROW(solver.addConstraint("", x + y + z <= solver.getConstant(12)));
    ASSERT_NO_THROW(solver.addConstraint("", solver.getConstant(0.5) * y + z - x == solver.getConstant(5)));
    ASSERT_NO_THROW(solver.addConstraint("", y - x <= solver.getConstant(5.5)));
    ASSERT_NO_THROW(solver.update());
    
    ASSERT_NO_THROW(solver.optimize());
    ASSERT_TRUE(solver.isOptimal());
    ASSERT_FALSE(solver.isUnbounded());
    ASSERT_FALSE(solver.isInfeasible());
    bool xValue = false;
    ASSERT_NO_THROW(xValue = solver.getBinaryValue(x));
    ASSERT_EQ(true, xValue);
    int_fast64_t yValue = 0;
    ASSERT_NO_THROW(yValue = solver.getIntegerValue(y));
    ASSERT_EQ(6, yValue);
    double zValue = 0;
    ASSERT_NO_THROW(zValue = solver.getContinuousValue(z));
    ASSERT_LT(std::abs(zValue - 3), storm::settings::generalSettings().getPrecision());
    double objectiveValue = 0;
    ASSERT_NO_THROW(objectiveValue = solver.getObjectiveValue());
    ASSERT_LT(std::abs(objectiveValue - 14), storm::settings::generalSettings().getPrecision());
}

TEST(GurobiLpSolver, MILPOptimizeMin) {
    storm::solver::GurobiLpSolver solver(storm::solver::LpSolver::ModelSense::Minimize);
    storm::expressions::Variable x;
    storm::expressions::Variable y;
    storm::expressions::Variable z;
    ASSERT_NO_THROW(x = solver.addBinaryVariable("x", -1));
    ASSERT_NO_THROW(y = solver.addLowerBoundedIntegerVariable("y", 0, 2));
    ASSERT_NO_THROW(z = solver.addBoundedContinuousVariable("z", 0, 5, -1));
    ASSERT_NO_THROW(solver.update());
    
    ASSERT_NO_THROW(solver.addConstraint("", x + y + z <= solver.getConstant(12)));
    ASSERT_NO_THROW(solver.addConstraint("", solver.getConstant(0.5) * y + z - x <= solver.getConstant(5)));
    ASSERT_NO_THROW(solver.addConstraint("", y - x <= solver.getConstant(5.5)));
    ASSERT_NO_THROW(solver.update());
    
    ASSERT_NO_THROW(solver.optimize());
    ASSERT_TRUE(solver.isOptimal());
    ASSERT_FALSE(solver.isUnbounded());
    ASSERT_FALSE(solver.isInfeasible());
    bool xValue = false;
    ASSERT_NO_THROW(xValue = solver.getBinaryValue(x));
    ASSERT_EQ(true, xValue);
    int_fast64_t yValue = 0;
    ASSERT_NO_THROW(yValue = solver.getIntegerValue(y));
    ASSERT_EQ(0, yValue);
    double zValue = 0;
    ASSERT_NO_THROW(zValue = solver.getContinuousValue(z));
    ASSERT_LT(std::abs(zValue - 5), storm::settings::generalSettings().getPrecision());
    double objectiveValue = 0;
    ASSERT_NO_THROW(objectiveValue = solver.getObjectiveValue());
    ASSERT_LT(std::abs(objectiveValue - (-6)), storm::settings::generalSettings().getPrecision());
}

TEST(GurobiLpSolver, LPInfeasible) {
    storm::solver::GurobiLpSolver solver(storm::solver::LpSolver::ModelSense::Maximize);
    storm::expressions::Variable x;
    storm::expressions::Variable y;
    storm::expressions::Variable z;
    ASSERT_NO_THROW(x = solver.addBoundedContinuousVariable("x", 0, 1, -1));
    ASSERT_NO_THROW(y = solver.addLowerBoundedContinuousVariable("y", 0, 2));
    ASSERT_NO_THROW(z = solver.addLowerBoundedContinuousVariable("z", 0, 1));
    ASSERT_NO_THROW(solver.update());
    
    ASSERT_NO_THROW(solver.addConstraint("", x + y + z <= solver.getConstant(12)));
    ASSERT_NO_THROW(solver.addConstraint("", solver.getConstant(0.5) * y + z - x == solver.getConstant(5)));
    ASSERT_NO_THROW(solver.addConstraint("", y - x <= solver.getConstant(5.5)));
    ASSERT_NO_THROW(solver.addConstraint("", y > solver.getConstant(7)));
    ASSERT_NO_THROW(solver.update());
    
    ASSERT_NO_THROW(solver.optimize());
    ASSERT_FALSE(solver.isOptimal());
    ASSERT_FALSE(solver.isUnbounded());
    ASSERT_TRUE(solver.isInfeasible());
    double xValue = 0;
    ASSERT_THROW(xValue = solver.getContinuousValue(x), storm::exceptions::InvalidAccessException);
    double yValue = 0;
    ASSERT_THROW(yValue = solver.getContinuousValue(y), storm::exceptions::InvalidAccessException);
    double zValue = 0;
    ASSERT_THROW(zValue = solver.getContinuousValue(z), storm::exceptions::InvalidAccessException);
    double objectiveValue = 0;
    ASSERT_THROW(objectiveValue = solver.getObjectiveValue(), storm::exceptions::InvalidAccessException);
}

TEST(GurobiLpSolver, MILPInfeasible) {
    storm::solver::GurobiLpSolver solver(storm::solver::LpSolver::ModelSense::Maximize);
    storm::expressions::Variable x;
    storm::expressions::Variable y;
    storm::expressions::Variable z;
    ASSERT_NO_THROW(x = solver.addBoundedContinuousVariable("x", 0, 1, -1));
    ASSERT_NO_THROW(y = solver.addLowerBoundedContinuousVariable("y", 0, 2));
    ASSERT_NO_THROW(z = solver.addLowerBoundedContinuousVariable("z", 0, 1));
    ASSERT_NO_THROW(solver.update());
    
    ASSERT_NO_THROW(solver.addConstraint("", x + y + z <= solver.getConstant(12)));
    ASSERT_NO_THROW(solver.addConstraint("", solver.getConstant(0.5) * y + z - x == solver.getConstant(5)));
    ASSERT_NO_THROW(solver.addConstraint("", y - x <= solver.getConstant(5.5)));
    ASSERT_NO_THROW(solver.addConstraint("", y > solver.getConstant(7)));
    ASSERT_NO_THROW(solver.update());
    
    ASSERT_NO_THROW(solver.optimize());
    ASSERT_FALSE(solver.isOptimal());
    ASSERT_FALSE(solver.isUnbounded());
    ASSERT_TRUE(solver.isInfeasible());
    double xValue = 0;
    ASSERT_THROW(xValue = solver.getContinuousValue(x), storm::exceptions::InvalidAccessException);
    double yValue = 0;
    ASSERT_THROW(yValue = solver.getContinuousValue(y), storm::exceptions::InvalidAccessException);
    double zValue = 0;
    ASSERT_THROW(zValue = solver.getContinuousValue(z), storm::exceptions::InvalidAccessException);
    double objectiveValue = 0;
    ASSERT_THROW(objectiveValue = solver.getObjectiveValue(), storm::exceptions::InvalidAccessException);
}

TEST(GurobiLpSolver, LPUnbounded) {
    storm::solver::GurobiLpSolver solver(storm::solver::LpSolver::ModelSense::Maximize);
    storm::expressions::Variable x;
    storm::expressions::Variable y;
    storm::expressions::Variable z;
    ASSERT_NO_THROW(x = solver.addBoundedContinuousVariable("x", 0, 1, -1));
    ASSERT_NO_THROW(y = solver.addLowerBoundedContinuousVariable("y", 0, 2));
    ASSERT_NO_THROW(z = solver.addLowerBoundedContinuousVariable("z", 0, 1));
    ASSERT_NO_THROW(solver.update());
    
    ASSERT_NO_THROW(solver.addConstraint("", x + y - z <= solver.getConstant(12)));
    ASSERT_NO_THROW(solver.addConstraint("", y - x <= solver.getConstant(5.5)));
    ASSERT_NO_THROW(solver.update());
    
    ASSERT_NO_THROW(solver.optimize());
    ASSERT_FALSE(solver.isOptimal());
    ASSERT_TRUE(solver.isUnbounded());
    ASSERT_FALSE(solver.isInfeasible());
    double xValue = 0;
    ASSERT_THROW(xValue = solver.getContinuousValue(x), storm::exceptions::InvalidAccessException);
    double yValue = 0;
    ASSERT_THROW(yValue = solver.getContinuousValue(y), storm::exceptions::InvalidAccessException);
    double zValue = 0;
    ASSERT_THROW(zValue = solver.getContinuousValue(z), storm::exceptions::InvalidAccessException);
    double objectiveValue = 0;
    ASSERT_THROW(objectiveValue = solver.getObjectiveValue(), storm::exceptions::InvalidAccessException);
}

TEST(GurobiLpSolver, MILPUnbounded) {
    storm::solver::GurobiLpSolver solver(storm::solver::LpSolver::ModelSense::Maximize);
    storm::expressions::Variable x;
    storm::expressions::Variable y;
    storm::expressions::Variable z;
    ASSERT_NO_THROW(x = solver.addBinaryVariable("x", -1));
    ASSERT_NO_THROW(y = solver.addLowerBoundedContinuousVariable("y", 0, 2));
    ASSERT_NO_THROW(z = solver.addLowerBoundedContinuousVariable("z", 0, 1));
    ASSERT_NO_THROW(solver.update());
    
    ASSERT_NO_THROW(solver.addConstraint("", x + y - z <= solver.getConstant(12)));
    ASSERT_NO_THROW(solver.addConstraint("", y - x <= solver.getConstant(5.5)));
    ASSERT_NO_THROW(solver.update());
    
    ASSERT_NO_THROW(solver.optimize());
    ASSERT_FALSE(solver.isOptimal());
    ASSERT_TRUE(solver.isUnbounded());
    ASSERT_FALSE(solver.isInfeasible());
    bool xValue = false;
    ASSERT_THROW(xValue = solver.getBinaryValue(x), storm::exceptions::InvalidAccessException);
    int_fast64_t yValue = 0;
    ASSERT_THROW(yValue = solver.getIntegerValue(y), storm::exceptions::InvalidAccessException);
    double zValue = 0;
    ASSERT_THROW(zValue = solver.getContinuousValue(z), storm::exceptions::InvalidAccessException);
    double objectiveValue = 0;
    ASSERT_THROW(objectiveValue = solver.getObjectiveValue(), storm::exceptions::InvalidAccessException);
}
#endif