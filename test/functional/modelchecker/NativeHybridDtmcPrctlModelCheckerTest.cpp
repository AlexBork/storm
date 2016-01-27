#include "gtest/gtest.h"
#include "storm-config.h"

#include "src/parser/FormulaParser.h"
#include "src/logic/Formulas.h"
#include "src/utility/solver.h"
#include "src/modelchecker/prctl/HybridDtmcPrctlModelChecker.h"
#include "src/modelchecker/results/HybridQuantitativeCheckResult.h"
#include "src/modelchecker/results/SymbolicQualitativeCheckResult.h"
#include "src/modelchecker/results/SymbolicQuantitativeCheckResult.h"
#include "src/parser/PrismParser.h"
#include "src/builder/DdPrismModelBuilder.h"
#include "src/models/symbolic/StandardRewardModel.h"
#include "src/models/symbolic/Dtmc.h"
#include "src/settings/SettingsManager.h"
#include "src/settings/modules/GeneralSettings.h"

#include "src/settings/modules/GmmxxEquationSolverSettings.h"

#include "src/settings/modules/NativeEquationSolverSettings.h"

TEST(NativeHybridDtmcPrctlModelCheckerTest, Die_Cudd) {
    storm::prism::Program program = storm::parser::PrismParser::parse(STORM_CPP_TESTS_BASE_PATH "/functional/builder/die.pm");
    
    // A parser that we use for conveniently constructing the formulas.
    storm::parser::FormulaParser formulaParser;

    // Build the die model with its reward model.
#ifdef WINDOWS
	storm::builder::DdPrismModelBuilder<storm::dd::DdType::CUDD>::Options options;
#else
	typename storm::builder::DdPrismModelBuilder<storm::dd::DdType::CUDD>::Options options;
#endif
    options.buildAllRewardModels = false;
    options.rewardModelsToBuild.insert("coin_flips");
    std::shared_ptr<storm::models::symbolic::Model<storm::dd::DdType::CUDD>> model = storm::builder::DdPrismModelBuilder<storm::dd::DdType::CUDD>().translateProgram(program, options);
    EXPECT_EQ(13ul, model->getNumberOfStates());
    EXPECT_EQ(20ul, model->getNumberOfTransitions());
    
    ASSERT_EQ(model->getType(), storm::models::ModelType::Dtmc);
    
    std::shared_ptr<storm::models::symbolic::Dtmc<storm::dd::DdType::CUDD>> dtmc = model->as<storm::models::symbolic::Dtmc<storm::dd::DdType::CUDD>>();
    
    storm::modelchecker::HybridDtmcPrctlModelChecker<storm::dd::DdType::CUDD, double> checker(*dtmc, std::unique_ptr<storm::utility::solver::LinearEquationSolverFactory<double>>(new storm::utility::solver::NativeLinearEquationSolverFactory<double>()));
    
    std::shared_ptr<const storm::logic::Formula> formula = formulaParser.parseSingleFormulaFromString("P=? [F \"one\"]");
    
    std::unique_ptr<storm::modelchecker::CheckResult> result = checker.check(*formula);
    result->filter(storm::modelchecker::SymbolicQualitativeCheckResult<storm::dd::DdType::CUDD>(model->getReachableStates(), model->getInitialStates()));
    storm::modelchecker::HybridQuantitativeCheckResult<storm::dd::DdType::CUDD>& quantitativeResult1 = result->asHybridQuantitativeCheckResult<storm::dd::DdType::CUDD, double>();
    
    EXPECT_NEAR(1.0/6.0, quantitativeResult1.getMin(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    EXPECT_NEAR(1.0/6.0, quantitativeResult1.getMax(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    
    formula = formulaParser.parseSingleFormulaFromString("P=? [F \"two\"]");
    
    result = checker.check(*formula);
    result->filter(storm::modelchecker::SymbolicQualitativeCheckResult<storm::dd::DdType::CUDD>(model->getReachableStates(), model->getInitialStates()));
    storm::modelchecker::HybridQuantitativeCheckResult<storm::dd::DdType::CUDD>& quantitativeResult2 = result->asHybridQuantitativeCheckResult<storm::dd::DdType::CUDD, double>();
    
    EXPECT_NEAR(1.0/6.0, quantitativeResult2.getMin(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    EXPECT_NEAR(1.0/6.0, quantitativeResult2.getMax(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    
    formula = formulaParser.parseSingleFormulaFromString("P=? [F \"three\"]");
    
    result = checker.check(*formula);
    result->filter(storm::modelchecker::SymbolicQualitativeCheckResult<storm::dd::DdType::CUDD>(model->getReachableStates(), model->getInitialStates()));
    storm::modelchecker::HybridQuantitativeCheckResult<storm::dd::DdType::CUDD>& quantitativeResult3 = result->asHybridQuantitativeCheckResult<storm::dd::DdType::CUDD, double>();
    
    EXPECT_NEAR(1.0/6.0, quantitativeResult3.getMin(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    EXPECT_NEAR(1.0/6.0, quantitativeResult3.getMax(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    
    formula = formulaParser.parseSingleFormulaFromString("R=? [F \"done\"]");
    
    result = checker.check(*formula);
    result->filter(storm::modelchecker::SymbolicQualitativeCheckResult<storm::dd::DdType::CUDD>(model->getReachableStates(), model->getInitialStates()));
    storm::modelchecker::HybridQuantitativeCheckResult<storm::dd::DdType::CUDD>& quantitativeResult4 = result->asHybridQuantitativeCheckResult<storm::dd::DdType::CUDD, double>();
    
    EXPECT_NEAR(3.6666646003723145, quantitativeResult4.getMin(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    EXPECT_NEAR(3.6666646003723145, quantitativeResult4.getMax(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
}

TEST(NativeHybridDtmcPrctlModelCheckerTest, Die_Sylvan) {
    storm::prism::Program program = storm::parser::PrismParser::parse(STORM_CPP_TESTS_BASE_PATH "/functional/builder/die.pm");
    
    // A parser that we use for conveniently constructing the formulas.
    storm::parser::FormulaParser formulaParser;
    
    // Build the die model with its reward model.
#ifdef WINDOWS
    storm::builder::DdPrismModelBuilder<storm::dd::DdType::Sylvan>::Options options;
#else
    typename storm::builder::DdPrismModelBuilder<storm::dd::DdType::Sylvan>::Options options;
#endif
    options.buildAllRewardModels = false;
    options.rewardModelsToBuild.insert("coin_flips");
    std::shared_ptr<storm::models::symbolic::Model<storm::dd::DdType::Sylvan>> model = storm::builder::DdPrismModelBuilder<storm::dd::DdType::Sylvan>().translateProgram(program, options);
    EXPECT_EQ(13ul, model->getNumberOfStates());
    EXPECT_EQ(20ul, model->getNumberOfTransitions());
    
    ASSERT_EQ(model->getType(), storm::models::ModelType::Dtmc);
    
    std::shared_ptr<storm::models::symbolic::Dtmc<storm::dd::DdType::Sylvan>> dtmc = model->as<storm::models::symbolic::Dtmc<storm::dd::DdType::Sylvan>>();
    
    storm::modelchecker::HybridDtmcPrctlModelChecker<storm::dd::DdType::Sylvan, double> checker(*dtmc, std::unique_ptr<storm::utility::solver::LinearEquationSolverFactory<double>>(new storm::utility::solver::NativeLinearEquationSolverFactory<double>()));
    
    std::shared_ptr<const storm::logic::Formula> formula = formulaParser.parseSingleFormulaFromString("P=? [F \"one\"]");
    
    std::unique_ptr<storm::modelchecker::CheckResult> result = checker.check(*formula);
    result->filter(storm::modelchecker::SymbolicQualitativeCheckResult<storm::dd::DdType::Sylvan>(model->getReachableStates(), model->getInitialStates()));
    storm::modelchecker::HybridQuantitativeCheckResult<storm::dd::DdType::Sylvan>& quantitativeResult1 = result->asHybridQuantitativeCheckResult<storm::dd::DdType::Sylvan, double>();
    
    EXPECT_NEAR(1.0/6.0, quantitativeResult1.getMin(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    EXPECT_NEAR(1.0/6.0, quantitativeResult1.getMax(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    
    formula = formulaParser.parseSingleFormulaFromString("P=? [F \"two\"]");
    
    result = checker.check(*formula);
    result->filter(storm::modelchecker::SymbolicQualitativeCheckResult<storm::dd::DdType::Sylvan>(model->getReachableStates(), model->getInitialStates()));
    storm::modelchecker::HybridQuantitativeCheckResult<storm::dd::DdType::Sylvan>& quantitativeResult2 = result->asHybridQuantitativeCheckResult<storm::dd::DdType::Sylvan, double>();
    
    EXPECT_NEAR(1.0/6.0, quantitativeResult2.getMin(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    EXPECT_NEAR(1.0/6.0, quantitativeResult2.getMax(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    
    formula = formulaParser.parseSingleFormulaFromString("P=? [F \"three\"]");
    
    result = checker.check(*formula);
    result->filter(storm::modelchecker::SymbolicQualitativeCheckResult<storm::dd::DdType::Sylvan>(model->getReachableStates(), model->getInitialStates()));
    storm::modelchecker::HybridQuantitativeCheckResult<storm::dd::DdType::Sylvan>& quantitativeResult3 = result->asHybridQuantitativeCheckResult<storm::dd::DdType::Sylvan, double>();
    
    EXPECT_NEAR(1.0/6.0, quantitativeResult3.getMin(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    EXPECT_NEAR(1.0/6.0, quantitativeResult3.getMax(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    
    formula = formulaParser.parseSingleFormulaFromString("R=? [F \"done\"]");
    
    result = checker.check(*formula);
    result->filter(storm::modelchecker::SymbolicQualitativeCheckResult<storm::dd::DdType::Sylvan>(model->getReachableStates(), model->getInitialStates()));
    storm::modelchecker::HybridQuantitativeCheckResult<storm::dd::DdType::Sylvan>& quantitativeResult4 = result->asHybridQuantitativeCheckResult<storm::dd::DdType::Sylvan, double>();
    
    EXPECT_NEAR(3.6666646003723145, quantitativeResult4.getMin(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    EXPECT_NEAR(3.6666646003723145, quantitativeResult4.getMax(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
}

TEST(NativeHybridDtmcPrctlModelCheckerTest, Crowds_Cudd) {
    storm::prism::Program program = storm::parser::PrismParser::parse(STORM_CPP_TESTS_BASE_PATH "/functional/builder/crowds-5-5.pm");
    
    // A parser that we use for conveniently constructing the formulas.
    storm::parser::FormulaParser formulaParser;
    
    std::shared_ptr<storm::models::symbolic::Model<storm::dd::DdType::CUDD>> model = storm::builder::DdPrismModelBuilder<storm::dd::DdType::CUDD>().translateProgram(program);
    EXPECT_EQ(8607ul, model->getNumberOfStates());
    EXPECT_EQ(15113ul, model->getNumberOfTransitions());
    
    ASSERT_EQ(model->getType(), storm::models::ModelType::Dtmc);
    
    std::shared_ptr<storm::models::symbolic::Dtmc<storm::dd::DdType::CUDD>> dtmc = model->as<storm::models::symbolic::Dtmc<storm::dd::DdType::CUDD>>();
    
    storm::modelchecker::HybridDtmcPrctlModelChecker<storm::dd::DdType::CUDD, double> checker(*dtmc, std::unique_ptr<storm::utility::solver::LinearEquationSolverFactory<double>>(new storm::utility::solver::NativeLinearEquationSolverFactory<double>()));
    
    std::shared_ptr<const storm::logic::Formula> formula = formulaParser.parseSingleFormulaFromString("P=? [F \"observe0Greater1\"]");
    
    std::unique_ptr<storm::modelchecker::CheckResult> result = checker.check(*formula);
    result->filter(storm::modelchecker::SymbolicQualitativeCheckResult<storm::dd::DdType::CUDD>(model->getReachableStates(), model->getInitialStates()));
    storm::modelchecker::HybridQuantitativeCheckResult<storm::dd::DdType::CUDD>& quantitativeResult1 = result->asHybridQuantitativeCheckResult<storm::dd::DdType::CUDD, double>();
    
    EXPECT_NEAR(0.33288205191646525, quantitativeResult1.getMin(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    EXPECT_NEAR(0.33288205191646525, quantitativeResult1.getMax(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    
    formula = formulaParser.parseSingleFormulaFromString("P=? [F \"observeIGreater1\"]");
    
    result = checker.check(*formula);
    result->filter(storm::modelchecker::SymbolicQualitativeCheckResult<storm::dd::DdType::CUDD>(model->getReachableStates(), model->getInitialStates()));
    storm::modelchecker::HybridQuantitativeCheckResult<storm::dd::DdType::CUDD>& quantitativeResult2 = result->asHybridQuantitativeCheckResult<storm::dd::DdType::CUDD, double>();
    
    EXPECT_NEAR(0.15222066094730619, quantitativeResult2.getMin(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    EXPECT_NEAR(0.15222066094730619, quantitativeResult2.getMax(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    
    formula = formulaParser.parseSingleFormulaFromString("P=? [F \"observeOnlyTrueSender\"]");
    
    result = checker.check(*formula);
    result->filter(storm::modelchecker::SymbolicQualitativeCheckResult<storm::dd::DdType::CUDD>(model->getReachableStates(), model->getInitialStates()));
    storm::modelchecker::HybridQuantitativeCheckResult<storm::dd::DdType::CUDD>& quantitativeResult3 = result->asHybridQuantitativeCheckResult<storm::dd::DdType::CUDD, double>();
    
    EXPECT_NEAR(0.32153900158185761, quantitativeResult3.getMin(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    EXPECT_NEAR(0.32153900158185761, quantitativeResult3.getMax(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
}

TEST(NativeHybridDtmcPrctlModelCheckerTest, Crowds_Sylvan) {
    storm::prism::Program program = storm::parser::PrismParser::parse(STORM_CPP_TESTS_BASE_PATH "/functional/builder/crowds-5-5.pm");
    
    // A parser that we use for conveniently constructing the formulas.
    storm::parser::FormulaParser formulaParser;
    
    std::shared_ptr<storm::models::symbolic::Model<storm::dd::DdType::Sylvan>> model = storm::builder::DdPrismModelBuilder<storm::dd::DdType::Sylvan>().translateProgram(program);
    EXPECT_EQ(8607ul, model->getNumberOfStates());
    EXPECT_EQ(15113ul, model->getNumberOfTransitions());
    
    ASSERT_EQ(model->getType(), storm::models::ModelType::Dtmc);
    
    std::shared_ptr<storm::models::symbolic::Dtmc<storm::dd::DdType::Sylvan>> dtmc = model->as<storm::models::symbolic::Dtmc<storm::dd::DdType::Sylvan>>();
    
    storm::modelchecker::HybridDtmcPrctlModelChecker<storm::dd::DdType::Sylvan, double> checker(*dtmc, std::unique_ptr<storm::utility::solver::LinearEquationSolverFactory<double>>(new storm::utility::solver::NativeLinearEquationSolverFactory<double>()));
    
    std::shared_ptr<const storm::logic::Formula> formula = formulaParser.parseSingleFormulaFromString("P=? [F \"observe0Greater1\"]");
    
    std::unique_ptr<storm::modelchecker::CheckResult> result = checker.check(*formula);
    result->filter(storm::modelchecker::SymbolicQualitativeCheckResult<storm::dd::DdType::Sylvan>(model->getReachableStates(), model->getInitialStates()));
    storm::modelchecker::HybridQuantitativeCheckResult<storm::dd::DdType::Sylvan>& quantitativeResult1 = result->asHybridQuantitativeCheckResult<storm::dd::DdType::Sylvan, double>();
    
    EXPECT_NEAR(0.33288205191646525, quantitativeResult1.getMin(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    EXPECT_NEAR(0.33288205191646525, quantitativeResult1.getMax(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    
    formula = formulaParser.parseSingleFormulaFromString("P=? [F \"observeIGreater1\"]");
    
    result = checker.check(*formula);
    result->filter(storm::modelchecker::SymbolicQualitativeCheckResult<storm::dd::DdType::Sylvan>(model->getReachableStates(), model->getInitialStates()));
    storm::modelchecker::HybridQuantitativeCheckResult<storm::dd::DdType::Sylvan>& quantitativeResult2 = result->asHybridQuantitativeCheckResult<storm::dd::DdType::Sylvan, double>();
    
    EXPECT_NEAR(0.15222066094730619, quantitativeResult2.getMin(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    EXPECT_NEAR(0.15222066094730619, quantitativeResult2.getMax(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    
    formula = formulaParser.parseSingleFormulaFromString("P=? [F \"observeOnlyTrueSender\"]");
    
    result = checker.check(*formula);
    result->filter(storm::modelchecker::SymbolicQualitativeCheckResult<storm::dd::DdType::Sylvan>(model->getReachableStates(), model->getInitialStates()));
    storm::modelchecker::HybridQuantitativeCheckResult<storm::dd::DdType::Sylvan>& quantitativeResult3 = result->asHybridQuantitativeCheckResult<storm::dd::DdType::Sylvan, double>();
    
    EXPECT_NEAR(0.32153900158185761, quantitativeResult3.getMin(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    EXPECT_NEAR(0.32153900158185761, quantitativeResult3.getMax(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
}

TEST(NativeHybridDtmcPrctlModelCheckerTest, SynchronousLeader_Cudd) {
    storm::prism::Program program = storm::parser::PrismParser::parse(STORM_CPP_TESTS_BASE_PATH "/functional/builder/leader-3-5.pm");
    
    // A parser that we use for conveniently constructing the formulas.
    storm::parser::FormulaParser formulaParser;
    
    // Build the die model with its reward model.
#ifdef WINDOWS
	storm::builder::DdPrismModelBuilder<storm::dd::DdType::CUDD>::Options options;
#else
	typename storm::builder::DdPrismModelBuilder<storm::dd::DdType::CUDD>::Options options;
#endif
    options.buildAllRewardModels = false;
    options.rewardModelsToBuild.insert("num_rounds");
    std::shared_ptr<storm::models::symbolic::Model<storm::dd::DdType::CUDD>> model = storm::builder::DdPrismModelBuilder<storm::dd::DdType::CUDD>().translateProgram(program, options);
    EXPECT_EQ(273ul, model->getNumberOfStates());
    EXPECT_EQ(397ul, model->getNumberOfTransitions());
    
    ASSERT_EQ(model->getType(), storm::models::ModelType::Dtmc);
    
    std::shared_ptr<storm::models::symbolic::Dtmc<storm::dd::DdType::CUDD>> dtmc = model->as<storm::models::symbolic::Dtmc<storm::dd::DdType::CUDD>>();
    
    storm::modelchecker::HybridDtmcPrctlModelChecker<storm::dd::DdType::CUDD, double> checker(*dtmc, std::unique_ptr<storm::utility::solver::LinearEquationSolverFactory<double>>(new storm::utility::solver::NativeLinearEquationSolverFactory<double>()));
    
    std::shared_ptr<const storm::logic::Formula> formula = formulaParser.parseSingleFormulaFromString("P=? [F \"elected\"]");
    
    std::unique_ptr<storm::modelchecker::CheckResult> result = checker.check(*formula);
    result->filter(storm::modelchecker::SymbolicQualitativeCheckResult<storm::dd::DdType::CUDD>(model->getReachableStates(), model->getInitialStates()));
    storm::modelchecker::SymbolicQuantitativeCheckResult<storm::dd::DdType::CUDD>& quantitativeResult1 = result->asSymbolicQuantitativeCheckResult<storm::dd::DdType::CUDD, double>();
    
    EXPECT_NEAR(1.0, quantitativeResult1.getMin(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    EXPECT_NEAR(1.0, quantitativeResult1.getMax(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    
    formula = formulaParser.parseSingleFormulaFromString("P=? [F<=20 \"elected\"]");
    
    result = checker.check(*formula);
    result->filter(storm::modelchecker::SymbolicQualitativeCheckResult<storm::dd::DdType::CUDD>(model->getReachableStates(), model->getInitialStates()));
    storm::modelchecker::HybridQuantitativeCheckResult<storm::dd::DdType::CUDD>& quantitativeResult2 = result->asHybridQuantitativeCheckResult<storm::dd::DdType::CUDD, double>();
    
    EXPECT_NEAR(0.99999989760000074, quantitativeResult2.getMin(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    EXPECT_NEAR(0.99999989760000074, quantitativeResult2.getMax(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    
    formula = formulaParser.parseSingleFormulaFromString("R=? [F \"elected\"]");
    
    result = checker.check(*formula);
    result->filter(storm::modelchecker::SymbolicQualitativeCheckResult<storm::dd::DdType::CUDD>(model->getReachableStates(), model->getInitialStates()));
    storm::modelchecker::HybridQuantitativeCheckResult<storm::dd::DdType::CUDD>& quantitativeResult3 = result->asHybridQuantitativeCheckResult<storm::dd::DdType::CUDD, double>();
    
    EXPECT_NEAR(1.0416666666666643, quantitativeResult3.getMin(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    EXPECT_NEAR(1.0416666666666643, quantitativeResult3.getMax(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
}

TEST(NativeHybridDtmcPrctlModelCheckerTest, SynchronousLeader_Sylvan) {
    storm::prism::Program program = storm::parser::PrismParser::parse(STORM_CPP_TESTS_BASE_PATH "/functional/builder/leader-3-5.pm");
    
    // A parser that we use for conveniently constructing the formulas.
    storm::parser::FormulaParser formulaParser;
    
    // Build the die model with its reward model.
#ifdef WINDOWS
    storm::builder::DdPrismModelBuilder<storm::dd::DdType::Sylvan>::Options options;
#else
    typename storm::builder::DdPrismModelBuilder<storm::dd::DdType::Sylvan>::Options options;
#endif
    options.buildAllRewardModels = false;
    options.rewardModelsToBuild.insert("num_rounds");
    std::shared_ptr<storm::models::symbolic::Model<storm::dd::DdType::Sylvan>> model = storm::builder::DdPrismModelBuilder<storm::dd::DdType::Sylvan>().translateProgram(program, options);
    EXPECT_EQ(273ul, model->getNumberOfStates());
    EXPECT_EQ(397ul, model->getNumberOfTransitions());
    
    ASSERT_EQ(model->getType(), storm::models::ModelType::Dtmc);
    
    std::shared_ptr<storm::models::symbolic::Dtmc<storm::dd::DdType::Sylvan>> dtmc = model->as<storm::models::symbolic::Dtmc<storm::dd::DdType::Sylvan>>();
    
    storm::modelchecker::HybridDtmcPrctlModelChecker<storm::dd::DdType::Sylvan, double> checker(*dtmc, std::unique_ptr<storm::utility::solver::LinearEquationSolverFactory<double>>(new storm::utility::solver::NativeLinearEquationSolverFactory<double>()));
    
    std::shared_ptr<const storm::logic::Formula> formula = formulaParser.parseSingleFormulaFromString("P=? [F \"elected\"]");
    
    std::unique_ptr<storm::modelchecker::CheckResult> result = checker.check(*formula);
    result->filter(storm::modelchecker::SymbolicQualitativeCheckResult<storm::dd::DdType::Sylvan>(model->getReachableStates(), model->getInitialStates()));
    storm::modelchecker::SymbolicQuantitativeCheckResult<storm::dd::DdType::Sylvan>& quantitativeResult1 = result->asSymbolicQuantitativeCheckResult<storm::dd::DdType::Sylvan, double>();
    
    EXPECT_NEAR(1.0, quantitativeResult1.getMin(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    EXPECT_NEAR(1.0, quantitativeResult1.getMax(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    
    formula = formulaParser.parseSingleFormulaFromString("P=? [F<=20 \"elected\"]");
    
    result = checker.check(*formula);
    result->filter(storm::modelchecker::SymbolicQualitativeCheckResult<storm::dd::DdType::Sylvan>(model->getReachableStates(), model->getInitialStates()));
    storm::modelchecker::HybridQuantitativeCheckResult<storm::dd::DdType::Sylvan>& quantitativeResult2 = result->asHybridQuantitativeCheckResult<storm::dd::DdType::Sylvan, double>();
    
    EXPECT_NEAR(0.99999989760000074, quantitativeResult2.getMin(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    EXPECT_NEAR(0.99999989760000074, quantitativeResult2.getMax(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    
    formula = formulaParser.parseSingleFormulaFromString("R=? [F \"elected\"]");
    
    result = checker.check(*formula);
    result->filter(storm::modelchecker::SymbolicQualitativeCheckResult<storm::dd::DdType::Sylvan>(model->getReachableStates(), model->getInitialStates()));
    storm::modelchecker::HybridQuantitativeCheckResult<storm::dd::DdType::Sylvan>& quantitativeResult3 = result->asHybridQuantitativeCheckResult<storm::dd::DdType::Sylvan, double>();
    
    EXPECT_NEAR(1.0416666666666643, quantitativeResult3.getMin(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
    EXPECT_NEAR(1.0416666666666643, quantitativeResult3.getMax(), storm::settings::gmmxxEquationSolverSettings().getPrecision());
}