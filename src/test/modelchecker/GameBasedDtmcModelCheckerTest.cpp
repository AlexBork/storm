#include "gtest/gtest.h"
#include "storm-config.h"

#include "storm/parser/FormulaParser.h"
#include "storm/logic/Formulas.h"
#include "storm/utility/solver.h"
#include "storm/modelchecker/abstraction/GameBasedMdpModelChecker.h"
#include "storm/modelchecker/results/ExplicitQualitativeCheckResult.h"
#include "storm/modelchecker/results/ExplicitQuantitativeCheckResult.h"
#include "storm/parser/PrismParser.h"
#include "storm/builder/DdPrismModelBuilder.h"
#include "storm/models/symbolic/StandardRewardModel.h"
#include "storm/models/symbolic/Dtmc.h"
#include "storm/settings/SettingsManager.h"

#include "storm/settings/modules/NativeEquationSolverSettings.h"

#include "storm/settings/modules/GeneralSettings.h"

TEST(GameBasedDtmcModelCheckerTest, Die_Cudd) {
    storm::prism::Program program = storm::parser::PrismParser::parse(STORM_TEST_RESOURCES_DIR "/dtmc/die.pm");
	auto checker = std::make_shared<storm::modelchecker::GameBasedMdpModelChecker<storm::dd::DdType::CUDD, storm::models::symbolic::Dtmc<storm::dd::DdType::CUDD>>>(program);
    
    // A parser that we use for conveniently constructing the formulas.
    storm::parser::FormulaParser formulaParser;

    std::shared_ptr<storm::logic::Formula const> formula = formulaParser.parseSingleFormulaFromString("P=? [F \"one\"]");
	storm::modelchecker::CheckTask<storm::logic::Formula, double> task(*formula, true);
    
    std::unique_ptr<storm::modelchecker::CheckResult> result = checker->check(task);
    storm::modelchecker::ExplicitQuantitativeCheckResult<double>& quantitativeResult1 = result->asExplicitQuantitativeCheckResult<double>();
    
    EXPECT_NEAR(1.0/6.0, quantitativeResult1[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());
    
    formula = formulaParser.parseSingleFormulaFromString("P=? [F \"two\"]");
	task = storm::modelchecker::CheckTask<storm::logic::Formula, double>(*formula, true);
    
    result = checker->check(task);
    storm::modelchecker::ExplicitQuantitativeCheckResult<double>& quantitativeResult2 = result->asExplicitQuantitativeCheckResult<double>();
    
    EXPECT_NEAR(1.0/6.0, quantitativeResult2[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());
    
    formula = formulaParser.parseSingleFormulaFromString("P=? [F \"three\"]");
	task = storm::modelchecker::CheckTask<storm::logic::Formula, double>(*formula, true);
    
    result = checker->check(task);
    storm::modelchecker::ExplicitQuantitativeCheckResult<double>& quantitativeResult3 = result->asExplicitQuantitativeCheckResult<double>();
    
    EXPECT_NEAR(1.0/6.0, quantitativeResult3[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());
}

TEST(GameBasedDtmcModelCheckerTest, Die_Sylvan) {
    storm::prism::Program program = storm::parser::PrismParser::parse(STORM_TEST_RESOURCES_DIR "/dtmc/die.pm");
    
    // A parser that we use for conveniently constructing the formulas.
    storm::parser::FormulaParser formulaParser;

	auto checker = std::make_shared<storm::modelchecker::GameBasedMdpModelChecker<storm::dd::DdType::Sylvan, storm::models::symbolic::Dtmc<storm::dd::DdType::Sylvan>>>(program);
    
    std::shared_ptr<storm::logic::Formula const> formula = formulaParser.parseSingleFormulaFromString("P=? [F \"one\"]");
	storm::modelchecker::CheckTask<storm::logic::Formula, double> task(*formula, true);
    
    std::unique_ptr<storm::modelchecker::CheckResult> result = checker->check(task);
    storm::modelchecker::ExplicitQuantitativeCheckResult<double>& quantitativeResult1 = result->asExplicitQuantitativeCheckResult<double>();
    
    EXPECT_NEAR(1.0/6.0, quantitativeResult1[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());
    
    formula = formulaParser.parseSingleFormulaFromString("P=? [F \"two\"]");
	task = storm::modelchecker::CheckTask<storm::logic::Formula, double>(*formula, true);
    
    result = checker->check(task);
    storm::modelchecker::ExplicitQuantitativeCheckResult<double>& quantitativeResult2 = result->asExplicitQuantitativeCheckResult<double>();
    
    EXPECT_NEAR(1.0/6.0, quantitativeResult2[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());
    
    formula = formulaParser.parseSingleFormulaFromString("P=? [F \"three\"]");
	task = storm::modelchecker::CheckTask<storm::logic::Formula, double>(*formula, true);
    
    result = checker->check(task);
    storm::modelchecker::ExplicitQuantitativeCheckResult<double>& quantitativeResult3 = result->asExplicitQuantitativeCheckResult<double>();
    
    EXPECT_NEAR(1.0/6.0, quantitativeResult3[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());
}

TEST(GameBasedDtmcModelCheckerTest, SynchronousLeader_Cudd) {
    storm::prism::Program program = storm::parser::PrismParser::parse(STORM_TEST_RESOURCES_DIR "/dtmc/leader-3-5.pm");
    program = program.substituteConstants();
    
    // A parser that we use for conveniently constructing the formulas.
    storm::parser::FormulaParser formulaParser;
    
	auto checker = std::make_shared<storm::modelchecker::GameBasedMdpModelChecker<storm::dd::DdType::CUDD, storm::models::symbolic::Dtmc<storm::dd::DdType::CUDD>>>(program);

    std::shared_ptr<storm::logic::Formula const> formula = formulaParser.parseSingleFormulaFromString("P=? [F \"elected\"]");
	storm::modelchecker::CheckTask<storm::logic::Formula, double> task(*formula, true);
    
    std::unique_ptr<storm::modelchecker::CheckResult> result = checker->check(task);
    storm::modelchecker::ExplicitQuantitativeCheckResult<double>& quantitativeResult1 = result->asExplicitQuantitativeCheckResult<double>();
    
    EXPECT_NEAR(1.0, quantitativeResult1[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());
}

TEST(GameBasedDtmcModelCheckerTest, SynchronousLeader_Sylvan) {
    storm::prism::Program program = storm::parser::PrismParser::parse(STORM_TEST_RESOURCES_DIR "/dtmc/leader-3-5.pm");
    program = program.substituteConstants();

    // A parser that we use for conveniently constructing the formulas.
    storm::parser::FormulaParser formulaParser;
    
	auto checker = std::make_shared<storm::modelchecker::GameBasedMdpModelChecker<storm::dd::DdType::Sylvan, storm::models::symbolic::Dtmc<storm::dd::DdType::Sylvan>>>(program);

    std::shared_ptr<storm::logic::Formula const> formula = formulaParser.parseSingleFormulaFromString("P=? [F \"elected\"]");
	storm::modelchecker::CheckTask<storm::logic::Formula, double> task(*formula, true);
    
    std::unique_ptr<storm::modelchecker::CheckResult> result = checker->check(task);
    storm::modelchecker::ExplicitQuantitativeCheckResult<double>& quantitativeResult1 = result->asExplicitQuantitativeCheckResult<double>();
    
    EXPECT_NEAR(1.0, quantitativeResult1[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());
}
