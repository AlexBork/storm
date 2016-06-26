#include "gtest/gtest.h"
#include "storm-config.h"

#include "src/parser/FormulaParser.h"
#include "src/logic/Formulas.h"
#include "src/solver/StandardMinMaxLinearEquationSolver.h"
#include "src/models/sparse/StandardRewardModel.h"
#include "src/modelchecker/prctl/SparseMdpPrctlModelChecker.h"
#include "src/modelchecker/results/ExplicitQuantitativeCheckResult.h"
#include "src/settings/SettingsManager.h"
#include "src/settings/modules/GeneralSettings.h"

#include "src/settings/modules/GmmxxEquationSolverSettings.h"

#include "src/settings/modules/NativeEquationSolverSettings.h"
#include "src/parser/AutoParser.h"
#include "src/parser/PrismParser.h"
#include "src/builder/ExplicitModelBuilder.h"

TEST(GmmxxMdpPrctlModelCheckerTest, Dice) {
    std::shared_ptr<storm::models::sparse::Model<double>> abstractModel = storm::parser::AutoParser<>::parseModel(STORM_CPP_BASE_PATH "/examples/mdp/two_dice/two_dice.tra", STORM_CPP_BASE_PATH "/examples/mdp/two_dice/two_dice.lab", "", STORM_CPP_BASE_PATH "/examples/mdp/two_dice/two_dice.flip.trans.rew");

    // A parser that we use for conveniently constructing the formulas.
    storm::parser::FormulaParser formulaParser;

    ASSERT_EQ(abstractModel->getType(), storm::models::ModelType::Mdp);

    std::shared_ptr<storm::models::sparse::Mdp<double>> mdp = abstractModel->as<storm::models::sparse::Mdp<double>>();

    ASSERT_EQ(mdp->getNumberOfStates(), 169ull);
    ASSERT_EQ(mdp->getNumberOfTransitions(), 436ull);

    storm::modelchecker::SparseMdpPrctlModelChecker<storm::models::sparse::Mdp<double>> checker(*mdp, std::make_unique<storm::solver::GmmxxMinMaxLinearEquationSolverFactory<double>>());

    std::shared_ptr<storm::logic::Formula const> formula = formulaParser.parseSingleFormulaFromString("Pmin=? [F \"two\"]");

    std::unique_ptr<storm::modelchecker::CheckResult> result = checker.check(*formula);
    storm::modelchecker::ExplicitQuantitativeCheckResult<double>& quantitativeResult1 = result->asExplicitQuantitativeCheckResult<double>();

    EXPECT_NEAR(0.0277777612209320068, quantitativeResult1[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());

    formula = formulaParser.parseSingleFormulaFromString("Pmax=? [F \"two\"]");

    result = checker.check(*formula);
    storm::modelchecker::ExplicitQuantitativeCheckResult<double>& quantitativeResult2 = result->asExplicitQuantitativeCheckResult<double>();

    EXPECT_NEAR(0.0277777612209320068, quantitativeResult2[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());

    formula = formulaParser.parseSingleFormulaFromString("Pmin=? [F \"three\"]");

    result = checker.check(*formula);
    storm::modelchecker::ExplicitQuantitativeCheckResult<double>& quantitativeResult3 = result->asExplicitQuantitativeCheckResult<double>();

    EXPECT_NEAR(0.0555555224418640136, quantitativeResult3[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());

    formula = formulaParser.parseSingleFormulaFromString("Pmax=? [F \"three\"]");

    result = checker.check(*formula);
    storm::modelchecker::ExplicitQuantitativeCheckResult<double>& quantitativeResult4 = result->asExplicitQuantitativeCheckResult<double>();

    EXPECT_NEAR(0.0555555224418640136, quantitativeResult4[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());

    formula = formulaParser.parseSingleFormulaFromString("Pmin=? [F \"four\"]");

    result = checker.check(*formula);
    storm::modelchecker::ExplicitQuantitativeCheckResult<double>& quantitativeResult5 = result->asExplicitQuantitativeCheckResult<double>();

    EXPECT_NEAR(0.083333283662796020508, quantitativeResult5[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());

    formula = formulaParser.parseSingleFormulaFromString("Pmax=? [F \"four\"]");

    result = checker.check(*formula);
    storm::modelchecker::ExplicitQuantitativeCheckResult<double>& quantitativeResult6 = result->asExplicitQuantitativeCheckResult<double>();

    EXPECT_NEAR(0.083333283662796020508, quantitativeResult6[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());

    formula = formulaParser.parseSingleFormulaFromString("Rmin=? [F \"done\"]");

    result = checker.check(*formula);
    storm::modelchecker::ExplicitQuantitativeCheckResult<double>& quantitativeResult7 = result->asExplicitQuantitativeCheckResult<double>();

    EXPECT_NEAR(7.333329499, quantitativeResult7[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());

    formula = formulaParser.parseSingleFormulaFromString("Rmax=? [F \"done\"]");

    result = checker.check(*formula);
    storm::modelchecker::ExplicitQuantitativeCheckResult<double>& quantitativeResult8 = result->asExplicitQuantitativeCheckResult<double>();

    EXPECT_NEAR(7.333329499, quantitativeResult8[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());

    abstractModel = storm::parser::AutoParser<>::parseModel(STORM_CPP_BASE_PATH "/examples/mdp/two_dice/two_dice.tra", STORM_CPP_BASE_PATH "/examples/mdp/two_dice/two_dice.lab", STORM_CPP_BASE_PATH "/examples/mdp/two_dice/two_dice.flip.state.rew", "");

    ASSERT_EQ(abstractModel->getType(), storm::models::ModelType::Mdp);

    std::shared_ptr<storm::models::sparse::Mdp<double>> stateRewardMdp = abstractModel->as<storm::models::sparse::Mdp<double>>();

    storm::modelchecker::SparseMdpPrctlModelChecker<storm::models::sparse::Mdp<double>> stateRewardModelChecker(*mdp, std::make_unique<storm::solver::GmmxxMinMaxLinearEquationSolverFactory<double>>());

    formula = formulaParser.parseSingleFormulaFromString("Rmin=? [F \"done\"]");

    result = stateRewardModelChecker.check(*formula);
    storm::modelchecker::ExplicitQuantitativeCheckResult<double>& quantitativeResult9 = result->asExplicitQuantitativeCheckResult<double>();

    EXPECT_NEAR(7.333329499, quantitativeResult9[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());

    formula = formulaParser.parseSingleFormulaFromString("Rmax=? [F \"done\"]");

    result = stateRewardModelChecker.check(*formula);
    storm::modelchecker::ExplicitQuantitativeCheckResult<double>& quantitativeResult10 = result->asExplicitQuantitativeCheckResult<double>();

    EXPECT_NEAR(7.333329499, quantitativeResult10[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());

    abstractModel = storm::parser::AutoParser<>::parseModel(STORM_CPP_BASE_PATH "/examples/mdp/two_dice/two_dice.tra", STORM_CPP_BASE_PATH "/examples/mdp/two_dice/two_dice.lab", STORM_CPP_BASE_PATH "/examples/mdp/two_dice/two_dice.flip.state.rew", STORM_CPP_BASE_PATH "/examples/mdp/two_dice/two_dice.flip.trans.rew");

    ASSERT_EQ(abstractModel->getType(), storm::models::ModelType::Mdp);

    std::shared_ptr<storm::models::sparse::Mdp<double>> stateAndTransitionRewardMdp = abstractModel->as<storm::models::sparse::Mdp<double>>();

    storm::modelchecker::SparseMdpPrctlModelChecker<storm::models::sparse::Mdp<double>> stateAndTransitionRewardModelChecker(*stateAndTransitionRewardMdp, std::make_unique<storm::solver::GmmxxMinMaxLinearEquationSolverFactory<double>>());

    formula = formulaParser.parseSingleFormulaFromString("Rmin=? [F \"done\"]");

    result = stateAndTransitionRewardModelChecker.check(*formula);
    storm::modelchecker::ExplicitQuantitativeCheckResult<double>& quantitativeResult11 = result->asExplicitQuantitativeCheckResult<double>();

    EXPECT_NEAR(14.666658998, quantitativeResult11[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());

    formula = formulaParser.parseSingleFormulaFromString("Rmax=? [F \"done\"]");

    result = stateAndTransitionRewardModelChecker.check(*formula);
    storm::modelchecker::ExplicitQuantitativeCheckResult<double>& quantitativeResult12 = result->asExplicitQuantitativeCheckResult<double>();

    EXPECT_NEAR(14.666658998, quantitativeResult12[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());
}

TEST(GmmxxMdpPrctlModelCheckerTest, AsynchronousLeader) {
    std::shared_ptr<storm::models::sparse::Model<double>> abstractModel = storm::parser::AutoParser<>::parseModel(STORM_CPP_BASE_PATH "/examples/mdp/asynchronous_leader/leader4.tra", STORM_CPP_BASE_PATH "/examples/mdp/asynchronous_leader/leader4.lab", "", STORM_CPP_BASE_PATH "/examples/mdp/asynchronous_leader/leader4.trans.rew");

    // A parser that we use for conveniently constructing the formulas.
    storm::parser::FormulaParser formulaParser;

    ASSERT_EQ(storm::models::ModelType::Mdp, abstractModel->getType());

    std::shared_ptr<storm::models::sparse::Mdp<double>> mdp = abstractModel->as<storm::models::sparse::Mdp<double>>();

    ASSERT_EQ(3172ull, mdp->getNumberOfStates());
    ASSERT_EQ(7144ull, mdp->getNumberOfTransitions());

    storm::modelchecker::SparseMdpPrctlModelChecker<storm::models::sparse::Mdp<double>> checker(*mdp, std::make_unique<storm::solver::GmmxxMinMaxLinearEquationSolverFactory<double>>());

    std::shared_ptr<storm::logic::Formula const> formula = formulaParser.parseSingleFormulaFromString("Pmin=? [F \"elected\"]");

    std::unique_ptr<storm::modelchecker::CheckResult> result = checker.check(*formula);
    storm::modelchecker::ExplicitQuantitativeCheckResult<double>& quantitativeResult1 = result->asExplicitQuantitativeCheckResult<double>();

    EXPECT_NEAR(1, quantitativeResult1[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());

    formula = formulaParser.parseSingleFormulaFromString("Pmax=? [F \"elected\"]");

    result = checker.check(*formula);
    storm::modelchecker::ExplicitQuantitativeCheckResult<double>& quantitativeResult2 = result->asExplicitQuantitativeCheckResult<double>();

    EXPECT_NEAR(1, quantitativeResult2[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());

    formula = formulaParser.parseSingleFormulaFromString("Pmin=? [F<=25 \"elected\"]");

    result = checker.check(*formula);
    storm::modelchecker::ExplicitQuantitativeCheckResult<double>& quantitativeResult3 = result->asExplicitQuantitativeCheckResult<double>();

    EXPECT_NEAR(0.0625, quantitativeResult3[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());

    formula = formulaParser.parseSingleFormulaFromString("Pmax=? [F<=25 \"elected\"]");

    result = checker.check(*formula);
    storm::modelchecker::ExplicitQuantitativeCheckResult<double>& quantitativeResult4 = result->asExplicitQuantitativeCheckResult<double>();

    EXPECT_NEAR(0.0625, quantitativeResult4[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());

    formula = formulaParser.parseSingleFormulaFromString("Rmin=? [F \"elected\"]");

    result = checker.check(*formula);
    storm::modelchecker::ExplicitQuantitativeCheckResult<double>& quantitativeResult5 = result->asExplicitQuantitativeCheckResult<double>();

    EXPECT_NEAR(4.285689611, quantitativeResult5[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());

    formula = formulaParser.parseSingleFormulaFromString("Rmax=? [F \"elected\"]");

    result = checker.check(*formula);
    storm::modelchecker::ExplicitQuantitativeCheckResult<double>& quantitativeResult6 = result->asExplicitQuantitativeCheckResult<double>();

    EXPECT_NEAR(4.285689611, quantitativeResult6[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());
}

TEST(GmmxxMdpPrctlModelCheckerTest, SchedulerGeneration) {
    storm::prism::Program program = storm::parser::PrismParser::parse(STORM_CPP_TESTS_BASE_PATH "/functional/modelchecker/scheduler_generation.nm");

    // A parser that we use for conveniently constructing the formulas.
    storm::parser::FormulaParser formulaParser;
    
    storm::generator::NextStateGeneratorOptions options;
    options.setBuildAllLabels();
    std::shared_ptr<storm::models::sparse::Model<double>> model = storm::builder::ExplicitModelBuilder<double>(program, options).build();
    EXPECT_EQ(4ul, model->getNumberOfStates());
    EXPECT_EQ(11ul, model->getNumberOfTransitions());
    
    ASSERT_EQ(model->getType(), storm::models::ModelType::Mdp);

    std::shared_ptr<storm::models::sparse::Mdp<double>> mdp = model->as<storm::models::sparse::Mdp<double>>();

    EXPECT_EQ(7ul, mdp->getNumberOfChoices());

    auto solverFactory = std::make_unique<storm::solver::GmmxxMinMaxLinearEquationSolverFactory<double>>();
    solverFactory->getSettings().setSolutionMethod(storm::solver::StandardMinMaxLinearEquationSolverSettings::SolutionMethod::PolicyIteration);
    storm::modelchecker::SparseMdpPrctlModelChecker<storm::models::sparse::Mdp<double>> checker(*mdp, std::move(solverFactory));
    
    std::shared_ptr<storm::logic::Formula const> formula = formulaParser.parseSingleFormulaFromString("Pmin=? [F \"target\"]");
    
    storm::modelchecker::CheckTask<storm::logic::Formula> checkTask(*formula);
    checkTask.setProduceSchedulers(true);
    
    std::unique_ptr<storm::modelchecker::CheckResult> result = checker.check(checkTask);
    
    ASSERT_TRUE(result->isExplicitQuantitativeCheckResult());
    ASSERT_TRUE(result->asExplicitQuantitativeCheckResult<double>().hasScheduler());
    storm::storage::Scheduler const& scheduler = result->asExplicitQuantitativeCheckResult<double>().getScheduler();
    EXPECT_EQ(0, scheduler.getChoice(0));
    EXPECT_EQ(1, scheduler.getChoice(1));
    EXPECT_EQ(0, scheduler.getChoice(2));
    EXPECT_EQ(0, scheduler.getChoice(3));
    
    formula = formulaParser.parseSingleFormulaFromString("Pmax=? [F \"target\"]");
    
    checkTask = storm::modelchecker::CheckTask<storm::logic::Formula>(*formula);
    checkTask.setProduceSchedulers(true);
    result = checker.check(checkTask);

    ASSERT_TRUE(result->isExplicitQuantitativeCheckResult());
    ASSERT_TRUE(result->asExplicitQuantitativeCheckResult<double>().hasScheduler());
    storm::storage::Scheduler const& scheduler2 = result->asExplicitQuantitativeCheckResult<double>().getScheduler();
    EXPECT_EQ(1, scheduler2.getChoice(0));
    EXPECT_EQ(2, scheduler2.getChoice(1));
    EXPECT_EQ(0, scheduler2.getChoice(2));
    EXPECT_EQ(0, scheduler2.getChoice(3));
}

TEST(GmmxxMdpPrctlModelCheckerTest, TinyRewards) {
    storm::prism::Program program = storm::parser::PrismParser::parse(STORM_CPP_TESTS_BASE_PATH "/functional/modelchecker/tiny_rewards.nm");

    // A parser that we use for conveniently constructing the formulas.
    storm::parser::FormulaParser formulaParser;
    
    std::shared_ptr<storm::models::sparse::Model<double>> model = storm::builder::ExplicitModelBuilder<double>(program, storm::generator::NextStateGeneratorOptions(true, true)).build();
    EXPECT_EQ(3ul, model->getNumberOfStates());
    EXPECT_EQ(4ul, model->getNumberOfTransitions());
    
    ASSERT_EQ(model->getType(), storm::models::ModelType::Mdp);

    std::shared_ptr<storm::models::sparse::Mdp<double>> mdp = model->as<storm::models::sparse::Mdp<double>>();

    EXPECT_EQ(4ul, mdp->getNumberOfChoices());

    storm::modelchecker::SparseMdpPrctlModelChecker<storm::models::sparse::Mdp<double>> checker(*mdp, std::make_unique<storm::solver::GmmxxMinMaxLinearEquationSolverFactory<double>>());
    
    std::shared_ptr<storm::logic::Formula const> formula = formulaParser.parseSingleFormulaFromString("Rmin=? [F \"target\"]");
    
    storm::modelchecker::CheckTask<storm::logic::Formula> checkTask(*formula);
    
    std::unique_ptr<storm::modelchecker::CheckResult> result = checker.check(checkTask);
    
    ASSERT_TRUE(result->isExplicitQuantitativeCheckResult());
    EXPECT_NEAR(1,result->asExplicitQuantitativeCheckResult<double>().getValueVector()[0], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());
    EXPECT_NEAR(1,result->asExplicitQuantitativeCheckResult<double>().getValueVector()[1], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());
    EXPECT_NEAR(0,result->asExplicitQuantitativeCheckResult<double>().getValueVector()[2], storm::settings::getModule<storm::settings::modules::NativeEquationSolverSettings>().getPrecision());
    
}
