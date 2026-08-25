#include "storm-config.h"
#include "test/storm_gtest.h"

#include "storm-parsers/api/storm-parsers.h"
#include "storm-parsers/parser/AutoParser.h"
#include "storm-parsers/parser/DeterministicModelParser.h"
#include "storm-parsers/parser/MarkovAutomatonParser.h"
#include "storm/api/storm.h"
#include "storm/modelchecker/results/ExplicitQuantitativeCheckResult.h"
#include "storm/models/sparse/Ctmc.h"
#include "storm/models/sparse/MarkovAutomaton.h"
#include "storm/models/sparse/StandardRewardModel.h"
#include "storm/transformer/TransitionToActionRewardTransformer.h"

namespace {
double computeInitialReward(std::shared_ptr<storm::models::sparse::Model<double>> const& model, std::string const& formulaString) {
    auto const formula = storm::api::extractFormulasFromProperties(storm::api::parseProperties(formulaString)).front();
    auto const result = storm::api::verifyWithSparseEngine(model, storm::api::createTask<double>(formula, true));
    return result->asExplicitQuantitativeCheckResult<double>()[*model->getInitialStates().begin()];
}

TEST(TransitionToActionRewardTransformerTest, DtmcDie) {
    auto const model = storm::parser::AutoParser<>::parseModel(STORM_TEST_RESOURCES_DIR "/dtmc/die.tra", STORM_TEST_RESOURCES_DIR "/dtmc/die.lab", "",
                                                               STORM_TEST_RESOURCES_DIR "/rew/die.coin_flips.trans.rew");

    auto const transformed = storm::transformer::transformTransitionToActionRewards<double>(model, {""});

    EXPECT_EQ(25ull, transformed.model->getNumberOfStates());
    EXPECT_EQ(32ull, transformed.model->getNumberOfTransitions());
    EXPECT_EQ((std::vector<uint64_t>{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24}), transformed.originalToNewStateIndices);
    EXPECT_TRUE(transformed.model->getRewardModel("").hasStateActionRewards());
    EXPECT_FALSE(transformed.model->getRewardModel("").hasTransitionRewards());

    EXPECT_NEAR(11.0 / 3.0, computeInitialReward(transformed.model, "R=? [F \"done\"]"), 1e-6);
}

TEST(TransitionToActionRewardTransformerTest, CtmcDie) {
    auto const model = std::make_shared<storm::models::sparse::Ctmc<double>>(storm::parser::DeterministicModelParser<>::parseCtmc(
        STORM_TEST_RESOURCES_DIR "/tra/die.tra", STORM_TEST_RESOURCES_DIR "/lab/die.lab", "", STORM_TEST_RESOURCES_DIR "/rew/die.coin_flips.trans.rew"));

    auto const transformed = storm::transformer::transformTransitionToActionRewards<double>(model, {""});

    EXPECT_EQ(storm::models::ModelType::Ctmc, transformed.model->getType());
    EXPECT_EQ(25ull, transformed.model->getNumberOfStates());
    EXPECT_EQ(32ull, transformed.model->getNumberOfTransitions());
    EXPECT_TRUE(transformed.model->getRewardModel("").hasStateActionRewards());
    EXPECT_FALSE(transformed.model->getRewardModel("").hasTransitionRewards());

    EXPECT_NEAR(11.0 / 3.0, computeInitialReward(transformed.model, "R=? [F \"done\"]"), 1e-6);
}

TEST(TransitionToActionRewardTransformerTest, MdpTwoDice) {
    auto const model = storm::parser::AutoParser<>::parseModel(STORM_TEST_RESOURCES_DIR "/tra/two_dice.tra", STORM_TEST_RESOURCES_DIR "/lab/two_dice.lab", "",
                                                               STORM_TEST_RESOURCES_DIR "/rew/two_dice.flip.trans.rew");

    auto const transformed = storm::transformer::transformTransitionToActionRewards<double>(model, {""});

    EXPECT_EQ(storm::models::ModelType::Mdp, transformed.model->getType());
    EXPECT_EQ(337ull, transformed.model->getNumberOfStates());
    EXPECT_EQ(604ull, transformed.model->getNumberOfTransitions());
    ASSERT_EQ(169ull, transformed.originalToNewStateIndices.size());
    for (uint64_t state = 0; state < transformed.originalToNewStateIndices.size(); ++state) {
        EXPECT_EQ(2 * state, transformed.originalToNewStateIndices[state]);
    }
    EXPECT_TRUE(transformed.model->getRewardModel("").hasStateActionRewards());
    EXPECT_FALSE(transformed.model->getRewardModel("").hasTransitionRewards());

    EXPECT_NEAR(22.0 / 3.0, computeInitialReward(transformed.model, "Rmin=? [F \"done\"]"), 1e-6);
}

TEST(TransitionToActionRewardTransformerTest, MarkovAutomatonGeneral) {
    auto const model = std::make_shared<storm::models::sparse::MarkovAutomaton<double>>(storm::parser::MarkovAutomatonParser<>::parseMarkovAutomaton(
        STORM_TEST_RESOURCES_DIR "/tra/ma_general.tra", STORM_TEST_RESOURCES_DIR "/lab/ma_general.lab", STORM_TEST_RESOURCES_DIR "/rew/ma_general.state.rew"));
    auto transitionRewards = model->getTransitionMatrix();
    for (auto& entry : transitionRewards) {
        entry.setValue(1.0);
    }
    auto const stateRewards = model->getRewardModel("").getOptionalStateRewardVector();
    model->getRewardModel("") = storm::models::sparse::StandardRewardModel<double>(std::move(stateRewards), std::nullopt, std::move(transitionRewards));

    auto const transformed = storm::transformer::transformTransitionToActionRewards<double>(model, {""});

    EXPECT_EQ(storm::models::ModelType::MarkovAutomaton, transformed.model->getType());
    EXPECT_EQ(12ull, transformed.model->getNumberOfStates());
    EXPECT_EQ(18ull, transformed.model->getNumberOfTransitions());
    EXPECT_EQ((std::vector<uint64_t>{1, 3, 5, 7, 9, 11}), transformed.originalToNewStateIndices);
    EXPECT_TRUE(transformed.model->getRewardModel("").hasStateActionRewards());
    EXPECT_FALSE(transformed.model->getRewardModel("").hasTransitionRewards());

    auto transformedMa = transformed.model->as<storm::models::sparse::MarkovAutomaton<double>>();
    EXPECT_EQ(2ull, transformedMa->getMarkovianStates().getNumberOfSetBits());
    EXPECT_TRUE(transformedMa->isMarkovianState(1));
    EXPECT_EQ(2.0, transformedMa->getExitRate(1));
    EXPECT_TRUE(transformedMa->isMarkovianState(5));
    EXPECT_EQ(15.0, transformedMa->getExitRate(5));
    for (uint64_t state = 0; state < transformedMa->getNumberOfStates(); state += 2) {
        EXPECT_TRUE(transformedMa->isProbabilisticState(state));
    }
}

}  // namespace
