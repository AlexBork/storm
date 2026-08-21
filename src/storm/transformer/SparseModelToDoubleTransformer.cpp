#include "SparseModelToDoubleTransformer.h"

#include "storm/exceptions/IllegalArgumentTypeException.h"
#include "storm/models/sparse/Ctmc.h"
#include "storm/models/sparse/Dtmc.h"
#include "storm/models/sparse/MarkovAutomaton.h"
#include "storm/models/sparse/Mdp.h"
#include "storm/models/sparse/Pomdp.h"
#include "storm/models/sparse/Smg.h"
#include "storm/models/sparse/StochasticTwoPlayerGame.h"
#include "storm/storage/sparse/ModelComponents.h"
#include "storm/utility/macros.h"
#include "storm/utility/vector.h"

namespace storm::transformer {
std::shared_ptr<storm::models::sparse::Model<double>> sparseRationalModelToDouble(
    std::shared_ptr<storm::models::sparse::Model<storm::RationalNumber>> const& inputModel) {
    STORM_LOG_THROW(inputModel, storm::exceptions::IllegalArgumentTypeException, "Cannot transform a null model.");
    storm::storage::sparse::ModelComponents<double> convertedComponents;
    convertedComponents.transitionMatrix = inputModel->getTransitionMatrix().toValueType<double>();
    convertedComponents.choiceLabeling = inputModel->getOptionalChoiceLabeling();
    convertedComponents.stateLabeling = inputModel->getStateLabeling();
    convertedComponents.stateValuations = inputModel->getOptionalStateValuations();
    convertedComponents.choiceOrigins = inputModel->getOptionalChoiceOrigins();
    for (auto const& [rewardModelName, rewardModel] : inputModel->getRewardModels()) {
        // Transform reward models
        std::optional<std::vector<double>> optionalStateRewardVector = std::nullopt;
        std::optional<std::vector<double>> optionalStateActionRewardVector = std::nullopt;
        std::optional<storm::storage::SparseMatrix<double>> optionalTransitionRewardMatrix = std::nullopt;
        if (rewardModel.hasStateRewards()) {
            std::vector<double> resultVector;
            resultVector.reserve(rewardModel.getStateRewardVector().size());
            for (auto const& oldValue : rewardModel.getStateRewardVector()) {
                resultVector.push_back(storm::utility::convertNumber<double>(oldValue));
            }
            optionalStateRewardVector = resultVector;
        }
        if (rewardModel.hasStateActionRewards()) {
            std::vector<double> resultVector;
            resultVector.reserve(rewardModel.getStateActionRewardVector().size());
            for (auto const& oldValue : rewardModel.getStateActionRewardVector()) {
                resultVector.push_back(storm::utility::convertNumber<double>(oldValue));
            }
            optionalStateActionRewardVector = resultVector;
        }
        if (rewardModel.hasTransitionRewards()) {
            optionalTransitionRewardMatrix = rewardModel.getTransitionRewardMatrix().toValueType<double>();
        }
        convertedComponents.rewardModels.emplace(
            rewardModelName, storm::models::sparse::StandardRewardModel<double>(
                                 std::move(optionalStateRewardVector), std::move(optionalStateActionRewardVector), std::move(optionalTransitionRewardMatrix)));
    }
    switch (inputModel->getType()) {
        case storm::models::ModelType::Dtmc:
            return std::make_shared<storm::models::sparse::Dtmc<double>>(storm::models::sparse::Dtmc<double>(convertedComponents));
        case storm::models::ModelType::Mdp:
            return std::make_shared<storm::models::sparse::Mdp<double>>(storm::models::sparse::Mdp<double>(convertedComponents));
        case storm::models::ModelType::Ctmc: {
            auto ctmc = inputModel->as<storm::models::sparse::Ctmc<storm::RationalNumber>>();
            std::vector<double> resultVector;
            resultVector.reserve(ctmc->getExitRateVector().size());
            for (auto const& oldValue : ctmc->getExitRateVector()) {
                resultVector.push_back(storm::utility::convertNumber<double>(oldValue));
            }
            convertedComponents.exitRates = resultVector;
            // Markov automata store probabilities in their transition matrix and rates separately in exitRates.
            convertedComponents.rateTransitions = false;
            return std::make_shared<storm::models::sparse::Ctmc<double>>(storm::models::sparse::Ctmc<double>(convertedComponents));
        }
        case storm::models::ModelType::MarkovAutomaton: {
            auto ma = inputModel->as<storm::models::sparse::MarkovAutomaton<storm::RationalNumber>>();
            std::vector<double> resultVector;
            resultVector.reserve(ma->getExitRates().size());
            for (auto const& oldValue : ma->getExitRates()) {
                resultVector.push_back(storm::utility::convertNumber<double>(oldValue));
            }
            convertedComponents.exitRates = resultVector;
            convertedComponents.rateTransitions = true;
            convertedComponents.markovianStates = ma->getMarkovianStates();
            return std::make_shared<storm::models::sparse::MarkovAutomaton<double>>(storm::models::sparse::MarkovAutomaton<double>(convertedComponents));
        }
        case storm::models::ModelType::Pomdp: {
            auto pomdp = inputModel->as<storm::models::sparse::Pomdp<storm::RationalNumber>>();
            convertedComponents.observabilityClasses = pomdp->getObservations();
            convertedComponents.observationValuations = pomdp->getOptionalObservationValuations();
            return std::make_shared<models::sparse::Pomdp<double>>(models::sparse::Pomdp<double>(convertedComponents, pomdp->isCanonic()));
        }
        case storm::models::ModelType::Smg: {
            auto smg = inputModel->as<storm::models::sparse::Smg<storm::RationalNumber>>();
            convertedComponents.statePlayerIndications = smg->getStatePlayerIndications();
            convertedComponents.playerNameToIndexMap = smg->getPlayerNamesToIndex();
            return std::make_shared<storm::models::sparse::Smg<double>>(models::sparse::Smg<double>(convertedComponents));
        }
        case storm::models::ModelType::S2pg: {
            auto s2pg = inputModel->as<storm::models::sparse::StochasticTwoPlayerGame<storm::RationalNumber>>();
            convertedComponents.player1Matrix = s2pg->getPlayer1Matrix();
            return std::make_shared<storm::models::sparse::StochasticTwoPlayerGame<double>>(
                models::sparse::StochasticTwoPlayerGame<double>(convertedComponents));
        }
        default:
            STORM_LOG_THROW(false, storm::exceptions::IllegalArgumentTypeException,
                            "Value type transformation is not supported for models of type " << inputModel->getType() << ".");
    }
    return nullptr;
}
}  // namespace storm::transformer
