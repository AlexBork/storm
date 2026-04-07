#pragma once
#include "storm/models/sparse/Pomdp.h"
#include "storm/utility/logging.h"

namespace storm {
namespace pomdp {
namespace analysis {

template<typename ValueType>
bool detectRewardAwarePomdp(std::shared_ptr<models::sparse::Pomdp<ValueType>> const& pomdp) {
    if (pomdp->getNumberOfRewardModels() == 0) {
        STORM_LOG_WARN("POMDP has no defined reward models and is therefore not reward-aware.");
        return false;
    }
    for (auto const& rewardModelPair : pomdp->getRewardModels()) {
        std::vector<std::vector<std::unordered_map<uint32_t, ValueType>>> observationActionObservationReward(pomdp->getNrObservations());
        for (uint32_t obs : pomdp->getObservations()) {
            auto statesWithObs = pomdp->getStatesWithObservation(obs);
            auto reprState = statesWithObs[0];
            auto getNrChoicesInObs = pomdp->getNumberOfChoices(reprState);
            observationActionObservationReward[obs] = std::vector<std::unordered_map<uint32_t, ValueType>>(getNrChoicesInObs);
        }
        [[maybe_unused]] auto const& rewardModel = rewardModelPair.second;
        STORM_LOG_THROW(!rewardModel.hasTransitionRewards(), storm::exceptions::NotSupportedException,
                        "Transition rewards are currently not supported in this context.");

        for (uint64_t state = 0; state < pomdp->getNumberOfStates(); ++state) {
            uint64_t localChoiceIndex = 0ul;
            for (auto choice : pomdp->getTransitionMatrix().getRowGroupIndices(state)) {
                for (auto const& entry : pomdp->getTransitionMatrix().getRow(choice)) {
                    auto transitionReward = storm::utility::zero<ValueType>();
                    if (rewardModel.hasStateRewards()) {
                        transitionReward += rewardModel.getStateReward(state);
                    }
                    if (rewardModel.hasStateActionRewards()) {
                        transitionReward += rewardModel.getStateActionReward(choice);
                    }
                    uint32_t originObservation = pomdp->getObservation(state);
                    uint32_t targetObservation = pomdp->getObservation(entry.getColumn());
                    if (observationActionObservationReward[originObservation][localChoiceIndex].contains(targetObservation)) {
                        if (observationActionObservationReward[originObservation][localChoiceIndex][targetObservation] != transitionReward) {
                            return false;
                        }
                    } else {
                        observationActionObservationReward[originObservation][localChoiceIndex][targetObservation] = transitionReward;
                    }
                }
                ++localChoiceIndex;
            }
        }
    }
    return true;
}
}  // namespace analysis
}  // namespace pomdp
}  // namespace storm