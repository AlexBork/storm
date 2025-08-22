#include "storm-pomdp/beliefs/exploration/BeliefExploration.h"

#include "storm-pomdp/beliefs/exploration/BeliefExplorationMatrix.h"
#include "storm-pomdp/beliefs/exploration/BeliefExplorationMode.h"

#include "storm-pomdp/beliefs/storage/Belief.h"
#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/models/sparse/Pomdp.h"
#include "storm/utility/OptionalRef.h"
#include "storm/utility/SignalHandler.h"
#include "storm/utility/macros.h"
#include "storm/utility/vector.h"

namespace storm::pomdp::beliefs {

template<typename BeliefMdpValueType, typename PomdpType, typename BeliefType>
template<typename InfoType, typename NextStateHandleType>
bool BeliefExploration<BeliefMdpValueType, PomdpType, BeliefType>::performExploration(InfoType& info, NextStateHandleType&& exploreNextStates,
                                                                                      TerminalBeliefCallback const& terminalBeliefCallback,
                                                                                      TerminationCallback const& terminationCallback) {
    while (info.queue.hasNext()) {
        // Check if we terminate prematurely
        if ((terminationCallback && terminationCallback()) || storm::utility::resources::isTerminate()) {
            STORM_LOG_ASSERT(storm::utility::resources::isTerminate() || info.queue.getContents() == info.getFrontierBeliefs(),
                             "Frontier beliefs inconsistent.");
            return false;  // Terminate prematurely
        }

        // Get the next belief to explore and perform some checks
        auto const currentBeliefId = info.queue.popNext();
        STORM_LOG_ASSERT(info.discoveredBeliefs.containsId(currentBeliefId), "Unknown belief id");
        STORM_LOG_ASSERT(info.exploredBeliefs.count(currentBeliefId) == 0, "Belief #" << currentBeliefId << " already explored.");
        STORM_LOG_ASSERT(info.terminalBeliefValues.count(currentBeliefId) == 0, "Belief #" << currentBeliefId << " already found to be terminal.");
        // do not take the current belief as reference since it will be invalidated when collecting more beliefs
        auto const currentBelief = info.discoveredBeliefs.getBeliefFromId(currentBeliefId);
        STORM_LOG_TRACE("Explore belief " << currentBeliefId << " : " << currentBelief.toString());
        // Check if the current belief is terminal
        if (terminalBeliefCallback) {
            if (auto terminal = terminalBeliefCallback(currentBelief); terminal.has_value()) {
                info.terminalBeliefValues.emplace(currentBeliefId, std::move(terminal.value()));
                continue;
            }
        }

        // Explore for each action the successors of the current belief with that action. Potentially also add rewards.
        info.exploredBeliefs.emplace(currentBeliefId, info.matrix.groups());
        auto const numActions = firstStateNextStateGenerator.getBeliefNumberOfActions(currentBelief);
        for (uint64_t localActionIndex = 0; localActionIndex < numActions; ++localActionIndex) {
            exploreNextStates(currentBelief, localActionIndex);
            info.matrix.endCurrentRow();
            if (firstStateNextStateGenerator.hasRewardModel()) {
                info.actionRewards.emplace_back(
                    storm::utility::convertNumber<BeliefMdpValueType>(firstStateNextStateGenerator.getBeliefActionReward(currentBelief, localActionIndex)));
            }
            if (info.generateChoiceLabeling) {
                info.matrix.choiceLabels.push_back(firstStateNextStateGenerator.getBeliefActionChoiceLabels(currentBelief, localActionIndex));
            }
        }
        info.matrix.endCurrentRowGroup();
    }
    return true;
}

template<typename BeliefMdpValueType, typename PomdpType, typename BeliefType>
BeliefExploration<BeliefMdpValueType, PomdpType, BeliefType>::BeliefExploration(PomdpType const& pomdp) : firstStateNextStateGenerator(pomdp) {
    // Intentionally left empty.
}

template class BeliefExploration<double, storm::models::sparse::Pomdp<double>, Belief<double>>;
template class BeliefExploration<double, storm::models::sparse::Pomdp<double>, Belief<storm::RationalNumber>>;
template class BeliefExploration<storm::RationalNumber, storm::models::sparse::Pomdp<storm::RationalNumber>, Belief<double>>;
template class BeliefExploration<storm::RationalNumber, storm::models::sparse::Pomdp<storm::RationalNumber>, Belief<storm::RationalNumber>>;

template void BeliefExploration<double, storm::models::sparse::Pomdp<double>, Belief<double>>::resumeExploration(
    StandardExplorationInformation<double, Belief<double>>& info, TerminalBeliefCallback const& terminalBeliefCallback,
    TerminationCallback const& terminationCallback, storm::OptionalRef<std::string const> rewardModelName, storm::OptionalRef<NoAbstractionType> abstraction);
template void BeliefExploration<double, storm::models::sparse::Pomdp<double>, Belief<double>>::resumeExploration(
    StandardExplorationInformation<double, Belief<double>>& info, TerminalBeliefCallback const& terminalBeliefCallback,
    TerminationCallback const& terminationCallback, storm::OptionalRef<std::string const> rewardModelName,
    storm::OptionalRef<FreudenthalTriangulationBeliefAbstraction<Belief<double>>> abstraction);

template void BeliefExploration<double, storm::models::sparse::Pomdp<double>, Belief<storm::RationalNumber>>::resumeExploration(
    StandardExplorationInformation<double, Belief<storm::RationalNumber>>& info, TerminalBeliefCallback const& terminalBeliefCallback,
    TerminationCallback const& terminationCallback, storm::OptionalRef<std::string const> rewardModelName, storm::OptionalRef<NoAbstractionType> abstraction);
template void BeliefExploration<double, storm::models::sparse::Pomdp<double>, Belief<storm::RationalNumber>>::resumeExploration(
    StandardExplorationInformation<double, Belief<storm::RationalNumber>>& info, TerminalBeliefCallback const& terminalBeliefCallback,
    TerminationCallback const& terminationCallback, storm::OptionalRef<std::string const> rewardModelName,
    storm::OptionalRef<FreudenthalTriangulationBeliefAbstraction<Belief<double>>> abstraction);
}  // namespace storm::pomdp::beliefs