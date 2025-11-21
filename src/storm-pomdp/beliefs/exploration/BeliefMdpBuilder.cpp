#include "storm-pomdp/beliefs/exploration/BeliefMdpBuilder.h"

#include "storm-pomdp/beliefs/storage/Belief.h"

#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/models/sparse/Mdp.h"
#include "storm/models/sparse/StandardRewardModel.h"
#include "storm/storage/SparseMatrix.h"
#include "storm/storage/sparse/ModelComponents.h"

#include "storm/exceptions/UnexpectedException.h"

namespace storm::pomdp::beliefs {

std::shared_ptr<storm::logic::Formula const> createFormulaForBeliefMdp(PropertyInformation const& propertyInformation) {
    STORM_LOG_ASSERT(propertyInformation.kind == PropertyInformation::Kind::ReachabilityProbability ||
                         propertyInformation.kind == PropertyInformation::Kind::ExpectedTotalReachabilityReward,
                     "Unexpected kind of property.");
    switch (propertyInformation.kind) {
        case PropertyInformation::Kind::ReachabilityProbability: {
            auto target = std::make_shared<storm::logic::AtomicLabelFormula const>("target");
            auto eventuallyTarget = std::make_shared<storm::logic::EventuallyFormula const>(target, storm::logic::FormulaContext::Probability);
            return std::make_shared<storm::logic::ProbabilityOperatorFormula const>(eventuallyTarget,
                                                                                    storm::logic::OperatorInformation(propertyInformation.dir));
        }
        case PropertyInformation::Kind::ExpectedTotalReachabilityReward: {
            auto bottom = std::make_shared<storm::logic::AtomicLabelFormula const>("target");
            auto eventuallyBottom = std::make_shared<storm::logic::EventuallyFormula const>(bottom, storm::logic::FormulaContext::Reward,
                                                                                            storm::logic::RewardAccumulation(true, false, false));
            return std::make_shared<storm::logic::RewardOperatorFormula const>(eventuallyBottom, propertyInformation.rewardModelName.value(),
                                                                               storm::logic::OperatorInformation(propertyInformation.dir));
        }
    }
    STORM_LOG_THROW(false, storm::exceptions::UnexpectedException, "Unhandled case.");
}

template<typename BeliefMdpValueType, typename BeliefType, typename... ExtraTransitionData>
std::pair<std::shared_ptr<storm::models::sparse::Mdp<BeliefMdpValueType>>, std::unordered_map<uint64_t, BeliefId>> buildBeliefMdpWithImplicitCutoffs(
    ExplorationInformation<BeliefMdpValueType, BeliefType, ExtraTransitionData...> const& explorationInformation,
    PropertyInformation const& propertyInformation, std::function<BeliefMdpValueType(BeliefType const&)> const& computeCutOffValue) {
    STORM_LOG_ASSERT(propertyInformation.kind == PropertyInformation::Kind::ReachabilityProbability ||
                         propertyInformation.kind == PropertyInformation::Kind::ExpectedTotalReachabilityReward,
                     "Unexpected kind of property.");
    bool const reachabilityProbability = propertyInformation.kind == PropertyInformation::Kind::ReachabilityProbability;

    constexpr bool clippingUsed = std::is_same_v<ExplorationInformation<BeliefMdpValueType, BeliefType, ExtraTransitionData...>,
                                                 ClippingExplorationInformation<BeliefMdpValueType, BeliefType>>;

    uint64_t const numExtraStates = reachabilityProbability || clippingUsed ? 2ull : 1ull;
    uint64_t const numStates = explorationInformation.matrix.groups() + numExtraStates;
    uint64_t const numChoices = explorationInformation.matrix.rows() + numExtraStates;
    uint64_t const targetState = numStates - numExtraStates;
    uint64_t const bottomState = numStates - 1;
    std::optional<models::sparse::ChoiceLabeling> optionalChoiceLabeling;
    std::vector<BeliefMdpValueType> actionRewards;
    if (!reachabilityProbability) {
        actionRewards.reserve(numChoices);
        actionRewards.insert(actionRewards.end(), explorationInformation.actionRewards.begin(), explorationInformation.actionRewards.end());
        actionRewards.push_back(storm::utility::zero<BeliefMdpValueType>());
        if (clippingUsed) {
            actionRewards.push_back(storm::utility::zero<BeliefMdpValueType>());
        }
        STORM_LOG_ASSERT(numChoices == actionRewards.size(),
                         "Unexpected size of action rewards: Expected " << numChoices << " got " << actionRewards.size() << ".");
    }
    if (explorationInformation.matrix.hasChoiceLabels()) {
        optionalChoiceLabeling = models::sparse::ChoiceLabeling(numChoices);
    }
    storm::storage::SparseMatrixBuilder<BeliefMdpValueType> transitionBuilder(numChoices, numStates, 0, true, true, numStates);
    for (uint64_t state = 0; state < numStates - numExtraStates; ++state) {
        uint64_t choice = explorationInformation.matrix.rowGroupIndices[state];
        transitionBuilder.newRowGroup(choice);
        for (uint64_t const groupEnd = explorationInformation.matrix.rowGroupIndices[state + 1]; choice < groupEnd; ++choice) {
            auto probabilityToBottom = storm::utility::zero<BeliefMdpValueType>();
            auto probabilityToTarget = storm::utility::zero<BeliefMdpValueType>();
            if (optionalChoiceLabeling.has_value()) {
                for (auto const& label : explorationInformation.matrix.choiceLabels.at(choice)) {
                    if (!optionalChoiceLabeling.value().containsLabel(label)) {
                        optionalChoiceLabeling.value().addLabel(label);
                    }
                    optionalChoiceLabeling.value().addLabelToChoice(label, choice);
                }
            }
            for (uint64_t entryIndex = explorationInformation.matrix.rowIndications[choice];
                 entryIndex < explorationInformation.matrix.rowIndications[choice + 1]; ++entryIndex) {
                auto const& entry = explorationInformation.matrix.transitions[entryIndex];
                if (auto explIt = explorationInformation.exploredBeliefs.find(entry.targetBelief); explIt != explorationInformation.exploredBeliefs.end()) {
                    // Transition to explored belief
                    transitionBuilder.addNextValue(choice, explIt->second, entry.probability);
                    if constexpr (clippingUsed) {
                        // In case of clipping exploration, we have extra data that indicates whether the transition is a clipping transition
                        auto const& clippingProbability = std::get<0>(entry.data);
                        auto const& rewardPenalty = std::get<1>(entry.data);

                        if (clippingProbability) {
                            if (reachabilityProbability) {
                                probabilityToBottom += *clippingProbability;
                            } else if (rewardPenalty) {
                                if (storm::utility::isInfinity(*rewardPenalty)) {
                                    /* Infinite reward on transitions is not correctly handled by the model checker. Therefore, we treat it by adding a
                                     * transition to the bottom state which due to the semantics of expected reward until reaching a target has infinite
                                     * expected reward.This causes the expected reward for the transition to become infinite. */
                                    probabilityToBottom += *clippingProbability;
                                } else {
                                    actionRewards[choice] += *rewardPenalty;
                                    probabilityToTarget += *clippingProbability;
                                }
                            } else {
                                probabilityToTarget += *clippingProbability;
                            }
                        }
                    }
                } else {
                    // Transition to unexplored belief (either terminal or cut-off)
                    BeliefMdpValueType successorValue;
                    if (auto terminalIt = explorationInformation.terminalBeliefValues.find(entry.targetBelief);
                        terminalIt != explorationInformation.terminalBeliefValues.end()) {
                        successorValue = entry.probability * terminalIt->second;  // terminal value determined during exploration
                    } else {
                        // Transition to cut-off belief
                        BeliefType const& successorBelief = explorationInformation.discoveredBeliefs.getBeliefFromId(entry.targetBelief);
                        successorValue = entry.probability * computeCutOffValue(successorBelief);  // Cut-off value
                    }
                    if (reachabilityProbability) {
                        probabilityToTarget += successorValue;
                        probabilityToBottom += entry.probability - successorValue;
                    } else {
                        probabilityToTarget += entry.probability;
                        actionRewards[choice] += successorValue;
                    }
                }
            }
            if (!storm::utility::isZero(probabilityToTarget)) {
                transitionBuilder.addNextValue(choice, targetState, probabilityToTarget);
            }
            if (!storm::utility::isZero(probabilityToBottom)) {
                transitionBuilder.addNextValue(choice, bottomState, probabilityToBottom);
            }
        }
    }

    transitionBuilder.newRowGroup(numChoices - numExtraStates);
    transitionBuilder.addNextValue(numChoices - numExtraStates, targetState, storm::utility::one<BeliefMdpValueType>());
    if (optionalChoiceLabeling.has_value()) {
        if (!optionalChoiceLabeling.value().containsLabel("__loop__")) {
            optionalChoiceLabeling.value().addLabel("__loop__");
        }
        optionalChoiceLabeling.value().addLabelToChoice("__loop__", numChoices - numExtraStates);
    }
    if (reachabilityProbability) {
        transitionBuilder.newRowGroup(numChoices - 1);
        transitionBuilder.addNextValue(numChoices - 1, bottomState, storm::utility::one<BeliefMdpValueType>());
        if (optionalChoiceLabeling.has_value()) {
            if (!optionalChoiceLabeling.value().containsLabel("__loop__")) {
                optionalChoiceLabeling.value().addLabel("__loop__");
            }
            optionalChoiceLabeling.value().addLabelToChoice("__loop__", numChoices - 1);
        }
    }

    storm::models::sparse::StateLabeling stateLabeling(numStates);
    stateLabeling.addLabel("target");
    stateLabeling.addLabelToState("target", targetState);
    stateLabeling.addLabel("init");
    stateLabeling.addLabelToState("init", explorationInformation.exploredBeliefs.at(explorationInformation.initialBeliefId));

    if (reachabilityProbability) {
        stateLabeling.addLabel("bottom");
        stateLabeling.addLabelToState("bottom", bottomState);
    }
    storm::storage::sparse::ModelComponents<BeliefMdpValueType> components(transitionBuilder.build(), std::move(stateLabeling));

    if (optionalChoiceLabeling.has_value()) {
        components.choiceLabeling = std::move(optionalChoiceLabeling.value());
    }

    if (!reachabilityProbability) {
        storm::models::sparse::StandardRewardModel<BeliefMdpValueType> rewardModel(std::nullopt, std::move(actionRewards));
        components.rewardModels.emplace(propertyInformation.rewardModelName.value(), std::move(rewardModel));
    }

    // If requested, populate the stateToBeliefMap: for each MDP state index, record the corresponding BeliefId or InvalidBeliefId
    std::unordered_map<uint64_t, BeliefId> stateToBeliefMap;
    // Explored beliefs
    for (auto const& [beliefId, stateIndex] : explorationInformation.exploredBeliefs) {
        stateToBeliefMap[stateIndex] = beliefId;
    }

    return std::make_pair(std::make_shared<storm::models::sparse::Mdp<BeliefMdpValueType>>(std::move(components)), std::move(stateToBeliefMap));
}

template<typename BeliefMdpValueType, typename BeliefType, typename... ExtraTransitionData>
std::pair<std::shared_ptr<models::sparse::Mdp<BeliefMdpValueType>>, std::unordered_map<uint64_t, BeliefId>> buildBeliefMdp(
    ExplorationInformation<BeliefMdpValueType, BeliefType, ExtraTransitionData...> const& explorationInformation,
    PropertyInformation const& propertyInformation,
    std::function<std::unordered_map<std::string, BeliefMdpValueType>(BeliefType const&)> const& computeCutOffValueMap) {
    STORM_LOG_ASSERT(propertyInformation.kind == PropertyInformation::Kind::ReachabilityProbability ||
                         propertyInformation.kind == PropertyInformation::Kind::ExpectedTotalReachabilityReward,
                     "Unexpected kind of property.");

    bool const reachabilityProbability = propertyInformation.kind == PropertyInformation::Kind::ReachabilityProbability;
    // First gather all cut-off information
    uint64_t nrCutOffChoices = 0ull;
    std::unordered_map<BeliefId, std::unordered_map<std::string, BeliefMdpValueType>> cutOffInformationMap;
    for (auto const& frontierBeliefId : explorationInformation.getFrontierBeliefs()) {
        auto const& frontierBelief = explorationInformation.discoveredBeliefs.getBeliefFromId(frontierBeliefId);
        cutOffInformationMap[frontierBeliefId] = computeCutOffValueMap(frontierBelief);
        nrCutOffChoices += cutOffInformationMap[frontierBeliefId].size();
    }

    constexpr bool clippingUsed = std::is_same_v<ExplorationInformation<BeliefMdpValueType, BeliefType, ExtraTransitionData...>,
                                                 ClippingExplorationInformation<BeliefMdpValueType, BeliefType>>;

    // Possible optimisation: check if bottom is really needed for clipping
    uint64_t const numBottomTargetStates = reachabilityProbability || clippingUsed ? 2ull : 1ull;
    uint64_t const numExtraStates = numBottomTargetStates + explorationInformation.getFrontierBeliefs().size();
    uint64_t const numStates = explorationInformation.matrix.groups() + numExtraStates;
    uint64_t const numChoices = explorationInformation.matrix.rows() + numBottomTargetStates + nrCutOffChoices;
    uint64_t const targetState = numStates - numBottomTargetStates;
    uint64_t const bottomState = numStates - 1;
    std::optional<models::sparse::ChoiceLabeling> optionalChoiceLabeling;
    if (explorationInformation.matrix.hasChoiceLabels()) {
        optionalChoiceLabeling = models::sparse::ChoiceLabeling(numChoices);
    }

    std::vector<BeliefMdpValueType> actionRewards;
    if (!reachabilityProbability) {
        actionRewards.reserve(numChoices);
        actionRewards.insert(actionRewards.end(), explorationInformation.actionRewards.begin(), explorationInformation.actionRewards.end());
        // Insert 0 for all cut-off choices and bottom state
        actionRewards.insert(actionRewards.end(), nrCutOffChoices + numBottomTargetStates, storm::utility::zero<BeliefMdpValueType>());
        STORM_LOG_ASSERT(numChoices == actionRewards.size(),
                         "Unexpected size of action rewards: Expected " << numChoices << " got " << actionRewards.size() << ".");
    }

    std::unordered_map<BeliefId, uint64_t> frontierBeliefToStateMap;
    std::unordered_map<uint64_t, BeliefId> stateToFrontierBeliefMap;
    uint64_t nextStateId = numStates - numExtraStates;

    storm::storage::SparseMatrixBuilder<BeliefMdpValueType> transitionBuilder(numChoices, numStates, 0, true, true, numStates);
    // Treat explored beliefs
    for (uint64_t state = 0; state < numStates - numExtraStates; ++state) {
        uint64_t choice = explorationInformation.matrix.rowGroupIndices[state];
        transitionBuilder.newRowGroup(choice);
        for (uint64_t const groupEnd = explorationInformation.matrix.rowGroupIndices[state + 1]; choice < groupEnd; ++choice) {
            auto probabilityToBottom = storm::utility::zero<BeliefMdpValueType>();
            auto probabilityToTarget = storm::utility::zero<BeliefMdpValueType>();
            for (uint64_t entryIndex = explorationInformation.matrix.rowIndications[choice];
                 entryIndex < explorationInformation.matrix.rowIndications[choice + 1]; ++entryIndex) {
                auto const& entry = explorationInformation.matrix.transitions[entryIndex];
                if (auto explIt = explorationInformation.exploredBeliefs.find(entry.targetBelief); explIt != explorationInformation.exploredBeliefs.end()) {
                    // Transition to explored belief
                    transitionBuilder.addNextValue(choice, explIt->second, entry.probability);
                    if constexpr (clippingUsed) {
                        // In case of clipping exploration, we have extra data that indicates whether the transition is a clipping transition
                        auto const& clippingProbability = std::get<0>(entry.data);
                        auto const& rewardPenalty = std::get<1>(entry.data);

                        if (clippingProbability) {
                            if (reachabilityProbability) {
                                probabilityToBottom += *clippingProbability;
                            } else if (rewardPenalty) {
                                if (storm::utility::isInfinity(*rewardPenalty)) {
                                    /* Infinite reward on transitions is not correctly handled by the model checker. Therefore, we treat it by adding a
                                     * transition to the bottom state which due to the semantics of expected reward until reaching a target has infinite
                                     * expected reward.This causes the expected reward for the transition to become infinite. */
                                    probabilityToBottom += *clippingProbability;
                                } else {
                                    actionRewards[choice] += *rewardPenalty;
                                    probabilityToTarget += *clippingProbability;
                                }
                            } else {
                                probabilityToTarget += *clippingProbability;
                            }
                        }
                    }
                } else {
                    // Transition to unexplored belief (either terminal or cut-off)
                    BeliefMdpValueType successorValue;
                    if (auto terminalIt = explorationInformation.terminalBeliefValues.find(entry.targetBelief);
                        terminalIt != explorationInformation.terminalBeliefValues.end()) {
                        successorValue = entry.probability * terminalIt->second;  // terminal value determined during exploration
                        if (reachabilityProbability) {
                            probabilityToTarget += successorValue;
                            probabilityToBottom += entry.probability - successorValue;
                        } else {
                            probabilityToTarget += entry.probability;
                            actionRewards[choice] += successorValue;
                        }
                    } else {
                        // Transition to frontier belief
                        auto [insertIterator, inserted] = frontierBeliefToStateMap.insert({entry.targetBelief, nextStateId});
                        if (inserted) {
                            stateToFrontierBeliefMap[nextStateId] = entry.targetBelief;
                            ++nextStateId;
                        }
                        transitionBuilder.addNextValue(choice, insertIterator->second, entry.probability);
                    }
                }
            }
            // Add transition to bottom/target state if necessary
            if (!storm::utility::isZero(probabilityToTarget)) {
                transitionBuilder.addNextValue(choice, targetState, probabilityToTarget);
            }
            if (!storm::utility::isZero(probabilityToBottom)) {
                transitionBuilder.addNextValue(choice, bottomState, probabilityToBottom);
            }
            if (optionalChoiceLabeling.has_value()) {
                for (auto const& label : explorationInformation.matrix.choiceLabels.at(choice)) {
                    if (!optionalChoiceLabeling.value().containsLabel(label)) {
                        optionalChoiceLabeling.value().addLabel(label);
                    }
                    optionalChoiceLabeling.value().addLabelToChoice(label, choice);
                }
            }
        }
    }
    // Treat frontier beliefs
    uint64_t choice = explorationInformation.matrix.rows();
    for (uint64_t state = numStates - numExtraStates; state < numStates - numBottomTargetStates; ++state) {
        transitionBuilder.newRowGroup(choice);
        std::unordered_map<std::string, BeliefMdpValueType> cutOffInformationForBelief = cutOffInformationMap.at(stateToFrontierBeliefMap.at(state));
        for (auto const& entry : cutOffInformationForBelief) {
            if (reachabilityProbability) {
                transitionBuilder.addNextValue(choice, targetState, entry.second);
                transitionBuilder.addNextValue(choice, bottomState, storm::utility::one<BeliefMdpValueType>() - entry.second);
            } else {
                transitionBuilder.addNextValue(choice, targetState, storm::utility::one<BeliefMdpValueType>());
                actionRewards[choice] += entry.second;
            }
            if (optionalChoiceLabeling.has_value()) {
                if (!optionalChoiceLabeling.value().containsLabel(entry.first)) {
                    optionalChoiceLabeling.value().addLabel(entry.first);
                }
                optionalChoiceLabeling.value().addLabelToChoice(entry.first, choice);
            }
            ++choice;
        }
    }

    // Treat extra states
    transitionBuilder.newRowGroup(numChoices - numBottomTargetStates);
    if (optionalChoiceLabeling.has_value()) {
        if (!optionalChoiceLabeling.value().containsLabel("__loop__")) {
            optionalChoiceLabeling.value().addLabel("__loop__");
        }
        optionalChoiceLabeling.value().addLabelToChoice("__loop__", numChoices - numBottomTargetStates);
    }
    transitionBuilder.addNextValue(numChoices - numBottomTargetStates, targetState, storm::utility::one<BeliefMdpValueType>());

    if (reachabilityProbability || clippingUsed) {
        transitionBuilder.newRowGroup(numChoices - 1);
        transitionBuilder.addNextValue(numChoices - 1, bottomState, storm::utility::one<BeliefMdpValueType>());
        if (optionalChoiceLabeling.has_value()) {
            if (!optionalChoiceLabeling.value().containsLabel("__loop__")) {
                optionalChoiceLabeling.value().addLabel("__loop__");
            }
            optionalChoiceLabeling.value().addLabelToChoice("__loop__", numChoices - 1);
        }
    }

    storm::models::sparse::StateLabeling stateLabeling(numStates);
    stateLabeling.addLabel("target");
    stateLabeling.addLabelToState("target", targetState);
    stateLabeling.addLabel("init");
    stateLabeling.addLabelToState("init", explorationInformation.exploredBeliefs.at(explorationInformation.initialBeliefId));
    stateLabeling.addLabel("truncated");
    for (uint64_t state = numStates - numExtraStates; state < numStates - numBottomTargetStates; ++state) {
        stateLabeling.addLabelToState("truncated", state);
    }

    if (reachabilityProbability || clippingUsed) {
        stateLabeling.addLabel("bottom");
        stateLabeling.addLabelToState("bottom", bottomState);
    }
    storm::storage::sparse::ModelComponents<BeliefMdpValueType> components(transitionBuilder.build(), std::move(stateLabeling));

    if (!reachabilityProbability) {
        storm::models::sparse::StandardRewardModel<BeliefMdpValueType> rewardModel(std::nullopt, std::move(actionRewards));
        components.rewardModels.emplace(propertyInformation.rewardModelName.value(), std::move(rewardModel));
    }
    if (optionalChoiceLabeling.has_value()) {
        components.choiceLabeling = optionalChoiceLabeling.value();
    }

    // If requested, populate the stateToBeliefMap for the generic buildBeliefMdp variant
    std::unordered_map<uint64_t, BeliefId> stateToBeliefMap;
    // Explored beliefs
    for (auto const& [beliefId, stateIndex] : explorationInformation.exploredBeliefs) {
        stateToBeliefMap[stateIndex] = beliefId;
    }
    // Frontier beliefs: mapping was built as stateToFrontierBeliefMap
    for (auto const& [stateIndex, beliefId] : stateToFrontierBeliefMap) {
        stateToBeliefMap[stateIndex] = beliefId;
    }

    return std::make_pair(std::make_shared<storm::models::sparse::Mdp<BeliefMdpValueType>>(std::move(components)), std::move(stateToBeliefMap));
}

template std::pair<std::shared_ptr<storm::models::sparse::Mdp<double>>, std::unordered_map<uint64_t, BeliefId>> buildBeliefMdpWithImplicitCutoffs(
    ExplorationInformation<double, Belief<double>> const& explorationInformation, PropertyInformation const& propertyInformation,
    std::function<double(Belief<double> const&)> const& computeCutOffValue);

template std::pair<std::shared_ptr<storm::models::sparse::Mdp<storm::RationalNumber>>, std::unordered_map<uint64_t, BeliefId>>
buildBeliefMdpWithImplicitCutoffs(ExplorationInformation<storm::RationalNumber, Belief<double>> const& explorationInformation,
                                  PropertyInformation const& propertyInformation,
                                  std::function<__gmp_expr<__mpq_struct[1], __mpq_struct[1]>(Belief<double> const&)> const& computeCutOffValue);

template std::pair<std::shared_ptr<storm::models::sparse::Mdp<double>>, std::unordered_map<uint64_t, BeliefId>> buildBeliefMdpWithImplicitCutoffs(
    ExplorationInformation<double, Belief<storm::RationalNumber>> const& explorationInformation, PropertyInformation const& propertyInformation,
    std::function<double(Belief<__gmp_expr<__mpq_struct[1], __mpq_struct[1]>> const&)> const& computeCutOffValue);

template std::pair<std::shared_ptr<storm::models::sparse::Mdp<storm::RationalNumber>>, std::unordered_map<uint64_t, BeliefId>>
buildBeliefMdpWithImplicitCutoffs(
    ExplorationInformation<storm::RationalNumber, Belief<storm::RationalNumber>> const& explorationInformation, PropertyInformation const& propertyInformation,
    std::function<__gmp_expr<__mpq_struct[1], __mpq_struct[1]>(Belief<__gmp_expr<__mpq_struct[1], __mpq_struct[1]>> const&)> const& computeCutOffValue);

template std::pair<std::shared_ptr<models::sparse::Mdp<double>>, std::unordered_map<uint64_t, BeliefId>> buildBeliefMdp(
    ExplorationInformation<double, Belief<double>> const& explorationInformation, PropertyInformation const& propertyInformation,
    std::function<std::unordered_map<std::string, double>(Belief<double> const&)> const& computeCutOffValueMap);

template std::pair<std::shared_ptr<models::sparse::Mdp<__gmp_expr<__mpq_struct[1], __mpq_struct[1]>>>, std::unordered_map<uint64_t, BeliefId>> buildBeliefMdp(
    ExplorationInformation<storm::RationalNumber, Belief<double>> const& explorationInformation, PropertyInformation const& propertyInformation,
    std::function<std::unordered_map<std::string, __gmp_expr<__mpq_struct[1], __mpq_struct[1]>>(Belief<double> const&)> const& computeCutOffValueMap);

template std::pair<std::shared_ptr<models::sparse::Mdp<double>>, std::unordered_map<uint64_t, BeliefId>> buildBeliefMdp(
    ExplorationInformation<double, Belief<storm::RationalNumber>> const& explorationInformation, PropertyInformation const& propertyInformation,
    std::function<std::unordered_map<std::string, double>(Belief<__gmp_expr<__mpq_struct[1], __mpq_struct[1]>> const&)> const& computeCutOffValueMap);

template std::pair<std::shared_ptr<models::sparse::Mdp<storm::RationalNumber>>, std::unordered_map<uint64_t, BeliefId>> buildBeliefMdp(
    ExplorationInformation<storm::RationalNumber, Belief<storm::RationalNumber>> const& explorationInformation, PropertyInformation const& propertyInformation,
    std::function<std::unordered_map<std::string, storm::RationalNumber>(Belief<storm::RationalNumber> const&)> const& computeCutOffValueMap);

template std::pair<std::shared_ptr<storm::models::sparse::Mdp<double>>, std::unordered_map<uint64_t, BeliefId>> buildBeliefMdpWithImplicitCutoffs(
    ClippingExplorationInformation<double, Belief<double>> const& explorationInformation, PropertyInformation const& propertyInformation,
    std::function<double(Belief<double> const&)> const& computeCutOffValue);

template std::pair<std::shared_ptr<storm::models::sparse::Mdp<storm::RationalNumber>>, std::unordered_map<uint64_t, BeliefId>>
buildBeliefMdpWithImplicitCutoffs(ClippingExplorationInformation<storm::RationalNumber, Belief<double>> const& explorationInformation,
                                  PropertyInformation const& propertyInformation,
                                  std::function<storm::RationalNumber(Belief<double> const&)> const& computeCutOffValue);

template std::pair<std::shared_ptr<storm::models::sparse::Mdp<double>>, std::unordered_map<uint64_t, BeliefId>> buildBeliefMdpWithImplicitCutoffs(
    ClippingExplorationInformation<double, Belief<storm::RationalNumber>> const& explorationInformation, PropertyInformation const& propertyInformation,
    std::function<double(Belief<storm::RationalNumber> const&)> const& computeCutOffValue);

template std::pair<std::shared_ptr<storm::models::sparse::Mdp<storm::RationalNumber>>, std::unordered_map<uint64_t, BeliefId>>
buildBeliefMdpWithImplicitCutoffs(ClippingExplorationInformation<storm::RationalNumber, Belief<storm::RationalNumber>> const& explorationInformation,
                                  PropertyInformation const& propertyInformation,
                                  std::function<storm::RationalNumber(Belief<storm::RationalNumber> const&)> const& computeCutOffValue);

template std::pair<std::shared_ptr<models::sparse::Mdp<double>>, std::unordered_map<uint64_t, BeliefId>> buildBeliefMdp(
    ClippingExplorationInformation<double, Belief<double>> const& explorationInformation, PropertyInformation const& propertyInformation,
    std::function<std::unordered_map<std::string, double>(Belief<double> const&)> const& computeCutOffValueMap);

template std::pair<std::shared_ptr<models::sparse::Mdp<storm::RationalNumber>>, std::unordered_map<uint64_t, BeliefId>> buildBeliefMdp(
    ClippingExplorationInformation<storm::RationalNumber, Belief<double>> const& explorationInformation, PropertyInformation const& propertyInformation,
    std::function<std::unordered_map<std::string, storm::RationalNumber>(Belief<double> const&)> const& computeCutOffValueMap);

template std::pair<std::shared_ptr<models::sparse::Mdp<double>>, std::unordered_map<uint64_t, BeliefId>> buildBeliefMdp(
    ClippingExplorationInformation<double, Belief<storm::RationalNumber>> const& explorationInformation, PropertyInformation const& propertyInformation,
    std::function<std::unordered_map<std::string, double>(Belief<storm::RationalNumber> const&)> const& computeCutOffValueMap);

template std::pair<std::shared_ptr<models::sparse::Mdp<storm::RationalNumber>>, std::unordered_map<uint64_t, BeliefId>> buildBeliefMdp(
    ClippingExplorationInformation<storm::RationalNumber, Belief<storm::RationalNumber>> const& explorationInformation,
    PropertyInformation const& propertyInformation,
    std::function<std::unordered_map<std::string, storm::RationalNumber>(Belief<storm::RationalNumber> const&)> const& computeCutOffValueMap);

}  // namespace storm::pomdp::beliefs
