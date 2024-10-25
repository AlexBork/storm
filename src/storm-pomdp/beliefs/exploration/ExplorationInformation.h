#pragma once
#include "storm-pomdp/beliefs/exploration/BeliefExplorationMatrix.h"
#include "storm-pomdp/beliefs/exploration/ExplorationQueue.h"
#include "storm-pomdp/beliefs/storage/BeliefCollector.h"
#include "storm-pomdp/beliefs/utility/types.h"

namespace storm::pomdp::beliefs {
struct ActionLabelingInformation {
    std::unordered_map<BeliefId, std::unordered_map<uint64_t, std::set<std::string>>> beliefActionToChoiceLabels;

    void addChoiceLabel(BeliefId const& beliefId, uint64_t const& localActionIndex, std::string const& label) {
        beliefActionToChoiceLabels[beliefId][localActionIndex].insert(label);
    }

    [[nodiscard]] std::set<std::string> getChoiceLabels(BeliefId const& beliefId, uint64_t const& localActionIndex) {
        return beliefActionToChoiceLabels[beliefId][localActionIndex];
    }
};

template<typename BeliefMdpValueType, typename BeliefType, typename... ExtraTransitionData>
struct ExplorationInformation {
    BeliefExplorationMatrix<BeliefMdpValueType, ExtraTransitionData...> matrix;
    std::vector<BeliefMdpValueType> actionRewards;
    storm::pomdp::beliefs::BeliefCollector<BeliefType> discoveredBeliefs;
    std::unordered_map<BeliefId, BeliefStateType> exploredBeliefs;
    std::unordered_map<BeliefId, BeliefMdpValueType> terminalBeliefValues;
    BeliefId initialBeliefId;
    ExplorationQueue queue;
    uint64_t nrObservationsInPomdp;
    std::optional<ActionLabelingInformation> optionalActionLabelingInfo;

    [[nodiscard]] std::unordered_set<BeliefId> getFrontierBeliefs() const {
        std::unordered_set<BeliefId> resFrontierBeliefs;
        for (uint64_t id = 0; id < discoveredBeliefs.getNumberOfBeliefIds(); id++) {
            if (exploredBeliefs.count(id) == 0 && terminalBeliefValues.count(id) == 0) {
                resFrontierBeliefs.insert(id);
            }
        }
        return resFrontierBeliefs;
    }
};

template<typename BeliefMdpValueType, typename BeliefType>
using StandardExplorationInformation = ExplorationInformation<BeliefMdpValueType, BeliefType>;

template<typename BeliefMdpValueType, typename BeliefType>
using RewardAwareExplorationInformation = ExplorationInformation<BeliefMdpValueType, BeliefType, std::vector<BeliefMdpValueType>>;
}  // namespace storm::pomdp::beliefs