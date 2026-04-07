#include "storm-pomdp/beliefs/policy/ObservationBasedFiniteStateController.h"
#include "storm/api/export.h"
#include "storm/models/sparse/Dtmc.h"

namespace storm {
namespace pomdp::policy {
template<typename PomdpValueType, typename FscValueType>
storm::models::sparse::Dtmc<PomdpValueType> applyObservationBasedFSCToPomdp(
    storm::models::sparse::Pomdp<PomdpValueType> const& pomdp, storm::pomdp::policy::ObservationBasedFiniteStateController<FscValueType> const& fsc,
    bool treatUnspecifiedChoiceAsDontCare = false) {
    std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> idToPomdpStateAndFscNode;
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, uint64_t>> pomdpStateAndFscNodeToNewStateId;
    // TODO change for vector?
    std::map<uint64_t, std::unordered_map<uint64_t, PomdpValueType>> transitionProbabilities;
    std::unordered_map<std::string, std::vector<PomdpValueType>> stateRewards;

    uint64_t initialStateId = 0ul;
    uint64_t nextStateId = 1ul;
    uint64_t nrEntries = 0ul;
    uint64_t initialPomdpState = pomdp.getInitialStates().getNextSetIndex(0ul);
    idToPomdpStateAndFscNode[initialStateId] = {initialPomdpState, fsc.getInitialNodeId()};
    pomdpStateAndFscNodeToNewStateId[initialPomdpState][fsc.getInitialNodeId()] = initialStateId;
    STORM_LOG_DEBUG("Add initial state " << initialStateId << " (P: " << initialPomdpState << ", F: " << fsc.getInitialNodeId() << ")");

    std::optional<std::unordered_map<uint64_t, uint64_t>> pomdpObservationIdToFscObservationId = std::nullopt;
    std::optional<std::unordered_map<uint64_t, std::unordered_map<uint64_t, uint64_t>>> pomdpActionIdToFscActionIdMap = std::nullopt;
    // If the POMDP and FSC have naming information, use it to create mappings between actions and observations
    // Otherwise, we assume that IDs correspond directly!
    if (pomdp.hasChoiceLabeling() && pomdp.hasObservationValuations() && fsc.hasIdToActionNameMap() && fsc.hasIdToObservationNameMap()) {
        // TODO
        // Create mapping between POMDP observations and FSC observations
        // Create mapping between POMDP actions and FSC actions
    } else {
        STORM_LOG_WARN(
            "The POMDP or the FSC is missing naming information for actions or observations. Assuming that IDs correspond directly (this might lead to "
            "unexpected results if they don't).");
    }

    std::queue<uint64_t> statesToExplore;
    statesToExplore.push(initialStateId);
    while (!statesToExplore.empty()) {
        uint64_t currentStateId = statesToExplore.front();
        auto [currentPomdpState, currentFscNode] = idToPomdpStateAndFscNode[currentStateId];
        statesToExplore.pop();
        STORM_LOG_DEBUG("Explore state " << currentStateId << " (P: " << currentPomdpState << ", F: " << currentFscNode << ")");
        storm::storage::Distribution<FscValueType, uint64_t> actionDistribution;
        uint64_t nextFSCNode;
        if (fsc.hasOutputForObservationInNode(currentFscNode, pomdp.getObservation(currentPomdpState))) {
            std::tie(actionDistribution, nextFSCNode) =
                fsc.getActionDistributionAndSuccessorForObservationInNode(currentFscNode, pomdp.getObservation(currentPomdpState));
        } else {
            STORM_LOG_THROW(treatUnspecifiedChoiceAsDontCare, storm::exceptions::UnexpectedException,
                            "Expected FSC to have an output for observation " << pomdp.getObservation(currentPomdpState) << " in FSC node " << currentFscNode
                                                                              << ", but it does not.");
            // If requested, treat unspecified choice as don't care: pick the first enabled action deterministically and stay in the same FSC node
            nextFSCNode = currentFscNode;
            actionDistribution = storm::storage::Distribution<FscValueType, uint64_t>();
            actionDistribution.addProbability(0ul, storm::utility::one<FscValueType>());
        }
        for (auto const& actionEntry : actionDistribution) {
            for (auto const& [rewardModelName, rewardModel] : pomdp.getRewardModels()) {
                stateRewards[rewardModelName].push_back(storm::utility::zero<PomdpValueType>());
                if (rewardModel.hasStateRewards()) {
                    stateRewards[rewardModelName][currentStateId] += rewardModel.getStateRewardVector()[currentPomdpState];
                }
                // We take the expected reward over the action distribution, so not all properties are preserved.
                if (rewardModel.hasStateActionRewards()) {
                    uint64_t globalActionIndex = pomdp.getTransitionMatrix().getRowGroupIndices()[currentPomdpState] + actionEntry.first;
                    stateRewards[rewardModelName][currentStateId] += rewardModel.getStateActionRewardVector()[globalActionIndex] * actionEntry.second;
                }
                if (rewardModel.hasTransitionRewards()) {
                    STORM_LOG_WARN("Transition rewards are not supported when applying an FSC to a POMDP. They will be ignored.");
                }
            }
            for (auto const& transitionEntry : pomdp.getTransitionMatrix().getRow(currentPomdpState, actionEntry.first)) {
                uint64_t successorPomdpState = transitionEntry.getColumn();
                uint64_t successorNewStateId;
                if (pomdpStateAndFscNodeToNewStateId.contains(successorPomdpState) &&
                    pomdpStateAndFscNodeToNewStateId[successorPomdpState].contains(nextFSCNode)) {
                    successorNewStateId = pomdpStateAndFscNodeToNewStateId[successorPomdpState][nextFSCNode];
                } else {
                    successorNewStateId = nextStateId;
                    idToPomdpStateAndFscNode[successorNewStateId] = {
                        successorPomdpState,
                        nextFSCNode,
                    };
                    pomdpStateAndFscNodeToNewStateId[successorPomdpState][nextFSCNode] = successorNewStateId;
                    STORM_LOG_DEBUG("Add state " << successorNewStateId << " (P: " << successorPomdpState << ", F: " << nextFSCNode << ")");
                    ++nextStateId;
                    statesToExplore.push(successorNewStateId);
                }
                if (transitionProbabilities.contains(currentStateId) && transitionProbabilities[currentStateId].contains(successorNewStateId)) {
                    // Add transition from currentStateId to successorNewStateId with probability from POMDP
                    transitionProbabilities[currentStateId][successorNewStateId] += transitionEntry.getValue() * actionEntry.second;
                } else {
                    transitionProbabilities[currentStateId][successorNewStateId] = transitionEntry.getValue() * actionEntry.second;
                    ++nrEntries;
                }
            }
        }
    }

    // Build transition matrix from collected transitions
    storm::storage::SparseMatrixBuilder<PomdpValueType> matrixBuilder(nextStateId, nextStateId, nrEntries);
    for (auto const& fromEntry : transitionProbabilities) {
        uint64_t fromStateId = fromEntry.first;
        for (auto const& toEntry : fromEntry.second) {
            uint64_t toStateId = toEntry.first;
            PomdpValueType prob = toEntry.second;
            matrixBuilder.addNextValue(fromStateId, toStateId, prob);
        }
    }
    auto transitionMatrix = matrixBuilder.build();
    storm::storage::sparse::ModelComponents<PomdpValueType> modelComponents;
    modelComponents.transitionMatrix = transitionMatrix;

    // Add state labeling
    storm::models::sparse::StateLabeling stateLabeling(nextStateId);
    stateLabeling.addLabel("init");
    stateLabeling.addLabelToState("init", initialStateId);
    for (uint64_t pomdpState = 0; pomdpState < nextStateId; ++pomdpState) {
        if (pomdpStateAndFscNodeToNewStateId.contains(pomdpState)) {
            for (const auto& newStateId : pomdpStateAndFscNodeToNewStateId[pomdpState] | std::views::values) {
                for (auto const& label : pomdp.getStateLabeling().getLabelsOfState(pomdpState)) {
                    if (label != "init") {
                        if (!stateLabeling.containsLabel(label)) {
                            stateLabeling.addLabel(label);
                        }
                        stateLabeling.addLabelToState(label, newStateId);
                    }
                }
            }
        }
    }
    modelComponents.stateLabeling = stateLabeling;

    // build reward models
    for (auto const& [rewardModelName, rewardValues] : stateRewards) {
        storm::models::sparse::StandardRewardModel<PomdpValueType> rewardModel(rewardValues);
        modelComponents.rewardModels[rewardModelName] = rewardModel;
    }

    return storm::models::sparse::Dtmc<PomdpValueType>(modelComponents);
}
}  // namespace pomdp::policy
}  // namespace storm
