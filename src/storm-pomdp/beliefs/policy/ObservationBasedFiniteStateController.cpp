#include "ObservationBasedFiniteStateController.h"

#include <utility>

#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/storage/Distribution.h"
#include "storm/utility/constants.h"
#include "storm/utility/macros.h"

namespace storm::pomdp::policy {

template<typename ValueType>
ObservationBasedFiniteStateController<ValueType>::ObservationBasedFiniteStateController(uint64_t const initialNode) : initialNodeId(initialNode) {
    // Intentionally left empty
}

template<typename ValueType>
ObservationBasedFiniteStateController<ValueType>::ObservationBasedFiniteStateController(
    uint64_t const initialNode, std::optional<std::unordered_map<uint64_t, std::string>> idToObservationNameMap,
    std::optional<std::unordered_map<uint64_t, std::unordered_map<uint64_t, std::string>>> idToActionNameMap)
    : initialNodeId(initialNode), idToObservationName(std::move(idToObservationNameMap)), idToActionName(std::move(idToActionNameMap)) {
    // Intentionally left empty
}

template<typename ValueType>
void ObservationBasedFiniteStateController<ValueType>::addDeterministicActionTransition(uint64_t const originId, uint64_t const observationId,
                                                                                        uint64_t const actionId, uint64_t const targetId) {
    auto update = std::make_unique<DeterministicActionUpdate>();
    update->action = actionId;
    update->nextMemoryNode = targetId;
    addActionOutputUpdate(originId, observationId, std::move(update));
}

template<typename ValueType>
void ObservationBasedFiniteStateController<ValueType>::addRandomisedActionTransition(uint64_t const originId, uint64_t const observationId,
                                                                                     storm::storage::Distribution<ValueType, uint64_t> actionDistribution,
                                                                                     uint64_t targetId) {
    isDeterministicPolicy = false;
    auto update = std::make_unique<RandomisedActionUpdate<ValueType>>();
    update->actionDistribution = std::move(actionDistribution);
    update->nextMemoryNode = targetId;
    addActionOutputUpdate(originId, observationId, std::move(update));
}

template<typename ValueType>
std::pair<uint64_t, uint64_t> ObservationBasedFiniteStateController<ValueType>::getActionAndSuccessorForObservationInNode(uint64_t const originId,
                                                                                                                          uint64_t const observationId) const {
    FSCOutputUpdate const& outputUpdate = getActionOutputUpdate(originId, observationId);
    STORM_LOG_ASSERT(!outputUpdate.randomisedActionOutput(),
                     "Output for observation (ID " << observationId << ") in node (ID " << originId << ") is randomised.");
    auto const& detActionUpdate = dynamic_cast<DeterministicActionUpdate const&>(outputUpdate);
    return {detActionUpdate.action, detActionUpdate.nextMemoryNode};
}

template<typename ValueType>
uint64_t ObservationBasedFiniteStateController<ValueType>::getActionForObservationInNode(uint64_t const originId, uint64_t const observationId) const {
    return getActionAndSuccessorForObservationInNode(originId, observationId).first;
}

template<typename ValueType>
uint64_t ObservationBasedFiniteStateController<ValueType>::getSuccessorForObservationInNode(uint64_t const originId, uint64_t const observationId) const {
    return getActionOutputUpdate(originId, observationId).nextMemoryNode;
}

template<typename ValueType>
std::pair<storm::storage::Distribution<ValueType, uint64_t>, uint64_t>
ObservationBasedFiniteStateController<ValueType>::getActionDistributionAndSuccessorForObservationInNode(uint64_t const originId,
                                                                                                        uint64_t const observationId) const {
    if (FSCOutputUpdate const& outputUpdate = getActionOutputUpdate(originId, observationId); outputUpdate.randomisedActionOutput()) {
        auto const& randActionUpdate = dynamic_cast<RandomisedActionUpdate<ValueType> const&>(outputUpdate);
        return {randActionUpdate.actionDistribution, randActionUpdate.nextMemoryNode};
    } else {
        auto const& detActionUpdate = dynamic_cast<DeterministicActionUpdate const&>(outputUpdate);
        storm::storage::Distribution<ValueType, uint64_t> diracForAction;
        diracForAction.addProbability(detActionUpdate.action, storm::utility::one<ValueType>());
        return {diracForAction, detActionUpdate.nextMemoryNode};
    }
}

template<typename ValueType>
storm::storage::Distribution<ValueType, uint64_t> ObservationBasedFiniteStateController<ValueType>::getActionDistributionForObservationInNode(
    uint64_t const originId, uint64_t const observationId) const {
    return getActionDistributionAndSuccessorForObservationInNode(originId, observationId).first;
}

template<typename ValueType>
void ObservationBasedFiniteStateController<ValueType>::setIdToObservationNameMap(std::unordered_map<uint64_t, std::string> const& idToObservationNameMap) {
    idToObservationName = idToObservationNameMap;
}

template<typename ValueType>
void ObservationBasedFiniteStateController<ValueType>::setIdToActionNameMap(
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, std::string>> const& idToActionNameMap) {
    idToActionName = idToActionNameMap;
}

template<typename ValueType>
std::string ObservationBasedFiniteStateController<ValueType>::getObservationName(uint64_t const observationId) const {
    STORM_LOG_ASSERT(idToObservationName, "idToObservationName is not set.");
    STORM_LOG_ASSERT(!idToObservationName->empty(), "idToObservationName is empty.");
    STORM_LOG_ASSERT(idToObservationName->contains(observationId), "Observation ID " << observationId << " not found.");
    return idToObservationName->at(observationId);
}

template<typename ValueType>
std::string ObservationBasedFiniteStateController<ValueType>::getActionName(uint64_t const observationId, uint64_t const actionId) const {
    STORM_LOG_ASSERT(idToActionName, "idToActionName is not set.");
    STORM_LOG_ASSERT(!idToActionName->empty(), "idToActionName is empty.");
    STORM_LOG_ASSERT(idToActionName->contains(observationId), "Observation ID " << observationId << " not found.");
    return idToActionName->at(observationId).at(actionId);
}

template<typename ValueType>
std::optional<uint64_t> ObservationBasedFiniteStateController<ValueType>::getObservationIdByName(std::string const& observationName) const {
    if (!idToObservationName) {
        return std::nullopt;
    }
    for (auto const& [observationId, name] : *idToObservationName) {
        if (name == observationName) {
            return observationId;
        }
    }
    return std::nullopt;
}

template<typename ValueType>
std::optional<uint64_t> ObservationBasedFiniteStateController<ValueType>::getActionIdByName(uint64_t const observationId, std::string const& actionName) const {
    if (!idToActionName || !idToActionName->contains(observationId)) {
        return std::nullopt;
    }
    for (auto const& [actionId, name] : idToActionName->at(observationId)) {
        if (name == actionName) {
            return actionId;
        }
    }
    return std::nullopt;
}

template<typename ValueType>
void ObservationBasedFiniteStateController<ValueType>::addActionOutputUpdate(uint64_t const originId, uint64_t const observationId,
                                                                             std::unique_ptr<FSCOutputUpdate> update) {
    transitions[originId][observationId] = std::move(update);
}

template<typename ValueType>
bool ObservationBasedFiniteStateController<ValueType>::hasOutputForObservationInNode(uint64_t const originId, uint64_t const observationId) const {
    return transitions.contains(originId) && transitions.at(originId).contains(observationId);
}

template<typename ValueType>
FSCOutputUpdate const& ObservationBasedFiniteStateController<ValueType>::getActionOutputUpdate(uint64_t const originId, uint64_t const observationId) const {
    STORM_LOG_ASSERT(transitions.contains(originId), "Origin node (ID " << originId << " not found.");
    STORM_LOG_ASSERT(hasOutputForObservationInNode(originId, observationId),
                     "No output for observation (ID " << observationId << ") in node (ID " << originId << ").");
    return *transitions.at(originId).at(observationId);
}

template<typename ValueType>
bool ObservationBasedFiniteStateController<ValueType>::isDeterministic() const {
    return isDeterministicPolicy;
}

template<typename ValueType>
bool ObservationBasedFiniteStateController<ValueType>::outputIsRandomised(uint64_t const originId, uint64_t const observationId) const {
    return transitions.at(originId).at(observationId)->randomisedActionOutput();
}

template<typename ValueType>
uint64_t ObservationBasedFiniteStateController<ValueType>::getInitialNodeId() const {
    return initialNodeId;
}

template<typename ValueType>
bool ObservationBasedFiniteStateController<ValueType>::hasIdToObservationNameMap() const {
    return static_cast<bool>(idToObservationName);
}

template<typename ValueType>
bool ObservationBasedFiniteStateController<ValueType>::hasIdToActionNameMap() const {
    return static_cast<bool>(idToActionName);
}

template<typename ValueType>
std::string ObservationBasedFiniteStateController<ValueType>::toString() const {
    std::string result = "Observation-Based Finite State Controller:\n";
    result += "Initial Node ID: " + std::to_string(initialNodeId) + "\n";
    for (auto const& [originId, actionOutputMap] : transitions) {
        result += "Node " + std::to_string(originId) + ":\n";
        for (auto const& [observationId, outputUpdatePtr] : actionOutputMap) {
            if (idToObservationName) {
                result += "(" + std::to_string(observationId) + ")" + idToObservationName->at(observationId) + " -> ";
            } else {
                result += std::to_string(observationId) + " -> ";
            }
            if (outputUpdatePtr->randomisedActionOutput()) {
                result += "({ ";
                for (auto randomisedOutputUpdatePtr = std::static_pointer_cast<RandomisedActionUpdate<ValueType>>(outputUpdatePtr);
                     auto const& [actionId, prob] : randomisedOutputUpdatePtr->actionDistribution) {
                    if (idToActionName) {
                        result += "(" + std::to_string(actionId) + ") " + idToActionName->at(observationId).at(actionId) + " : " +
                                  storm::utility::to_string(prob) + ", ";
                    } else {
                        result += "(" + std::to_string(actionId) + ") : " + storm::utility::to_string(prob) + ", ";
                    }
                }
                result += "}, " + std::to_string(outputUpdatePtr->nextMemoryNode) + ")\n";
            } else {
                auto deterministicOutputUpdatePtr = std::static_pointer_cast<DeterministicActionUpdate>(outputUpdatePtr);
                result += "(";
                if (idToActionName) {
                    result += "(" + std::to_string(deterministicOutputUpdatePtr->action) + ") " +
                              idToActionName->at(observationId).at(deterministicOutputUpdatePtr->action) + ", " +
                              std::to_string(outputUpdatePtr->nextMemoryNode) + ")\n";
                } else {
                    result += "(" + std::to_string(deterministicOutputUpdatePtr->action) + "), " + std::to_string(outputUpdatePtr->nextMemoryNode) + ")\n";
                }
            }
        }
    }
    return result;
}

template<typename ValueType>
void ObservationBasedFiniteStateController<ValueType>::writeDotToStream(std::ostream& outStream) const {
    outStream << "digraph fsc {\n";
    outStream << "\trankdir = LR;\n";
    outStream << "\tnode [shape = circle];\n";
    outStream << "\tinit [shape = point];\n";
    outStream << "\tinit -> " << initialNodeId << ";\n";

    std::unordered_set<uint64_t> nodes;
    nodes.insert(initialNodeId);
    for (auto const& [originId, actionOutputMap] : transitions) {
        nodes.insert(originId);
        for (auto const& [observationId, outputUpdatePtr] : actionOutputMap) {
            nodes.insert(outputUpdatePtr->nextMemoryNode);
            outStream << "\t" << originId << " -> " << outputUpdatePtr->nextMemoryNode << " [label=\"";
            if (idToObservationName && idToObservationName->contains(observationId)) {
                outStream << getObservationName(observationId);
            } else {
                outStream << observationId;
            }
            outStream << " / ";
            if (outputUpdatePtr->randomisedActionOutput()) {
                auto const& randActionUpdate = dynamic_cast<RandomisedActionUpdate<ValueType> const&>(*outputUpdatePtr);
                outStream << "{";
                bool first = true;
                for (auto const& [actionId, prob] : randActionUpdate.actionDistribution) {
                    if (!first) {
                        outStream << ", ";
                    }
                    first = false;
                    if (idToActionName && idToActionName->contains(observationId) && idToActionName->at(observationId).contains(actionId)) {
                        outStream << idToActionName->at(observationId).at(actionId);
                    } else {
                        outStream << actionId;
                    }
                    outStream << ":" << storm::utility::to_string(prob);
                }
                outStream << "}";
            } else {
                auto const& detActionUpdate = dynamic_cast<DeterministicActionUpdate const&>(*outputUpdatePtr);
                if (idToActionName && idToActionName->contains(observationId) && idToActionName->at(observationId).contains(detActionUpdate.action)) {
                    outStream << idToActionName->at(observationId).at(detActionUpdate.action);
                } else {
                    outStream << detActionUpdate.action;
                }
            }
            outStream << "\"];\n";
        }
    }
    for (auto nodeId : nodes) {
        if (nodeId == initialNodeId) {
            outStream << "\t" << nodeId << " [shape = doublecircle];\n";
        }
    }
    outStream << "}\n";
}

template class ObservationBasedFiniteStateController<double>;
template class ObservationBasedFiniteStateController<storm::RationalNumber>;

}  // namespace storm::pomdp::policy
