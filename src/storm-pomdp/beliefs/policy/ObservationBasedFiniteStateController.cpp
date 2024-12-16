#include "ObservationBasedFiniteStateController.h"

#include <storm/utility/constants.h>
#include <storm/utility/macros.h>

namespace storm::pomdp::policy {

template<typename ValueType>
ObservationBasedFiniteStateController<ValueType>::ObservationBasedFiniteStateController(std::unordered_map<uint64_t, std::string> const& idToObservationNameMap,
                                                                                        std::unordered_map<uint64_t, std::string> const& idToActionNameMap)
    : idToObservationName(idToObservationNameMap), idToActionName(idToActionNameMap) {
    // Intentionally left empty
}

template<typename ValueType>
void ObservationBasedFiniteStateController<ValueType>::addDeterministicActionTransition(uint64_t originId, uint64_t observationId, uint64_t actionId,
                                                                                        uint64_t targetId) {
    DeterministicActionUpdate update;
    update.action = actionId;
    update.nextMemoryNode = targetId;
    addActionOutputUpdate(originId, observationId, update);
}

template<typename ValueType>
void ObservationBasedFiniteStateController<ValueType>::addRandomisedActionTransition(uint64_t originId, uint64_t observationId,
                                                                                     storm::storage::Distribution<ValueType, uint64_t> actionDistribution,
                                                                                     uint64_t targetId) {
    isDeterministicPolicy = false;
    RandomisedActionUpdate<ValueType> update(actionDistribution, targetId);
    addActionOutputUpdate(originId, observationId, update);
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
    return getActionAndSuccessorForObservationInNode(originId, observationId).first;
}

template<typename ValueType>
void ObservationBasedFiniteStateController<ValueType>::setIdToObservationNameMap(std::unordered_map<uint64_t, std::string> const& idToObservationNameMap) {
    idToObservationName = idToObservationNameMap;
}

template<typename ValueType>
void ObservationBasedFiniteStateController<ValueType>::setIdToOActionNameMap(std::unordered_map<uint64_t, std::string> const& idToActionNameMap) {
    idToObservationName = idToActionNameMap;
}

template<typename ValueType>
std::string ObservationBasedFiniteStateController<ValueType>::getObservationName(uint64_t const observationId) const {
    STORM_LOG_ASSERT(!idToObservationName.empty(), "idToObservationName is empty.");
    STORM_LOG_ASSERT(idToObservationName.contains(observationId), "Observation ID " << observationId << " not found.");
    return idToObservationName.at(observationId);
}

template<typename ValueType>
std::string ObservationBasedFiniteStateController<ValueType>::getActionName(uint64_t const actionId) const {
    STORM_LOG_ASSERT(!idToActionName.empty(), "idToActionName is empty.");
    STORM_LOG_ASSERT(idToActionName.contains(actionId), "Action ID " << actionId << " not found.");
    return idToActionName.at(actionId);
}

template<typename ValueType>
uint64_t ObservationBasedFiniteStateController<ValueType>::getIdForObservation(std::string const observationName) const {
    STORM_LOG_ASSERT(!idToObservationName.empty(), "idToObservationName is empty.");
    return 0;
}

template<typename ValueType>
uint64_t ObservationBasedFiniteStateController<ValueType>::getIdForAction(std::string const actionName) const {
    return 0;
}

template<typename ValueType>
void ObservationBasedFiniteStateController<ValueType>::addActionOutputUpdate(uint64_t const originId, uint64_t const observationId,
                                                                             FSCOutputUpdate const& update) {
    transitions[originId][observationId] = update;
}

template<typename ValueType>
bool ObservationBasedFiniteStateController<ValueType>::hasOutputForObservationInNode(uint64_t const originId, uint64_t const observationId) const {
    return transitions.at(originId).contains(observationId);
}

template<typename ValueType>
FSCOutputUpdate const& ObservationBasedFiniteStateController<ValueType>::getActionOutputUpdate(uint64_t const originId, uint64_t const observationId) const {
    STORM_LOG_ASSERT(transitions.contains(originId), "Origin node (ID " << originId << " not found.");
    STORM_LOG_ASSERT(hasOutputForObservationInNode(originId, observationId),
                     "No output for observation (ID " << observationId << ") in node (ID " << originId << ").");
    return transitions.at(originId).at(observationId);
}

template<typename ValueType>
bool ObservationBasedFiniteStateController<ValueType>::isDeterministic() const {
    return isDeterministicPolicy;
}

template<typename ValueType>
bool ObservationBasedFiniteStateController<ValueType>::outputIsRandomised(uint64_t const originId, uint64_t const observationId) const {
    return transitions.at(originId).at(observationId).randomisedActionOutput();
}

}  // namespace storm::pomdp::policy
