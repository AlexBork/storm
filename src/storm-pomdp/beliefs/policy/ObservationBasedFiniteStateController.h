#pragma once

namespace storm {
namespace storage {
template<typename ValueType, typename StateType>
class Distribution;
}
namespace pomdp::policy {
struct FSCOutputUpdate {
    virtual ~FSCOutputUpdate() = default;
    uint64_t nextMemoryNode;
    virtual bool randomisedActionOutput() const {
        return false;
    }
};

template<typename ValueType>
struct RandomisedActionUpdate final : FSCOutputUpdate {
    storm::storage::Distribution<ValueType, uint64_t> actionDistribution;
    bool randomisedActionOutput() const override {
        return true;
    }
};

struct DeterministicActionUpdate final : FSCOutputUpdate {
    uint64_t action;
};

// TODO: allow randomised memory structures?

template<typename ValueType>
class ObservationBasedFiniteStateController {
    using ActionOutputUpdate = std::unordered_map<uint64_t, FSCOutputUpdate>;

   public:
    ObservationBasedFiniteStateController() = default;
    ObservationBasedFiniteStateController(std::unordered_map<uint64_t, std::string> const& idToObservationNameMap,
                                          std::unordered_map<uint64_t, std::string> const& idToActionNameMap);

    void addDeterministicActionTransition(uint64_t originId, uint64_t observationId, uint64_t actionId, uint64_t targetId);
    void addRandomisedActionTransition(uint64_t originId, uint64_t observationId, storm::storage::Distribution<ValueType, uint64_t> actionDistribution,
                                       uint64_t targetId);

    std::pair<uint64_t, uint64_t> getActionAndSuccessorForObservationInNode(uint64_t const originId, uint64_t const observationId) const;
    uint64_t getActionForObservationInNode(uint64_t const originId, uint64_t const observationId) const;
    uint64_t getSuccessorForObservationInNode(uint64_t const originId, uint64_t const observationId) const;

    std::pair<storm::storage::Distribution<ValueType, uint64_t>, uint64_t> getActionDistributionAndSuccessorForObservationInNode(
        uint64_t const originId, uint64_t const observationId) const;
    storm::storage::Distribution<ValueType, uint64_t> getActionDistributionForObservationInNode(uint64_t const originId, uint64_t const observationId) const;

    bool hasOutputForObservationInNode(uint64_t const originId, uint64_t const observationId) const;

    void setIdToObservationNameMap(std::unordered_map<uint64_t, std::string> const& idToObservationNameMap);
    void setIdToOActionNameMap(std::unordered_map<uint64_t, std::string> const& idToActionNameMap);

    std::string getObservationName(uint64_t const observationId) const;
    std::string getActionName(uint64_t const actionId) const;

    uint64_t getIdForObservation(std::string const observationName) const;
    uint64_t getIdForAction(std::string const actionName) const;

    bool isDeterministic() const;

    bool outputIsRandomised(uint64_t const originId, uint64_t const observationId) const;

   private:
    void addActionOutputUpdate(uint64_t const originId, uint64_t const observationId, FSCOutputUpdate const& update);
    FSCOutputUpdate const& getActionOutputUpdate(uint64_t const originId, uint64_t const observationId) const;

    std::unordered_map<uint64_t, ActionOutputUpdate> transitions;
    std::unordered_map<uint64_t, std::string> idToObservationName;
    std::unordered_map<uint64_t, std::string> idToActionName;
    bool isDeterministicPolicy = true;
};

}  // namespace pomdp::policy
}  // namespace storm
