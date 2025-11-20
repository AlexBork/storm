#pragma once
#include "storm/utility/constants.h"

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
// Extend here if more types of updates are needed (e.g. stochastic transitions for memory nodes)

template<typename ValueType>
class ObservationBasedFiniteStateController {
    using ActionOutputUpdate = std::unordered_map<uint64_t, std::shared_ptr<FSCOutputUpdate>>;

   public:
    explicit ObservationBasedFiniteStateController(uint64_t initialNode);
    ObservationBasedFiniteStateController(uint64_t initialNode, std::optional<std::unordered_map<uint64_t, std::string>> idToObservationNameMap,
                                          std::optional<std::unordered_map<uint64_t, std::unordered_map<uint64_t, std::string>>> idToActionNameMap);

    std::pair<uint64_t, uint64_t> getActionAndSuccessorForObservationInNode(uint64_t originId, uint64_t observationId) const;
    uint64_t getActionForObservationInNode(uint64_t originId, uint64_t observationId) const;
    uint64_t getSuccessorForObservationInNode(uint64_t originId, uint64_t observationId) const;

    std::pair<storm::storage::Distribution<ValueType, uint64_t>, uint64_t> getActionDistributionAndSuccessorForObservationInNode(uint64_t originId,
                                                                                                                                 uint64_t observationId) const;
    storm::storage::Distribution<ValueType, uint64_t> getActionDistributionForObservationInNode(uint64_t originId, uint64_t observationId) const;

    bool hasOutputForObservationInNode(uint64_t originId, uint64_t observationId) const;

    void addDeterministicActionTransition(uint64_t originId, uint64_t observationId, uint64_t actionId, uint64_t targetId);
    void addRandomisedActionTransition(uint64_t originId, uint64_t observationId, storm::storage::Distribution<ValueType, uint64_t> actionDistribution,
                                       uint64_t targetId);

    void setIdToObservationNameMap(std::unordered_map<uint64_t, std::string> const& idToObservationNameMap);
    void setIdToActionNameMap(std::unordered_map<uint64_t, std::unordered_map<uint64_t, std::string>> const& idToActionNameMap);

    std::string getObservationName(uint64_t observationId) const;
    std::string getActionName(uint64_t observationId, uint64_t actionId) const;

    bool isDeterministic() const;

    bool outputIsRandomised(uint64_t originId, uint64_t observationId) const;

    std::string toString() const;

   private:
    void addActionOutputUpdate(uint64_t originId, uint64_t observationId, std::unique_ptr<FSCOutputUpdate> update);
    FSCOutputUpdate const& getActionOutputUpdate(uint64_t originId, uint64_t observationId) const;

    uint64_t initialNodeId;
    std::unordered_map<uint64_t, ActionOutputUpdate> transitions;
    std::optional<std::unordered_map<uint64_t, std::string>> idToObservationName = std::nullopt;
    std::optional<std::unordered_map<uint64_t, std::unordered_map<uint64_t, std::string>>> idToActionName = std::nullopt;
    bool isDeterministicPolicy = true;
};

}  // namespace pomdp::policy
}  // namespace storm
