#pragma once
#include "storm/utility/constants.h"

namespace storm {
namespace storage {
template<typename ValueType, typename StateType>
class Distribution;
}
namespace pomdp::policy {

/**
 * Describes the successor memory node selected by an FSC output update.
 *
 * Derived update types additionally describe either a deterministic action or a distribution over actions.
 */
struct FSCOutputUpdate {
    virtual ~FSCOutputUpdate() = default;

    /** The memory node entered after applying this output update. */
    uint64_t nextMemoryNode;

    /**
     * Indicates whether this update selects an action randomly.
     *
     * @return True if the update contains an action distribution, and false if it contains a single action.
     */
    virtual bool randomisedActionOutput() const {
        return false;
    }
};

/**
 * An FSC output update that chooses an action according to a probability distribution.
 *
 * @tparam ValueType The type used for the probabilities in the action distribution.
 */
template<typename ValueType>
struct RandomisedActionUpdate final : FSCOutputUpdate {
    /** The distribution over local action indices. */
    storm::storage::Distribution<ValueType, uint64_t> actionDistribution;

    /**
     * @return True, as this update contains an action distribution.
     */
    bool randomisedActionOutput() const override {
        return true;
    }
};

/** An FSC output update that deterministically chooses a single action. */
struct DeterministicActionUpdate final : FSCOutputUpdate {
    /** The local index of the selected action. */
    uint64_t action;
};
// Extend here if more types of updates are needed (e.g. stochastic transitions for memory nodes)

/**
 * A finite-state controller (FSC) for POMDPs that bases its decisions on observations.
 * Each node in the FSC corresponds to a memory state, and for each observation received in that node,
 * the FSC specifies an action to take and the successor node to transition to.
 * Observation and action IDs do not need to be continuously numbered, e.g. if the FSC stems from a belief MDP where some observations and actions are not
 * used. Action IDs correspond to local action indices for the respective observation. Optional mappings from IDs to names allow the controller to be
 * applied to POMDPs whose numeric IDs differ.
 *
 * @tparam ValueType The type used for probabilities in randomised action distributions.
 */
template<typename ValueType>
class ObservationBasedFiniteStateController {
    using ActionOutputUpdate = std::unordered_map<uint64_t, std::shared_ptr<FSCOutputUpdate>>;

   public:
    /**
     * Constructs an empty controller with the given initial memory node.
     *
     * @param initialNode The ID of the initial memory node.
     */
    explicit ObservationBasedFiniteStateController(uint64_t initialNode);

    /**
     * Constructs an empty controller with an initial memory node and optional names for observations and actions.
     *
     * @param initialNode The ID of the initial memory node.
     * @param idToObservationNameMap A mapping from observation IDs to names, or no mapping if observations are identified only by their IDs.
     * @param idToActionNameMap For each observation ID, a mapping from local action IDs to names, or no mapping if actions are identified only by their IDs.
     */
    ObservationBasedFiniteStateController(uint64_t initialNode, std::optional<std::unordered_map<uint64_t, std::string>> idToObservationNameMap,
                                          std::optional<std::unordered_map<uint64_t, std::unordered_map<uint64_t, std::string>>> idToActionNameMap);

    /**
     * Retrieves the deterministic action and successor memory node for an observation in a memory node.
     *
     * @param originId The ID of the current memory node.
     * @param observationId The ID of the current observation.
     * @return The local action ID and the successor memory-node ID.
     * @pre The controller has a deterministic output for @p observationId in @p originId.
     */
    std::pair<uint64_t, uint64_t> getActionAndSuccessorForObservationInNode(uint64_t originId, uint64_t observationId) const;

    /**
     * Retrieves the deterministic action for an observation in a memory node.
     *
     * @param originId The ID of the current memory node.
     * @param observationId The ID of the current observation.
     * @return The local ID of the selected action.
     * @pre The controller has a deterministic output for @p observationId in @p originId.
     */
    uint64_t getActionForObservationInNode(uint64_t originId, uint64_t observationId) const;

    /**
     * Retrieves the successor memory node for an observation in a memory node.
     *
     * @param originId The ID of the current memory node.
     * @param observationId The ID of the current observation.
     * @return The ID of the successor memory node.
     * @pre The controller has an output for @p observationId in @p originId.
     */
    uint64_t getSuccessorForObservationInNode(uint64_t originId, uint64_t observationId) const;

    /**
     * Retrieves the action distribution and successor memory node for an observation in a memory node.
     * A deterministic output is returned as a Dirac distribution over its selected action.
     *
     * @param originId The ID of the current memory node.
     * @param observationId The ID of the current observation.
     * @return The distribution over local action IDs and the successor memory-node ID.
     * @pre The controller has an output for @p observationId in @p originId.
     */
    std::pair<storm::storage::Distribution<ValueType, uint64_t>, uint64_t> getActionDistributionAndSuccessorForObservationInNode(uint64_t originId,
                                                                                                                                 uint64_t observationId) const;

    /**
     * Retrieves the action distribution for an observation in a memory node.
     * A deterministic output is returned as a Dirac distribution over its selected action.
     *
     * @param originId The ID of the current memory node.
     * @param observationId The ID of the current observation.
     * @return The distribution over local action IDs.
     * @pre The controller has an output for @p observationId in @p originId.
     */
    storm::storage::Distribution<ValueType, uint64_t> getActionDistributionForObservationInNode(uint64_t originId, uint64_t observationId) const;

    /**
     * Checks whether an output is defined for an observation in a memory node.
     *
     * @param originId The ID of the memory node.
     * @param observationId The ID of the observation.
     * @return True if an output is defined for the node-observation pair.
     */
    bool hasOutputForObservationInNode(uint64_t originId, uint64_t observationId) const;

    /**
     * Adds a deterministic output for an observation in a memory node.
     * An existing output for the same node-observation pair is replaced.
     *
     * @param originId The ID of the current memory node.
     * @param observationId The ID of the current observation.
     * @param actionId The local ID of the action to select.
     * @param targetId The ID of the successor memory node.
     */
    void addDeterministicActionTransition(uint64_t originId, uint64_t observationId, uint64_t actionId, uint64_t targetId);

    /**
     * Adds a randomised output for an observation in a memory node.
     * An existing output for the same node-observation pair is replaced.
     *
     * @param originId The ID of the current memory node.
     * @param observationId The ID of the current observation.
     * @param actionDistribution The distribution over local action IDs.
     * @param targetId The ID of the successor memory node.
     */
    void addRandomisedActionTransition(uint64_t originId, uint64_t observationId, storm::storage::Distribution<ValueType, uint64_t> actionDistribution,
                                       uint64_t targetId);

    /**
     * Replaces the mapping from observation IDs to names.
     *
     * @param idToObservationNameMap The new observation-name mapping.
     */
    void setIdToObservationNameMap(std::unordered_map<uint64_t, std::string> const& idToObservationNameMap);

    /**
     * Replaces the mappings from local action IDs to names for each observation.
     *
     * @param idToActionNameMap The new action-name mappings, indexed first by observation ID and then by local action ID.
     */
    void setIdToActionNameMap(std::unordered_map<uint64_t, std::unordered_map<uint64_t, std::string>> const& idToActionNameMap);

    /**
     * Retrieves the name associated with an observation ID.
     *
     * @param observationId The ID of the observation.
     * @return The observation name.
     * @pre An observation-name mapping is set and contains @p observationId.
     */
    std::string getObservationName(uint64_t observationId) const;

    /**
     * Retrieves the name associated with a local action ID for an observation.
     *
     * @param observationId The ID of the observation.
     * @param actionId The local ID of the action.
     * @return The action name.
     * @pre An action-name mapping is set and contains @p observationId and @p actionId.
     */
    std::string getActionName(uint64_t observationId, uint64_t actionId) const;

    /**
     * Looks up an observation ID by name.
     *
     * @param observationName The observation name to find.
     * @return The corresponding observation ID, or std::nullopt if no observation-name mapping is set or the name is not found.
     */
    std::optional<uint64_t> getObservationIdByName(std::string const& observationName) const;

    /**
     * Looks up a local action ID by name for an observation.
     *
     * @param observationId The ID of the observation whose actions are searched.
     * @param actionName The action name to find.
     * @return The corresponding local action ID, or std::nullopt if no applicable action-name mapping or action name is found.
     */
    std::optional<uint64_t> getActionIdByName(uint64_t observationId, std::string const& actionName) const;

    /**
     * Reports whether all outputs added to the controller so far have been deterministic.
     *
     * @return True if no randomised output has been added.
     */
    bool isDeterministic() const;

    /**
     * Checks whether the output for an observation in a memory node is randomised.
     *
     * @param originId The ID of the memory node.
     * @param observationId The ID of the observation.
     * @return True if the output contains an action distribution and false if it contains a single action.
     * @pre The controller has an output for @p observationId in @p originId.
     */
    bool outputIsRandomised(uint64_t originId, uint64_t observationId) const;

    /**
     * @return The ID of the initial memory node.
     */
    uint64_t getInitialNodeId() const;

    /**
     * @return True if an observation-name mapping is set, including an empty mapping.
     */
    bool hasIdToObservationNameMap() const;

    /**
     * @return True if action-name mappings are set, including an empty mapping.
     */
    bool hasIdToActionNameMap() const;

    /**
     * Creates a human-readable representation of the controller, listing its initial node and outputs grouped by memory node.
     * Available observation and action names are included alongside their IDs.
     *
     * @return The human-readable representation.
     */
    std::string toString() const;

    /**
     * Writes a Graphviz DOT representation of the controller.
     * Edges represent output updates and are labeled with the observation followed by the deterministic action or action distribution. Available names are
     * used in place of IDs.
     *
     * @param outStream The stream to write to.
     */
    void writeDotToStream(std::ostream& outStream) const;

    /**
     * Writes a JSON representation of the controller.
     * The root object contains `initial-node`, `deterministic`, and `nodes`. Each node contains its `id` and `transitions`; each transition records an
     * `observation`, `successor`, and an `actions` array whose entries contain an action `id` and `probability`. Available observation and action names are
     * included as `name` fields.
     *
     * @param outStream The stream to write to.
     */
    void writeJsonToStream(std::ostream& outStream) const;

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
