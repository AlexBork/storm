#include "storm-pomdp/builder/BeliefMdpExplorer.h"

#include "storm-parsers/api/properties.h"
#include "storm/api/properties.h"

#include "storm/modelchecker/results/CheckResult.h"
#include "storm/modelchecker/results/ExplicitQualitativeCheckResult.h"
#include "storm/modelchecker/results/ExplicitQuantitativeCheckResult.h"
#include "storm/modelchecker/hints/ExplicitModelCheckerHint.cpp"
#include "storm/models/sparse/Pomdp.h"
#include "storm/storage/SparseMatrix.h"
#include "storm/utility/constants.h"
#include "storm/utility/macros.h"
#include "storm/utility/SignalHandler.h"

namespace storm {
    namespace builder {
        template<typename PomdpType, typename BeliefValueType>
        BeliefMdpExplorer<PomdpType, BeliefValueType>::SuccessorObservationInformation::SuccessorObservationInformation(ValueType const &obsProb, ValueType const &maxProb, uint64_t const &count) : observationProbability(obsProb), maxProbabilityToSuccessorWithObs(maxProb), successorWithObsCount(count) {
            // Intentionally left empty.
        }

        template<typename PomdpType, typename BeliefValueType>
        void BeliefMdpExplorer<PomdpType, BeliefValueType>::SuccessorObservationInformation::join(SuccessorObservationInformation other) { /// Does not join support (for performance reasons)
            observationProbability += other.observationProbability;
            maxProbabilityToSuccessorWithObs = std::max(maxProbabilityToSuccessorWithObs, other.maxProbabilityToSuccessorWithObs);
            successorWithObsCount += other.successorWithObsCount;
        }

        template<typename PomdpType, typename BeliefValueType>
        BeliefMdpExplorer<PomdpType, BeliefValueType>::BeliefMdpExplorer(std::shared_ptr<BeliefManagerType> beliefManager,
                                                                         storm::pomdp::modelchecker::PreprocessingPomdpValueBounds<ValueType> const &pomdpValueBounds,
                                                                         ExplorationHeuristic explorationHeuristic) : beliefManager(beliefManager),
                                                                                                                      pomdpValueBounds(pomdpValueBounds),
                                                                                                                      explHeuristic(explorationHeuristic),
                                                                                                                      status(Status::Uninitialized) {
            // Intentionally left empty
        }

        template<typename PomdpType, typename BeliefValueType>
        typename BeliefMdpExplorer<PomdpType, BeliefValueType>::BeliefManagerType const &BeliefMdpExplorer<PomdpType, BeliefValueType>::getBeliefManager() const {
            return *beliefManager;
        }

        template<typename PomdpType, typename BeliefValueType>
        void
        BeliefMdpExplorer<PomdpType, BeliefValueType>::startNewExploration(boost::optional<ValueType> extraTargetStateValue, boost::optional<ValueType> extraBottomStateValue) {
            status = Status::Exploring;
            // Reset data from potential previous explorations
            prio = storm::utility::zero<ValueType>();
            nextId = 0;
            mdpStateToBeliefIdMap.clear();
            beliefIdToMdpStateMap.clear();
            exploredBeliefIds.clear();
            exploredBeliefIds.grow(beliefManager->getNumberOfBeliefIds(), false);
            mdpStatesToExplorePrioState.clear();
            mdpStatesToExploreStatePrio.clear();
            stateRemapping.clear();
            lowerValueBounds.clear();
            upperValueBounds.clear();
            values.clear();
            exploredMdpTransitions.clear();
            exploredChoiceIndices.clear();
            probabilityEstimation.clear();
            mdpActionRewards.clear();
            targetStates.clear();
            truncatedStates.clear();
            clippedStates.clear();
            delayedExplorationChoices.clear();
            clippingTransitionRewards.clear();
            mdpStateToChoiceLabelsMap.clear();
            optimalChoices = boost::none;
            optimalChoicesReachableMdpStates = boost::none;
            exploredMdp = nullptr;
            internalAddRowGroupIndex(); // Mark the start of the first row group

            // Add some states with special treatment (if requested)
            if (extraBottomStateValue) {
                currentMdpState = getCurrentNumberOfMdpStates();
                extraBottomState = currentMdpState;
                mdpStateToBeliefIdMap.push_back(beliefManager->noId());
                probabilityEstimation.push_back(storm::utility::zero<ValueType>());
                insertValueHints(extraBottomStateValue.get(), extraBottomStateValue.get());

                internalAddTransition(getStartOfCurrentRowGroup(), extraBottomState.get(), storm::utility::one<ValueType>());
                mdpStateToChoiceLabelsMap[getStartOfCurrentRowGroup()][0] = "loop";
                internalAddRowGroupIndex();
                ++nextId;
            } else {
                extraBottomState = boost::none;
            }
            if (extraTargetStateValue) {
                currentMdpState = getCurrentNumberOfMdpStates();
                extraTargetState = currentMdpState;
                mdpStateToBeliefIdMap.push_back(beliefManager->noId());
                probabilityEstimation.push_back(storm::utility::zero<ValueType>());
                insertValueHints(extraTargetStateValue.get(), extraTargetStateValue.get());

                internalAddTransition(getStartOfCurrentRowGroup(), extraTargetState.get(), storm::utility::one<ValueType>());
                mdpStateToChoiceLabelsMap[getStartOfCurrentRowGroup()][0] = "loop";
                internalAddRowGroupIndex();

                targetStates.grow(getCurrentNumberOfMdpStates(), false);
                targetStates.set(extraTargetState.get(), true);
                ++nextId;
            } else {
                extraTargetState = boost::none;
            }
            currentMdpState = noState();

            // Set up the initial state.
            initialMdpState = getOrAddMdpState(beliefManager->getInitialBelief());
        }

        template<typename PomdpType, typename BeliefValueType>
        void BeliefMdpExplorer<PomdpType, BeliefValueType>::restartExploration() {
            STORM_LOG_ASSERT(status == Status::ModelChecked || status == Status::ModelFinished, "Method call is invalid in current status.");
            status = Status::Exploring;
            // We will not erase old states during the exploration phase, so most state-based data (like mappings between MDP and Belief states) remain valid.
            prio = storm::utility::zero<ValueType>();
            stateRemapping.clear();
            exploredBeliefIds.clear();
            exploredBeliefIds.grow(beliefManager->getNumberOfBeliefIds(), false);
            exploredMdpTransitions.clear();
            exploredMdpTransitions.resize(exploredMdp->getNumberOfChoices());
            clippingTransitionRewards.clear();
            exploredChoiceIndices = exploredMdp->getNondeterministicChoiceIndices();
            mdpActionRewards.clear();
            probabilityEstimation.clear();
            if (exploredMdp->hasRewardModel()) {
                // Can be overwritten during exploration
                mdpActionRewards = exploredMdp->getUniqueRewardModel().getStateActionRewardVector();
            }
            targetStates = storm::storage::BitVector(getCurrentNumberOfMdpStates(), false);
            truncatedStates = storm::storage::BitVector(getCurrentNumberOfMdpStates(), false);
            clippedStates = storm::storage::BitVector(getCurrentNumberOfMdpStates(), false);
            delayedExplorationChoices.clear();
            mdpStatesToExplorePrioState.clear();
            mdpStatesToExploreStatePrio.clear();

            // The extra states are not changed
            if (extraBottomState) {
                currentMdpState = extraBottomState.get();
                restoreOldBehaviorAtCurrentState(0);
            }
            if (extraTargetState) {
                currentMdpState = extraTargetState.get();
                restoreOldBehaviorAtCurrentState(0);
                targetStates.set(extraTargetState.get(), true);
            }
            currentMdpState = noState();

            // Set up the initial state.
            initialMdpState = getOrAddMdpState(beliefManager->getInitialBelief());
        }

        template<typename PomdpType, typename BeliefValueType>
        bool BeliefMdpExplorer<PomdpType, BeliefValueType>::hasUnexploredState() const {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            return !mdpStatesToExploreStatePrio.empty();
        }

        template<typename PomdpType, typename BeliefValueType>
        std::vector<uint64_t> BeliefMdpExplorer<PomdpType, BeliefValueType>::getUnexploredStates() {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            std::vector<uint64_t> res;
            for(auto const &entry : mdpStatesToExploreStatePrio){
                res.push_back(entry.first);
            }
            return res;
        }

        template<typename PomdpType, typename BeliefValueType>
        typename BeliefMdpExplorer<PomdpType, BeliefValueType>::BeliefId BeliefMdpExplorer<PomdpType, BeliefValueType>::exploreNextState() {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            // Mark the end of the previously explored row group.
            if (currentMdpState != noState() && !currentStateHasOldBehavior()) {
                internalAddRowGroupIndex();
            }

            // Pop from the queue.
            currentMdpState = mdpStatesToExplorePrioState.rbegin()->second;
            auto currprio = mdpStatesToExplorePrioState.rbegin()->first;
            auto range = mdpStatesToExplorePrioState.equal_range(currprio);
            for(auto i = range.first; i != range.second; ++i) {
                if(i->second == currentMdpState){
                    mdpStatesToExplorePrioState.erase(i);
                    break;
                }
            }
            mdpStatesToExploreStatePrio.erase(currentMdpState);
            if (currentMdpState != nextId && !currentStateHasOldBehavior()) {
                stateRemapping[currentMdpState] = nextId;
                STORM_LOG_DEBUG(
                        "Explore state " << currentMdpState << " [Bel " << getCurrentBeliefId() << " " << beliefManager->toString(getCurrentBeliefId()) << "] as state with ID " << nextId << " (Prio: " << storm::utility::to_string(currprio) << ")");
            } else {
                STORM_LOG_DEBUG(
                        "Explore state " << currentMdpState << " [Bel " << getCurrentBeliefId() << " " << beliefManager->toString(getCurrentBeliefId()) << "]" << " (Prio: " << storm::utility::to_string(currprio) << ")");
            }

            if(!currentStateHasOldBehavior()) {
                ++nextId;
            }
            if(explHeuristic == ExplorationHeuristic::ProbabilityPrio){
                probabilityEstimation.push_back(currprio);
            }

            return mdpStateToBeliefIdMap[currentMdpState];
        }

        template<typename PomdpType, typename BeliefValueType>
        void BeliefMdpExplorer<PomdpType, BeliefValueType>::addTransitionsToExtraStates(uint64_t const &localActionIndex, ValueType const &targetStateValue,
                                                                                        ValueType const &bottomStateValue) {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            STORM_LOG_ASSERT(!currentStateHasOldBehavior() || localActionIndex < exploredChoiceIndices[currentMdpState + 1] - exploredChoiceIndices[currentMdpState],
                             "Action index " << localActionIndex << " was not valid at state " << currentMdpState << " of the previously explored MDP.");
            uint64_t row = getStartOfCurrentRowGroup() + localActionIndex;
            if (!storm::utility::isZero(bottomStateValue)) {
                STORM_LOG_ASSERT(extraBottomState.is_initialized(), "Requested a transition to the extra bottom state but there is none.");
                internalAddTransition(row, extraBottomState.get(), bottomStateValue);
            }
            if (!storm::utility::isZero(targetStateValue)) {
                STORM_LOG_ASSERT(extraTargetState.is_initialized(), "Requested a transition to the extra target state but there is none.");
                internalAddTransition(row, extraTargetState.get(), targetStateValue);
            }
        }

        template<typename PomdpType, typename BeliefValueType>
        void BeliefMdpExplorer<PomdpType, BeliefValueType>::addSelfloopTransition(uint64_t const &localActionIndex, ValueType const &value) {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            STORM_LOG_ASSERT(!currentStateHasOldBehavior() || localActionIndex < exploredChoiceIndices[currentMdpState + 1] - exploredChoiceIndices[currentMdpState],
                             "Action index " << localActionIndex << " was not valid at state " << currentMdpState << " of the previously explored MDP.");
            uint64_t row = getStartOfCurrentRowGroup() + localActionIndex;
            internalAddTransition(row, getCurrentMdpState(), value);
        }

        template<typename PomdpType, typename BeliefValueType>
        bool BeliefMdpExplorer<PomdpType, BeliefValueType>::addTransitionToBelief(uint64_t const &localActionIndex, BeliefId const &transitionTarget, ValueType const &value,
                                                                                  bool ignoreNewBeliefs) {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            STORM_LOG_ASSERT(!currentStateHasOldBehavior() || localActionIndex < exploredChoiceIndices[currentMdpState + 1] - exploredChoiceIndices[currentMdpState],
                             "Action index " << localActionIndex << " was not valid at state " << currentMdpState << " of the previously explored MDP.");

            MdpStateType column;
            if (ignoreNewBeliefs) {
                column = getExploredMdpState(transitionTarget);
                if (column == noState()) {
                    return false;
                }
            } else {
                column = getOrAddMdpState(transitionTarget, value);
            }
            uint64_t row = getStartOfCurrentRowGroup() + localActionIndex;
            internalAddTransition(row, column, value);
            return true;
        }

        template<typename PomdpType, typename BeliefValueType>
        void BeliefMdpExplorer<PomdpType, BeliefValueType>::computeRewardAtCurrentState(uint64_t const &localActionIndex, ValueType extraReward) {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            if (getCurrentNumberOfMdpChoices() > mdpActionRewards.size()) {
                mdpActionRewards.resize(getCurrentNumberOfMdpChoices(), storm::utility::zero<ValueType>());
            }
            uint64_t row = getStartOfCurrentRowGroup() + localActionIndex;
            mdpActionRewards[row] = beliefManager->getBeliefActionReward(getCurrentBeliefId(), localActionIndex) + extraReward;
        }

        template<typename PomdpType, typename BeliefValueType>
        void BeliefMdpExplorer<PomdpType, BeliefValueType>::addRewardToCurrentState(uint64_t const &localActionIndex, ValueType rewardValue) {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            if (getCurrentNumberOfMdpChoices() > mdpActionRewards.size()) {
                mdpActionRewards.resize(getCurrentNumberOfMdpChoices(), storm::utility::zero<ValueType>());
            }
            uint64_t row = getStartOfCurrentRowGroup() + localActionIndex;
            mdpActionRewards[row] = rewardValue;
        }

        template<typename PomdpType, typename BeliefValueType>
        void BeliefMdpExplorer<PomdpType, BeliefValueType>::addClippingRewardToCurrentState(uint64_t const &localActionIndex, ValueType rewardValue) {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            uint64_t row = getStartOfCurrentRowGroup() + localActionIndex;
            clippingTransitionRewards[row] = rewardValue;
        }

        template<typename PomdpType, typename BeliefValueType>
        void BeliefMdpExplorer<PomdpType, BeliefValueType>::setCurrentStateIsTarget() {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            targetStates.grow(getCurrentNumberOfMdpStates(), false);
            targetStates.set(getCurrentMdpState(), true);
        }

        template<typename PomdpType, typename BeliefValueType>
        void BeliefMdpExplorer<PomdpType, BeliefValueType>::setCurrentStateIsTruncated() {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            truncatedStates.grow(getCurrentNumberOfMdpStates(), false);
            truncatedStates.set(getCurrentMdpState(), true);
        }

        template<typename PomdpType, typename BeliefValueType>
        void BeliefMdpExplorer<PomdpType, BeliefValueType>::setCurrentStateIsClipped() {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            setCurrentStateIsTruncated();
            clippedStates.grow(getCurrentNumberOfMdpStates(), false);
            clippedStates.set(getCurrentMdpState(), true);
        }

        template<typename PomdpType, typename BeliefValueType>
        void BeliefMdpExplorer<PomdpType, BeliefValueType>::setCurrentChoiceIsDelayed(uint64_t const &localActionIndex) {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            delayedExplorationChoices.grow(getCurrentNumberOfMdpChoices(), false);
            delayedExplorationChoices.set(getStartOfCurrentRowGroup() + localActionIndex, true);
        }

        template<typename PomdpType, typename BeliefValueType>
        bool BeliefMdpExplorer<PomdpType, BeliefValueType>::currentStateHasOldBehavior() const {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            STORM_LOG_ASSERT(getCurrentMdpState() != noState(), "Method 'currentStateHasOldBehavior' called but there is no current state.");
            return exploredMdp && getCurrentMdpState() < exploredMdp->getNumberOfStates();
        }

        template<typename PomdpType, typename BeliefValueType>
        bool BeliefMdpExplorer<PomdpType, BeliefValueType>::getCurrentStateWasTruncated() const {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            STORM_LOG_ASSERT(getCurrentMdpState() != noState(), "Method 'actionAtCurrentStateWasOptimal' called but there is no current state.");
            STORM_LOG_ASSERT(currentStateHasOldBehavior(), "Method 'actionAtCurrentStateWasOptimal' called but current state has no old behavior");
            STORM_LOG_ASSERT(exploredMdp, "No 'old' mdp available");
            return exploredMdp->getStateLabeling().getStateHasLabel("truncated", getCurrentMdpState());
        }

        template<typename PomdpType, typename BeliefValueType>
        bool BeliefMdpExplorer<PomdpType, BeliefValueType>::getCurrentStateWasClipped() const {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            STORM_LOG_ASSERT(getCurrentMdpState() != noState(), "Method 'actionAtCurrentStateWasOptimal' called but there is no current state.");
            STORM_LOG_ASSERT(currentStateHasOldBehavior(), "Method 'actionAtCurrentStateWasOptimal' called but current state has no old behavior");
            STORM_LOG_ASSERT(exploredMdp, "No 'old' mdp available");
            return exploredMdp->getStateLabeling().getStateHasLabel("clipped", getCurrentMdpState());
        }

        template<typename PomdpType, typename BeliefValueType>
        bool BeliefMdpExplorer<PomdpType, BeliefValueType>::stateIsOptimalSchedulerReachable(MdpStateType mdpState) const {
            STORM_LOG_ASSERT(status == Status::ModelChecked, "Method call is invalid in current status.");
            STORM_LOG_ASSERT(optimalChoicesReachableMdpStates.is_initialized(),
                             "Method 'stateIsOptimalSchedulerReachable' called but 'computeOptimalChoicesAndReachableMdpStates' was not called before.");
            return optimalChoicesReachableMdpStates->get(mdpState);
        }

        template<typename PomdpType, typename BeliefValueType>
        bool BeliefMdpExplorer<PomdpType, BeliefValueType>::actionIsOptimal(uint64_t const &globalActionIndex) const {
            STORM_LOG_ASSERT(status == Status::ModelChecked, "Method call is invalid in current status.");
            STORM_LOG_ASSERT(optimalChoices.is_initialized(), "Method 'actionIsOptimal' called but 'computeOptimalChoicesAndReachableMdpStates' was not called before.");
            return optimalChoices->get(globalActionIndex);
        }

        template<typename PomdpType, typename BeliefValueType>
        bool BeliefMdpExplorer<PomdpType, BeliefValueType>::currentStateIsOptimalSchedulerReachable() const {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            STORM_LOG_ASSERT(getCurrentMdpState() != noState(), "Method 'currentStateIsOptimalSchedulerReachable' called but there is no current state.");
            STORM_LOG_ASSERT(currentStateHasOldBehavior(), "Method 'currentStateIsOptimalSchedulerReachable' called but current state has no old behavior");
            STORM_LOG_ASSERT(optimalChoicesReachableMdpStates.is_initialized(),
                             "Method 'currentStateIsOptimalSchedulerReachable' called but 'computeOptimalChoicesAndReachableMdpStates' was not called before.");
            return optimalChoicesReachableMdpStates->get(getCurrentMdpState());
        }

        template<typename PomdpType, typename BeliefValueType>
        bool BeliefMdpExplorer<PomdpType, BeliefValueType>::actionAtCurrentStateWasOptimal(uint64_t const &localActionIndex) const {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            STORM_LOG_ASSERT(getCurrentMdpState() != noState(), "Method 'actionAtCurrentStateWasOptimal' called but there is no current state.");
            STORM_LOG_ASSERT(currentStateHasOldBehavior(), "Method 'actionAtCurrentStateWasOptimal' called but current state has no old behavior");
            STORM_LOG_ASSERT(optimalChoices.is_initialized(),
                             "Method 'currentStateIsOptimalSchedulerReachable' called but 'computeOptimalChoicesAndReachableMdpStates' was not called before.");
            uint64_t choice = getStartOfCurrentRowGroup() + localActionIndex;
            return optimalChoices->get(choice);
        }

        template<typename PomdpType, typename BeliefValueType>
        bool BeliefMdpExplorer<PomdpType, BeliefValueType>::getCurrentStateActionExplorationWasDelayed(uint64_t const &localActionIndex) const {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            STORM_LOG_ASSERT(getCurrentMdpState() != noState(), "Method 'actionAtCurrentStateWasOptimal' called but there is no current state.");
            STORM_LOG_ASSERT(currentStateHasOldBehavior(), "Method 'actionAtCurrentStateWasOptimal' called but current state has no old behavior");
            STORM_LOG_ASSERT(exploredMdp, "No 'old' mdp available");
            uint64_t choice = exploredMdp->getNondeterministicChoiceIndices()[getCurrentMdpState()] + localActionIndex;
            return exploredMdp->hasChoiceLabeling() && exploredMdp->getChoiceLabeling().getChoiceHasLabel("delayed", choice);
        }

        template<typename PomdpType, typename BeliefValueType>
        void BeliefMdpExplorer<PomdpType, BeliefValueType>::restoreOldBehaviorAtCurrentState(uint64_t const &localActionIndex) {
            STORM_LOG_ASSERT(currentStateHasOldBehavior(), "Cannot restore old behavior as the current state does not have any.");
            STORM_LOG_ASSERT(localActionIndex < exploredChoiceIndices[currentMdpState + 1] - exploredChoiceIndices[currentMdpState],
                             "Action index " << localActionIndex << " was not valid at state " << currentMdpState << " of the previously explored MDP.");

            uint64_t choiceIndex = exploredChoiceIndices[getCurrentMdpState()] + localActionIndex;
            STORM_LOG_ASSERT(choiceIndex < exploredChoiceIndices[getCurrentMdpState() + 1], "Invalid local action index.");

            // Insert the transitions
            for (auto const &transition : exploredMdp->getTransitionMatrix().getRow(choiceIndex)) {
                internalAddTransition(choiceIndex, transition.getColumn(), transition.getValue());
                // Check whether exploration is needed
                auto beliefId = getBeliefId(transition.getColumn());
                if (beliefId != beliefManager->noId()) { // Not the extra target or bottom state
                    if (!exploredBeliefIds.get(beliefId)) {
                        // This belief needs exploration
                        exploredBeliefIds.set(beliefId, true);
                        // TODO set priority correctly
                        ValueType currentPrio;
                        switch (explHeuristic) {
                            case ExplorationHeuristic::BreadthFirst :
                                currentPrio = prio;
                                prio = prio - storm::utility::one<ValueType>();
                                break;
                            case ExplorationHeuristic::LowerBoundPrio :
                                currentPrio = getLowerValueBoundAtCurrentState();
                                break;
                            case ExplorationHeuristic::UpperBoundPrio :
                                currentPrio = getUpperValueBoundAtCurrentState();
                                break;
                            case ExplorationHeuristic::GapPrio        :
                                currentPrio = getUpperValueBoundAtCurrentState() - getLowerValueBoundAtCurrentState();
                                break;
                            case ExplorationHeuristic::ProbabilityPrio:
                                if(getCurrentMdpState() != noState()) {
                                    currentPrio = probabilityEstimation[getCurrentMdpState()] * transition.getValue();
                                } else {
                                    currentPrio = storm::utility::one<ValueType>();
                                }
                                break;
                            default :
                                STORM_LOG_THROW(false, storm::exceptions::NotImplementedException, "Other heuristics not implemented yet");
                        }
                        //mdpStatesToExplore.push(std::make_pair(currentPrio, transition.getColumn()));
                        mdpStatesToExploreStatePrio[transition.getColumn()] = currentPrio;
                        mdpStatesToExplorePrioState.emplace(currentPrio, transition.getColumn());
                    }
                }
            }

            // Actually, nothing needs to be done for rewards since we already initialize the vector with the "old" values
        }

        template<typename PomdpType, typename BeliefValueType>
        void BeliefMdpExplorer<PomdpType, BeliefValueType>::finishExploration() {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            STORM_LOG_ASSERT(!hasUnexploredState(), "Finishing exploration not possible if there are still unexplored states.");

            // Complete the exploration
            // Finish the last row grouping in case the last explored state was new
            if (!currentStateHasOldBehavior()) {
                internalAddRowGroupIndex();
            }
            // Resize state- and choice based vectors to the correct size
            targetStates.resize(getCurrentNumberOfMdpStates(), false);
            truncatedStates.resize(getCurrentNumberOfMdpStates(), false);
            clippedStates.resize(getCurrentNumberOfMdpStates(), false);
            if (!mdpActionRewards.empty()) {
                mdpActionRewards.resize(getCurrentNumberOfMdpChoices(), storm::utility::zero<ValueType>());
            }

            // We are not exploring anymore
            currentMdpState = noState();

            // If this was a restarted exploration, we might still have unexplored states (which were only reachable and explored in a previous build).
            // We get rid of these before rebuilding the model
            if (exploredMdp) {
                dropUnexploredStates();
            }

            // The potentially computed optimal choices and the set of states that are reachable under these choices are not valid anymore.
            optimalChoices = boost::none;
            optimalChoicesReachableMdpStates = boost::none;

            // Apply state remapping to the Belief-State maps
            if (!stateRemapping.empty()) {
                std::vector<BeliefId> remappedStateToBeliefIdMap(mdpStateToBeliefIdMap);
                for (auto const &entry : stateRemapping) {
                    remappedStateToBeliefIdMap[entry.second] = mdpStateToBeliefIdMap[entry.first];
                }
                mdpStateToBeliefIdMap = remappedStateToBeliefIdMap;
                for (auto const &beliefMdpState : beliefIdToMdpStateMap) {
                    if (stateRemapping.find(beliefMdpState.second) != stateRemapping.end()) {
                        beliefIdToMdpStateMap[beliefMdpState.first] = stateRemapping[beliefMdpState.second];
                    }
                }
                if(!mdpStateToChoiceLabelsMap.empty()) {
                    std::map<BeliefId, std::map<uint64_t, std::string>> temp(mdpStateToChoiceLabelsMap);
                    for (auto const &entry : stateRemapping) {
                        temp[entry.second] = mdpStateToChoiceLabelsMap[entry.first];
                    }
                    mdpStateToChoiceLabelsMap = temp;
                }
            }

            // Create the transition matrix
            uint64_t entryCount = 0;
            for (auto const &row : exploredMdpTransitions) {
                entryCount += row.size();
            }
            storm::storage::SparseMatrixBuilder<ValueType> builder(getCurrentNumberOfMdpChoices(), getCurrentNumberOfMdpStates(), entryCount, true, true,
                                                                   getCurrentNumberOfMdpStates());
            for (uint64_t groupIndex = 0; groupIndex < exploredChoiceIndices.size() - 1; ++groupIndex) {
                uint64_t rowIndex = exploredChoiceIndices[groupIndex];
                uint64_t groupEnd = exploredChoiceIndices[groupIndex + 1];
                builder.newRowGroup(rowIndex);
                for (; rowIndex < groupEnd; ++rowIndex) {
                    for (auto const &entry : exploredMdpTransitions[rowIndex]) {
                        if (stateRemapping.find(entry.first) == stateRemapping.end()) {
                            builder.addNextValue(rowIndex, entry.first, entry.second);
                        } else {
                            builder.addNextValue(rowIndex, stateRemapping[entry.first], entry.second);
                        }
                    }
                }
            }
            auto mdpTransitionMatrix = builder.build();

            // Create a standard labeling
            storm::models::sparse::StateLabeling mdpLabeling(getCurrentNumberOfMdpStates());
            mdpLabeling.addLabel("init");
            mdpLabeling.addLabelToState("init", initialMdpState);
            targetStates.resize(getCurrentNumberOfMdpStates(), false);
            mdpLabeling.addLabel("target", std::move(targetStates));
            truncatedStates.resize(getCurrentNumberOfMdpStates(), false);
            mdpLabeling.addLabel("truncated", std::move(truncatedStates));
            clippedStates.resize(getCurrentNumberOfMdpStates(), false);
            mdpLabeling.addLabel("clipped", std::move(clippedStates));

            for(uint64_t state = 0; state < getCurrentNumberOfMdpStates(); ++state){
                if(state == extraBottomState || state == extraTargetState){
                    if(!mdpLabeling.containsLabel("__extra")) {
                        mdpLabeling.addLabel("__extra");
                    }
                    mdpLabeling.addLabelToState("__extra", state);
                } else {
                    STORM_LOG_DEBUG("Observation of MDP state " << state << " : " << beliefManager->getObservationLabel(mdpStateToBeliefIdMap[state]) << "\n");
                    std::string obsLabel = beliefManager->getObservationLabel(mdpStateToBeliefIdMap[state]);
                    if (!obsLabel.empty()) {
                        if (!mdpLabeling.containsLabel(obsLabel)) {
                            mdpLabeling.addLabel(obsLabel);
                        }
                        mdpLabeling.addLabelToState(obsLabel, state);
                    }
                }
            }

            // Create a standard reward model (if rewards are available)
            std::unordered_map<std::string, storm::models::sparse::StandardRewardModel<ValueType>> mdpRewardModels;
            if (!mdpActionRewards.empty()) {
                mdpActionRewards.resize(getCurrentNumberOfMdpChoices(), storm::utility::zero<ValueType>());
                if(!clippingTransitionRewards.empty()){
                    storm::storage::SparseMatrixBuilder<ValueType> rewardBuilder(getCurrentNumberOfMdpChoices(), getCurrentNumberOfMdpStates(), clippingTransitionRewards.size(), true, true, getCurrentNumberOfMdpStates());
                    for (uint64_t groupIndex = 0; groupIndex < exploredChoiceIndices.size() - 1; ++groupIndex) {
                        uint64_t rowIndex = exploredChoiceIndices[groupIndex];
                        uint64_t groupEnd = exploredChoiceIndices[groupIndex + 1];
                        rewardBuilder.newRowGroup(rowIndex);
                        for (; rowIndex < groupEnd; ++rowIndex) {
                            if(clippingTransitionRewards.find(rowIndex) != clippingTransitionRewards.end()){
                                STORM_LOG_ASSERT(extraTargetState.is_initialized(), "Requested a transition to the extra target state but there is none.");
                                rewardBuilder.addNextValue(rowIndex, extraTargetState.get(), clippingTransitionRewards[rowIndex]);
                            }
                        }
                    }
                    auto transitionRewardMatrix = rewardBuilder.build();
                    mdpRewardModels.emplace("default",
                                            storm::models::sparse::StandardRewardModel<ValueType>(boost::optional<std::vector<ValueType>>(), std::move(mdpActionRewards), std::move(transitionRewardMatrix)));
                } else {
                    mdpRewardModels.emplace("default",
                                            storm::models::sparse::StandardRewardModel<ValueType>(boost::optional<std::vector<ValueType>>(), std::move(mdpActionRewards)));
                }
            }

            // Create model components
            storm::storage::sparse::ModelComponents<ValueType> modelComponents(std::move(mdpTransitionMatrix), std::move(mdpLabeling), std::move(mdpRewardModels));

            // Potentially create a choice labeling
            if(!mdpStateToChoiceLabelsMap.empty()){
                modelComponents.choiceLabeling = storm::models::sparse::ChoiceLabeling(getCurrentNumberOfMdpChoices());
                for(auto const & stateMap : mdpStateToChoiceLabelsMap){
                    auto rowGroup = stateMap.first;
                    for(auto const & actionLabel : stateMap.second){
                        if(! modelComponents.choiceLabeling->containsLabel(actionLabel.second)){
                            modelComponents.choiceLabeling->addLabel(actionLabel.second);
                        }
                        modelComponents.choiceLabeling->addLabelToChoice(actionLabel.second, exploredChoiceIndices[rowGroup] + actionLabel.first);
                    }
                }
            }

            if (!delayedExplorationChoices.empty()) {
                modelComponents.choiceLabeling = storm::models::sparse::ChoiceLabeling(getCurrentNumberOfMdpChoices());
                delayedExplorationChoices.resize(getCurrentNumberOfMdpChoices(), false);
                modelComponents.choiceLabeling->addLabel("delayed", std::move(delayedExplorationChoices));
            }

            // Create the final model.
            exploredMdp = std::make_shared<storm::models::sparse::Mdp<ValueType>>(std::move(modelComponents));
            status = Status::ModelFinished;
            STORM_LOG_DEBUG(
                    "Explored Mdp with " << exploredMdp->getNumberOfStates() << " states (" << clippedStates.getNumberOfSetBits() << " of which were clipped and "
                                         << truncatedStates.getNumberOfSetBits() - clippedStates.getNumberOfSetBits() << " of which were flagged as truncated).");
        }

        template<typename PomdpType, typename BeliefValueType>
        void BeliefMdpExplorer<PomdpType, BeliefValueType>::dropUnexploredStates() {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            STORM_LOG_ASSERT(!hasUnexploredState(), "Finishing exploration not possible if there are still unexplored states.");

            STORM_LOG_ASSERT(exploredMdp, "Method called although no 'old' MDP is available.");
            // Find the states (and corresponding choices) that were not explored.
            // These correspond to "empty" MDP transitions
            storm::storage::BitVector relevantMdpStates(getCurrentNumberOfMdpStates(), true), relevantMdpChoices(getCurrentNumberOfMdpChoices(), true);
            std::vector<MdpStateType> toRelevantStateIndexMap(getCurrentNumberOfMdpStates(), noState());
            MdpStateType nextRelevantIndex = 0;
            for (uint64_t groupIndex = 0; groupIndex < exploredChoiceIndices.size() - 1; ++groupIndex) {
                uint64_t rowIndex = exploredChoiceIndices[groupIndex];
                // Check first row in group
                if (exploredMdpTransitions[rowIndex].empty()) {
                    relevantMdpChoices.set(rowIndex, false);
                    relevantMdpStates.set(groupIndex, false);
                } else {
                    toRelevantStateIndexMap[groupIndex] = nextRelevantIndex;
                    ++nextRelevantIndex;
                }
                uint64_t groupEnd = exploredChoiceIndices[groupIndex + 1];
                // process remaining rows in group
                for (++rowIndex; rowIndex < groupEnd; ++rowIndex) {
                    // Assert that all actions at the current state were consistently explored or unexplored.
                    STORM_LOG_ASSERT(exploredMdpTransitions[rowIndex].empty() != relevantMdpStates.get(groupIndex),
                                     "Actions at 'old' MDP state " << groupIndex << " were only partly explored.");
                    if (exploredMdpTransitions[rowIndex].empty()) {
                        relevantMdpChoices.set(rowIndex, false);
                    }
                }
            }

            if (relevantMdpStates.full()) {
                // All states are relevant so nothing to do
                return;
            }

            nextId -= (relevantMdpStates.size() - relevantMdpStates.getNumberOfSetBits());

            // Translate various components to the "new" MDP state set
            storm::utility::vector::filterVectorInPlace(mdpStateToBeliefIdMap, relevantMdpStates);
            { // beliefIdToMdpStateMap
                for (auto belIdToMdpStateIt = beliefIdToMdpStateMap.begin(); belIdToMdpStateIt != beliefIdToMdpStateMap.end();) {
                    if (relevantMdpStates.get(belIdToMdpStateIt->second)) {
                        // Translate current entry and move on to the next one.
                        belIdToMdpStateIt->second = toRelevantStateIndexMap[belIdToMdpStateIt->second];
                        ++belIdToMdpStateIt;
                    } else {
                        STORM_LOG_ASSERT(!exploredBeliefIds.get(belIdToMdpStateIt->first),
                                         "Inconsistent exploration information: Unexplored MDPState corresponds to explored beliefId");
                        // Delete current entry and move on to the next one.
                        // This works because std::map::erase does not invalidate other iterators within the map!
                        beliefIdToMdpStateMap.erase(belIdToMdpStateIt++);
                    }
                }
            }
            { // exploredMdpTransitions
                storm::utility::vector::filterVectorInPlace(exploredMdpTransitions, relevantMdpChoices);
                // Adjust column indices. Unfortunately, the fastest way seems to be to "rebuild" the map
                // It might payoff to do this when building the matrix.
                for (auto &transitions : exploredMdpTransitions) {
                    std::map<MdpStateType, ValueType> newTransitions;
                    for (auto const &entry : transitions) {
                        STORM_LOG_ASSERT(relevantMdpStates.get(entry.first), "Relevant state has transition to irrelevant state.");
                        newTransitions.emplace_hint(newTransitions.end(), toRelevantStateIndexMap[entry.first], entry.second);
                    }
                    transitions = std::move(newTransitions);
                }
            }
            { // exploredChoiceIndices
                MdpStateType newState = 0;
                assert(exploredChoiceIndices[0] == 0u);
                // Loop invariant: all indices up to exploredChoiceIndices[newState] consider the new row indices and all other entries are not touched.
                for (auto const &oldState : relevantMdpStates) {
                    if (oldState != newState) {
                        assert(oldState > newState);
                        uint64_t groupSize = exploredChoiceIndices[oldState + 1] - exploredChoiceIndices[oldState];
                        exploredChoiceIndices[newState + 1] = exploredChoiceIndices[newState] + groupSize;
                    }
                    ++newState;
                }
                exploredChoiceIndices.resize(newState + 1);
            }
            if (!mdpActionRewards.empty()) {
                storm::utility::vector::filterVectorInPlace(mdpActionRewards, relevantMdpChoices);
            }
            if (extraBottomState) {
                extraBottomState = toRelevantStateIndexMap[extraBottomState.get()];
            }
            if (extraTargetState) {
                extraTargetState = toRelevantStateIndexMap[extraTargetState.get()];
            }
            targetStates = targetStates % relevantMdpStates;
            truncatedStates = truncatedStates % relevantMdpStates;
            clippedStates = clippedStates % relevantMdpStates;
            initialMdpState = toRelevantStateIndexMap[initialMdpState];

            storm::utility::vector::filterVectorInPlace(lowerValueBounds, relevantMdpStates);
            storm::utility::vector::filterVectorInPlace(upperValueBounds, relevantMdpStates);
            storm::utility::vector::filterVectorInPlace(values, relevantMdpStates);

        }

        template<typename PomdpType, typename BeliefValueType>
        std::shared_ptr<storm::models::sparse::Mdp<typename BeliefMdpExplorer<PomdpType, BeliefValueType>::ValueType>>
        BeliefMdpExplorer<PomdpType, BeliefValueType>::getExploredMdp() const {
            STORM_LOG_ASSERT(status == Status::ModelFinished || status == Status::ModelChecked, "Method call is invalid in current status.");
            STORM_LOG_ASSERT(exploredMdp, "Tried to get the explored MDP but exploration was not finished yet.");
            return exploredMdp;
        }

        template<typename PomdpType, typename BeliefValueType>
        typename BeliefMdpExplorer<PomdpType, BeliefValueType>::MdpStateType BeliefMdpExplorer<PomdpType, BeliefValueType>::getCurrentNumberOfMdpStates() const {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            return mdpStateToBeliefIdMap.size();
        }

        template<typename PomdpType, typename BeliefValueType>
        typename BeliefMdpExplorer<PomdpType, BeliefValueType>::MdpStateType BeliefMdpExplorer<PomdpType, BeliefValueType>::getCurrentNumberOfMdpChoices() const {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            return exploredMdpTransitions.size();
        }

        template<typename PomdpType, typename BeliefValueType>
        typename BeliefMdpExplorer<PomdpType, BeliefValueType>::MdpStateType BeliefMdpExplorer<PomdpType, BeliefValueType>::getStartOfCurrentRowGroup() const {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            return exploredChoiceIndices[getCurrentMdpState()];
        }

        template<typename PomdpType, typename BeliefValueType>
        typename BeliefMdpExplorer<PomdpType, BeliefValueType>::ValueType BeliefMdpExplorer<PomdpType, BeliefValueType>::getLowerValueBoundAtCurrentState() const {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            return lowerValueBounds[getCurrentMdpState()];
        }

        template<typename PomdpType, typename BeliefValueType>
        typename BeliefMdpExplorer<PomdpType, BeliefValueType>::ValueType BeliefMdpExplorer<PomdpType, BeliefValueType>::getUpperValueBoundAtCurrentState() const {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            return upperValueBounds[getCurrentMdpState()];
        }

        template<typename PomdpType, typename BeliefValueType>
        typename BeliefMdpExplorer<PomdpType, BeliefValueType>::ValueType
        BeliefMdpExplorer<PomdpType, BeliefValueType>::computeLowerValueBoundAtBelief(BeliefId const &beliefId) const {
            STORM_LOG_ASSERT(!pomdpValueBounds.lower.empty(), "Requested lower value bounds but none were available.");
            auto it = pomdpValueBounds.lower.begin();
            ValueType result = beliefManager->getWeightedSum(beliefId, *it);
            for (++it; it != pomdpValueBounds.lower.end(); ++it) {
                result = std::max(result, beliefManager->getWeightedSum(beliefId, *it));
            }
            return result;
        }

        template<typename PomdpType, typename BeliefValueType>
        typename BeliefMdpExplorer<PomdpType, BeliefValueType>::ValueType
        BeliefMdpExplorer<PomdpType, BeliefValueType>::computeUpperValueBoundAtBelief(BeliefId const &beliefId) const {
            STORM_LOG_ASSERT(!pomdpValueBounds.upper.empty(), "Requested upper value bounds but none were available.");
            auto it = pomdpValueBounds.upper.begin();
            ValueType result = beliefManager->getWeightedSum(beliefId, *it);
            for (++it; it != pomdpValueBounds.upper.end(); ++it) {
                result = std::min(result, beliefManager->getWeightedSum(beliefId, *it));
            }
            return result;
        }

        template<typename PomdpType, typename BeliefValueType>
        typename BeliefMdpExplorer<PomdpType, BeliefValueType>::ValueType
        BeliefMdpExplorer<PomdpType, BeliefValueType>::computeLowerValueBoundForScheduler(BeliefId const &beliefId, uint64_t schedulerId) const {
            STORM_LOG_ASSERT(!pomdpValueBounds.lower.empty(), "Requested lower value bounds but none were available.");
            STORM_LOG_ASSERT(pomdpValueBounds.lower.size() > schedulerId, "Requested lower value bound for scheduler with ID " << schedulerId << " not available.");
            return beliefManager->getWeightedSum(beliefId, pomdpValueBounds.lower[schedulerId]);;
        }

        template<typename PomdpType, typename BeliefValueType>
        typename BeliefMdpExplorer<PomdpType, BeliefValueType>::ValueType
        BeliefMdpExplorer<PomdpType, BeliefValueType>::computeUpperValueBoundForScheduler(BeliefId const &beliefId, uint64_t schedulerId) const {
            STORM_LOG_ASSERT(!pomdpValueBounds.upper.empty(), "Requested upper value bounds but none were available.");
            STORM_LOG_ASSERT(pomdpValueBounds.upper.size() > schedulerId, "Requested upper value bound for scheduler with ID " << schedulerId << " not available.");
            return beliefManager->getWeightedSum(beliefId, pomdpValueBounds.upper[schedulerId]);
        }

        template<typename PomdpType, typename BeliefValueType>
        typename BeliefMdpExplorer<PomdpType, BeliefValueType>::ValueType
        BeliefMdpExplorer<PomdpType, BeliefValueType>::computeParametricBoundAtBelief(BeliefId const &beliefId) const {
            STORM_LOG_ASSERT(!pomdpValueBounds.parametric.empty(), "Parametric bounds not available.");
            return beliefManager->getWeightedSum(beliefId, pomdpValueBounds.parametric);
        }

        template<typename PomdpType, typename BeliefValueType>
        void BeliefMdpExplorer<PomdpType, BeliefValueType>::computeValuesOfExploredMdp(storm::solver::OptimizationDirection const &dir) {
            STORM_LOG_ASSERT(status == Status::ModelFinished, "Method call is invalid in current status.");
            STORM_LOG_ASSERT(exploredMdp, "Tried to compute values but the MDP is not explored");
            auto property = createStandardProperty(dir, exploredMdp->hasRewardModel());
            auto task = createStandardCheckTask(property);

            std::unique_ptr<storm::modelchecker::CheckResult> res(storm::api::verifyWithSparseEngine<ValueType>(exploredMdp, task));
            if (res) {
                values = std::move(res->asExplicitQuantitativeCheckResult<ValueType>().getValueVector());
                scheduler = std::make_shared<storm::storage::Scheduler<ValueType>>(res->asExplicitQuantitativeCheckResult<ValueType>().getScheduler());
                STORM_LOG_WARN_COND_DEBUG(storm::utility::vector::compareElementWise(lowerValueBounds, values, std::less_equal<ValueType>()),
                                          "Computed values are smaller than the lower bound.");
                STORM_LOG_WARN_COND_DEBUG(storm::utility::vector::compareElementWise(upperValueBounds, values, std::greater_equal<ValueType>()),
                                          "Computed values are larger than the upper bound.");


                // ============================================================================================
                // ===PRINTING BELIEF MDP INFO (REMOVE AFTER TODO)=============================================
                // ============================================================================================

                std::ofstream myfile;
                myfile.open("belief-mdp.txt");

                storm::models::sparse::NondeterministicModel<ValueType> nd_model = (storm::models::sparse::NondeterministicModel<ValueType>) *exploredMdp;

                storm::storage::BitVector actionSelection = res->asExplicitQuantitativeCheckResult<ValueType>().getScheduler().computeActionSupport(nd_model.getNondeterministicChoiceIndices());
                storm::storage::BitVector allStates(nd_model.getNumberOfStates(), true);

                //Belief MDP state information
                myfile << "Belief MDP state information:\n\n";

                for (MdpStateType i = beliefIdToMdpStateMap[0]; i < getCurrentNumberOfMdpStates(); i++)
                {
                    {
                        myfile << "belief: " << getBeliefManager().toString(getBeliefId(i)) << 
                                    "\n\tbelief MDP state: " << i << 
                                    "\n\tobservation id: " << beliefManager->getBeliefObservation(getBeliefId(i)) << 
                                    "\n\tchosen action: " << res->asExplicitQuantitativeCheckResult<ValueType>().getScheduler().getChoice(i).getDeterministicChoice() << 
                                    "\n\tstate value: " << values[i] << "\n\n";
                    }
                }

                myfile.close();
            } else {
                STORM_LOG_ASSERT(storm::utility::resources::isTerminate(), "Empty check result!");
                STORM_LOG_ERROR("No result obtained while checking.");
            }
            status = Status::ModelChecked;
        }

        template<typename PomdpType, typename BeliefValueType>
        bool BeliefMdpExplorer<PomdpType, BeliefValueType>::hasComputedValues() const {
            return status == Status::ModelChecked;
        }

        template<typename PomdpType, typename BeliefValueType>
        std::vector<typename BeliefMdpExplorer<PomdpType, BeliefValueType>::ValueType> const &BeliefMdpExplorer<PomdpType, BeliefValueType>::getValuesOfExploredMdp() const {
            STORM_LOG_ASSERT(status == Status::ModelChecked, "Method call is invalid in current status.");
            return values;
        }

        template<typename PomdpType, typename BeliefValueType>
        const std::shared_ptr<storm::storage::Scheduler<typename BeliefMdpExplorer<PomdpType, BeliefValueType>::ValueType>> &BeliefMdpExplorer<PomdpType, BeliefValueType>::getSchedulerForExploredMdp() const {
            STORM_LOG_ASSERT(status == Status::ModelChecked, "Method call is invalid in current status.");
            return scheduler;
        }

        template<typename PomdpType, typename BeliefValueType>
        typename BeliefMdpExplorer<PomdpType, BeliefValueType>::ValueType const &BeliefMdpExplorer<PomdpType, BeliefValueType>::getComputedValueAtInitialState() const {
            STORM_LOG_ASSERT(status == Status::ModelChecked, "Method call is invalid in current status.");
            STORM_LOG_ASSERT(exploredMdp, "Tried to get a value but no MDP was explored.");
            return getValuesOfExploredMdp()[exploredMdp->getInitialStates().getNextSetIndex(0)];
        }

        template<typename PomdpType, typename BeliefValueType>
        typename BeliefMdpExplorer<PomdpType, BeliefValueType>::MdpStateType BeliefMdpExplorer<PomdpType, BeliefValueType>::getBeliefId(MdpStateType exploredMdpState) const {
            STORM_LOG_ASSERT(status != Status::Uninitialized, "Method call is invalid in current status.");
            return mdpStateToBeliefIdMap[exploredMdpState];
        }

        template<typename PomdpType, typename BeliefValueType>
        void BeliefMdpExplorer<PomdpType, BeliefValueType>::gatherSuccessorObservationInformationAtCurrentState(uint64_t localActionIndex,
                                                                                                                std::map<uint32_t, SuccessorObservationInformation> &gatheredSuccessorObservations) {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            STORM_LOG_ASSERT(currentStateHasOldBehavior(), "Method call is invalid since the current state has no old behavior");
            uint64_t mdpChoice = getStartOfCurrentRowGroup() + localActionIndex;
            gatherSuccessorObservationInformationAtMdpChoice(mdpChoice, gatheredSuccessorObservations);

        }

        template<typename PomdpType, typename BeliefValueType>
        void BeliefMdpExplorer<PomdpType, BeliefValueType>::gatherSuccessorObservationInformationAtMdpChoice(uint64_t mdpChoice,
                                                                                                             std::map<uint32_t, SuccessorObservationInformation> &gatheredSuccessorObservations) {
            STORM_LOG_ASSERT(exploredMdp, "Method call is invalid if no MDP has been explored before");
            for (auto const &entry : exploredMdp->getTransitionMatrix().getRow(mdpChoice)) {
                auto const &beliefId = getBeliefId(entry.getColumn());
                if (beliefId != beliefManager->noId()) {
                    auto const &obs = beliefManager->getBeliefObservation(beliefId);
                    SuccessorObservationInformation info(entry.getValue(), entry.getValue(), 1);
                    auto obsInsertion = gatheredSuccessorObservations.emplace(obs, info);
                    if (!obsInsertion.second) {
                        // There already is an entry for this observation, so join the two informations
                        obsInsertion.first->second.join(info);
                    }
                    beliefManager->joinSupport(beliefId, obsInsertion.first->second.support);
                }
            }
        }

        template<typename PomdpType, typename BeliefValueType>
        bool BeliefMdpExplorer<PomdpType, BeliefValueType>::currentStateHasSuccessorObservationInObservationSet(uint64_t localActionIndex,
                                                                                                                storm::storage::BitVector const &observationSet) {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            STORM_LOG_ASSERT(currentStateHasOldBehavior(), "Method call is invalid since the current state has no old behavior");
            uint64_t mdpChoice = getStartOfCurrentRowGroup() + localActionIndex;
            for(auto const &entry : exploredMdp->getTransitionMatrix().getRow(mdpChoice)) {
                auto const &beliefId = getBeliefId(entry.getColumn());
                if (observationSet.get(beliefManager->getBeliefObservation(beliefId))) {
                    return true;
                }
            }
            return false;
        }

        template<typename PomdpType, typename BeliefValueType>
        void BeliefMdpExplorer<PomdpType, BeliefValueType>::takeCurrentValuesAsUpperBounds() {
            STORM_LOG_ASSERT(status == Status::ModelChecked, "Method call is invalid in current status.");
            upperValueBounds = values;
        }

        template<typename PomdpType, typename BeliefValueType>
        void BeliefMdpExplorer<PomdpType, BeliefValueType>::takeCurrentValuesAsLowerBounds() {
            STORM_LOG_ASSERT(status == Status::ModelChecked, "Method call is invalid in current status.");
            lowerValueBounds = values;
        }

        template<typename PomdpType, typename BeliefValueType>
        void BeliefMdpExplorer<PomdpType, BeliefValueType>::computeOptimalChoicesAndReachableMdpStates(ValueType const &ancillaryChoicesEpsilon, bool relativeDifference) {
            STORM_LOG_ASSERT(status == Status::ModelChecked, "Method call is invalid in current status.");
            STORM_LOG_ASSERT(exploredMdp, "Method call is invalid in if no MDP is available.");
            STORM_LOG_ASSERT(!optimalChoices.is_initialized(), "Tried to compute optimal scheduler but this has already been done before.");
            STORM_LOG_ASSERT(!optimalChoicesReachableMdpStates.is_initialized(),
                             "Tried to compute states that are reachable under an optimal scheduler but this has already been done before.");

            // First find the choices that are optimal
            optimalChoices = storm::storage::BitVector(exploredMdp->getNumberOfChoices(), false);
            auto const &choiceIndices = exploredMdp->getNondeterministicChoiceIndices();
            auto const &transitions = exploredMdp->getTransitionMatrix();
            auto const &targetStates = exploredMdp->getStates("target");
            for (uint64_t mdpState = 0; mdpState < exploredMdp->getNumberOfStates(); ++mdpState) {
                if (targetStates.get(mdpState)) {
                    // Target states can be skipped.
                    continue;
                } else {
                    auto const &stateValue = values[mdpState];
                    for (uint64_t globalChoice = choiceIndices[mdpState]; globalChoice < choiceIndices[mdpState + 1]; ++globalChoice) {
                        ValueType choiceValue = transitions.multiplyRowWithVector(globalChoice, values);
                        if (exploredMdp->hasRewardModel()) {
                            choiceValue += exploredMdp->getUniqueRewardModel().getStateActionReward(globalChoice);
                        }
                        ValueType absDiff = storm::utility::abs<ValueType>((choiceValue - stateValue));
                        if ((relativeDifference && absDiff <= ancillaryChoicesEpsilon * stateValue) || (!relativeDifference && absDiff <= ancillaryChoicesEpsilon)) {
                            optimalChoices->set(globalChoice, true);
                        }
                    }
                    STORM_LOG_ASSERT(optimalChoices->getNextSetIndex(choiceIndices[mdpState]) < optimalChoices->size(), "Could not find an optimal choice.");
                }
            }

            // Then, find the states that are reachable via these choices
            optimalChoicesReachableMdpStates = storm::utility::graph::getReachableStates(transitions, exploredMdp->getInitialStates(), ~targetStates, targetStates, false, 0,
                                                                                         optimalChoices.get());
        }

        template<typename PomdpType, typename BeliefValueType>
        bool BeliefMdpExplorer<PomdpType, BeliefValueType>::beliefHasMdpState(BeliefId const &beliefId)  const {
            return getExploredMdpState(beliefId) != noState();
        }

        template<typename PomdpType, typename BeliefValueType>
        typename BeliefMdpExplorer<PomdpType, BeliefValueType>::MdpStateType BeliefMdpExplorer<PomdpType, BeliefValueType>::noState() const {
            return std::numeric_limits<MdpStateType>::max();
        }

        template<typename PomdpType, typename BeliefValueType>
        std::shared_ptr<storm::logic::Formula const>
        BeliefMdpExplorer<PomdpType, BeliefValueType>::createStandardProperty(storm::solver::OptimizationDirection const &dir, bool computeRewards) {
            std::string propertyString = computeRewards ? "R" : "P";
            propertyString += storm::solver::minimize(dir) ? "min" : "max";
            propertyString += "=? [F \"target\"]";
            std::vector<storm::jani::Property> propertyVector = storm::api::parseProperties(propertyString);
            return storm::api::extractFormulasFromProperties(propertyVector).front();
        }

        template<typename PomdpType, typename BeliefValueType>
        storm::modelchecker::CheckTask<storm::logic::Formula, typename BeliefMdpExplorer<PomdpType, BeliefValueType>::ValueType>
        BeliefMdpExplorer<PomdpType, BeliefValueType>::createStandardCheckTask(std::shared_ptr<storm::logic::Formula const> &property) {
            //Note: The property should not run out of scope after calling this because the task only stores the property by reference.
            // Therefore, this method needs the property by reference (and not const reference)
            auto task = storm::api::createTask<ValueType>(property, false);
            auto hint = storm::modelchecker::ExplicitModelCheckerHint<ValueType>();
            hint.setResultHint(values);
            auto hintPtr = std::make_shared<storm::modelchecker::ExplicitModelCheckerHint<ValueType>>(hint);
            task.setHint(hintPtr);
            task.setProduceSchedulers();
            return task;
        }

        template<typename PomdpType, typename BeliefValueType>
        typename BeliefMdpExplorer<PomdpType, BeliefValueType>::MdpStateType BeliefMdpExplorer<PomdpType, BeliefValueType>::getCurrentMdpState() const {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            if (stateRemapping.find(currentMdpState) != stateRemapping.end()) {
                return stateRemapping.at(currentMdpState);
            } else {
                return currentMdpState;
            }
        }

        template<typename PomdpType, typename BeliefValueType>
        typename BeliefMdpExplorer<PomdpType, BeliefValueType>::MdpStateType BeliefMdpExplorer<PomdpType, BeliefValueType>::getCurrentBeliefId() const {
            STORM_LOG_ASSERT(status == Status::Exploring, "Method call is invalid in current status.");
            return getBeliefId(currentMdpState);
        }

        template<typename PomdpType, typename BeliefValueType>
        void BeliefMdpExplorer<PomdpType, BeliefValueType>::internalAddTransition(uint64_t const &row, MdpStateType const &column, ValueType const &value) {
            STORM_LOG_ASSERT(row <= exploredMdpTransitions.size(), "Skipped at least one row.");
            if (row == exploredMdpTransitions.size()) {
                exploredMdpTransitions.emplace_back();
            }
            STORM_LOG_ASSERT(exploredMdpTransitions[row].count(column) == 0, "Trying to insert multiple transitions to the same state.");
            exploredMdpTransitions[row][column] = value;
        }

        template<typename PomdpType, typename BeliefValueType>
        void BeliefMdpExplorer<PomdpType, BeliefValueType>::internalAddRowGroupIndex() {
            exploredChoiceIndices.push_back(getCurrentNumberOfMdpChoices());
        }

        template<typename PomdpType, typename BeliefValueType>
        typename BeliefMdpExplorer<PomdpType, BeliefValueType>::MdpStateType BeliefMdpExplorer<PomdpType, BeliefValueType>::getExploredMdpState(BeliefId const &beliefId) const {
            if (beliefId < exploredBeliefIds.size() && exploredBeliefIds.get(beliefId)) {
                return beliefIdToMdpStateMap.at(beliefId);
            } else {
                return noState();
            }
        }

        template<typename PomdpType, typename BeliefValueType>
        void BeliefMdpExplorer<PomdpType, BeliefValueType>::insertValueHints(ValueType const &lowerBound, ValueType const &upperBound) {
            lowerValueBounds.push_back(lowerBound);
            upperValueBounds.push_back(upperBound);
            // Take the middle value as a hint
            values.push_back((lowerBound + upperBound) / storm::utility::convertNumber<ValueType, uint64_t>(2));
            STORM_LOG_ASSERT(lowerValueBounds.size() == getCurrentNumberOfMdpStates(), "Value vectors have different size then number of available states.");
            STORM_LOG_ASSERT(lowerValueBounds.size() == upperValueBounds.size() && values.size() == upperValueBounds.size(), "Value vectors have inconsistent size.");
        }

        template<typename PomdpType, typename BeliefValueType>
        typename BeliefMdpExplorer<PomdpType, BeliefValueType>::MdpStateType BeliefMdpExplorer<PomdpType, BeliefValueType>::getOrAddMdpState(BeliefId const &beliefId, ValueType const &transitionValue) {
            exploredBeliefIds.grow(beliefId + 1, false);
            if (exploredBeliefIds.get(beliefId)) {
                if(explHeuristic == ExplorationHeuristic::ProbabilityPrio && mdpStatesToExploreStatePrio.find(beliefIdToMdpStateMap[beliefId]) != mdpStatesToExploreStatePrio.end()){
                    // We check if the value is higher than the current priority and update if necessary
                    auto newPrio = probabilityEstimation[getCurrentMdpState()] * transitionValue;
                    if(newPrio > mdpStatesToExploreStatePrio[beliefIdToMdpStateMap[beliefId]]) {
                        // Erase the state from the "queue" map and re-insert it with the new value
                        auto range = mdpStatesToExplorePrioState.equal_range(mdpStatesToExploreStatePrio[beliefIdToMdpStateMap[beliefId]]);
                        for (auto i = range.first; i != range.second; ++i) {
                            if (i->second == beliefIdToMdpStateMap[beliefId]) {
                                mdpStatesToExplorePrioState.erase(i);
                                break;
                            }
                        }
                        mdpStatesToExplorePrioState.emplace(newPrio, beliefIdToMdpStateMap[beliefId]);
                        mdpStatesToExploreStatePrio[beliefIdToMdpStateMap[beliefId]] = newPrio;
                    }
                }
                return beliefIdToMdpStateMap[beliefId];
            } else {
                // This state needs exploration
                exploredBeliefIds.set(beliefId, true);

                // If this is a restart of the exploration, we still might have an MDP state for the belief
                if (exploredMdp) {
                    auto findRes = beliefIdToMdpStateMap.find(beliefId);
                    if (findRes != beliefIdToMdpStateMap.end()) {
                        ValueType currentPrio;
                        switch (explHeuristic) {
                            case ExplorationHeuristic::BreadthFirst :
                                currentPrio = prio;
                                prio = prio - storm::utility::one<ValueType>();
                                break;
                            case ExplorationHeuristic::LowerBoundPrio :
                                currentPrio = getLowerValueBoundAtCurrentState();
                                break;
                            case ExplorationHeuristic::UpperBoundPrio :
                                currentPrio = getUpperValueBoundAtCurrentState();
                                break;
                            case ExplorationHeuristic::GapPrio        :
                                currentPrio = getUpperValueBoundAtCurrentState() - getLowerValueBoundAtCurrentState();
                                break;
                            case ExplorationHeuristic::ProbabilityPrio:
                                if(getCurrentMdpState() != noState()) {
                                    currentPrio = probabilityEstimation[getCurrentMdpState()] * transitionValue;
                                } else {
                                    currentPrio = storm::utility::one<ValueType>();
                                }
                                break;
                            default :
                                STORM_LOG_THROW(false, storm::exceptions::NotImplementedException, "Other heuristics not implemented yet");
                        }
                        mdpStatesToExploreStatePrio[findRes->second] = currentPrio;
                        mdpStatesToExplorePrioState.emplace(currentPrio, findRes->second);
                        return findRes->second;
                    }
                }
                // At this point we need to add a new MDP state
                MdpStateType result = getCurrentNumberOfMdpStates();
                assert(getCurrentNumberOfMdpStates() == mdpStateToBeliefIdMap.size());
                mdpStateToBeliefIdMap.push_back(beliefId);
                beliefIdToMdpStateMap[beliefId] = result;
                insertValueHints(computeLowerValueBoundAtBelief(beliefId), computeUpperValueBoundAtBelief(beliefId));
                ValueType currentPrio;
                switch (explHeuristic) {
                    case ExplorationHeuristic::BreadthFirst :
                        currentPrio = prio;
                        prio = prio - storm::utility::one<ValueType>();
                        break;
                    case ExplorationHeuristic::LowerBoundPrio :
                        currentPrio = getLowerValueBoundAtCurrentState();
                        break;
                    case ExplorationHeuristic::UpperBoundPrio :
                        currentPrio = getUpperValueBoundAtCurrentState();
                        break;
                    case ExplorationHeuristic::GapPrio        :
                        currentPrio = getUpperValueBoundAtCurrentState() - getLowerValueBoundAtCurrentState();
                        break;
                    case ExplorationHeuristic::ProbabilityPrio:
                        if(getCurrentMdpState() != noState()) {
                            currentPrio = probabilityEstimation[getCurrentMdpState()] * transitionValue;
                        } else {
                            currentPrio = storm::utility::one<ValueType>();
                        }
                        break;
                    default :
                        STORM_LOG_THROW(false, storm::exceptions::NotImplementedException, "Other heuristics not implemented yet");
                }
                mdpStatesToExploreStatePrio[result] = currentPrio;
                mdpStatesToExplorePrioState.emplace(currentPrio, result);
                return result;
            }
        }

        template<typename PomdpType, typename BeliefValueType>
        void BeliefMdpExplorer<PomdpType, BeliefValueType>::addChoiceLabelToCurrentState(uint64_t const &localActionIndex, std::string label) {
            mdpStateToChoiceLabelsMap[currentMdpState][localActionIndex] = label;
        }

        template<typename PomdpType, typename BeliefValueType>
        std::vector<typename BeliefMdpExplorer<PomdpType, BeliefValueType>::BeliefId> BeliefMdpExplorer<PomdpType, BeliefValueType>::getBeliefsInMdp() {
            return mdpStateToBeliefIdMap;
        }

        template<typename PomdpType, typename BeliefValueType>
        std::vector<typename BeliefMdpExplorer<PomdpType, BeliefValueType>::BeliefId> BeliefMdpExplorer<PomdpType, BeliefValueType>::getBeliefsWithObservationInMdp(uint32_t obs) const {
            std::vector<BeliefId> res;
            for (auto const &belief : mdpStateToBeliefIdMap) {
                if(belief != beliefManager->noId()) {
                    if (beliefManager->getBeliefObservation(belief) == obs) {
                        res.push_back(belief);
                    }
                }
            }
            return res;
        }

        template<typename PomdpType, typename BeliefValueType>
        bool BeliefMdpExplorer<PomdpType, BeliefValueType>::hasParametricBounds(){
            return !pomdpValueBounds.parametric.empty();
        }

        template<typename PomdpType, typename BeliefValueType>
        typename BeliefMdpExplorer<PomdpType, BeliefValueType>::ValueType BeliefMdpExplorer<PomdpType, BeliefValueType>::getParametricBoundAtPOMDPState(uint64_t const &pomdpState){
            return pomdpValueBounds.getParametricBound(pomdpState);
        }

        template<typename PomdpType, typename BeliefValueType>
        typename BeliefMdpExplorer<PomdpType, BeliefValueType>::ValueType BeliefMdpExplorer<PomdpType, BeliefValueType>::getTrivialUpperBoundAtPOMDPState(uint64_t const &pomdpState){
            return pomdpValueBounds.getSmallestUpperBound(pomdpState);
        }

        template<typename PomdpType, typename BeliefValueType>
        typename BeliefMdpExplorer<PomdpType, BeliefValueType>::ValueType BeliefMdpExplorer<PomdpType, BeliefValueType>::getTrivialLowerBoundAtPOMDPState(uint64_t const &pomdpState){
            return pomdpValueBounds.getHighestLowerBound(pomdpState);
        }

        template<typename PomdpType, typename BeliefValueType>
        void BeliefMdpExplorer<PomdpType, BeliefValueType>::setExtremeValueBound(storm::pomdp::modelchecker::ExtremePOMDPValueBound<ValueType> valueBound){
            extremeValueBound = valueBound;
        }

        template<typename PomdpType, typename BeliefValueType>
        typename BeliefMdpExplorer<PomdpType, BeliefValueType>::ValueType BeliefMdpExplorer<PomdpType, BeliefValueType>::getExtremeValueBoundAtPOMDPState(const uint64_t &pomdpState){
            return extremeValueBound.getValueForState(pomdpState);
        }

        template<typename PomdpType, typename BeliefValueType>
        storm::storage::BitVector BeliefMdpExplorer<PomdpType, BeliefValueType>::getStateExtremeBoundIsInfinite(){
            return extremeValueBound.isInfinite;
        }

        template<typename PomdpType, typename BeliefValueType>
        uint64_t BeliefMdpExplorer<PomdpType, BeliefValueType>::getNrSchedulersForUpperBounds() {
            return pomdpValueBounds.upper.size();
        }

        template<typename PomdpType, typename BeliefValueType>
        uint64_t BeliefMdpExplorer<PomdpType, BeliefValueType>::getNrSchedulersForLowerBounds() {
            return pomdpValueBounds.lower.size();
        }

        template<typename PomdpType, typename BeliefValueType>
        storm::storage::Scheduler<typename BeliefMdpExplorer<PomdpType, BeliefValueType>::ValueType>
        BeliefMdpExplorer<PomdpType, BeliefValueType>::getLowerValueBoundScheduler(uint64_t schedulerId) const {
            STORM_LOG_ASSERT(!pomdpValueBounds.lowerSchedulers.empty(), "Requested lower bound scheduler but none were available.");
            STORM_LOG_ASSERT(pomdpValueBounds.lowerSchedulers.size() > schedulerId, "Requested lower value bound scheduler with ID " << schedulerId << " not available.");
            return pomdpValueBounds.lowerSchedulers[schedulerId];
        }

        template<typename PomdpType, typename BeliefValueType>
        storm::storage::Scheduler<typename BeliefMdpExplorer<PomdpType, BeliefValueType>::ValueType>
        BeliefMdpExplorer<PomdpType, BeliefValueType>::getUpperValueBoundScheduler(uint64_t schedulerId) const {
            STORM_LOG_ASSERT(!pomdpValueBounds.upperSchedulers.empty(), "Requested upper bound scheduler but none were available.");
            STORM_LOG_ASSERT(pomdpValueBounds.upperSchedulers.size() > schedulerId, "Requested upper value bound scheduler with ID " << schedulerId << " not available.");
            return pomdpValueBounds.upperSchedulers[schedulerId];
        }

        template<typename PomdpType, typename BeliefValueType>
        std::vector<storm::storage::Scheduler<typename BeliefMdpExplorer<PomdpType, BeliefValueType>::ValueType>>
        BeliefMdpExplorer<PomdpType, BeliefValueType>::getLowerValueBoundSchedulers() const {
            STORM_LOG_ASSERT(!pomdpValueBounds.lowerSchedulers.empty(), "Requested lower bound schedulers but none were available.");
            return pomdpValueBounds.lowerSchedulers;
        }

        template<typename PomdpType, typename BeliefValueType>
        std::vector<storm::storage::Scheduler<typename BeliefMdpExplorer<PomdpType, BeliefValueType>::ValueType>>
        BeliefMdpExplorer<PomdpType, BeliefValueType>::getUpperValueBoundSchedulers() const {
            STORM_LOG_ASSERT(!pomdpValueBounds.upperSchedulers.empty(), "Requested upper bound schedulers but none were available.");
            return pomdpValueBounds.upperSchedulers;
        }

        template<typename PomdpType, typename BeliefValueType>
        std::vector<uint32_t> BeliefMdpExplorer<PomdpType, BeliefValueType>::getObservationForMdpStates() const {
            std::vector<uint32_t> obs;
            uint64_t nrMdpStates = exploredMdp->getNumberOfStates();
            for(uint64_t mdpState = 0; mdpState< nrMdpStates; ++mdpState) {
                uint64_t beliefId = this->getBeliefId(mdpState);
                if (beliefId != beliefManager->noId()) {
                    obs.push_back(this->beliefManager->getBeliefObservation(beliefId));
                } else {
                    obs.push_back(-1);
                }
            }
            STORM_LOG_ASSERT(obs.size() == nrMdpStates, "There should be an entry for every MDP state");
            return obs;
        }

        template
        class BeliefMdpExplorer<storm::models::sparse::Pomdp<double>>;

        template
        class BeliefMdpExplorer<storm::models::sparse::Pomdp<double>, storm::RationalNumber>;

        template
        class BeliefMdpExplorer<storm::models::sparse::Pomdp<storm::RationalNumber>, double>;

        template
        class BeliefMdpExplorer<storm::models::sparse::Pomdp<storm::RationalNumber>>;
    }
}

