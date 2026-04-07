#include "storm-pomdp/beliefs/verification/BeliefBasedModelChecker.h"

#include "storm-pomdp/beliefs/abstraction/ClippingBeliefAbstraction.h"
#include "storm-pomdp/beliefs/abstraction/FreudenthalTriangulationBeliefAbstraction.h"
#include "storm-pomdp/beliefs/abstraction/RewardBoundedBeliefSplitter.h"
#include "storm-pomdp/beliefs/exploration/BeliefExploration.h"
#include "storm-pomdp/beliefs/exploration/BeliefMdpBuilder.h"
#include "storm-pomdp/beliefs/policy/ApplyObservationBasedFSCToPomdp.h"
#include "storm-pomdp/beliefs/policy/PolicyExtractor.h"
#include "storm-pomdp/beliefs/storage/Belief.h"
#include "storm-pomdp/beliefs/verification/BeliefBasedModelCheckerOptions.h"
#include "storm/api/verification.h"
#include "storm/modelchecker/results/ExplicitQuantitativeCheckResult.h"
#include "storm/models/sparse/Pomdp.h"
#include "storm/transformer/GoalStateMerger.h"
#include "storm/transformer/TransitionToActionRewardTransformer.h"
#include "storm/utility/OptionalRef.h"
#include "storm/utility/Stopwatch.h"
#include "storm/utility/constants.h"
#include "storm/utility/graph.h"
#include "storm/utility/macros.h"

namespace storm::pomdp::beliefs {

template<typename PomdpModelType, typename BeliefValueType, typename BeliefMdpValueType>
BeliefBasedModelChecker<PomdpModelType, BeliefValueType, BeliefMdpValueType>::BeliefBasedModelChecker(PomdpModelType const& pomdp) : inputPomdp(pomdp) {
    STORM_LOG_ERROR_COND(inputPomdp.isCanonic(), "Input Pomdp is not known to be canonic. This might lead to unexpected verification results.");
}

template<typename PomdpModelType, typename BeliefValueType, typename BeliefMdpValueType>
BeliefBasedModelChecker<PomdpModelType, BeliefValueType, BeliefMdpValueType>::BeliefBasedModelChecker(
    PomdpModelType const& pomdp, storm::models::sparse::Pomdp<storm::RationalNumber> const&)
    : inputPomdp(pomdp) {
    STORM_LOG_THROW(false, storm::exceptions::NotSupportedException,
                    "The requested constructor is only available for checking floating point POMDPs with exact beliefs.");
}

template<>
BeliefBasedModelChecker<storm::models::sparse::Pomdp<double>, storm::RationalNumber, double>::BeliefBasedModelChecker(
    storm::models::sparse::Pomdp<double> const& pomdp, storm::models::sparse::Pomdp<storm::RationalNumber> const& exactInputPomdp)
    : inputPomdp(pomdp), exactPomdp(exactInputPomdp) {
    STORM_LOG_ERROR_COND(inputPomdp.isCanonic(), "Input POMDP is not known to be canonic. This might lead to unexpected verification results.");
    STORM_LOG_ERROR_COND(exactPomdp->isCanonic(), "Exact POMDP is not known to be canonic. This might lead to unexpected verification results.");
}

template<typename PomdpModelType, typename BeliefType, typename BeliefMdpValueType, typename InfoType>
typename BeliefExploration<BeliefMdpValueType, PomdpModelType, BeliefType>::TerminationCallback getTerminationCallback(
    storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<BeliefMdpValueType> const& options, InfoType& info, storm::utility::Stopwatch& swExplore) {
    switch (options.getTerminationCriterion()) {
        case MAX_EXPLORATION_SIZE:
            if (options.implicitCutOffs) {
                return [&info, maxSize = options.maxExplorationSize.value()]() { return info.exploredBeliefs.size() > maxSize; };
            } else {
                return [&info, maxSize = options.maxExplorationSize.value()]() { return info.discoveredBeliefs.getNumberOfBeliefIds() > maxSize; };
            }
        case MAX_EXPLORATION_TIME:
            return [&swExplore, maxDuration = options.maxExplorationTime.value()]() {
                return static_cast<unsigned>(abs(swExplore.getTimeInSeconds())) > maxDuration;
            };
        case MAX_EXPLORATION_SIZE_AND_TIME:
            if (options.implicitCutOffs) {
                return [&info, &swExplore, maxSize = options.maxExplorationSize.value(), maxDuration = options.maxExplorationTime.value()]() {
                    return info.exploredBeliefs.size() > maxSize || static_cast<unsigned>(abs(swExplore.getTimeInSeconds())) > maxDuration;
                };
            } else {
                return [&info, &swExplore, maxSize = options.maxExplorationSize.value(), maxDuration = options.maxExplorationTime.value()]() {
                    return info.discoveredBeliefs.getNumberOfBeliefIds() > maxSize || static_cast<unsigned>(abs(swExplore.getTimeInSeconds())) > maxDuration;
                };
            }
        case NONE:
            // Unlimited unfolding (useful for known finite belief MDPs)
            return []() { return false; };
        default:
            STORM_LOG_ERROR("Unknown termination criterion for belief exploration.");
            return []() { return false; };
    }
}

template<typename PomdpModelType, typename BeliefType, typename BeliefMdpValueType>
typename BeliefExploration<BeliefMdpValueType, PomdpModelType, BeliefType>::TerminalBeliefCallback getTerminalBeliefCallback(
    PropertyInformation const& propertyInformation, storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<BeliefMdpValueType> const& options,
    storm::pomdp::storage::PreprocessingPomdpValueBounds<typename PomdpModelType::ValueType> const& valueBounds) {
    using PomdpValueType = PomdpModelType::ValueType;
    if (propertyInformation.kind == PropertyInformation::Kind::ExpectedTotalReachabilityReward) {
        if (options.maxGapToCut.has_value()) {
            // Terminate if the gap is small enough
            return
                [&propertyInformation, &valueBounds, maxGapToCut = options.maxGapToCut.value()](BeliefType const& belief) -> std::optional<BeliefMdpValueType> {
                    if (propertyInformation.targetObservations.contains(belief.observation())) {
                        return storm::utility::zero<BeliefMdpValueType>();
                    } else {
                        // TODO add scheduler information if requested
                        auto smallestUpper = storm::utility::infinity<PomdpValueType>();
                        for (auto const& valueList : valueBounds.upper) {
                            smallestUpper = std::min(smallestUpper, belief.template getWeightedSum<PomdpValueType>(valueList));
                        }
                        PomdpValueType largestLower = -storm::utility::infinity<PomdpValueType>();
                        for (auto const& valueList : valueBounds.lower) {
                            largestLower = storm::utility::max(largestLower, belief.template getWeightedSum<PomdpValueType>(valueList));
                        }
                        if (storm::utility::abs<PomdpValueType>(smallestUpper - largestLower) <= maxGapToCut) {
                            if constexpr (std::is_same_v<PomdpValueType, BeliefMdpValueType>) {
                                return propertyInformation.dir == solver::OptimizationDirection::Maximize ? largestLower : smallestUpper;
                            } else {
                                return storm::utility::convertNumber<BeliefMdpValueType>(
                                    propertyInformation.dir == solver::OptimizationDirection::Maximize ? largestLower : smallestUpper);
                            }
                        }
                        return std::nullopt;
                    }
                };
        } else {
            return [&propertyInformation](BeliefType const& belief) -> std::optional<BeliefMdpValueType> {
                if (propertyInformation.targetObservations.contains(belief.observation())) {
                    return storm::utility::zero<BeliefMdpValueType>();
                } else {
                    return std::nullopt;
                }
            };
        }
    } else if (propertyInformation.kind == PropertyInformation::Kind::RewardBoundedReachabilityProbability) {
        return [](BeliefType const& belief) -> std::optional<BeliefMdpValueType> {
            // For reward-bounded properties, we cannot be sure that a target belief is terminal as we are not bound-aware at this point
            return std::nullopt;
        };
    } else if (options.maxGapToCut.has_value()) {
        // Terminate if the gap is small enough
        return [&propertyInformation, &valueBounds, maxGapToCut = options.maxGapToCut.value()](BeliefType const& belief) -> std::optional<BeliefMdpValueType> {
            if (propertyInformation.targetObservations.contains(belief.observation())) {
                return storm::utility::one<BeliefMdpValueType>();
            } else {
                // TODO add scheduler information if requested
                auto smallestUpper = storm::utility::infinity<PomdpValueType>();
                for (auto const& valueList : valueBounds.upper) {
                    smallestUpper = std::min(smallestUpper, belief.template getWeightedSum<PomdpValueType>(valueList));
                }
                PomdpValueType largestLower = -storm::utility::infinity<PomdpValueType>();
                for (auto const& valueList : valueBounds.lower) {
                    largestLower = storm::utility::max(largestLower, belief.template getWeightedSum<PomdpValueType>(valueList));
                }
                if (storm::utility::abs<PomdpValueType>(smallestUpper - largestLower) <= maxGapToCut) {
                    if constexpr (std::is_same_v<PomdpValueType, BeliefMdpValueType>) {
                        return propertyInformation.dir == solver::OptimizationDirection::Maximize ? largestLower : smallestUpper;
                    } else {
                        return storm::utility::convertNumber<BeliefMdpValueType>(
                            propertyInformation.dir == solver::OptimizationDirection::Maximize ? largestLower : smallestUpper);
                    }
                }
                return std::nullopt;
            }
        };
    } else {
        return [&propertyInformation](BeliefType const& belief) -> std::optional<BeliefMdpValueType> {
            if (propertyInformation.targetObservations.contains(belief.observation())) {
                return storm::utility::one<BeliefMdpValueType>();
            } else {
                return std::nullopt;
            };
        };
    }
}

template<typename PomdpModelType, typename BeliefType, typename BeliefMdpValueType, typename InfoType>
std::pair<std::shared_ptr<models::sparse::Mdp<BeliefMdpValueType>>, std::unordered_map<uint64_t, BeliefId>> buildBeliefMdpFromInfo(
    PropertyInformation const& propertyInformation, BeliefBasedModelCheckerOptions<BeliefMdpValueType> const& options,
    storm::pomdp::storage::PreprocessingPomdpValueBounds<typename PomdpModelType::ValueType> const& valueBounds, InfoType const& info) {
    using PomdpValueType = PomdpModelType::ValueType;
    if (options.implicitCutOffs) {
        std::function<BeliefMdpValueType(BeliefType const&)> computeCutOffValue = [&valueBounds, &propertyInformation](BeliefType const& belief) {
            // TODO: extend with different sources for cut-offs
            auto result = storm::utility::infinity<PomdpValueType>();
            if (propertyInformation.dir == storm::OptimizationDirection::Minimize) {
                for (auto const& valueList : valueBounds.upper) {
                    result = std::min(result, belief.template getWeightedSum<PomdpValueType>(valueList));
                }
            } else {
                result = -storm::utility::infinity<PomdpValueType>();
                for (auto const& valueList : valueBounds.lower) {
                    result = std::max(result, belief.template getWeightedSum<PomdpValueType>(valueList));
                }
            }
            if constexpr (std::is_same_v<PomdpValueType, BeliefMdpValueType>) {
                return result;
            } else {
                return storm::utility::convertNumber<BeliefMdpValueType>(result);
            }
        };
        return buildBeliefMdpWithImplicitCutoffs(info, propertyInformation, computeCutOffValue);
    } else {
        std::function<std::unordered_map<std::string, BeliefMdpValueType>(BeliefType const&)> computeCutOffValueMap =
            [&valueBounds, &propertyInformation](BeliefType const& belief) {
                // TODO: extend with different sources for cut-offs
                uint64_t const nrCutoffPolicies =
                    propertyInformation.dir == storm::OptimizationDirection::Minimize ? valueBounds.upper.size() : valueBounds.lower.size();
                std::unordered_map<std::string, BeliefMdpValueType> result;
                for (uint64_t i = 0; i < nrCutoffPolicies; ++i) {
                    auto val = belief.template getWeightedSum<PomdpValueType>(
                        propertyInformation.dir == storm::OptimizationDirection::Minimize ? valueBounds.upper.at(i) : valueBounds.lower.at(i));
                    if constexpr (std::is_same_v<PomdpValueType, BeliefMdpValueType>) {
                        result["__sched_" + std::to_string(i)] = val;
                    } else {
                        result["__sched_" + std::to_string(i)] = storm::utility::convertNumber<BeliefMdpValueType>(val);
                    }
                }
                return result;
            };
        return buildBeliefMdp(info, propertyInformation, computeCutOffValueMap);
    }
}

template<typename PomdpModelType, typename BeliefType, typename BeliefMdpValueType, typename AbstractionType,
         typename InfoType = StandardExplorationInformation<BeliefMdpValueType, BeliefType>>
std::pair<BeliefMdpValueType, bool> checkUnfoldOrDiscretize(storm::Environment const& env, PomdpModelType const& pomdp,
                                                            PropertyInformation const& propertyInformation,
                                                            storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<BeliefMdpValueType> const& options,
                                                            storage::BeliefExplorationBounds<typename PomdpModelType::ValueType> const& valueBounds,
                                                            storm::OptionalRef<AbstractionType> abstraction = {}) {
    STORM_LOG_ASSERT(propertyInformation.kind == PropertyInformation::Kind::ReachabilityProbability ||
                         propertyInformation.kind == PropertyInformation::Kind::ExpectedTotalReachabilityReward,
                     "Unexpected kind of property.");

    STORM_PRINT_AND_LOG("Exploring the belief space...\n");

    // First, explore the beliefs and its successors
    using BeliefExplorationType = BeliefExploration<BeliefMdpValueType, PomdpModelType, BeliefType>;
    storm::utility::Stopwatch swExplore(true);
    BeliefExplorationType exploration(pomdp);

    auto info = exploration.template initializeExploration<InfoType>(pomdp.getNrObservations(), options.explorationQueueOrder);
    info.generateChoiceLabeling = options.buildChoiceLabeling;

    // Determine terminationCallback based on options
    typename BeliefExplorationType::TerminationCallback terminationCallback =
        getTerminationCallback<PomdpModelType, BeliefType, BeliefMdpValueType>(options, info, swExplore);

    // Determine terminalBeliefCallback based on options
    typename BeliefExplorationType::TerminalBeliefCallback terminalBeliefCallback =
        getTerminalBeliefCallback<PomdpModelType, BeliefType, BeliefMdpValueType>(propertyInformation, options, *valueBounds.preprocessingBounds);

    if constexpr (std::is_same_v<InfoType, ClippingExplorationInformation<BeliefMdpValueType, BeliefType>>) {
        STORM_LOG_ASSERT(options.useClipping, "Clipping exploration information requires clipping to be enabled.");
        if (propertyInformation.kind == PropertyInformation::Kind::ExpectedTotalReachabilityReward) {
            exploration.resumeClippingExploration(info, terminalBeliefCallback, terminationCallback, propertyInformation.rewardModelName.value(), abstraction);
        } else {
            exploration.resumeClippingExploration(info, terminalBeliefCallback, terminationCallback, storm::NullRef, abstraction);
        }
        STORM_LOG_TRACE("Starting clipping phase.");
        STORM_LOG_ASSERT(options.clippingResolutions.has_value(), "Clipping requested, but no resolution vector given.");
        std::vector<uint64_t> resolutions(options.clippingResolutions.value());
        if (propertyInformation.kind == PropertyInformation::Kind::ExpectedTotalReachabilityReward) {
            STORM_LOG_ASSERT(valueBounds.extremeBounds.has_value(),
                             "Clipping for expected total reachability reward requires extreme value bounds to be given.");
            ClippingBeliefAbstraction<BeliefType> clippingAbstraction(
                std::move(resolutions), std::move(valueBounds.extremeBounds->template copyValues<typename BeliefType::ValueType>()),
                std::move(storm::storage::BitVector(valueBounds.extremeBounds->isInfinite)));
            exploration.resumeClippingExploration(
                info, terminalBeliefCallback, []() { return false; }, propertyInformation.rewardModelName.value(), storm::OptionalRef(clippingAbstraction));
        } else {
            ClippingBeliefAbstraction<BeliefType> clippingAbstraction(std::move(resolutions));
            exploration.resumeClippingExploration(
                info, terminalBeliefCallback, []() { return false; }, storm::NullRef, storm::OptionalRef(clippingAbstraction));
        }
        STORM_LOG_TRACE("Finished clipping phase.");
    } else {
        if (propertyInformation.kind == PropertyInformation::Kind::ExpectedTotalReachabilityReward) {
            exploration.resumeExploration(info, terminalBeliefCallback, terminationCallback, propertyInformation.rewardModelName.value(), abstraction);
        } else {
            exploration.resumeExploration(info, terminalBeliefCallback, terminationCallback, storm::NullRef, abstraction);
        }
    }
    swExplore.stop();
    bool earlyExplorationStop = info.queue.hasNext();
    if (earlyExplorationStop) {
        STORM_PRINT_AND_LOG("Exploration stopped before all beliefs were explored. " << info.discoveredBeliefs.getNumberOfBeliefIds() << " beliefs discovered. "
                                                                                     << info.exploredBeliefs.size() << " beliefs explored.\n");
    }

    // Second, build the Belief MDP from the exploration information
    STORM_PRINT_AND_LOG("Constructing the belief MDP...\n");
    storm::utility::Stopwatch swBuild(true);
    auto [beliefMdp, stateToBeliefMap] =
        buildBeliefMdpFromInfo<PomdpModelType, BeliefType, BeliefMdpValueType, InfoType>(propertyInformation, options, *valueBounds.preprocessingBounds, info);
    swBuild.stop();
    beliefMdp->printModelInformationToStream(std::cout);

    // Finally, perform model checking on the belief MDP.
    storm::utility::Stopwatch swCheck(true);
    auto formula = createFormulaForBeliefMdp(propertyInformation);
    storm::modelchecker::CheckTask<storm::logic::Formula, BeliefMdpValueType> task(*formula, true);
    task.setProduceSchedulers(options.generatePolicy);
    std::unique_ptr<storm::modelchecker::CheckResult> res(storm::api::verifyWithSparseEngine<BeliefMdpValueType>(env, beliefMdp, task));
    swCheck.stop();
    STORM_PRINT_AND_LOG("Time for exploring beliefs: " << swExplore << ".\n");
    STORM_PRINT_AND_LOG("Time for building the belief MDP: " << swBuild << ".\n");
    STORM_PRINT_AND_LOG("Time for analyzing the belief MDP: " << swCheck << ".\n");
    STORM_LOG_ASSERT(res, "Model checking of belief MDP did not return any result.");
    STORM_LOG_ASSERT(res->isExplicitQuantitativeCheckResult(), "Model checking of belief MDP did not return result of expected type.");
    STORM_LOG_ASSERT(beliefMdp->getInitialStates().getNumberOfSetBits() == 1, "Unexpected number of initial states for belief Mdp.");
    auto const initState = beliefMdp->getInitialStates().getNextSetIndex(0);
    if (options.generatePolicy) {
        STORM_LOG_ASSERT(res->asQuantitativeCheckResult<BeliefMdpValueType>().hasScheduler() && options.buildChoiceLabeling,
                         "Model checking of belief MDP did not return a policy.");
        std::unordered_map<uint64_t, BeliefObservationType> beliefStateObservationMap;
        for (const auto& [state, beliefId] : stateToBeliefMap) {
            beliefStateObservationMap[state] = info.discoveredBeliefs.getBeliefFromId(beliefId).observation();
        }
        policy::PolicyExtractor<PomdpModelType, typename BeliefType::ValueType, BeliefMdpValueType> policyExtractor(
            pomdp, *beliefMdp, beliefStateObservationMap, res->asExplicitQuantitativeCheckResult<BeliefMdpValueType>().getScheduler(),
            propertyInformation.dir == storm::OptimizationDirection::Minimize ? valueBounds.preprocessingBounds->upperSchedulers
                                                                              : valueBounds.preprocessingBounds->lowerSchedulers);
        auto fsc = policyExtractor.exportPolicyAsFiniteStateController();
        // STORM_PRINT(fsc.toString() << "\n");
        storm::pomdp::policy::applyObservationBasedFSCToPomdp(pomdp, policyExtractor.exportPolicyAsFiniteStateController(), true);
    }
    return {res->asExplicitQuantitativeCheckResult<BeliefMdpValueType>()[initState], !earlyExplorationStop};
}

template<typename PomdpModelType, typename BeliefType, typename BeliefMdpValueType>
std::pair<BeliefMdpValueType, bool> checkRewardAwareUnfoldOrDiscretize(
    storm::Environment const& env, PomdpModelType const& pomdp, PropertyInformation const& propertyInformation,
    storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<BeliefMdpValueType> const& options,
    storage::BeliefExplorationBounds<typename PomdpModelType::ValueType> const& valueBounds,
    RewardBoundedBeliefSplitter<BeliefMdpValueType, PomdpModelType, BeliefType>& rewardSplitter,
    storm::OptionalRef<FreudenthalTriangulationBeliefAbstraction<BeliefType>> abstraction = {}) {
    STORM_LOG_ASSERT(propertyInformation.kind == PropertyInformation::Kind::RewardBoundedReachabilityProbability, "Unexpected kind of property.");
    STORM_LOG_ASSERT(rewardSplitter.getNumberOfSetRewardModels() != 0, "rewardSplitter must have a reward model set for reward-aware belief MDP construction.");

    STORM_PRINT_AND_LOG("Exploring the belief space...\n");

    // First, explore the beliefs and its successors
    using BeliefExplorationType = BeliefExploration<BeliefMdpValueType, PomdpModelType, BeliefType>;
    storm::utility::Stopwatch swExplore(true);
    BeliefExplorationType exploration(pomdp);
    using InfoType = RewardAwareExplorationInformation<BeliefMdpValueType, BeliefType>;
    auto info = exploration.template initializeExploration<InfoType>(pomdp.getNrObservations(), options.explorationQueueOrder);

    // Determine terminationCallback based on options
    typename BeliefExplorationType::TerminationCallback terminationCallback =
        getTerminationCallback<PomdpModelType, BeliefType, BeliefMdpValueType>(options, info, swExplore);

    // Determine terminalBeliefCallback based on options
    typename BeliefExplorationType::TerminalBeliefCallback terminalBeliefCallback =
        getTerminalBeliefCallback<PomdpModelType, BeliefType, BeliefMdpValueType>(propertyInformation, options, *valueBounds.preprocessingBounds);

    exploration.resumeRewardAwareExploration(info, terminalBeliefCallback, terminationCallback, rewardSplitter, abstraction);
    swExplore.stop();
    bool const earlyExplorationStop = info.queue.hasNext();
    if (earlyExplorationStop) {
        STORM_PRINT_AND_LOG("Exploration stopped before all beliefs were explored. " << info.discoveredBeliefs.getNumberOfBeliefIds() << " beliefs discovered. "
                                                                                     << info.exploredBeliefs.size() << " beliefs explored.\n");
    }

    // Second, build the Belief MDP from the exploration information
    STORM_PRINT_AND_LOG("Constructing the belief MDP...\n");
    storm::utility::Stopwatch swBuild(true);
    auto [beliefMdp, stateToBeliefMap] =
        buildBeliefMdpFromInfo<PomdpModelType, BeliefType, BeliefMdpValueType, InfoType>(propertyInformation, options, *valueBounds.preprocessingBounds, info);
    swBuild.stop();
    beliefMdp->printModelInformationToStream(std::cout);

    // Finally, perform model checking on the belief MDP.
    auto formula = createFormulaForBeliefMdp(propertyInformation);
    STORM_PRINT_AND_LOG("Analyzing property '" << *formula << "' on the belief MDP...\n");
    storm::utility::Stopwatch swCheck(true);
    std::shared_ptr<storm::models::sparse::Mdp<BeliefMdpValueType>> processedMdp = beliefMdp;
    if (propertyInformation.kind == PropertyInformation::Kind::RewardBoundedReachabilityProbability) {
        std::vector<std::string> rewardModelNames;
        for (auto const& bnd : propertyInformation.rewardBounds) {
            rewardModelNames.push_back(bnd.rewardModelName);
        }
        processedMdp = storm::transformer::transformTransitionToActionRewards<BeliefMdpValueType>(processedMdp, rewardModelNames)
                           .model->template as<storm::models::sparse::Mdp<BeliefMdpValueType>>();
        double increase = static_cast<double>(processedMdp->getNumberOfStates()) / static_cast<double>(beliefMdp->getNumberOfStates());
        STORM_PRINT_AND_LOG("Transformation of transition rewards resulted in a model with " << processedMdp->getNumberOfStates() << " states. " << increase
                                                                                             << " times more states than the original belief MDP.\n");

        // Cut away states that can not reach the target
        auto targetStates = processedMdp->getStateLabeling().getStates("target");
        storm::storage::BitVector allStates(targetStates.size(), true);
        storm::storage::BitVector probGreaterZeroStates;
        if (storm::solver::maximize(propertyInformation.dir)) {
            probGreaterZeroStates = storm::utility::graph::performProbGreater0E(processedMdp->getBackwardTransitions(), allStates, targetStates);
        } else {
            probGreaterZeroStates =
                storm::utility::graph::performProbGreater0A(processedMdp->getTransitionMatrix(), processedMdp->getTransitionMatrix().getRowGroupIndices(),
                                                            processedMdp->getBackwardTransitions(), allStates, targetStates);
        }
        auto mergingResult = storm::transformer::GoalStateMerger(*processedMdp)
                                 .mergeTargetAndSinkStates(probGreaterZeroStates, ~probGreaterZeroStates, ~allStates, rewardModelNames);
        processedMdp = mergingResult.model;
        STORM_PRINT_AND_LOG("Merging of sink states resulted in a model with " << processedMdp->getNumberOfStates() << " states.\n");
    }
    storm::modelchecker::CheckTask<storm::logic::Formula, BeliefMdpValueType> task(*formula, true);
    std::unique_ptr<storm::modelchecker::CheckResult> res(storm::api::verifyWithSparseEngine<BeliefMdpValueType>(env, processedMdp, task));
    swCheck.stop();
    STORM_PRINT_AND_LOG("Time for exploring beliefs: " << swExplore << ".\n");
    STORM_PRINT_AND_LOG("Time for building the belief MDP: " << swBuild << ".\n");
    STORM_PRINT_AND_LOG("Time for analyzing the belief MDP: " << swCheck << ".\n");
    STORM_LOG_ASSERT(res, "Model checking of belief MDP did not return any result.");
    STORM_LOG_ASSERT(res->isExplicitQuantitativeCheckResult(), "Model checking of belief MDP did not return result of expected type.");
    STORM_LOG_ASSERT(processedMdp->getInitialStates().getNumberOfSetBits() == 1, "Unexpected number of initial states for (processed) belief Mdp.");
    auto const initState = processedMdp->getInitialStates().getNextSetIndex(0);
    return {res->asExplicitQuantitativeCheckResult<BeliefMdpValueType>()[initState], !earlyExplorationStop};
}

template<typename PomdpModelType, typename BeliefValueType, typename BeliefMdpValueType>
std::pair<BeliefMdpValueType, bool> BeliefBasedModelChecker<PomdpModelType, BeliefValueType, BeliefMdpValueType>::checkUnfold(
    storm::Environment const& env, PropertyInformation const& propertyInformation,
    storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<BeliefMdpValueType> const& options,
    storm::pomdp::storage::BeliefExplorationBounds<typename PomdpModelType::ValueType> const& valueBounds) {
    if (options.useClipping) {
        return checkUnfoldOrDiscretize<PomdpModelType, Belief<BeliefValueType>, BeliefMdpValueType, NoAbstractionType,
                                       ClippingExplorationInformation<BeliefMdpValueType, Belief<BeliefValueType>>>(env, inputPomdp, propertyInformation,
                                                                                                                    options, valueBounds);
    } else {
        return checkUnfoldOrDiscretize<PomdpModelType, Belief<BeliefValueType>, BeliefMdpValueType, NoAbstractionType>(env, inputPomdp, propertyInformation,
                                                                                                                       options, valueBounds);
    }
}

template<typename PomdpModelType, typename BeliefValueType, typename BeliefMdpValueType>
std::pair<BeliefMdpValueType, bool> BeliefBasedModelChecker<PomdpModelType, BeliefValueType, BeliefMdpValueType>::checkDiscretize(
    storm::Environment const& env, PropertyInformation const& propertyInformation,
    storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<BeliefMdpValueType> const& options, uint64_t resolution, bool useDynamic,
    storage::BeliefExplorationBounds<typename PomdpModelType::ValueType> const& valueBounds) {
    auto mode = useDynamic ? FreudenthalTriangulationMode::Dynamic : FreudenthalTriangulationMode::Static;
    FreudenthalTriangulationBeliefAbstraction<Belief<BeliefValueType>> abstraction(storm::utility::convertNumber<BeliefValueType>(resolution), mode);
    return checkUnfoldOrDiscretize<PomdpModelType, Belief<BeliefValueType>, BeliefMdpValueType>(env, inputPomdp, propertyInformation, options, valueBounds,
                                                                                                storm::OptionalRef(abstraction));
}

template<typename PomdpModelType, typename BeliefValueType, typename BeliefMdpValueType>
std::pair<BeliefMdpValueType, bool> BeliefBasedModelChecker<PomdpModelType, BeliefValueType, BeliefMdpValueType>::checkRewardAwareUnfold(
    storm::Environment const& env, PropertyInformation const& propertyInformation,
    storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<BeliefMdpValueType> const& options,
    storage::BeliefExplorationBounds<typename PomdpModelType::ValueType> const& valueBounds, std::vector<std::string> const& relevantRewardModelNames) {
    RewardBoundedBeliefSplitter<BeliefMdpValueType, PomdpModelType, Belief<BeliefValueType>> rewardBoundedBeliefSplitter(inputPomdp);
    if (relevantRewardModelNames.empty()) {
        rewardBoundedBeliefSplitter.setRewardModel();
    } else {
        rewardBoundedBeliefSplitter.setRewardModels(relevantRewardModelNames);
    }
    return checkRewardAwareUnfoldOrDiscretize<PomdpModelType, Belief<BeliefValueType>, BeliefMdpValueType>(env, inputPomdp, propertyInformation, options,
                                                                                                           valueBounds, rewardBoundedBeliefSplitter, {});
}

template<typename PomdpModelType, typename BeliefValueType, typename BeliefMdpValueType>
std::pair<BeliefMdpValueType, bool> BeliefBasedModelChecker<PomdpModelType, BeliefValueType, BeliefMdpValueType>::checkRewardAwareDiscretize(
    storm::Environment const& env, PropertyInformation const& propertyInformation,
    storm::pomdp::beliefs::BeliefBasedModelCheckerOptions<BeliefMdpValueType> const& options, uint64_t resolution, bool useDynamic,
    storage::BeliefExplorationBounds<typename PomdpModelType::ValueType> const& valueBounds, std::vector<std::string> const& relevantRewardModelNames) {
    auto mode = useDynamic ? FreudenthalTriangulationMode::Dynamic : FreudenthalTriangulationMode::Static;
    FreudenthalTriangulationBeliefAbstraction<Belief<BeliefValueType>> abstraction(storm::utility::convertNumber<BeliefValueType>(resolution), mode);
    RewardBoundedBeliefSplitter<BeliefMdpValueType, PomdpModelType, Belief<BeliefValueType>> rewardBoundedBeliefSplitter(inputPomdp);
    if (relevantRewardModelNames.empty()) {
        rewardBoundedBeliefSplitter.setRewardModel();
    } else {
        rewardBoundedBeliefSplitter.setRewardModels(relevantRewardModelNames);
    }
    return checkRewardAwareUnfoldOrDiscretize<PomdpModelType, Belief<BeliefValueType>, BeliefMdpValueType>(
        env, inputPomdp, propertyInformation, options, valueBounds, rewardBoundedBeliefSplitter, abstraction);
}

// TODO: Check which instantiations are actually necessary / reasonable.
template class BeliefBasedModelChecker<storm::models::sparse::Pomdp<double>, double, double>;
template class BeliefBasedModelChecker<storm::models::sparse::Pomdp<double>, storm::RationalNumber, double>;
template class BeliefBasedModelChecker<storm::models::sparse::Pomdp<storm::RationalNumber>, storm::RationalNumber, storm::RationalNumber>;
template class BeliefBasedModelChecker<storm::models::sparse::Pomdp<storm::RationalNumber>, storm::RationalNumber, double>;
template class BeliefBasedModelChecker<storm::models::sparse::Pomdp<double>, double, storm::RationalNumber>;
// Currently we don't consider this combination as having rational numbers in models, but floats in beliefs does not really help us
//  template class BeliefBasedModelChecker<storm::models::sparse::Pomdp<storm::RationalNumber>, double, storm::RationalNumber>;
}  // namespace storm::pomdp::beliefs
