 #include "src/storm/modelchecker/multiobjective/pcaa/SparsePcaaPreprocessor.h"

#include "src/storm/models/sparse/Mdp.h"
#include "src/storm/models/sparse/MarkovAutomaton.h"
#include "src/storm/models/sparse/StandardRewardModel.h"
#include "src/storm/modelchecker/propositional/SparsePropositionalModelChecker.h"
#include "src/storm/modelchecker/results/ExplicitQualitativeCheckResult.h"
#include "src/storm/storage/MaximalEndComponentDecomposition.h"
#include "src/storm/transformer/StateDuplicator.h"
#include "src/storm/transformer/SubsystemBuilder.h"
#include "src/storm/utility/macros.h"
#include "src/storm/utility/vector.h"
#include "src/storm/utility/graph.h"

#include "src/storm/exceptions/InvalidPropertyException.h"
#include "src/storm/exceptions/UnexpectedException.h"

namespace storm {
    namespace modelchecker {
        namespace multiobjective {
                
            template<typename SparseModelType>
            typename SparsePcaaPreprocessor<SparseModelType>::ReturnType SparsePcaaPreprocessor<SparseModelType>::preprocess(SparseModelType const& originalModel, storm::logic::MultiObjectiveFormula const& originalFormula) {
                
                ReturnType result(originalFormula, originalModel, SparseModelType(originalModel));
                result.newToOldStateIndexMapping = storm::utility::vector::buildVectorForRange(0, originalModel.getNumberOfStates());
                
                //Invoke preprocessing on the individual objectives
                for(auto const& subFormula : originalFormula.getSubformulas()){
                    STORM_LOG_DEBUG("Preprocessing objective " << *subFormula<< ".");
                    result.objectives.emplace_back();
                    PcaaObjective<ValueType>& currentObjective = result.objectives.back();
                    currentObjective.originalFormula = subFormula;
                    if(currentObjective.originalFormula->isOperatorFormula()) {
                        preprocessFormula(currentObjective.originalFormula->asOperatorFormula(), result, currentObjective);
                    } else {
                        STORM_LOG_THROW(false, storm::exceptions::InvalidPropertyException, "Could not preprocess the subformula " << *subFormula << " of " << originalFormula << " because it is not supported");
                    }
                }
                
                // Set the query type. In case of a quantitative query, also set the index of the objective to be optimized.
                // Note: If there are only zero (or one) objectives left, we should not consider a pareto query!
                storm::storage::BitVector objectivesWithoutThreshold(result.objectives.size());
                for(uint_fast64_t objIndex = 0; objIndex < result.objectives.size(); ++objIndex) {
                    objectivesWithoutThreshold.set(objIndex, !result.objectives[objIndex].threshold);
                }
                uint_fast64_t numOfObjectivesWithoutThreshold = objectivesWithoutThreshold.getNumberOfSetBits();
                if(numOfObjectivesWithoutThreshold == 0) {
                    result.queryType = ReturnType::QueryType::Achievability;
                } else if (numOfObjectivesWithoutThreshold == 1) {
                    result.queryType = ReturnType::QueryType::Quantitative;
                    result.indexOfOptimizingObjective = objectivesWithoutThreshold.getNextSetIndex(0);
                } else if (numOfObjectivesWithoutThreshold == result.objectives.size()) {
                    result.queryType = ReturnType::QueryType::Pareto;
                } else {
                    STORM_LOG_THROW(false, storm::exceptions::UnexpectedException, "The number of objectives without threshold is not valid. It should be either 0 (achievability query), 1 (quantitative query), or " << result.objectives.size() << " (Pareto Query). Got " << numOfObjectivesWithoutThreshold << " instead.");
                }
                
                //We can remove the original reward models to save some memory
                std::set<std::string> origRewardModels = originalFormula.getReferencedRewardModels();
                for (auto const& rewModel : origRewardModels){
                    result.preprocessedModel.removeRewardModel(rewModel);
                }
                
                //Get actions to which a positive or negative reward is assigned for at least one objective
                result.actionsWithPositiveReward = storm::storage::BitVector(result.preprocessedModel.getNumberOfChoices(), false);
                result.actionsWithNegativeReward = storm::storage::BitVector(result.preprocessedModel.getNumberOfChoices(), false);
                for(uint_fast64_t objIndex = 0; objIndex < result.objectives.size(); ++objIndex) {
                    if(!result.objectives[objIndex].upperTimeBound) {
                        if(result.objectives[objIndex].rewardsArePositive) {
                            result.actionsWithPositiveReward |= ~storm::utility::vector::filterZero(result.preprocessedModel.getRewardModel(result.objectives[objIndex].rewardModelName).getTotalRewardVector(result.preprocessedModel.getTransitionMatrix()));
                        } else {
                            result.actionsWithNegativeReward |= ~storm::utility::vector::filterZero(result.preprocessedModel.getRewardModel(result.objectives[objIndex].rewardModelName).getTotalRewardVector(result.preprocessedModel.getTransitionMatrix()));
                        }
                    }
                }
                
                auto backwardTransitions = result.preprocessedModel.getBackwardTransitions();
                analyzeEndComponents(result, backwardTransitions);
                ensureRewardFiniteness(result, backwardTransitions);
                
                return result;
            }
            
            template<typename SparseModelType>
            void SparsePcaaPreprocessor<SparseModelType>::updatePreprocessedModel(ReturnType& result, SparseModelType& newPreprocessedModel, std::vector<uint_fast64_t>& newToOldStateIndexMapping) {
                result.preprocessedModel = std::move(newPreprocessedModel);
                // the given newToOldStateIndexMapping reffers to the indices of the former preprocessedModel as 'old indices'
                for(auto & preprocessedModelStateIndex : newToOldStateIndexMapping){
                    preprocessedModelStateIndex = result.newToOldStateIndexMapping[preprocessedModelStateIndex];
                }
                result.newToOldStateIndexMapping = std::move(newToOldStateIndexMapping);
            }
            
            template<typename SparseModelType>
            void SparsePcaaPreprocessor<SparseModelType>::preprocessFormula(storm::logic::OperatorFormula const& formula, ReturnType& result, PcaaObjective<ValueType>& currentObjective) {
                
                // Get a unique name for the new reward model.
                currentObjective.rewardModelName = "objective" + std::to_string(result.objectives.size());
                while(result.preprocessedModel.hasRewardModel(currentObjective.rewardModelName)){
                    currentObjective.rewardModelName = "_" + currentObjective.rewardModelName;
                }
                
                currentObjective.toOriginalValueTransformationFactor = storm::utility::one<ValueType>();
                currentObjective.toOriginalValueTransformationOffset = storm::utility::zero<ValueType>();
                currentObjective.rewardsArePositive = true;
                
                bool formulaMinimizes = false;
                if(formula.hasBound()) {
                    currentObjective.threshold = storm::utility::convertNumber<ValueType>(formula.getBound().threshold);
                    currentObjective.thresholdIsStrict = storm::logic::isStrict(formula.getBound().comparisonType);
                    //Note that we minimize for upper bounds since we are looking for the EXISTENCE of a satisfying scheduler
                    formulaMinimizes = !storm::logic::isLowerBound(formula.getBound().comparisonType);
                } else if (formula.hasOptimalityType()){
                    formulaMinimizes = storm::solver::minimize(formula.getOptimalityType());
                } else {
                    STORM_LOG_THROW(false, storm::exceptions::InvalidPropertyException, "Current objective " << formula << " does not specify whether to minimize or maximize");
                }
                if(formulaMinimizes) {
                    // We negate all the values so we can consider the maximum for this objective
                    // Thus, all objectives will be maximized.
                    currentObjective.rewardsArePositive = false;
                    currentObjective.toOriginalValueTransformationFactor = -storm::utility::one<ValueType>();
                }
                
                if(formula.isProbabilityOperatorFormula()){
                    preprocessFormula(formula.asProbabilityOperatorFormula(), result, currentObjective);
                } else if(formula.isRewardOperatorFormula()){
                    preprocessFormula(formula.asRewardOperatorFormula(), result, currentObjective);
                } else if(formula.isTimeOperatorFormula()){
                    preprocessFormula(formula.asTimeOperatorFormula(), result, currentObjective);
                } else {
                    STORM_LOG_THROW(false, storm::exceptions::InvalidPropertyException, "Could not preprocess the objective " << formula << " because it is not supported");
                }
                
                // Transform the threshold for the preprocessed Model
                if(currentObjective.threshold) {
                    currentObjective.threshold = (currentObjective.threshold.get() - currentObjective.toOriginalValueTransformationOffset) / currentObjective.toOriginalValueTransformationFactor;
                }
            }
            
            template<typename SparseModelType>
            void SparsePcaaPreprocessor<SparseModelType>::preprocessFormula(storm::logic::ProbabilityOperatorFormula const& formula, ReturnType& result, PcaaObjective<ValueType>& currentObjective) {
                
                if(formula.getSubformula().isUntilFormula()){
                    preprocessFormula(formula.getSubformula().asUntilFormula(), result, currentObjective);
                } else if(formula.getSubformula().isBoundedUntilFormula()){
                    preprocessFormula(formula.getSubformula().asBoundedUntilFormula(), result, currentObjective);
                } else if(formula.getSubformula().isGloballyFormula()){
                    preprocessFormula(formula.getSubformula().asGloballyFormula(), result, currentObjective);
                } else if(formula.getSubformula().isEventuallyFormula()){
                    preprocessFormula(formula.getSubformula().asEventuallyFormula(), result, currentObjective);
                } else {
                    STORM_LOG_THROW(false, storm::exceptions::InvalidPropertyException, "The subformula of " << formula << " is not supported.");
                }
            }

            template<typename SparseModelType>
            void SparsePcaaPreprocessor<SparseModelType>::preprocessFormula(storm::logic::RewardOperatorFormula const& formula, ReturnType& result, PcaaObjective<ValueType>& currentObjective) {
                // Check if the reward model is uniquely specified
                STORM_LOG_THROW((formula.hasRewardModelName() && result.preprocessedModel.hasRewardModel(formula.getRewardModelName()))
                                || result.preprocessedModel.hasUniqueRewardModel(), storm::exceptions::InvalidPropertyException, "The reward model is not unique and the formula " << formula << " does not specify a reward model.");
                
                if(formula.getSubformula().isEventuallyFormula()){
                    preprocessFormula(formula.getSubformula().asEventuallyFormula(), result, currentObjective, formula.getOptionalRewardModelName());
                } else if(formula.getSubformula().isCumulativeRewardFormula()) {
                    preprocessFormula(formula.getSubformula().asCumulativeRewardFormula(), result, currentObjective, formula.getOptionalRewardModelName());
                } else if(formula.getSubformula().isTotalRewardFormula()) {
                    preprocessFormula(formula.getSubformula().asTotalRewardFormula(), result, currentObjective, formula.getOptionalRewardModelName());
                } else {
                    STORM_LOG_THROW(false, storm::exceptions::InvalidPropertyException, "The subformula of " << formula << " is not supported.");
                }
            }

            template<typename SparseModelType>
            void SparsePcaaPreprocessor<SparseModelType>::preprocessFormula(storm::logic::TimeOperatorFormula const& formula, ReturnType& result, PcaaObjective<ValueType>& currentObjective) {
                // Time formulas are only supported for Markov automata
                STORM_LOG_THROW(result.originalModel.isOfType(storm::models::ModelType::MarkovAutomaton), storm::exceptions::InvalidPropertyException, "Time operator formulas are only supported for Markov automata.");
                
                if(formula.getSubformula().isEventuallyFormula()){
                    preprocessFormula(formula.getSubformula().asEventuallyFormula(), result, currentObjective);
                } else {
                    STORM_LOG_THROW(false, storm::exceptions::InvalidPropertyException, "The subformula of " << formula << " is not supported.");
                }
            }
            
            template<typename SparseModelType>
            void SparsePcaaPreprocessor<SparseModelType>::preprocessFormula(storm::logic::UntilFormula const& formula, ReturnType& result, PcaaObjective<ValueType>& currentObjective) {
                CheckTask<storm::logic::Formula, ValueType> phiTask(formula.getLeftSubformula());
                CheckTask<storm::logic::Formula, ValueType> psiTask(formula.getRightSubformula());
                storm::modelchecker::SparsePropositionalModelChecker<SparseModelType> mc(result.preprocessedModel);
                STORM_LOG_THROW(mc.canHandle(phiTask) && mc.canHandle(psiTask), storm::exceptions::InvalidPropertyException, "The subformulas of " << formula << " should be propositional.");
                storm::storage::BitVector phiStates = mc.check(phiTask)->asExplicitQualitativeCheckResult().getTruthValuesVector();
                storm::storage::BitVector psiStates = mc.check(psiTask)->asExplicitQualitativeCheckResult().getTruthValuesVector();
                
                if(!(psiStates & result.preprocessedModel.getInitialStates()).empty() && !currentObjective.lowerTimeBound) {
                    // The probability is always one as the initial state is a target state.
                    // For this special case, the transformation to an expected reward objective fails.
                    // We could handle this with further preprocessing steps but as this case is boring anyway, we simply reject the input.
                    STORM_LOG_THROW(false, storm::exceptions::InvalidPropertyException, "The Probability for the objective " << currentObjective.originalFormula << " is always one as the rhs of the until formula is true in the initial state. Please omit this objective.");
                }
                
                auto duplicatorResult = storm::transformer::StateDuplicator<SparseModelType>::transform(result.preprocessedModel, ~phiStates | psiStates);
                updatePreprocessedModel(result, *duplicatorResult.model, duplicatorResult.newToOldStateIndexMapping);
                
                storm::storage::BitVector newPsiStates(result.preprocessedModel.getNumberOfStates(), false);
                for(auto const& oldPsiState : psiStates){
                    //note that psiStates are always located in the second copy
                    newPsiStates.set(duplicatorResult.secondCopyOldToNewStateIndexMapping[oldPsiState], true);
                }
                
                // build stateAction reward vector that gives (one*transitionProbability) reward whenever a transition leads from the firstCopy to a psiState
                std::vector<ValueType> objectiveRewards(result.preprocessedModel.getTransitionMatrix().getRowCount(), storm::utility::zero<ValueType>());
                for (auto const& firstCopyState : duplicatorResult.firstCopy) {
                    for (uint_fast64_t row = result.preprocessedModel.getTransitionMatrix().getRowGroupIndices()[firstCopyState]; row < result.preprocessedModel.getTransitionMatrix().getRowGroupIndices()[firstCopyState + 1]; ++row) {
                        objectiveRewards[row] = result.preprocessedModel.getTransitionMatrix().getConstrainedRowSum(row, newPsiStates);
                    }
                }
                if(!currentObjective.rewardsArePositive) {
                    storm::utility::vector::scaleVectorInPlace(objectiveRewards, -storm::utility::one<ValueType>());
                }
                result.preprocessedModel.addRewardModel(currentObjective.rewardModelName, RewardModelType(boost::none, objectiveRewards));
            }
            
            template<typename SparseModelType>
            void SparsePcaaPreprocessor<SparseModelType>::preprocessFormula(storm::logic::BoundedUntilFormula const& formula, ReturnType& result, PcaaObjective<ValueType>& currentObjective) {
                
                if(formula.hasDiscreteTimeBound()) {
                    currentObjective.upperTimeBound = storm::utility::convertNumber<ValueType>(formula.getDiscreteTimeBound());
                } else {
                    if(result.originalModel.isOfType(storm::models::ModelType::Mdp)) {
                        STORM_LOG_THROW(formula.getIntervalBounds().first == std::round(formula.getIntervalBounds().first), storm::exceptions::InvalidPropertyException, "Expected a boundedUntilFormula with discrete lower time bound but got " << formula << ".");
                        STORM_LOG_THROW(formula.getIntervalBounds().second == std::round(formula.getIntervalBounds().second), storm::exceptions::InvalidPropertyException, "Expected a boundedUntilFormula with discrete upper time bound but got " << formula << ".");
                    } else {
                        STORM_LOG_THROW(result.originalModel.isOfType(storm::models::ModelType::MarkovAutomaton), storm::exceptions::InvalidPropertyException, "Got a boundedUntilFormula which can not be checked for the current model type.");
                        STORM_LOG_THROW(formula.getIntervalBounds().second > formula.getIntervalBounds().first, storm::exceptions::InvalidPropertyException, "Neither empty nor point intervalls are allowed but got " << formula << ".");
                    }
                    if(!storm::utility::isZero(formula.getIntervalBounds().first)) {
                        currentObjective.lowerTimeBound = storm::utility::convertNumber<ValueType>(formula.getIntervalBounds().first);
                    }
                    currentObjective.upperTimeBound = storm::utility::convertNumber<ValueType>(formula.getIntervalBounds().second);
                }
                preprocessFormula(storm::logic::UntilFormula(formula.getLeftSubformula().asSharedPointer(), formula.getRightSubformula().asSharedPointer()), result, currentObjective);
            }
            
            template<typename SparseModelType>
            void SparsePcaaPreprocessor<SparseModelType>::preprocessFormula(storm::logic::GloballyFormula const& formula, ReturnType& result, PcaaObjective<ValueType>& currentObjective) {
                // The formula will be transformed to an until formula for the complementary event.
                // If the original formula minimizes, the complementary one will maximize and vice versa.
                // Hence, the decision whether to consider positive or negative rewards flips.
                currentObjective.rewardsArePositive = !currentObjective.rewardsArePositive;
                // To transform from the value of the preprocessed model back to the value of the original model, we have to add 1 to the result.
                // The transformation factor has already been set correctly.
                currentObjective.toOriginalValueTransformationOffset = storm::utility::one<ValueType>();
                
                auto negatedSubformula = std::make_shared<storm::logic::UnaryBooleanStateFormula>(storm::logic::UnaryBooleanStateFormula::OperatorType::Not, formula.getSubformula().asSharedPointer());
                
                preprocessFormula(storm::logic::UntilFormula(storm::logic::Formula::getTrueFormula(), negatedSubformula), result, currentObjective);
            }
            
            template<typename SparseModelType>
            void SparsePcaaPreprocessor<SparseModelType>::preprocessFormula(storm::logic::EventuallyFormula const& formula, ReturnType& result, PcaaObjective<ValueType>& currentObjective, boost::optional<std::string> const& optionalRewardModelName) {
                if(formula.isReachabilityProbabilityFormula()){
                    preprocessFormula(storm::logic::UntilFormula(storm::logic::Formula::getTrueFormula(), formula.getSubformula().asSharedPointer()), result, currentObjective);
                    return;
                }
                
                CheckTask<storm::logic::Formula, ValueType> targetTask(formula.getSubformula());
                storm::modelchecker::SparsePropositionalModelChecker<SparseModelType> mc(result.preprocessedModel);
                STORM_LOG_THROW(mc.canHandle(targetTask), storm::exceptions::InvalidPropertyException, "The subformula of " << formula << " should be propositional.");
                storm::storage::BitVector targetStates = mc.check(targetTask)->asExplicitQualitativeCheckResult().getTruthValuesVector();
                
                auto duplicatorResult = storm::transformer::StateDuplicator<SparseModelType>::transform(result.preprocessedModel, targetStates);
                updatePreprocessedModel(result, *duplicatorResult.model, duplicatorResult.newToOldStateIndexMapping);
                
                // Add a reward model that gives zero reward to the actions of states of the second copy.
                RewardModelType objectiveRewards(boost::none);
                if(formula.isReachabilityRewardFormula()) {
                    objectiveRewards = result.preprocessedModel.getRewardModel(optionalRewardModelName ? optionalRewardModelName.get() : "");
                    objectiveRewards.reduceToStateBasedRewards(result.preprocessedModel.getTransitionMatrix(), false);
                    if(objectiveRewards.hasStateRewards()) {
                        storm::utility::vector::setVectorValues(objectiveRewards.getStateRewardVector(), duplicatorResult.secondCopy, storm::utility::zero<ValueType>());
                    }
                    if(objectiveRewards.hasStateActionRewards()) {
                        for(auto secondCopyState : duplicatorResult.secondCopy) {
                            std::fill_n(objectiveRewards.getStateActionRewardVector().begin() + result.preprocessedModel.getTransitionMatrix().getRowGroupIndices()[secondCopyState], result.preprocessedModel.getTransitionMatrix().getRowGroupSize(secondCopyState), storm::utility::zero<ValueType>());
                        }
                    }
                } else if(formula.isReachabilityTimeFormula() && result.preprocessedModel.isOfType(storm::models::ModelType::MarkovAutomaton)) {
                    objectiveRewards = RewardModelType(std::vector<ValueType>(result.preprocessedModel.getNumberOfStates(), storm::utility::zero<ValueType>()));
                    storm::utility::vector::setVectorValues(objectiveRewards.getStateRewardVector(), dynamic_cast<storm::models::sparse::MarkovAutomaton<ValueType>*>(&result.preprocessedModel)->getMarkovianStates() & duplicatorResult.firstCopy, storm::utility::one<ValueType>());
                } else {
                    STORM_LOG_THROW(false, storm::exceptions::InvalidPropertyException, "The formula " << formula << " neither considers reachability probabilities nor reachability rewards " << (result.preprocessedModel.isOfType(storm::models::ModelType::MarkovAutomaton) ?  "nor reachability time" : "") << ". This is not supported.");
                }
                if(!currentObjective.rewardsArePositive){
                    if(objectiveRewards.hasStateRewards()) {
                        storm::utility::vector::scaleVectorInPlace(objectiveRewards.getStateRewardVector(), -storm::utility::one<ValueType>());
                    }
                    if(objectiveRewards.hasStateActionRewards()) {
                        storm::utility::vector::scaleVectorInPlace(objectiveRewards.getStateActionRewardVector(), -storm::utility::one<ValueType>());
                    }
                }
                result.preprocessedModel.addRewardModel(currentObjective.rewardModelName, std::move(objectiveRewards));
            }
            
            template<typename SparseModelType>
            void SparsePcaaPreprocessor<SparseModelType>::preprocessFormula(storm::logic::CumulativeRewardFormula const& formula, ReturnType& result, PcaaObjective<ValueType>& currentObjective, boost::optional<std::string> const& optionalRewardModelName) {
                STORM_LOG_THROW(result.originalModel.isOfType(storm::models::ModelType::Mdp), storm::exceptions::InvalidPropertyException, "Cumulative reward formulas are not supported for the given model type.");
                STORM_LOG_THROW(formula.hasDiscreteTimeBound(), storm::exceptions::InvalidPropertyException, "Expected a cumulativeRewardFormula with a discrete time bound but got " << formula << ".");
                STORM_LOG_THROW(formula.getDiscreteTimeBound()>0, storm::exceptions::InvalidPropertyException, "Expected a cumulativeRewardFormula with a positive discrete time bound but got " << formula << ".");
                currentObjective.upperTimeBound = storm::utility::convertNumber<ValueType>(formula.getDiscreteTimeBound());
                
                RewardModelType objectiveRewards = result.preprocessedModel.getRewardModel(optionalRewardModelName ? optionalRewardModelName.get() : "");
                objectiveRewards.reduceToStateBasedRewards(result.preprocessedModel.getTransitionMatrix(), false);
                if(!currentObjective.rewardsArePositive){
                    if(objectiveRewards.hasStateRewards()) {
                        storm::utility::vector::scaleVectorInPlace(objectiveRewards.getStateRewardVector(), -storm::utility::one<ValueType>());
                    }
                    if(objectiveRewards.hasStateActionRewards()) {
                        storm::utility::vector::scaleVectorInPlace(objectiveRewards.getStateActionRewardVector(), -storm::utility::one<ValueType>());
                    }
                }
                result.preprocessedModel.addRewardModel(currentObjective.rewardModelName, std::move(objectiveRewards));
            }
            
            template<typename SparseModelType>
            void SparsePcaaPreprocessor<SparseModelType>::preprocessFormula(storm::logic::TotalRewardFormula const& formula, ReturnType& result, PcaaObjective<ValueType>& currentObjective, boost::optional<std::string> const& optionalRewardModelName) {
                RewardModelType objectiveRewards = result.preprocessedModel.getRewardModel(optionalRewardModelName ? optionalRewardModelName.get() : "");
                objectiveRewards.reduceToStateBasedRewards(result.preprocessedModel.getTransitionMatrix(), false);
                if(!currentObjective.rewardsArePositive){
                    if(objectiveRewards.hasStateRewards()) {
                        storm::utility::vector::scaleVectorInPlace(objectiveRewards.getStateRewardVector(), -storm::utility::one<ValueType>());
                    }
                    if(objectiveRewards.hasStateActionRewards()) {
                        storm::utility::vector::scaleVectorInPlace(objectiveRewards.getStateActionRewardVector(), -storm::utility::one<ValueType>());
                    }
                }
                result.preprocessedModel.addRewardModel(currentObjective.rewardModelName, std::move(objectiveRewards));
            }
            
            
            template<typename SparseModelType>
            void SparsePcaaPreprocessor<SparseModelType>::analyzeEndComponents(ReturnType& result, storm::storage::SparseMatrix<ValueType> const& backwardTransitions) {
                
                result.ecActions = storm::storage::BitVector(result.preprocessedModel.getNumberOfChoices(), false);
                std::vector<storm::storage::MaximalEndComponent> ecs;
                auto mecDecomposition = storm::storage::MaximalEndComponentDecomposition<ValueType>(result.preprocessedModel.getTransitionMatrix(), backwardTransitions);
                STORM_LOG_ASSERT(!mecDecomposition.empty(), "Empty maximal end component decomposition.");
                ecs.reserve(mecDecomposition.size());
                for(auto& mec : mecDecomposition) {
                    for(auto const& stateActionsPair : mec) {
                        for(auto const& action : stateActionsPair.second) {
                            result.ecActions.set(action);
                        }
                    }
                    ecs.push_back(std::move(mec));
                }
                
                result.possiblyRecurrentStates = storm::storage::BitVector(result.preprocessedModel.getNumberOfStates(), false);
                storm::storage::BitVector neutralActions = ~(result.actionsWithNegativeReward | result.actionsWithPositiveReward);
                storm::storage::BitVector statesInCurrentECWithNeutralAction(result.preprocessedModel.getNumberOfStates(), false);
                for(uint_fast64_t ecIndex = 0; ecIndex < ecs.size(); ++ecIndex) { //we will insert new ecs in the vector (thus no iterators for the loop)
                    bool currentECIsNeutral = true;
                    for(auto const& stateActionsPair : ecs[ecIndex]) {
                        bool stateHasNeutralActionWithinEC = false;
                        for(auto const& action : stateActionsPair.second) {
                            stateHasNeutralActionWithinEC |= neutralActions.get(action);
                        }
                        statesInCurrentECWithNeutralAction.set(stateActionsPair.first, stateHasNeutralActionWithinEC);
                        currentECIsNeutral &= stateHasNeutralActionWithinEC;
                    }
                    if(currentECIsNeutral) {
                        result.possiblyRecurrentStates |= statesInCurrentECWithNeutralAction;
                    }else{
                        // Check if the ec contains neutral sub ecs. This is done by adding the subECs to our list of ECs
                        // A neutral subEC only consist of states that can stay in statesInCurrentECWithNeutralAction
                        statesInCurrentECWithNeutralAction = storm::utility::graph::performProb0E(result.preprocessedModel.getTransitionMatrix(), result.preprocessedModel.getTransitionMatrix().getRowGroupIndices(), backwardTransitions,statesInCurrentECWithNeutralAction, ~statesInCurrentECWithNeutralAction);
                        auto subECs = storm::storage::MaximalEndComponentDecomposition<ValueType>(result.preprocessedModel.getTransitionMatrix(), backwardTransitions, statesInCurrentECWithNeutralAction);
                        ecs.reserve(ecs.size() + subECs.size());
                        for(auto& ec : subECs){
                            ecs.push_back(std::move(ec));
                        }
                    }
                    statesInCurrentECWithNeutralAction.clear();
                }
            }
            
            template<typename SparseModelType>
            void SparsePcaaPreprocessor<SparseModelType>::ensureRewardFiniteness(ReturnType& result, storm::storage::SparseMatrix<ValueType> const& backwardTransitions) {
                STORM_LOG_THROW((result.actionsWithPositiveReward & result.ecActions).empty(), storm::exceptions::InvalidPropertyException, "Infinite reward: There is one maximizing objective for which infinite reward is possible. This is not supported.");
                
                //Check whether the states that can be visited inf. often are reachable with prob. 1 under some scheduler
                storm::storage::BitVector statesReachingNegativeRewardsFinitelyOftenForSomeScheduler = storm::utility::graph::performProb1E(result.preprocessedModel.getTransitionMatrix(), result.preprocessedModel.getTransitionMatrix().getRowGroupIndices(), backwardTransitions, storm::storage::BitVector(result.preprocessedModel.getNumberOfStates(), true), result.possiblyRecurrentStates);
                STORM_LOG_THROW(!(statesReachingNegativeRewardsFinitelyOftenForSomeScheduler & result.preprocessedModel.getInitialStates()).empty(), storm::exceptions::InvalidPropertyException, "Infinite Rewards: For every scheduler, the induced reward for one or more of the objectives that minimize rewards is infinity.");
                
                if(!statesReachingNegativeRewardsFinitelyOftenForSomeScheduler.full()) {
                    // Remove the states that for any scheduler have one objective with infinite expected reward.
                    auto subsystemBuilderResult = storm::transformer::SubsystemBuilder<SparseModelType>::transform(result.preprocessedModel, statesReachingNegativeRewardsFinitelyOftenForSomeScheduler, storm::storage::BitVector(result.preprocessedModel.getNumberOfChoices(), true));
                    updatePreprocessedModel(result, *subsystemBuilderResult.model, subsystemBuilderResult.newToOldStateIndexMapping);
                    result.ecActions = result.ecActions % subsystemBuilderResult.keptActions;
                    result.actionsWithPositiveReward = result.actionsWithPositiveReward % subsystemBuilderResult.keptActions;
                    result.actionsWithNegativeReward = result.actionsWithNegativeReward % subsystemBuilderResult.keptActions;
                    result.possiblyRecurrentStates = result.possiblyRecurrentStates % statesReachingNegativeRewardsFinitelyOftenForSomeScheduler;
                }
            }
        
        
        
            template class SparsePcaaPreprocessor<storm::models::sparse::Mdp<double>>;
            template class SparsePcaaPreprocessor<storm::models::sparse::MarkovAutomaton<double>>;
            
#ifdef STORM_HAVE_CARL
            template class SparsePcaaPreprocessor<storm::models::sparse::Mdp<storm::RationalNumber>>;
            template class SparsePcaaPreprocessor<storm::models::sparse::MarkovAutomaton<storm::RationalNumber>>;
#endif
        }
    }
}
