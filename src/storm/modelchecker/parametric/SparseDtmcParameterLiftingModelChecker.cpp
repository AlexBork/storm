#include "SparseDtmcParameterLiftingModelChecker.h"

#include "storm/adapters/CarlAdapter.h"
#include "storm/modelchecker/propositional/SparsePropositionalModelChecker.h"
#include "storm/modelchecker/results/ExplicitQualitativeCheckResult.h"
#include "storm/modelchecker/results/ExplicitQuantitativeCheckResult.h"
#include "storm/models/sparse/Dtmc.h"
#include "storm/models/sparse/StandardRewardModel.h"
#include "storm/utility/vector.h"
#include "storm/utility/graph.h"

#include "storm/exceptions/InvalidArgumentException.h"
#include "storm/exceptions/InvalidPropertyException.h"
#include "storm/exceptions/NotSupportedException.h"

namespace storm {
    namespace modelchecker {
        namespace parametric {
            
            template <typename SparseModelType, typename ConstantType>
            SparseDtmcParameterLiftingModelChecker<SparseModelType, ConstantType>::SparseDtmcParameterLiftingModelChecker(SparseModelType const& parametricModel) : SparseParameterLiftingModelChecker<SparseModelType, ConstantType>(parametricModel), solverFactory(std::make_unique<storm::solver::GeneralMinMaxLinearEquationSolverFactory<ConstantType>>()) {
            }
            
            template <typename SparseModelType, typename ConstantType>
            SparseDtmcParameterLiftingModelChecker<SparseModelType, ConstantType>::SparseDtmcParameterLiftingModelChecker(SparseModelType const& parametricModel, std::unique_ptr<storm::solver::MinMaxLinearEquationSolverFactory<ConstantType>>&& solverFactory) : SparseParameterLiftingModelChecker<SparseModelType, ConstantType>(parametricModel), solverFactory(std::move(solverFactory)) {
            }
            
            template <typename SparseModelType, typename ConstantType>
            bool SparseDtmcParameterLiftingModelChecker<SparseModelType, ConstantType>::canHandle(CheckTask<storm::logic::Formula, typename SparseModelType::ValueType> const& checkTask) const {
                return checkTask.getFormula().isInFragment(storm::logic::reachability().setRewardOperatorsAllowed(true).setReachabilityRewardFormulasAllowed(true).setBoundedUntilFormulasAllowed(true).setCumulativeRewardFormulasAllowed(true));
            }
            
            template <typename SparseModelType, typename ConstantType>
            void SparseDtmcParameterLiftingModelChecker<SparseModelType, ConstantType>::specifyBoundedUntilFormula(CheckTask<storm::logic::BoundedUntilFormula, ConstantType> const& checkTask) {
                
                // get the step bound
                STORM_LOG_THROW(!checkTask.getFormula().hasLowerBound(), storm::exceptions::NotSupportedException, "Lower step bounds are not supported.");
                STORM_LOG_THROW(checkTask.getFormula().hasUpperBound(), storm::exceptions::NotSupportedException, "Expected a bounded until formula with an upper bound.");
                STORM_LOG_THROW(checkTask.getFormula().isStepBounded(), storm::exceptions::NotSupportedException, "Expected a bounded until formula with step bounds.");
                stepBound = checkTask.getFormula().getUpperBound().evaluateAsInt();
                STORM_LOG_THROW(*stepBound > 0, storm::exceptions::NotSupportedException, "Can not apply parameter lifting on step bounded formula: The step bound has to be positive.");
                if (checkTask.getFormula().isUpperBoundStrict()) {
                    STORM_LOG_THROW(*stepBound > 0, storm::exceptions::NotSupportedException, "Expected a strict upper step bound that is greater than zero.");
                    --(*stepBound);
                }
                STORM_LOG_THROW(*stepBound > 0, storm::exceptions::NotSupportedException, "Can not apply parameter lifting on step bounded formula: The step bound has to be positive.");

                // get the results for the subformulas
                storm::modelchecker::SparsePropositionalModelChecker<SparseModelType> propositionalChecker(this->parametricModel);
                STORM_LOG_THROW(propositionalChecker.canHandle(checkTask.getFormula().getLeftSubformula()) && propositionalChecker.canHandle(checkTask.getFormula().getRightSubformula()), storm::exceptions::NotSupportedException, "Parameter lifting with non-propositional subformulas is not supported");
                storm::storage::BitVector phiStates = std::move(propositionalChecker.check(checkTask.getFormula().getLeftSubformula())->asExplicitQualitativeCheckResult().getTruthValuesVector());
                storm::storage::BitVector psiStates = std::move(propositionalChecker.check(checkTask.getFormula().getRightSubformula())->asExplicitQualitativeCheckResult().getTruthValuesVector());
                
                // get the maybeStates
                maybeStates = storm::utility::graph::performProbGreater0(this->parametricModel.getBackwardTransitions(), phiStates, psiStates, true, *stepBound);
                maybeStates &= ~psiStates;
                
                // set the result for all non-maybe states
                resultsForNonMaybeStates = std::vector<ConstantType>(this->parametricModel.getNumberOfStates(), storm::utility::zero<ConstantType>());
                storm::utility::vector::setVectorValues(resultsForNonMaybeStates, psiStates, storm::utility::one<ConstantType>());
                
                // if there are maybestates, create the parameterLifter
                if (!maybeStates.empty()) {
                    // Create the vector of one-step probabilities to go to target states.
                    std::vector<typename SparseModelType::ValueType> b = this->parametricModel.getTransitionMatrix().getConstrainedRowSumVector(storm::storage::BitVector(this->parametricModel.getTransitionMatrix().getRowCount(), true), psiStates);
                    
                    parameterLifter = std::make_unique<storm::transformer::ParameterLifter<typename SparseModelType::ValueType, ConstantType>>(this->parametricModel.getTransitionMatrix(), b, maybeStates, maybeStates);
                }
                
                // We know some bounds for the results so set them
                lowerResultBound = storm::utility::zero<ConstantType>();
                upperResultBound = storm::utility::one<ConstantType>();
            }
    
            template <typename SparseModelType, typename ConstantType>
            void SparseDtmcParameterLiftingModelChecker<SparseModelType, ConstantType>::specifyUntilFormula(CheckTask<storm::logic::UntilFormula, ConstantType> const& checkTask) {
                
                // get the results for the subformulas
                storm::modelchecker::SparsePropositionalModelChecker<SparseModelType> propositionalChecker(this->parametricModel);
                STORM_LOG_THROW(propositionalChecker.canHandle(checkTask.getFormula().getLeftSubformula()) && propositionalChecker.canHandle(checkTask.getFormula().getRightSubformula()), storm::exceptions::NotSupportedException, "Parameter lifting with non-propositional subformulas is not supported");
                storm::storage::BitVector phiStates = std::move(propositionalChecker.check(checkTask.getFormula().getLeftSubformula())->asExplicitQualitativeCheckResult().getTruthValuesVector());
                storm::storage::BitVector psiStates = std::move(propositionalChecker.check(checkTask.getFormula().getRightSubformula())->asExplicitQualitativeCheckResult().getTruthValuesVector());
                
                // get the maybeStates
                std::pair<storm::storage::BitVector, storm::storage::BitVector> statesWithProbability01 = storm::utility::graph::performProb01(this->parametricModel.getBackwardTransitions(), phiStates, psiStates);
                maybeStates = ~(statesWithProbability01.first | statesWithProbability01.second);
                
                // set the result for all non-maybe states
                resultsForNonMaybeStates = std::vector<ConstantType>(this->parametricModel.getNumberOfStates(), storm::utility::zero<ConstantType>());
                storm::utility::vector::setVectorValues(resultsForNonMaybeStates, statesWithProbability01.second, storm::utility::one<ConstantType>());
                
                // if there are maybestates, create the parameterLifter
                if (!maybeStates.empty()) {
                    // Create the vector of one-step probabilities to go to target states.
                    std::vector<typename SparseModelType::ValueType> b = this->parametricModel.getTransitionMatrix().getConstrainedRowSumVector(storm::storage::BitVector(this->parametricModel.getTransitionMatrix().getRowCount(), true), psiStates);
                    
                    parameterLifter = std::make_unique<storm::transformer::ParameterLifter<typename SparseModelType::ValueType, ConstantType>>(this->parametricModel.getTransitionMatrix(), b, maybeStates, maybeStates);
                }
                                
                // We know some bounds for the results so set them
                lowerResultBound = storm::utility::zero<ConstantType>();
                upperResultBound = storm::utility::one<ConstantType>();
            }
    
            template <typename SparseModelType, typename ConstantType>
            void SparseDtmcParameterLiftingModelChecker<SparseModelType, ConstantType>::specifyReachabilityRewardFormula(CheckTask<storm::logic::EventuallyFormula, ConstantType> const& checkTask) {
                
                // get the results for the subformula
                storm::modelchecker::SparsePropositionalModelChecker<SparseModelType> propositionalChecker(this->parametricModel);
                STORM_LOG_THROW(propositionalChecker.canHandle(checkTask.getFormula().getSubformula()), storm::exceptions::NotSupportedException, "Parameter lifting with non-propositional subformulas is not supported");
                storm::storage::BitVector targetStates = std::move(propositionalChecker.check(checkTask.getFormula().getSubformula())->asExplicitQualitativeCheckResult().getTruthValuesVector());
                
                // get the maybeStates
                storm::storage::BitVector infinityStates = storm::utility::graph::performProb1(this->parametricModel.getBackwardTransitions(), storm::storage::BitVector(this->parametricModel.getNumberOfStates(), true), targetStates);
                infinityStates.complement();
                maybeStates = ~(targetStates | infinityStates);
                
                // set the result for all the non-maybe states
                resultsForNonMaybeStates = std::vector<ConstantType>(this->parametricModel.getNumberOfStates(), storm::utility::zero<ConstantType>());
                storm::utility::vector::setVectorValues(resultsForNonMaybeStates, infinityStates, storm::utility::infinity<ConstantType>());
                
                // if there are maybestates, create the parameterLifter
                if (!maybeStates.empty()) {
                    // Create the reward vector
                    STORM_LOG_THROW((checkTask.isRewardModelSet() && this->parametricModel.hasRewardModel(checkTask.getRewardModel())) || (!checkTask.isRewardModelSet() && this->parametricModel.hasUniqueRewardModel()), storm::exceptions::InvalidPropertyException, "The reward model specified by the CheckTask is not available in the given model.");

                    typename SparseModelType::RewardModelType const& rewardModel = checkTask.isRewardModelSet() ? this->parametricModel.getRewardModel(checkTask.getRewardModel()) : this->parametricModel.getUniqueRewardModel();

                    std::vector<typename SparseModelType::ValueType> b = rewardModel.getTotalRewardVector(this->parametricModel.getTransitionMatrix());
                    
                    parameterLifter = std::make_unique<storm::transformer::ParameterLifter<typename SparseModelType::ValueType, ConstantType>>(this->parametricModel.getTransitionMatrix(), b, maybeStates, maybeStates);
                }
                
                // We only know a lower bound for the result
                lowerResultBound = storm::utility::zero<ConstantType>();

            }
    
            template <typename SparseModelType, typename ConstantType>
            void SparseDtmcParameterLiftingModelChecker<SparseModelType, ConstantType>::specifyCumulativeRewardFormula(CheckTask<storm::logic::CumulativeRewardFormula, ConstantType> const& checkTask) {
                
                // Obtain the stepBound
                stepBound = checkTask.getFormula().getBound().evaluateAsInt();
                if (checkTask.getFormula().isBoundStrict()) {
                    STORM_LOG_THROW(*stepBound > 0, storm::exceptions::NotSupportedException, "Expected a strict upper step bound that is greater than zero.");
                    --(*stepBound);
                }
                STORM_LOG_THROW(*stepBound > 0, storm::exceptions::NotSupportedException, "Can not apply parameter lifting on step bounded formula: The step bound has to be positive.");
                
                // Create the reward vector
                STORM_LOG_THROW((checkTask.isRewardModelSet() && this->parametricModel.hasRewardModel(checkTask.getRewardModel())) || (!checkTask.isRewardModelSet() && this->parametricModel.hasUniqueRewardModel()), storm::exceptions::InvalidPropertyException, "The reward model specified by the CheckTask is not available in the given model.");
                typename SparseModelType::RewardModelType const& rewardModel = checkTask.isRewardModelSet() ? this->parametricModel.getRewardModel(checkTask.getRewardModel()) : this->parametricModel.getUniqueRewardModel();
                std::vector<typename SparseModelType::ValueType> b = rewardModel.getTotalRewardVector(this->parametricModel.getTransitionMatrix());
                    
                parameterLifter = std::make_unique<storm::transformer::ParameterLifter<typename SparseModelType::ValueType, ConstantType>>(this->parametricModel.getTransitionMatrix(), b, storm::storage::BitVector(this->parametricModel.getTransitionMatrix().getRowCount(), true), storm::storage::BitVector(this->parametricModel.getTransitionMatrix().getColumnCount(), true));
                
                                
                // We only know a lower bound for the result
                lowerResultBound = storm::utility::zero<ConstantType>();
            }
            
            template <typename SparseModelType, typename ConstantType>
            std::unique_ptr<CheckResult> SparseDtmcParameterLiftingModelChecker<SparseModelType, ConstantType>::computeQuantitativeValues(storm::storage::ParameterRegion<typename SparseModelType::ValueType> const& region, storm::solver::OptimizationDirection const& dirForParameters) {
                
                if(maybeStates.empty()) {
                    return std::make_unique<storm::modelchecker::ExplicitQuantitativeCheckResult<ConstantType>>(resultsForNonMaybeStates);
                }
                
                parameterLifter->specifyRegion(region, dirForParameters);
                
                // Set up the solver
                auto solver = solverFactory->create(parameterLifter->getMatrix());
                solver->setTrackScheduler(true);
                if(lowerResultBound) solver->setLowerBound(lowerResultBound.get());
                if(upperResultBound) solver->setUpperBound(upperResultBound.get());
                if(storm::solver::minimize(dirForParameters) && minSched && !stepBound) solver->setSchedulerHint(std::move(minSched.get()));
                if(storm::solver::maximize(dirForParameters) && maxSched && !stepBound) solver->setSchedulerHint(std::move(maxSched.get()));
                
                // Invoke the solver
                if(stepBound) {
                    assert(*stepBound > 0);
                    x = std::vector<ConstantType>(maybeStates.getNumberOfSetBits(), storm::utility::zero<ConstantType>());
                    solver->repeatedMultiply(dirForParameters, x, &parameterLifter->getVector(), *stepBound);
                } else {
                    x.resize(maybeStates.getNumberOfSetBits(), storm::utility::zero<ConstantType>());
                    solver->solveEquations(dirForParameters, x, parameterLifter->getVector());
                    if(storm::solver::minimize(dirForParameters)) {
                        minSched = std::move(*solver->getScheduler());
                    } else {
                        maxSched = std::move(*solver->getScheduler());
                    }
                }
                
                // Get the result for the complete model (including maybestates)
                std::vector<ConstantType> result = resultsForNonMaybeStates;
                auto maybeStateResIt = x.begin();
                for(auto const& maybeState : maybeStates) {
                    result[maybeState] = *maybeStateResIt;
                    ++maybeStateResIt;
                }
                return std::make_unique<storm::modelchecker::ExplicitQuantitativeCheckResult<ConstantType>>(std::move(result));
            }
            
                
            template <typename SparseModelType, typename ConstantType>
            void SparseDtmcParameterLiftingModelChecker<SparseModelType, ConstantType>::reset() {
                maybeStates.resize(0);
                resultsForNonMaybeStates.clear();
                stepBound = boost::none;
                parameterLifter = nullptr;
                minSched = boost::none;
                maxSched = boost::none;
                x.clear();
                lowerResultBound = boost::none;
                upperResultBound = boost::none;
            }
    
            template class SparseDtmcParameterLiftingModelChecker<storm::models::sparse::Dtmc<storm::RationalFunction>, double>;

        }
    }
}