#include "storm/modelchecker/parametric/SparseMdpParameterLiftingModelChecker.h"

#include "storm/adapters/CarlAdapter.h"
#include "storm/modelchecker/propositional/SparsePropositionalModelChecker.h"
#include "storm/modelchecker/results/ExplicitQualitativeCheckResult.h"
#include "storm/modelchecker/results/ExplicitQuantitativeCheckResult.h"
#include "storm/models/sparse/Mdp.h"
#include "storm/models/sparse/StandardRewardModel.h"
#include "storm/utility/vector.h"
#include "storm/utility/graph.h"
#include "storm/logic/FragmentSpecification.h"

#include "storm/exceptions/InvalidArgumentException.h"
#include "storm/exceptions/InvalidPropertyException.h"
#include "storm/exceptions/NotSupportedException.h"

namespace storm {
    namespace modelchecker {
        namespace parametric {
            
            template <typename SparseModelType, typename ConstantType>
            SparseMdpParameterLiftingModelChecker<SparseModelType, ConstantType>::SparseMdpParameterLiftingModelChecker(SparseModelType const& parametricModel) : SparseParameterLiftingModelChecker<SparseModelType, ConstantType>(parametricModel), solverFactory(std::make_unique<storm::solver::GameSolverFactory<ConstantType>>()) {
            }
            
            template <typename SparseModelType, typename ConstantType>
            SparseMdpParameterLiftingModelChecker<SparseModelType, ConstantType>::SparseMdpParameterLiftingModelChecker(SparseModelType const& parametricModel, std::unique_ptr<storm::solver::GameSolverFactory<ConstantType>>&& solverFactory) : SparseParameterLiftingModelChecker<SparseModelType, ConstantType>(parametricModel), solverFactory(std::move(solverFactory)) {
            }
    
            template <typename SparseModelType, typename ConstantType>
            bool SparseMdpParameterLiftingModelChecker<SparseModelType, ConstantType>::canHandle(CheckTask<storm::logic::Formula, typename SparseModelType::ValueType> const& checkTask) const {
                return checkTask.getFormula().isInFragment(storm::logic::reachability().setRewardOperatorsAllowed(true).setReachabilityRewardFormulasAllowed(true).setBoundedUntilFormulasAllowed(true).setStepBoundedUntilFormulasAllowed(true).setCumulativeRewardFormulasAllowed(true));
            }
            
            template <typename SparseModelType, typename ConstantType>
            void SparseMdpParameterLiftingModelChecker<SparseModelType, ConstantType>::specifyBoundedUntilFormula(CheckTask<storm::logic::BoundedUntilFormula, ConstantType> const& checkTask) {
                
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
                maybeStates = storm::solver::minimize(checkTask.getOptimizationDirection()) ?
                              storm::utility::graph::performProbGreater0A(this->parametricModel.getTransitionMatrix(), this->parametricModel.getTransitionMatrix().getRowGroupIndices(), this->parametricModel.getBackwardTransitions(), phiStates, psiStates, true, *stepBound) :
                              storm::utility::graph::performProbGreater0E(this->parametricModel.getBackwardTransitions(), phiStates, psiStates, true, *stepBound);
                maybeStates &= ~psiStates;
                
                // set the result for all non-maybe states
                resultsForNonMaybeStates = std::vector<ConstantType>(this->parametricModel.getNumberOfStates(), storm::utility::zero<ConstantType>());
                storm::utility::vector::setVectorValues(resultsForNonMaybeStates, psiStates, storm::utility::one<ConstantType>());
                
                // if there are maybestates, create the parameterLifter
                if (!maybeStates.empty()) {
                    // Create the vector of one-step probabilities to go to target states.
                    std::vector<typename SparseModelType::ValueType> b = this->parametricModel.getTransitionMatrix().getConstrainedRowSumVector(storm::storage::BitVector(this->parametricModel.getTransitionMatrix().getRowCount(), true), psiStates);
                    
                    parameterLifter = std::make_unique<storm::transformer::ParameterLifter<typename SparseModelType::ValueType, ConstantType>>(this->parametricModel.getTransitionMatrix(), b, this->parametricModel.getTransitionMatrix().getRowIndicesOfRowGroups(maybeStates), maybeStates);
                    computePlayer1Matrix();
                    
                    applyPreviousResultAsHint = false;
                }
                
                // We know some bounds for the results
                lowerResultBound = storm::utility::zero<ConstantType>();
                upperResultBound = storm::utility::one<ConstantType>();
            }
    
            template <typename SparseModelType, typename ConstantType>
            void SparseMdpParameterLiftingModelChecker<SparseModelType, ConstantType>::specifyUntilFormula(CheckTask<storm::logic::UntilFormula, ConstantType> const& checkTask) {
                
                // get the results for the subformulas
                storm::modelchecker::SparsePropositionalModelChecker<SparseModelType> propositionalChecker(this->parametricModel);
                STORM_LOG_THROW(propositionalChecker.canHandle(checkTask.getFormula().getLeftSubformula()) && propositionalChecker.canHandle(checkTask.getFormula().getRightSubformula()), storm::exceptions::NotSupportedException, "Parameter lifting with non-propositional subformulas is not supported");
                storm::storage::BitVector phiStates = std::move(propositionalChecker.check(checkTask.getFormula().getLeftSubformula())->asExplicitQualitativeCheckResult().getTruthValuesVector());
                storm::storage::BitVector psiStates = std::move(propositionalChecker.check(checkTask.getFormula().getRightSubformula())->asExplicitQualitativeCheckResult().getTruthValuesVector());
                
                // get the maybeStates
                std::pair<storm::storage::BitVector, storm::storage::BitVector> statesWithProbability01 = storm::solver::minimize(checkTask.getOptimizationDirection()) ?
                                                                                                          storm::utility::graph::performProb01Min(this->parametricModel.getTransitionMatrix(), this->parametricModel.getTransitionMatrix().getRowGroupIndices(), this->parametricModel.getBackwardTransitions(), phiStates, psiStates) :
                                                                                                          storm::utility::graph::performProb01Max(this->parametricModel.getTransitionMatrix(), this->parametricModel.getTransitionMatrix().getRowGroupIndices(), this->parametricModel.getBackwardTransitions(), phiStates, psiStates);
                maybeStates = ~(statesWithProbability01.first | statesWithProbability01.second);
                
                // set the result for all non-maybe states
                resultsForNonMaybeStates = std::vector<ConstantType>(this->parametricModel.getNumberOfStates(), storm::utility::zero<ConstantType>());
                storm::utility::vector::setVectorValues(resultsForNonMaybeStates, statesWithProbability01.second, storm::utility::one<ConstantType>());
                
                // if there are maybestates, create the parameterLifter
                if (!maybeStates.empty()) {
                    // Create the vector of one-step probabilities to go to target states.
                    std::vector<typename SparseModelType::ValueType> b = this->parametricModel.getTransitionMatrix().getConstrainedRowSumVector(storm::storage::BitVector(this->parametricModel.getTransitionMatrix().getRowCount(), true), statesWithProbability01.second);
                    
                    parameterLifter = std::make_unique<storm::transformer::ParameterLifter<typename SparseModelType::ValueType, ConstantType>>(this->parametricModel.getTransitionMatrix(), b, this->parametricModel.getTransitionMatrix().getRowIndicesOfRowGroups(maybeStates), maybeStates);
                    computePlayer1Matrix();
                    
                    // Check whether there is an EC consisting of maybestates
                    applyPreviousResultAsHint = storm::solver::minimize(checkTask.getOptimizationDirection()) || // when minimizing, there can not be an EC within the maybestates
                                                storm::utility::graph::performProb1A(this->parametricModel.getTransitionMatrix(), this->parametricModel.getTransitionMatrix().getRowGroupIndices(), this->parametricModel.getBackwardTransitions(), maybeStates, ~maybeStates).full();
                }
                                
                // We know some bounds for the results
                lowerResultBound = storm::utility::zero<ConstantType>();
                upperResultBound = storm::utility::one<ConstantType>();
            }
    
            template <typename SparseModelType, typename ConstantType>
            void SparseMdpParameterLiftingModelChecker<SparseModelType, ConstantType>::specifyReachabilityRewardFormula(CheckTask<storm::logic::EventuallyFormula, ConstantType> const& checkTask) {
                
                // get the results for the subformula
                storm::modelchecker::SparsePropositionalModelChecker<SparseModelType> propositionalChecker(this->parametricModel);
                STORM_LOG_THROW(propositionalChecker.canHandle(checkTask.getFormula().getSubformula()), storm::exceptions::NotSupportedException, "Parameter lifting with non-propositional subformulas is not supported");
                storm::storage::BitVector targetStates = std::move(propositionalChecker.check(checkTask.getFormula().getSubformula())->asExplicitQualitativeCheckResult().getTruthValuesVector());
                
                // get the maybeStates
                storm::storage::BitVector infinityStates = storm::solver::minimize(checkTask.getOptimizationDirection()) ?
                                                           storm::utility::graph::performProb1E(this->parametricModel.getTransitionMatrix(), this->parametricModel.getTransitionMatrix().getRowGroupIndices(), this->parametricModel.getBackwardTransitions(), storm::storage::BitVector(this->parametricModel.getNumberOfStates(), true), targetStates) :
                                                           storm::utility::graph::performProb1A(this->parametricModel.getTransitionMatrix(), this->parametricModel.getTransitionMatrix().getRowGroupIndices(), this->parametricModel.getBackwardTransitions(), storm::storage::BitVector(this->parametricModel.getNumberOfStates(), true), targetStates);
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
                    
                    // We need to handle choices that lead to an infinity state.
                    // As a maybeState does not have reward infinity, such a choice will never be picked. Hence, we can unselect the corresponding rows
                    storm::storage::BitVector selectedRows = this->parametricModel.getTransitionMatrix().getRowIndicesOfRowGroups(maybeStates);
                    for (uint_fast64_t row : selectedRows) {
                        for (auto const& element : this->parametricModel.getTransitionMatrix().getRow(row)) {
                            if (infinityStates.get(element.getColumn())) {
                                selectedRows.set(row, false);
                                break;
                            }
                        }
                    }
                    
                    parameterLifter = std::make_unique<storm::transformer::ParameterLifter<typename SparseModelType::ValueType, ConstantType>>(this->parametricModel.getTransitionMatrix(), b, selectedRows, maybeStates);
                    computePlayer1Matrix();
                    
                    // Check whether there is an EC consisting of maybestates
                    applyPreviousResultAsHint = !storm::solver::minimize(checkTask.getOptimizationDirection()) || // when maximizing, there can not be an EC within the maybestates
                                                storm::utility::graph::performProb1A(this->parametricModel.getTransitionMatrix(), this->parametricModel.getTransitionMatrix().getRowGroupIndices(), this->parametricModel.getBackwardTransitions(), maybeStates, ~maybeStates).full();
                }
                
                // We only know a lower bound for the result
                lowerResultBound = storm::utility::zero<ConstantType>();

            }
    
            template <typename SparseModelType, typename ConstantType>
            void SparseMdpParameterLiftingModelChecker<SparseModelType, ConstantType>::specifyCumulativeRewardFormula(CheckTask<storm::logic::CumulativeRewardFormula, ConstantType> const& checkTask) {
                
                // Obtain the stepBound
                stepBound = checkTask.getFormula().getBound().evaluateAsInt();
                if (checkTask.getFormula().isBoundStrict()) {
                    STORM_LOG_THROW(*stepBound > 0, storm::exceptions::NotSupportedException, "Expected a strict upper step bound that is greater than zero.");
                    --(*stepBound);
                }
                STORM_LOG_THROW(*stepBound > 0, storm::exceptions::NotSupportedException, "Can not apply parameter lifting on step bounded formula: The step bound has to be positive.");
                
                //Every state is a maybeState
                maybeStates = storm::storage::BitVector(this->parametricModel.getTransitionMatrix().getColumnCount(), true);
                resultsForNonMaybeStates = std::vector<ConstantType>(this->parametricModel.getNumberOfStates());

                // Create the reward vector
                STORM_LOG_THROW((checkTask.isRewardModelSet() && this->parametricModel.hasRewardModel(checkTask.getRewardModel())) || (!checkTask.isRewardModelSet() && this->parametricModel.hasUniqueRewardModel()), storm::exceptions::InvalidPropertyException, "The reward model specified by the CheckTask is not available in the given model.");
                typename SparseModelType::RewardModelType const& rewardModel = checkTask.isRewardModelSet() ? this->parametricModel.getRewardModel(checkTask.getRewardModel()) : this->parametricModel.getUniqueRewardModel();
                std::vector<typename SparseModelType::ValueType> b = rewardModel.getTotalRewardVector(this->parametricModel.getTransitionMatrix());
                    
                parameterLifter = std::make_unique<storm::transformer::ParameterLifter<typename SparseModelType::ValueType, ConstantType>>(this->parametricModel.getTransitionMatrix(), b, storm::storage::BitVector(this->parametricModel.getTransitionMatrix().getRowCount(), true), maybeStates);
                computePlayer1Matrix();

                applyPreviousResultAsHint = false;
                
                // We only know a lower bound for the result
                lowerResultBound = storm::utility::zero<ConstantType>();
            }
            
            template <typename SparseModelType, typename ConstantType>
            std::unique_ptr<CheckResult> SparseMdpParameterLiftingModelChecker<SparseModelType, ConstantType>::computeQuantitativeValues(storm::storage::ParameterRegion<typename SparseModelType::ValueType> const& region, storm::solver::OptimizationDirection const& dirForParameters) {
                
                if(maybeStates.empty()) {
                    return std::make_unique<storm::modelchecker::ExplicitQuantitativeCheckResult<ConstantType>>(resultsForNonMaybeStates);
                }
                
                parameterLifter->specifyRegion(region, dirForParameters);
                
                // Set up the solver
                auto solver = solverFactory->create(player1Matrix, parameterLifter->getMatrix());
                if(lowerResultBound) solver->setLowerBound(lowerResultBound.get());
                if(upperResultBound) solver->setUpperBound(upperResultBound.get());
                if(applyPreviousResultAsHint) {
                    solver->setTrackSchedulers(true);
                    x.resize(maybeStates.getNumberOfSetBits(), storm::utility::zero<ConstantType>());
                    if(storm::solver::minimize(dirForParameters) && minSched && player1Sched) solver->setSchedulerHints(std::move(player1Sched.get()), std::move(minSched.get()));
                    if(storm::solver::maximize(dirForParameters) && maxSched && player1Sched) solver->setSchedulerHints(std::move(player1Sched.get()), std::move(maxSched.get()));
                } else {
                    x.assign(maybeStates.getNumberOfSetBits(), storm::utility::zero<ConstantType>());
                }
                if (this->currentCheckTask->isBoundSet() && this->currentCheckTask->getOptimizationDirection() == dirForParameters && solver->hasSchedulerHints()) {
                    // If we reach this point, we know that after applying the hints, the x-values can only become larger (if we maximize) or smaller (if we minimize).
                    std::unique_ptr<storm::solver::TerminationCondition<ConstantType>> termCond;
                    storm::storage::BitVector relevantStatesInSubsystem = this->currentCheckTask->isOnlyInitialStatesRelevantSet() ? this->parametricModel.getInitialStates() % maybeStates : storm::storage::BitVector(maybeStates.getNumberOfSetBits(), true);
                    if (storm::solver::minimize(dirForParameters)) {
                        // Terminate if the value for ALL relevant states is already below the threshold
                        termCond = std::make_unique<storm::solver::TerminateIfFilteredExtremumBelowThreshold<ConstantType>> (relevantStatesInSubsystem, this->currentCheckTask->getBoundThreshold(), true, false);
                    } else {
                        // Terminate if the value for ALL relevant states is already above the threshold
                        termCond = std::make_unique<storm::solver::TerminateIfFilteredExtremumExceedsThreshold<ConstantType>> (relevantStatesInSubsystem, true, this->currentCheckTask->getBoundThreshold(), true);
                    }
                    solver->setTerminationCondition(std::move(termCond));
                }
                
                // Invoke the solver
                if(stepBound) {
                    assert(*stepBound > 0);
                    solver->repeatedMultiply(this->currentCheckTask->getOptimizationDirection(), dirForParameters, x, &parameterLifter->getVector(), *stepBound);
                } else {
                    solver->solveGame(this->currentCheckTask->getOptimizationDirection(), dirForParameters, x, parameterLifter->getVector());
                    if(applyPreviousResultAsHint) {
                        if(storm::solver::minimize(dirForParameters)) {
                            minSched = std::move(*solver->getPlayer2Scheduler());
                        } else {
                            maxSched = std::move(*solver->getPlayer2Scheduler());
                        }
                        player1Sched = std::move(*solver->getPlayer1Scheduler());
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
            void SparseMdpParameterLiftingModelChecker<SparseModelType, ConstantType>::computePlayer1Matrix() {
                uint_fast64_t n = 0;
                for(auto const& maybeState : maybeStates) {
                    n += this->parametricModel.getTransitionMatrix().getRowGroupSize(maybeState);
                }
                // The player 1 matrix is the identity matrix of size n with the row groups as given by the original matrix
                storm::storage::SparseMatrixBuilder<storm::storage::sparse::state_type> matrixBuilder(n, n, n, true, true, maybeStates.getNumberOfSetBits());
                uint_fast64_t p1MatrixRow = 0;
                for (auto maybeState : maybeStates){
                    matrixBuilder.newRowGroup(p1MatrixRow);
                    for (uint_fast64_t row = this->parametricModel.getTransitionMatrix().getRowGroupIndices()[maybeState]; row < this->parametricModel.getTransitionMatrix().getRowGroupIndices()[maybeState + 1]; ++row){
                        matrixBuilder.addNextValue(p1MatrixRow, p1MatrixRow, storm::utility::one<storm::storage::sparse::state_type>());
                        ++p1MatrixRow;
                    }
                }
                player1Matrix = matrixBuilder.build();
            }
    
            template <typename SparseModelType, typename ConstantType>
            void SparseMdpParameterLiftingModelChecker<SparseModelType, ConstantType>::reset() {
                maybeStates.resize(0);
                resultsForNonMaybeStates.clear();
                stepBound = boost::none;
                player1Matrix = storm::storage::SparseMatrix<storm::storage::sparse::state_type>();
                parameterLifter = nullptr;
                minSched = boost::none;
                maxSched = boost::none;
                x.clear();
                lowerResultBound = boost::none;
                upperResultBound = boost::none;
                applyPreviousResultAsHint = false;
            }
            
            template <typename SparseModelType, typename ConstantType>
            boost::optional<storm::storage::TotalScheduler>& SparseMdpParameterLiftingModelChecker<SparseModelType, ConstantType>::getCurrentMinScheduler() {
                return minSched;
            }
                    
            template <typename SparseModelType, typename ConstantType>
            boost::optional<storm::storage::TotalScheduler>& SparseMdpParameterLiftingModelChecker<SparseModelType, ConstantType>::getCurrentMaxScheduler() {
                return maxSched;
            }
            
            template <typename SparseModelType, typename ConstantType>
            boost::optional<storm::storage::TotalScheduler>& SparseMdpParameterLiftingModelChecker<SparseModelType, ConstantType>::getCurrentPlayer1Scheduler() {
                return player1Sched;
            }
    
            template class SparseMdpParameterLiftingModelChecker<storm::models::sparse::Mdp<storm::RationalFunction>, double>;
            template class SparseMdpParameterLiftingModelChecker<storm::models::sparse::Mdp<storm::RationalFunction>, storm::RationalNumber>;
        }
    }
}
