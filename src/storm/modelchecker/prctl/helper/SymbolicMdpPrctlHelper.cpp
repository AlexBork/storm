#include "src/modelchecker/prctl/helper/SymbolicMdpPrctlHelper.h"

#include "src/solver/SymbolicMinMaxLinearEquationSolver.h"

#include "src/storage/dd/DdManager.h"
#include "src/storage/dd/Add.h"
#include "src/storage/dd/Bdd.h"

#include "src/utility/graph.h"
#include "src/utility/constants.h"


#include "src/models/symbolic/StandardRewardModel.h"

#include "src/modelchecker/results/SymbolicQualitativeCheckResult.h"
#include "src/modelchecker/results/SymbolicQuantitativeCheckResult.h"

#include "src/exceptions/InvalidPropertyException.h"
#include "src/exceptions/InvalidArgumentException.h"

namespace storm {
    namespace modelchecker {
        namespace helper {
            
            template<storm::dd::DdType DdType, typename ValueType>
            std::unique_ptr<CheckResult> SymbolicMdpPrctlHelper<DdType, ValueType>::computeUntilProbabilities(OptimizationDirection dir, storm::models::symbolic::NondeterministicModel<DdType, ValueType> const& model, storm::dd::Add<DdType, ValueType> const& transitionMatrix, storm::dd::Bdd<DdType> const& phiStates, storm::dd::Bdd<DdType> const& psiStates, bool qualitative, storm::utility::solver::SymbolicMinMaxLinearEquationSolverFactory<DdType, ValueType> const& linearEquationSolverFactory) {
                // We need to identify the states which have to be taken out of the matrix, i.e. all states that have
                // probability 0 and 1 of satisfying the until-formula.
                std::pair<storm::dd::Bdd<DdType>, storm::dd::Bdd<DdType>> statesWithProbability01;
                if (dir == OptimizationDirection::Minimize) {
                    statesWithProbability01 = storm::utility::graph::performProb01Min(model, phiStates, psiStates);
                } else {
                    statesWithProbability01 = storm::utility::graph::performProb01Max(model, phiStates, psiStates);
                }
                
                storm::dd::Bdd<DdType> maybeStates = !statesWithProbability01.first && !statesWithProbability01.second && model.getReachableStates();
                
                // Perform some logging.
                STORM_LOG_INFO("Found " << statesWithProbability01.first.getNonZeroCount() << " 'no' states.");
                STORM_LOG_INFO("Found " << statesWithProbability01.second.getNonZeroCount() << " 'yes' states.");
                STORM_LOG_INFO("Found " << maybeStates.getNonZeroCount() << " 'maybe' states.");
                
                // Check whether we need to compute exact probabilities for some states.
                if (qualitative) {
                    // Set the values for all maybe-states to 0.5 to indicate that their probability values are neither 0 nor 1.
                    return std::unique_ptr<CheckResult>(new storm::modelchecker::SymbolicQuantitativeCheckResult<DdType>(model.getReachableStates(), statesWithProbability01.second.template toAdd<ValueType>() + maybeStates.template toAdd<ValueType>() * model.getManager().getConstant(0.5)));
                } else {
                    // If there are maybe states, we need to solve an equation system.
                    if (!maybeStates.isZero()) {
                        // Create the matrix and the vector for the equation system.
                        storm::dd::Add<DdType, ValueType> maybeStatesAdd = maybeStates.template toAdd<ValueType>();
                        
                        // Start by cutting away all rows that do not belong to maybe states. Note that this leaves columns targeting
                        // non-maybe states in the matrix.
                        storm::dd::Add<DdType, ValueType> submatrix = transitionMatrix * maybeStatesAdd;
                        
                        // Then compute the vector that contains the one-step probabilities to a state with probability 1 for all
                        // maybe states.
                        storm::dd::Add<DdType, ValueType> prob1StatesAsColumn = statesWithProbability01.second.template toAdd<ValueType>();
                        prob1StatesAsColumn = prob1StatesAsColumn.swapVariables(model.getRowColumnMetaVariablePairs());
                        storm::dd::Add<DdType, ValueType> subvector = submatrix * prob1StatesAsColumn;
                        subvector = subvector.sumAbstract(model.getColumnVariables());
                        
                        // Finally cut away all columns targeting non-maybe states.
                        submatrix *= maybeStatesAdd.swapVariables(model.getRowColumnMetaVariablePairs());
                        
                        // Now solve the resulting equation system.
                        std::unique_ptr<storm::solver::SymbolicMinMaxLinearEquationSolver<DdType, ValueType>> solver = linearEquationSolverFactory.create(submatrix, maybeStates, model.getIllegalMask() && maybeStates, model.getRowVariables(), model.getColumnVariables(), model.getNondeterminismVariables(), model.getRowColumnMetaVariablePairs());
                        storm::dd::Add<DdType, ValueType> result = solver->solveEquations(dir == OptimizationDirection::Minimize, model.getManager().template getAddZero<ValueType>(), subvector);
                        
                        return std::unique_ptr<CheckResult>(new storm::modelchecker::SymbolicQuantitativeCheckResult<DdType>(model.getReachableStates(), statesWithProbability01.second.template toAdd<ValueType>() + result));
                    } else {
                        return std::unique_ptr<CheckResult>(new storm::modelchecker::SymbolicQuantitativeCheckResult<DdType>(model.getReachableStates(), statesWithProbability01.second.template toAdd<ValueType>()));
                    }
                }
            }
            
            template<storm::dd::DdType DdType, typename ValueType>
            std::unique_ptr<CheckResult> SymbolicMdpPrctlHelper<DdType, ValueType>::computeGloballyProbabilities(OptimizationDirection dir, storm::models::symbolic::NondeterministicModel<DdType, ValueType> const& model, storm::dd::Add<DdType, ValueType> const& transitionMatrix, storm::dd::Bdd<DdType> const& psiStates, bool qualitative, storm::utility::solver::SymbolicMinMaxLinearEquationSolverFactory<DdType, ValueType> const& linearEquationSolverFactory) {
                std::unique_ptr<CheckResult> result = computeUntilProbabilities(dir == OptimizationDirection::Minimize ? OptimizationDirection::Maximize : OptimizationDirection::Minimize, model, transitionMatrix, model.getReachableStates(), !psiStates && model.getReachableStates(), qualitative, linearEquationSolverFactory);
                result->asQuantitativeCheckResult<ValueType>().oneMinus();
                return result;
            }
            
            template<storm::dd::DdType DdType, typename ValueType>
            std::unique_ptr<CheckResult> SymbolicMdpPrctlHelper<DdType, ValueType>::computeNextProbabilities(OptimizationDirection dir, storm::models::symbolic::NondeterministicModel<DdType, ValueType> const& model, storm::dd::Add<DdType, ValueType> const& transitionMatrix, storm::dd::Bdd<DdType> const& nextStates) {
                storm::dd::Add<DdType, ValueType> result = transitionMatrix * nextStates.swapVariables(model.getRowColumnMetaVariablePairs()).template toAdd<ValueType>();
                return std::unique_ptr<CheckResult>(new SymbolicQuantitativeCheckResult<DdType>(model.getReachableStates(), result.sumAbstract(model.getColumnVariables())));
            }
            
            template<storm::dd::DdType DdType, typename ValueType>
            std::unique_ptr<CheckResult> SymbolicMdpPrctlHelper<DdType, ValueType>::computeBoundedUntilProbabilities(OptimizationDirection dir, storm::models::symbolic::NondeterministicModel<DdType, ValueType> const& model, storm::dd::Add<DdType, ValueType> const& transitionMatrix, storm::dd::Bdd<DdType> const& phiStates, storm::dd::Bdd<DdType> const& psiStates, uint_fast64_t stepBound, storm::utility::solver::SymbolicMinMaxLinearEquationSolverFactory<DdType, ValueType> const& linearEquationSolverFactory) {
                // We need to identify the states which have to be taken out of the matrix, i.e. all states that have
                // probability 0 or 1 of satisfying the until-formula.
                storm::dd::Bdd<DdType> statesWithProbabilityGreater0;
                if (dir == OptimizationDirection::Minimize) {
                    statesWithProbabilityGreater0 = storm::utility::graph::performProbGreater0A(model, transitionMatrix.notZero(), phiStates, psiStates);
                } else {
                    statesWithProbabilityGreater0 = storm::utility::graph::performProbGreater0E(model, transitionMatrix.notZero(), phiStates, psiStates);
                }
                storm::dd::Bdd<DdType> maybeStates = statesWithProbabilityGreater0 && !psiStates && model.getReachableStates();
                
                // If there are maybe states, we need to perform matrix-vector multiplications.
                if (!maybeStates.isZero()) {
                    // Create the matrix and the vector for the equation system.
                    storm::dd::Add<DdType, ValueType> maybeStatesAdd = maybeStates.template toAdd<ValueType>();
                    
                    // Start by cutting away all rows that do not belong to maybe states. Note that this leaves columns targeting
                    // non-maybe states in the matrix.
                    storm::dd::Add<DdType, ValueType> submatrix = transitionMatrix * maybeStatesAdd;
                    
                    // Then compute the vector that contains the one-step probabilities to a state with probability 1 for all
                    // maybe states.
                    storm::dd::Add<DdType, ValueType> prob1StatesAsColumn = psiStates.template toAdd<ValueType>().swapVariables(model.getRowColumnMetaVariablePairs());
                    storm::dd::Add<DdType, ValueType> subvector = (submatrix * prob1StatesAsColumn).sumAbstract(model.getColumnVariables());
                    
                    // Finally cut away all columns targeting non-maybe states.
                    submatrix *= maybeStatesAdd.swapVariables(model.getRowColumnMetaVariablePairs());
                    
                    std::unique_ptr<storm::solver::SymbolicMinMaxLinearEquationSolver<DdType, ValueType>> solver = linearEquationSolverFactory.create(submatrix, maybeStates, model.getIllegalMask() && maybeStates, model.getRowVariables(), model.getColumnVariables(), model.getNondeterminismVariables(), model.getRowColumnMetaVariablePairs());
                    storm::dd::Add<DdType, ValueType> result = solver->multiply(dir == OptimizationDirection::Minimize, model.getManager().template getAddZero<ValueType>(), &subvector, stepBound);
                    
                    return std::unique_ptr<CheckResult>(new storm::modelchecker::SymbolicQuantitativeCheckResult<DdType>(model.getReachableStates(), psiStates.template toAdd<ValueType>() + result));
                } else {
                    return std::unique_ptr<CheckResult>(new storm::modelchecker::SymbolicQuantitativeCheckResult<DdType>(model.getReachableStates(), psiStates.template toAdd<ValueType>()));
                }
            }
            
            template<storm::dd::DdType DdType, typename ValueType>
            std::unique_ptr<CheckResult> SymbolicMdpPrctlHelper<DdType, ValueType>::computeInstantaneousRewards(OptimizationDirection dir, storm::models::symbolic::NondeterministicModel<DdType, ValueType> const& model, storm::dd::Add<DdType, ValueType> const& transitionMatrix, RewardModelType const& rewardModel, uint_fast64_t stepBound, storm::utility::solver::SymbolicMinMaxLinearEquationSolverFactory<DdType, ValueType> const& linearEquationSolverFactory) {
                // Only compute the result if the model has at least one reward this->getModel().
                STORM_LOG_THROW(rewardModel.hasStateRewards(), storm::exceptions::InvalidPropertyException, "Missing reward model for formula. Skipping formula.");
                
                // Perform the matrix-vector multiplication.
                std::unique_ptr<storm::solver::SymbolicMinMaxLinearEquationSolver<DdType, ValueType>> solver = linearEquationSolverFactory.create(model.getTransitionMatrix(), model.getReachableStates(), model.getIllegalMask(), model.getRowVariables(), model.getColumnVariables(), model.getNondeterminismVariables(), model.getRowColumnMetaVariablePairs());
                storm::dd::Add<DdType, ValueType> result = solver->multiply(dir == OptimizationDirection::Minimize, rewardModel.getStateRewardVector(), nullptr, stepBound);

                return std::unique_ptr<CheckResult>(new SymbolicQuantitativeCheckResult<DdType>(model.getReachableStates(), result));
            }
            
            template<storm::dd::DdType DdType, typename ValueType>
            std::unique_ptr<CheckResult> SymbolicMdpPrctlHelper<DdType, ValueType>::computeCumulativeRewards(OptimizationDirection dir, storm::models::symbolic::NondeterministicModel<DdType, ValueType> const& model, storm::dd::Add<DdType, ValueType> const& transitionMatrix, RewardModelType const& rewardModel, uint_fast64_t stepBound, storm::utility::solver::SymbolicMinMaxLinearEquationSolverFactory<DdType, ValueType> const& linearEquationSolverFactory) {
                // Only compute the result if the model has at least one reward this->getModel().
                STORM_LOG_THROW(!rewardModel.empty(), storm::exceptions::InvalidPropertyException, "Missing reward model for formula. Skipping formula.");
                
                // Compute the reward vector to add in each step based on the available reward models.
                storm::dd::Add<DdType, ValueType> totalRewardVector = rewardModel.getTotalRewardVector(transitionMatrix, model.getColumnVariables());
                
                // Perform the matrix-vector multiplication.
                std::unique_ptr<storm::solver::SymbolicMinMaxLinearEquationSolver<DdType, ValueType>> solver = linearEquationSolverFactory.create(model.getTransitionMatrix(), model.getReachableStates(), model.getIllegalMask(), model.getRowVariables(), model.getColumnVariables(), model.getNondeterminismVariables(), model.getRowColumnMetaVariablePairs());
                storm::dd::Add<DdType, ValueType> result = solver->multiply(dir == OptimizationDirection::Minimize, model.getManager().template getAddZero<ValueType>(), &totalRewardVector, stepBound);
                
                return std::unique_ptr<CheckResult>(new SymbolicQuantitativeCheckResult<DdType>(model.getReachableStates(), result));
            }
            
            template<storm::dd::DdType DdType, typename ValueType>
            std::unique_ptr<CheckResult> SymbolicMdpPrctlHelper<DdType, ValueType>::computeReachabilityRewards(OptimizationDirection dir, storm::models::symbolic::NondeterministicModel<DdType, ValueType> const& model, storm::dd::Add<DdType, ValueType> const& transitionMatrix, RewardModelType const& rewardModel, storm::dd::Bdd<DdType> const& targetStates, bool qualitative, storm::utility::solver::SymbolicMinMaxLinearEquationSolverFactory<DdType, ValueType> const& linearEquationSolverFactory) {
                
                // Only compute the result if there is at least one reward model.
                STORM_LOG_THROW(!rewardModel.empty(), storm::exceptions::InvalidPropertyException, "Missing reward model for formula. Skipping formula.");
                
                // Determine which states have a reward of infinity by definition.
                storm::dd::Bdd<DdType> infinityStates;
                storm::dd::Bdd<DdType> transitionMatrixBdd = transitionMatrix.notZero();
                if (dir == OptimizationDirection::Minimize) {
                    infinityStates = storm::utility::graph::performProb1E(model, transitionMatrixBdd, model.getReachableStates(), targetStates, storm::utility::graph::performProbGreater0E(model, transitionMatrixBdd, model.getReachableStates(), targetStates));
                } else {
                    infinityStates = storm::utility::graph::performProb1A(model, transitionMatrixBdd, model.getReachableStates(), targetStates, storm::utility::graph::performProbGreater0A(model, transitionMatrixBdd, model.getReachableStates(), targetStates));
                }
                infinityStates = !infinityStates && model.getReachableStates();
                
                storm::dd::Bdd<DdType> maybeStates = (!targetStates && !infinityStates) && model.getReachableStates();
                STORM_LOG_INFO("Found " << infinityStates.getNonZeroCount() << " 'infinity' states.");
                STORM_LOG_INFO("Found " << targetStates.getNonZeroCount() << " 'target' states.");
                STORM_LOG_INFO("Found " << maybeStates.getNonZeroCount() << " 'maybe' states.");
                
                // Check whether we need to compute exact rewards for some states.
                if (qualitative) {
                    // Set the values for all maybe-states to 1 to indicate that their reward values
                    // are neither 0 nor infinity.
                    return std::unique_ptr<CheckResult>(new SymbolicQuantitativeCheckResult<DdType>(model.getReachableStates(), infinityStates.ite(model.getManager().getConstant(storm::utility::infinity<ValueType>()), model.getManager().template getAddZero<ValueType>()) + maybeStates.template toAdd<ValueType>() * model.getManager().getConstant(storm::utility::one<ValueType>())));
                } else {
                    // If there are maybe states, we need to solve an equation system.
                    if (!maybeStates.isZero()) {
                        // Create the matrix and the vector for the equation system.
                        storm::dd::Add<DdType, ValueType> maybeStatesAdd = maybeStates.template toAdd<ValueType>();
                        
                        // Start by cutting away all rows that do not belong to maybe states. Note that this leaves columns targeting
                        // non-maybe states in the matrix.
                        storm::dd::Add<DdType, ValueType> submatrix = transitionMatrix * maybeStatesAdd;
                        
                        // Then compute the state reward vector to use in the computation.
                        storm::dd::Add<DdType, ValueType> subvector = rewardModel.getTotalRewardVector(maybeStatesAdd, submatrix, model.getColumnVariables());
                        
                        // Since we are cutting away target and infinity states, we need to account for this by giving
                        // choices the value infinity that have some successor contained in the infinity states.
                        storm::dd::Bdd<DdType> choicesWithInfinitySuccessor = (maybeStates && transitionMatrixBdd && infinityStates.swapVariables(model.getRowColumnMetaVariablePairs())).existsAbstract(model.getColumnVariables());
                        subvector = choicesWithInfinitySuccessor.ite(model.getManager().template getInfinity<ValueType>(), subvector);
                        
                        // Finally cut away all columns targeting non-maybe states.
                        submatrix *= maybeStatesAdd.swapVariables(model.getRowColumnMetaVariablePairs());
                        
                        // Now solve the resulting equation system.
                        std::unique_ptr<storm::solver::SymbolicMinMaxLinearEquationSolver<DdType, ValueType>> solver = linearEquationSolverFactory.create(submatrix, maybeStates, model.getIllegalMask() && maybeStates, model.getRowVariables(), model.getColumnVariables(), model.getNondeterminismVariables(), model.getRowColumnMetaVariablePairs());
                        storm::dd::Add<DdType, ValueType> result = solver->solveEquations(dir == OptimizationDirection::Minimize, model.getManager().template getAddZero<ValueType>(), subvector);

                        return std::unique_ptr<CheckResult>(new storm::modelchecker::SymbolicQuantitativeCheckResult<DdType>(model.getReachableStates(), infinityStates.ite(model.getManager().getConstant(storm::utility::infinity<ValueType>()), result)));
                    } else {
                        return std::unique_ptr<CheckResult>(new storm::modelchecker::SymbolicQuantitativeCheckResult<DdType>(model.getReachableStates(), infinityStates.ite(model.getManager().getConstant(storm::utility::infinity<ValueType>()), model.getManager().template getAddZero<ValueType>())));
                    }
                }
            }
            
            template class SymbolicMdpPrctlHelper<storm::dd::DdType::CUDD, double>;
            template class SymbolicMdpPrctlHelper<storm::dd::DdType::Sylvan, double>;
        }
    }
}