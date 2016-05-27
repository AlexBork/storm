#ifndef STORM_MDPTRIVIALSTATEELIMINATOR_H
#define STORM_MDPTRIVIALSTATEELIMINATOR_H

#include "../utility/macros.h"

#include <chrono>
#include <memory>
#include <boost/optional.hpp>

#include "src/adapters/CarlAdapter.h"
#include "src/logic/Formulas.h"
#include "src/modelchecker/results/ExplicitQualitativeCheckResult.h"
#include "src/modelchecker/region/RegionCheckResult.h"
#include "src/modelchecker/propositional/SparsePropositionalModelChecker.h"
#include "src/models/sparse/StandardRewardModel.h"
#include "src/settings/SettingsManager.h"
#include "src/settings/modules/RegionSettings.h"
#include "src/solver/OptimizationDirection.h"
#include "src/storage/sparse/StateType.h"
#include "src/storage/FlexibleSparseMatrix.h"
#include "src/utility/constants.h"
#include "src/utility/graph.h"
#include "src/utility/macros.h"
#include "src/utility/vector.h"

#include "src/exceptions/InvalidArgumentException.h"
#include "src/exceptions/InvalidPropertyException.h"
#include "src/exceptions/InvalidStateException.h"
#include "src/exceptions/InvalidSettingsException.h"
#include "src/exceptions/NotImplementedException.h"
#include "src/exceptions/UnexpectedException.h"
#include "src/exceptions/NotSupportedException.h"
#include "src/logic/FragmentSpecification.h"

namespace storm {
    namespace transformer {
        
        /*
         * Duplicates the state space of the given model and redirects the incoming transitions of gateStates of the first copy to the gateStates of the second copy.
         * Only states reachable from the initial states are kept.
         */
        <template ModelType>
        class StateDuplicator {
        public:
            typedef typename SparseMdpModelType::ValueType ValueType;
            typedef typename SparseMdpModelType::RewardModelType RewardModelType;

            struct StateDuplicatorReturnType {
                ModelType model;                        // The resulting model
                storm::storage::BitVector firstCopy;    // The states of the resulting model that correspond to the first copy
                storm::storage::BitVector secondCopy;   // The states of the resulting model that correspond to the second copy
                storm::storage::BitVector gateStates;   // The gate states of the resulting model
                std::vector<uint_fast64_t> newToOldStateIndexMapping; // Gives for each state in the resulting model the corresponding state in the original model
                std::vector<uint_fast64_t> firstCopyOldToNewStateIndexMapping; //Maps old indices of states in the first copy to their new indices
                std::vector<uint_fast64_t> secondCopyOldToNewStateIndexMapping; //Maps old indices of states in the second copy to their new indices
                storm::storage::BitVector duplicatedStates;   // The states in the original model that have been duplicated
                storm::storage::BitVector reachableStates;   // The states in the original model that are reachable from the initial state
            };
            
            /*
             * Duplicates the state space of the given model and redirects the incoming transitions of gateStates of the first copy to the gateStates of the second copy.
             * 
             * Note that only reachable states are kept.
             * Gate states will always belong to the second copy.
             * Rewards and labels are duplicated accordingly, but the states in the second copy will not get the label for initial states.
             *
             * @param originalModel The model to be duplicated
             * @param gateStates The states for which the incoming transitions are redirected
             */
            static StateDuplicatorReturnType transform(ModelType const& originalModel, storm::storage::BitVector const& gateStates) {
                StateDuplicatorReturnType result;
                
                // Collect some data for the result
                initializeTransformation(originalModel, gateStates, result);
                
                // Transform the ingedients of the model
                
                storm::storage::SparseMatrix<ValueType> matrix = transformMatrix(originalModel.getTransitionMatrix(), originalModel, gateStates, result);
                
                //TODO
                return result;
            }
            
        private:
            
            static void initializeTransformation(ModelType const& originalModel, storm::storage::BitVector const& gateStates, StateDuplicatorReturnType& result) {
                
                storm::storage::BitVector noStates(originalModel.getNumberOfStates(), false)
                // Get the states that are reachable without visiting a gateState
                storm::storage::BitVector statesForFirstCopy = storm::utility::graph::getReachableStates(originalModel.getTransitionMatrix(), originalModel.getInitialStates(), ~gateStates, noStates);
                
                // Get the states reachable from gateStates
                storm::storage::BitVector statesForSecondCopy = storm::utility::graph::getReachableStates(originalModel.getTransitionMatrix(), gateStates, ~noStates, noStates);
            
                result.duplicatedStates = statesForFirstCopy & statesForSecondCopy;
                result.reachableStates = statesForFirstCopy | statesForSecondCopy;
            
                uint_fast64_t numStates = statesForFirstCopy.getNumberOfSetBits() + statesForSecondCopy.getNumberOfSetBits();
                result.firstCopy = statesForFirstCopy % reachableStates; // only consider reachable states
                result.firstCopy.resize(numStates, false); // the new states do NOT belong to the first copy
                result.secondCopy = (statesForSecondCopy & (~statesForFirstCopy)) % reachableStates; // only consider reachable states
                result.secondCopy.resize(numStates, true); // the new states DO belong to the second copy
                result.gateStates = gateStates;
                gateStates.resize(numStates, false); // there are no duplicated gateStates
                STORM_LOG_ASSERT((result.firstCopy^result.secondCopy).full()), "firstCopy and secondCopy do not partition the state space.");
            
                // Get the state mappings.
                // We initialize them with illegal values to assert that we don't get a valid
                // state when given e.g. an unreachable state or a state from the other copy.
                result.newToOldStateIndexMapping = std::vector<uint_fast64_t>(numStates, std::numeric_limits<uint_fast64_t>::max());
                result.firstCopyOldToNewStateIndexMapping = std::vector<uint_fast64_t>(originalModel.getNumberOfStates(), std::numeric_limits<uint_fast64_t>::max());
                result.secondCopyOldToNewStateIndexMapping = std::vector<uint_fast64_t>(originalModel.getNumberOfStates(), std::numeric_limits<uint_fast64_t>::max());
                uint_fast64_t newState = 0;
                for(auto const& oldState : reachableStates){
                    result.newToOldStateIndexMapping[newState] = oldState;
                    if(statesForFirstCopy.get(oldState)){
                        result.firstCopyOldToNewStateIndexMapping[oldState] = newState;
                    } else {
                        result.secondCopyOldToNewStateIndexMapping[oldState] = newState;
                    }
                    ++newState;
                }
                // The remaining states are duplicates. All these states belong to the second copy.
                
                for(auto const& oldState : result.duplicatedStates) {
                    result.newToOldStateIndexMapping[newState] = oldState;
                    result.secondCopyOldToNewStateIndexMapping[oldState] = newState;
                    ++newState
                }
                STORM_LOG_ASSERT(newState == numStates, "Unexpected state Indices");
            }
            
            static storm::storage::SparseMatrix<ValueType> transformMatrix(storm::storage::SparseMatrix<ValueType> const& originalMatrix, StateDuplicatorReturnType const& result) {
                // Build the builder
                uint_fast64_t numStates = result.newToOldStateIndexMapping.size();
                uint_fast64_t numRows = 0;
                uint_fast64_t numEntries = 0;
                for(auto const& oldState : result.newToOldStateIndexMapping){
                    numRows += originalMatrix.getRowGroupSize(oldState);
                    numEntries += stateOccurrences * originalMatrix.getRowGroupEntryCount(oldState);
                }
                storm::storage::SparseMatrixBuilder<ValueType> builder(numRows, numStates, numEntries, true, !originalMatrix.hasTrivialRowGrouping(), originalMatrix.hasTrivialRowGrouping() ? 0 : numStates);
                
                // Fill in the data
                uint_fast64_t newRow = 0;
                for(uint_fast64_t newState = 0; newState < numStates; ++newState){
                    if(!originalMatrix.hasTrivialRowGrouping()){
                        builder.newRowGroup(newRow);
                    }
                    uint_fast64_t oldState = result.newToOldStateIndexMapping[newState];
                    const& correctOldToNewMapping = result.firstCopy.get(newState) ? result.firstCopyOldToNewStateIndexMapping : result.secondCopyOldToNewStateIndexMapping;
                    for (uint_fast64_t oldRow = originalMatrix.getRowGroupIndices()[oldState]; oldRow < originalMatrix.getRowGroupIndices()[oldState+1]; ++oldRow){
                        for(auto const& entry : originalMatrix.getRow(oldRow)){
                            builder.addNextValue(newRow, correctOldToNewMapping[entry.getColumn()], entry.getValue());
                        }
                        ++newRow;
                    }
                }
                
                return builder.build();
            }
                
            static std::vector<ValueType> transformActionValueVector(std::vector<ValueType>const& originalVector, std::vector<uint_fast64_t> const& originalRowGroupIndices, StateDuplicatorReturnType const& result) {
                uint_fast64_t numActions = 0;
                for(auto const& oldState : result.newToOldStateIndexMapping){
                    numActions += originalRowGroupIndices[oldState+1] - originalRowGroupIndices[oldState];
                }
                std::vector<ValueType> v;
                v.reserve(numActions);
                for(auto const& oldState : result.newToOldStateIndexMapping){
                    for (uint_fast64_t oldAction = originalRowGroupIndices()[oldState]; oldAction < originaRowGroupIndices()[oldState+1]; ++oldAction){
                        v.push_back(originalVector[oldAction]);
                    }
                }
                STORM_LOG_ASSERT(v.size() == numActions, "Unexpected vector size.");
                return v;
            }
            
            static std::vector<ValueType> transformStateValueVector(std::vector<ValueType> const& originalVector, StateDuplicatorReturnType const& result) {
                uint_fast64_t numStates = result.newToOldStateIndexMapping.size();
                std::vector<ValueType> v;
                result.reserve(numStates);
                for(auto const& oldState : result.newToOldStateIndexMapping){
                    v.push_back(originalVector[oldState]);
                }
                STORM_LOG_ASSERT(result.size() == numStates, "Unexpected vector size.");
                return v;
            }
            
            static storm::storage::BitVector transformStateBitVector(storm::storage::BitVector const& originalBitVector, StateDuplicatorReturnType const& result) {
                uint_fast64_t numStates = result.newToOldStateIndexMapping.size();
                storm::storage::BitVector bv(numStates);
                for(uint_fast64_t newState = 0; newState < numStates; ++newState){
                    uint_fast64_t oldState = result.newToOldStateIndexMapping[newState];
                    bv.set(newState, originalBitVector.get(oldState));
                }
                return bv;
            }
            

             
        };
    }
}
