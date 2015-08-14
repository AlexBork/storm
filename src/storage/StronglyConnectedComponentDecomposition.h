#ifndef STORM_STORAGE_STRONGLYCONNECTEDCOMPONENTDECOMPOSITION_H_
#define STORM_STORAGE_STRONGLYCONNECTEDCOMPONENTDECOMPOSITION_H_

#include "src/storage/SparseMatrix.h"
#include "src/storage/Decomposition.h"
#include "src/storage/StronglyConnectedComponent.h"
#include "src/storage/BitVector.h"

namespace storm {
    namespace models {
        namespace sparse {
            // Forward declare the model class.
            template <typename ValueType, typename RewardModelType> class Model;
        }
    }
    
    namespace storage {
        
        /*!
         * This class represents the decomposition of a graph-like structure into its strongly connected components.
         */
        template <typename ValueType, typename RewardModelType = storm::models::sparse::StandardRewardModel<ValueType>>
        class StronglyConnectedComponentDecomposition : public Decomposition<StronglyConnectedComponent> {
        public:            
            /*
             * Creates an empty SCC decomposition.
             */
            StronglyConnectedComponentDecomposition();
            
            /*
             * Creates an SCC decomposition of the given model.
             *
             * @param model The model to decompose into SCCs.
             * @param dropNaiveSccs A flag that indicates whether trivial SCCs (i.e. SCCs consisting of just one state
             * without a self-loop) are to be kept in the decomposition.
             * @param onlyBottomSccs If set to true, only bottom SCCs, i.e. SCCs in which all states have no way of
             * leaving the SCC), are kept.
             */
            StronglyConnectedComponentDecomposition(storm::models::sparse::Model<ValueType, RewardModelType> const& model, bool dropNaiveSccs = false, bool onlyBottomSccs = false);
            
            /*
             * Creates an SCC decomposition of the given block in the given model.
             *
             * @param model The model whose block to decompose.
             * @param block The block to decompose into SCCs.
             * @param dropNaiveSccs A flag that indicates whether trivial SCCs (i.e. SCCs consisting of just one state
             * without a self-loop) are to be kept in the decomposition.
             * @param onlyBottomSccs If set to true, only bottom SCCs, i.e. SCCs in which all states have no way of
             * leaving the SCC), are kept.
             */
            StronglyConnectedComponentDecomposition(storm::models::sparse::Model<ValueType, RewardModelType> const& model, StateBlock const& block, bool dropNaiveSccs = false, bool onlyBottomSccs = false);
            
            /*
             * Creates an SCC decomposition of the given subsystem in the given model.
             *
             * @param model The model that contains the block.
             * @param subsystem A bit vector indicating which subsystem to consider for the decomposition into SCCs.
             * @param dropNaiveSccs A flag that indicates whether trivial SCCs (i.e. SCCs consisting of just one state
             * without a self-loop) are to be kept in the decomposition.
             * @param onlyBottomSccs If set to true, only bottom SCCs, i.e. SCCs in which all states have no way of
             * leaving the SCC), are kept.
             */
            StronglyConnectedComponentDecomposition(storm::models::sparse::Model<ValueType, RewardModelType> const& model, storm::storage::BitVector const& subsystem, bool dropNaiveSccs = false, bool onlyBottomSccs = false);

            /*
             * Creates an SCC decomposition of the given subsystem in the given system (whose transition relation is
             * given by a sparse matrix).
             *
             * @param transitionMatrix The transition matrix of the system to decompose.
             * @param block The block to decompose into SCCs.
             * @param dropNaiveSccs A flag that indicates whether trivial SCCs (i.e. SCCs consisting of just one state
             * without a self-loop) are to be kept in the decomposition.
             * @param onlyBottomSccs If set to true, only bottom SCCs, i.e. SCCs in which all states have no way of
             * leaving the SCC), are kept.
             */
            StronglyConnectedComponentDecomposition(storm::storage::SparseMatrix<ValueType> const& transitionMatrix, StateBlock const& block, bool dropNaiveSccs = false, bool onlyBottomSccs = false);
            
            /*
             * Creates an SCC decomposition of the given system (whose transition relation is given by a sparse matrix).
             *
             * @param transitionMatrix The transition matrix of the system to decompose.
             * @param dropNaiveSccs A flag that indicates whether trivial SCCs (i.e. SCCs consisting of just one state
             * without a self-loop) are to be kept in the decomposition.
             * @param onlyBottomSccs If set to true, only bottom SCCs, i.e. SCCs in which all states have no way of
             * leaving the SCC), are kept.
             */
            StronglyConnectedComponentDecomposition(storm::storage::SparseMatrix<ValueType> const& transitionMatrix, bool dropNaiveSccs = false, bool onlyBottomSccs = false);
            
            /*
             * Creates an SCC decomposition of the given subsystem in the given system (whose transition relation is 
             * given by a sparse matrix).
             *
             * @param transitionMatrix The transition matrix of the system to decompose.
             * @param subsystem A bit vector indicating which subsystem to consider for the decomposition into SCCs.
             * @param dropNaiveSccs A flag that indicates whether trivial SCCs (i.e. SCCs consisting of just one state
             * without a self-loop) are to be kept in the decomposition.
             * @param onlyBottomSccs If set to true, only bottom SCCs, i.e. SCCs in which all states have no way of
             * leaving the SCC), are kept.
             */
            StronglyConnectedComponentDecomposition(storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::BitVector const& subsystem, bool dropNaiveSccs = false, bool onlyBottomSccs = false);
            
            /*!
             * Creates an SCC decomposition by copying the given SCC decomposition.
             *
             * @oaram other The SCC decomposition to copy.
             */
            StronglyConnectedComponentDecomposition(StronglyConnectedComponentDecomposition const& other);
            
            /*!
             * Assigns the contents of the given SCC decomposition to the current one by copying its contents.
             *
             * @oaram other The SCC decomposition from which to copy-assign.
             */
            StronglyConnectedComponentDecomposition& operator=(StronglyConnectedComponentDecomposition const& other);
            
            /*!
             * Creates an SCC decomposition by moving the given SCC decomposition.
             *
             * @oaram other The SCC decomposition to move.
             */
            StronglyConnectedComponentDecomposition(StronglyConnectedComponentDecomposition&& other);
            
            /*!
             * Assigns the contents of the given SCC decomposition to the current one by moving its contents.
             *
             * @oaram other The SCC decomposition from which to copy-assign.
             */
            StronglyConnectedComponentDecomposition& operator=(StronglyConnectedComponentDecomposition&& other);
            
        private:
            /*!
             * Performs the SCC decomposition of the given model. As a side-effect this fills the vector of
             * blocks of the decomposition.
             *
             * @param model The model to decompose into SCCs.
             * @param dropNaiveSccs A flag that indicates whether trivial SCCs (i.e. SCCs consisting of just one state
             * without a self-loop) are to be kept in the decomposition.
             * @param onlyBottomSccs If set to true, only bottom SCCs, i.e. SCCs in which all states have no way of
             * leaving the SCC), are kept.
             */
            void performSccDecomposition(storm::models::sparse::Model<ValueType, RewardModelType> const& model, bool dropNaiveSccs, bool onlyBottomSccs);
            
            /*
             * Performs the SCC decomposition of the given block in the given model. As a side-effect this fills
             * the vector of blocks of the decomposition.
             *
             * @param transitionMatrix The transition matrix of the system to decompose.
             * @param subsystem A bit vector indicating which subsystem to consider for the decomposition into SCCs.
             * @param dropNaiveSccs A flag that indicates whether trivial SCCs (i.e. SCCs consisting of just one state
             * without a self-loop) are to be kept in the decomposition.
             * @param onlyBottomSccs If set to true, only bottom SCCs, i.e. SCCs in which all states have no way of
             * leaving the SCC), are kept.
             */
            void performSccDecomposition(storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::BitVector const& subsystem, bool dropNaiveSccs, bool onlyBottomSccs);
            
            /*!
             * Uses the algorithm by Gabow/Cheriyan/Mehlhorn ("Path-based strongly connected component algorithm") to
             * compute a mapping of states to their SCCs. All arguments given by (non-const) reference are modified by
             * the function as a side-effect.
             *
             * @param transitionMatrix The transition matrix of the system to decompose.
             * @param startState The starting state for the search of Tarjan's algorithm.
             * @param statesWithSelfLoop A bit vector that is to be filled with all states that have a self-loop. This
             * is later needed for identification of the naive SCCs.
             * @param subsystem The subsystem to search.
             * @param currentIndex The next free index that can be assigned to states.
             * @param hasPreorderNumber A bit that is used to keep track of the states that already have a preorder number.
             * @param preorderNumbers A vector storing the preorder number for each state.
             * @param s The stack S used by the algorithm.
             * @param p The stack S used by the algorithm.
             * @param stateHasScc A bit vector containing all states that have already been assigned to an SCC.
             * @param stateToSccMapping A mapping from states to the SCC indices they belong to. As a side effect of this
             * function this mapping is filled (for all states reachable from the starting state).
             * @param sccCount The number of SCCs that have been computed. As a side effect of this function, this count
             * is increased.
             */
            void performSccDecompositionGCM(storm::storage::SparseMatrix<ValueType> const& transitionMatrix, uint_fast64_t startState, storm::storage::BitVector& statesWithSelfLoop, storm::storage::BitVector const& subsystem, uint_fast64_t& currentIndex, storm::storage::BitVector& hasPreorderNumber, std::vector<uint_fast64_t>& preorderNumbers, std::vector<uint_fast64_t>& s, std::vector<uint_fast64_t>& p, storm::storage::BitVector& stateHasScc, std::vector<uint_fast64_t>& stateToSccMapping, uint_fast64_t& sccCount);
            
            // A comparator that is used to compare values.
            storm::utility::ConstantsComparator<ValueType> comparator;
        };
    }
}

#endif /* STORM_STORAGE_STRONGLYCONNECTEDCOMPONENTDECOMPOSITION_H_ */
