#include "SymbolicToSparseTransformer.h"

#include "storm/storage/dd/DdManager.h"
#include "storm/storage/dd/Add.h"
#include "storm/storage/dd/Bdd.h"
#include "storm/models/symbolic/StandardRewardModel.h"


#include "storm/models/sparse/StandardRewardModel.h"

namespace storm {
    namespace transformer {

        template<storm::dd::DdType Type, typename ValueType>
        std::shared_ptr<storm::models::sparse::Dtmc<ValueType>> SymbolicDtmcToSparseDtmcTransformer<Type, ValueType>::translate(storm::models::symbolic::Dtmc<Type, ValueType> const& symbolicDtmc) {
            this->odd = symbolicDtmc.getReachableStates().createOdd();
            storm::storage::SparseMatrix<ValueType> transitionMatrix = symbolicDtmc.getTransitionMatrix().toMatrix(this->odd, this->odd);
            std::unordered_map<std::string, storm::models::sparse::StandardRewardModel<ValueType>> rewardModels;
            for (auto const& rewardModelNameAndModel : symbolicDtmc.getRewardModels()) {
                boost::optional<std::vector<ValueType>> stateRewards;
                boost::optional<std::vector<ValueType>> stateActionRewards;
                boost::optional<storm::storage::SparseMatrix<ValueType>> transitionRewards;
                if (rewardModelNameAndModel.second.hasStateRewards()) {
                    stateRewards = rewardModelNameAndModel.second.getStateRewardVector().toVector(this->odd);
                }
                if (rewardModelNameAndModel.second.hasStateActionRewards()) {
                    stateActionRewards = rewardModelNameAndModel.second.getStateActionRewardVector().toVector(this->odd);
                }
                if (rewardModelNameAndModel.second.hasTransitionRewards()) {
                    transitionRewards = rewardModelNameAndModel.second.getTransitionRewardMatrix().toMatrix(this->odd, this->odd);
                }
                rewardModels.emplace(rewardModelNameAndModel.first,storm::models::sparse::StandardRewardModel<ValueType>(stateRewards, stateActionRewards, transitionRewards));
            }
            storm::models::sparse::StateLabeling labelling(transitionMatrix.getRowGroupCount());
            
            labelling.addLabel("init", symbolicDtmc.getInitialStates().toVector(this->odd));
            labelling.addLabel("deadlock", symbolicDtmc.getDeadlockStates().toVector(this->odd));
            for(auto const& label : symbolicDtmc.getLabels()) {
                labelling.addLabel(label, symbolicDtmc.getStates(label).toVector(this->odd));
            }
            return std::make_shared<storm::models::sparse::Dtmc<ValueType>>(transitionMatrix, labelling, rewardModels);
        }
        
        template<storm::dd::DdType Type, typename ValueType>
        storm::dd::Odd const& SymbolicDtmcToSparseDtmcTransformer<Type, ValueType>::getOdd() const {
            return this->odd;
        }
        
        template<storm::dd::DdType Type, typename ValueType>
        std::shared_ptr<storm::models::sparse::Mdp<ValueType>> SymbolicMdpToSparseMdpTransformer<Type, ValueType>::translate(storm::models::symbolic::Mdp<Type, ValueType> const& symbolicMdp) {
            storm::dd::Odd odd = symbolicMdp.getReachableStates().createOdd();
            storm::storage::SparseMatrix<ValueType> transitionMatrix = symbolicMdp.getTransitionMatrix().toMatrix(symbolicMdp.getNondeterminismVariables(), odd, odd);
            std::unordered_map<std::string, storm::models::sparse::StandardRewardModel<ValueType>> rewardModels;
            for (auto const& rewardModelNameAndModel : symbolicMdp.getRewardModels()) {
                boost::optional<std::vector<ValueType>> stateRewards;
                boost::optional<std::vector<ValueType>> stateActionRewards;
                boost::optional<storm::storage::SparseMatrix<ValueType>> transitionRewards;
                if (rewardModelNameAndModel.second.hasStateRewards()) {
                    stateRewards = rewardModelNameAndModel.second.getStateRewardVector().toVector(odd);
                }
                if (rewardModelNameAndModel.second.hasStateActionRewards()) {
                    stateActionRewards = rewardModelNameAndModel.second.getStateActionRewardVector().toVector(odd);
                }
                if (rewardModelNameAndModel.second.hasTransitionRewards()) {
                    transitionRewards = rewardModelNameAndModel.second.getTransitionRewardMatrix().toMatrix(symbolicMdp.getNondeterminismVariables(), odd, odd);
                }
                rewardModels.emplace(rewardModelNameAndModel.first,storm::models::sparse::StandardRewardModel<ValueType>(stateRewards, stateActionRewards, transitionRewards));
            }
            storm::models::sparse::StateLabeling labelling(transitionMatrix.getRowGroupCount());

            labelling.addLabel("init", symbolicMdp.getInitialStates().toVector(odd));
            labelling.addLabel("deadlock", symbolicMdp.getDeadlockStates().toVector(odd));
            for(auto const& label : symbolicMdp.getLabels()) {
                labelling.addLabel(label, symbolicMdp.getStates(label).toVector(odd));
            }
            return std::make_shared<storm::models::sparse::Mdp<ValueType>>(transitionMatrix, labelling, rewardModels);
        }

        template class SymbolicDtmcToSparseDtmcTransformer<storm::dd::DdType::CUDD, double>;
        template class SymbolicDtmcToSparseDtmcTransformer<storm::dd::DdType::Sylvan, double>;
        
        template class SymbolicDtmcToSparseDtmcTransformer<storm::dd::DdType::Sylvan, storm::RationalNumber>;
        template class SymbolicDtmcToSparseDtmcTransformer<storm::dd::DdType::Sylvan, storm::RationalFunction>;
        
        template class SymbolicMdpToSparseMdpTransformer<storm::dd::DdType::CUDD, double>;
        template class SymbolicMdpToSparseMdpTransformer<storm::dd::DdType::Sylvan, double>;

        template class SymbolicMdpToSparseMdpTransformer<storm::dd::DdType::Sylvan, storm::RationalNumber>;

    }
}
