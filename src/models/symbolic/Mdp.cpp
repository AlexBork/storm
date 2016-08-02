#include "src/models/symbolic/Mdp.h"

#include "src/storage/dd/DdManager.h"
#include "src/storage/dd/Add.h"
#include "src/storage/dd/Bdd.h"

#include "src/models/symbolic/StandardRewardModel.h"

namespace storm {
    namespace models {
        namespace symbolic {
            
            template<storm::dd::DdType Type, typename ValueType>
            Mdp<Type, ValueType>::Mdp(std::shared_ptr<storm::dd::DdManager<Type>> manager,
                                      storm::dd::Bdd<Type> reachableStates,
                                      storm::dd::Bdd<Type> initialStates,
                                      storm::dd::Bdd<Type> deadlockStates,
                                      storm::dd::Add<Type, ValueType> transitionMatrix,
                                      std::set<storm::expressions::Variable> const& rowVariables,
                                      std::shared_ptr<storm::adapters::AddExpressionAdapter<Type, ValueType>> rowExpressionAdapter,
                                      std::set<storm::expressions::Variable> const& columnVariables,
                                      std::shared_ptr<storm::adapters::AddExpressionAdapter<Type, ValueType>> columnExpressionAdapter,
                                      std::vector<std::pair<storm::expressions::Variable, storm::expressions::Variable>> const& rowColumnMetaVariablePairs,
                                      std::set<storm::expressions::Variable> const& nondeterminismVariables,
                                      std::map<std::string, storm::expressions::Expression> labelToExpressionMap,
                                      std::unordered_map<std::string, RewardModelType> const& rewardModels)
            : NondeterministicModel<Type, ValueType>(storm::models::ModelType::Mdp, manager, reachableStates, initialStates, deadlockStates, transitionMatrix, rowVariables, rowExpressionAdapter, columnVariables, columnExpressionAdapter, rowColumnMetaVariablePairs, nondeterminismVariables, labelToExpressionMap, rewardModels) {
                // Intentionally left empty.
            }
            
            // Explicitly instantiate the template class.
            template class Mdp<storm::dd::DdType::CUDD, double>;
            template class Mdp<storm::dd::DdType::Sylvan, double>;
            
        } // namespace symbolic
    } // namespace models
} // namespace storm
