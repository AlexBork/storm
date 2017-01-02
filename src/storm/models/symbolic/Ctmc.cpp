#include "storm/models/symbolic/Ctmc.h"

#include "storm/storage/dd/DdManager.h"
#include "storm/storage/dd/Add.h"
#include "storm/storage/dd/Bdd.h"

#include "storm/models/symbolic/StandardRewardModel.h"

namespace storm {
    namespace models {
        namespace symbolic {
            
            template<storm::dd::DdType Type, typename ValueType>
            Ctmc<Type, ValueType>::Ctmc(std::shared_ptr<storm::dd::DdManager<Type>> manager,
                                        storm::dd::Bdd<Type> reachableStates,
                                        storm::dd::Bdd<Type> initialStates,
                                        storm::dd::Bdd<Type> deadlockStates,
                                        storm::dd::Add<Type, ValueType> transitionMatrix,
                                        std::set<storm::expressions::Variable> const& rowVariables,
                                        std::shared_ptr<storm::adapters::AddExpressionAdapter<Type, ValueType>> rowExpressionAdapter,
                                        std::set<storm::expressions::Variable> const& columnVariables,
                                        std::shared_ptr<storm::adapters::AddExpressionAdapter<Type, ValueType>> columnExpressionAdapter,
                                        std::vector<std::pair<storm::expressions::Variable, storm::expressions::Variable>> const& rowColumnMetaVariablePairs,
                                        std::map<std::string, storm::expressions::Expression> labelToExpressionMap,
                                        std::unordered_map<std::string, RewardModelType> const& rewardModels)
            : DeterministicModel<Type, ValueType>(storm::models::ModelType::Ctmc, manager, reachableStates, initialStates, deadlockStates, transitionMatrix, rowVariables, rowExpressionAdapter, columnVariables, columnExpressionAdapter, rowColumnMetaVariablePairs, labelToExpressionMap, rewardModels) {
                exitRates = this->getTransitionMatrix().sumAbstract(this->getColumnVariables());
            }
            
            template<storm::dd::DdType Type, typename ValueType>
            storm::dd::Add<Type, ValueType> const& Ctmc<Type, ValueType>::getExitRateVector() const {
                return exitRates;
            }
            
            // Explicitly instantiate the template class.
            template class Ctmc<storm::dd::DdType::CUDD, double>;
            template class Ctmc<storm::dd::DdType::Sylvan, double>;
            
        } // namespace symbolic
    } // namespace models
} // namespace storm
