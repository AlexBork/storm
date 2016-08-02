#include "src/models/symbolic/StochasticTwoPlayerGame.h"

#include "src/storage/dd/DdManager.h"
#include "src/storage/dd/Add.h"
#include "src/storage/dd/Bdd.h"

#include "src/models/symbolic/StandardRewardModel.h"

namespace storm {
    namespace models {
        namespace symbolic {
            
            template<storm::dd::DdType Type, typename ValueType>
            StochasticTwoPlayerGame<Type, ValueType>::StochasticTwoPlayerGame(std::shared_ptr<storm::dd::DdManager<Type>> manager,
                                                                              storm::dd::Bdd<Type> reachableStates,
                                                                              storm::dd::Bdd<Type> initialStates,
                                                                              storm::dd::Bdd<Type> deadlockStates,
                                                                              storm::dd::Add<Type, ValueType> transitionMatrix,
                                                                              std::set<storm::expressions::Variable> const& rowVariables,
                                                                              std::shared_ptr<storm::adapters::AddExpressionAdapter<Type, ValueType>> rowExpressionAdapter,
                                                                              std::set<storm::expressions::Variable> const& columnVariables,
                                                                              std::shared_ptr<storm::adapters::AddExpressionAdapter<Type, ValueType>> columnExpressionAdapter,
                                                                              std::vector<std::pair<storm::expressions::Variable, storm::expressions::Variable>> const& rowColumnMetaVariablePairs,
                                                                              std::set<storm::expressions::Variable> const& player1Variables,
                                                                              std::set<storm::expressions::Variable> const& player2Variables,
                                                                              std::set<storm::expressions::Variable> const& nondeterminismVariables,
                                                                              std::map<std::string, storm::expressions::Expression> labelToExpressionMap,
                                                                              std::unordered_map<std::string, RewardModelType> const& rewardModels)
            : NondeterministicModel<Type, ValueType>(storm::models::ModelType::S2pg, manager, reachableStates, initialStates, deadlockStates, transitionMatrix, rowVariables, rowExpressionAdapter, columnVariables, columnExpressionAdapter, rowColumnMetaVariablePairs, nondeterminismVariables, labelToExpressionMap, rewardModels), player1Variables(player1Variables), player2Variables(player2Variables) {
                
                // Compute legal player 1 mask.
                illegalPlayer1Mask = transitionMatrix.notZero().existsAbstract(this->getColumnVariables()).existsAbstract(this->getPlayer2Variables());
                
                // Correct the mask for player 2. This is necessary, because it is not yet restricted to the legal choices of player 1.
                this->illegalMask &= illegalPlayer1Mask;
                
                // Then set the illegal mask for player 1 correctly.
                illegalPlayer1Mask = !illegalPlayer1Mask && reachableStates;
            }
            
            template<storm::dd::DdType Type, typename ValueType>
            storm::dd::Bdd<Type> StochasticTwoPlayerGame<Type, ValueType>::getIllegalPlayer1Mask() const {
                return illegalPlayer1Mask;
            }
            
            template<storm::dd::DdType Type, typename ValueType>
            storm::dd::Bdd<Type> StochasticTwoPlayerGame<Type, ValueType>::getIllegalPlayer2Mask() const {
                // For player 2, we can simply return the mask of the superclass.
                return this->getIllegalMask();
            }
            
            template<storm::dd::DdType Type, typename ValueType>
            std::set<storm::expressions::Variable> const& StochasticTwoPlayerGame<Type, ValueType>::getPlayer1Variables() const {
                return player1Variables;
            }
            
            template<storm::dd::DdType Type, typename ValueType>
            std::set<storm::expressions::Variable> const& StochasticTwoPlayerGame<Type, ValueType>::getPlayer2Variables() const {
                return player2Variables;
            }
            
            // Explicitly instantiate the template class.
            template class StochasticTwoPlayerGame<storm::dd::DdType::CUDD, double>;
            template class StochasticTwoPlayerGame<storm::dd::DdType::Sylvan, double>;
            
        } // namespace symbolic
    } // namespace models
} // namespace storm