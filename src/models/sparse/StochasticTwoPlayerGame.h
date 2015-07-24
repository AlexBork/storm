#ifndef STORM_MODELS_SPARSE_STOCHASTICTWOPLAYERGAME_H_
#define STORM_MODELS_SPARSE_STOCHASTICTWOPLAYERGAME_H_

#include "src/models/sparse/NondeterministicModel.h"
#include "src/utility/OsDetection.h"

namespace storm {
    namespace models {
        namespace sparse {
            
            /*!
             * This class represents a (discrete-time) stochastic two-player game.
             */
            template <typename ValueType>
            class StochasticTwoPlayerGame : public NondeterministicModel<ValueType> {
            public:
                /*!
                 * Constructs a model from the given data.
                 *
                 * @param player1Matrix The matrix representing the choices of player 1.
                 * @param player2Matrix The matrix representing the choices of player 2.
                 * @param stateLabeling The labeling of the states.
                 * @param optionalStateRewardVector The reward values associated with the states.
                 * @param optionalPlayer1ChoiceLabeling A vector that represents the labels associated with the choices of each player 1 state.
                 * @param optionalPlayer2ChoiceLabeling A vector that represents the labels associated with the choices of each player 2 state.
                 */
                StochasticTwoPlayerGame(storm::storage::SparseMatrix<storm::storage::sparse::state_type> const& player1Matrix,
                                        storm::storage::SparseMatrix<ValueType> const& player2Matrix,
                                        storm::models::sparse::StateLabeling const& stateLabeling,
                                        boost::optional<std::vector<ValueType>> const& optionalStateRewardVector = boost::optional<std::vector<ValueType>>(),
                                        boost::optional<std::vector<LabelSet>> const& optionalPlayer1ChoiceLabeling = boost::optional<std::vector<LabelSet>>(),
                                        boost::optional<std::vector<LabelSet>> const& optionalPlayer2ChoiceLabeling = boost::optional<std::vector<LabelSet>>());
                
                /*!
                 * Constructs a model by moving the given data.
                 *
                 * @param player1Matrix The matrix representing the choices of player 1.
                 * @param player2Matrix The matrix representing the choices of player 2.
                 * @param stateLabeling The labeling of the states.
                 * @param optionalStateRewardVector The reward values associated with the states.
                 * @param optionalPlayer1ChoiceLabeling A vector that represents the labels associated with the choices of each player 1 state.
                 * @param optionalPlayer2ChoiceLabeling A vector that represents the labels associated with the choices of each player 2 state.
                 */
                StochasticTwoPlayerGame(storm::storage::SparseMatrix<storm::storage::sparse::state_type>&& player1Matrix,
                                        storm::storage::SparseMatrix<ValueType>&& player2Matrix,
                                        storm::models::sparse::StateLabeling&& stateLabeling,
                                        boost::optional<std::vector<ValueType>>&& optionalStateRewardVector = boost::optional<std::vector<ValueType>>(),
                                        boost::optional<std::vector<LabelSet>>&& optionalPlayer1ChoiceLabeling = boost::optional<std::vector<LabelSet>>(),
                                        boost::optional<std::vector<LabelSet>>&& optionalPlayer2ChoiceLabeling = boost::optional<std::vector<LabelSet>>());
                
                StochasticTwoPlayerGame(StochasticTwoPlayerGame const& other) = default;
                StochasticTwoPlayerGame& operator=(StochasticTwoPlayerGame const& other) = default;
                
#ifndef WINDOWS
                StochasticTwoPlayerGame(StochasticTwoPlayerGame&& other) = default;
                StochasticTwoPlayerGame& operator=(StochasticTwoPlayerGame&& other) = default;
#endif
                
                /*!
                 * Retrieves the matrix representing the choices in player 1 states.
                 *
                 * @return A matrix representing the choices in player 1 states.
                 */
                storm::storage::SparseMatrix<storm::storage::sparse::state_type> const& getPlayer1Matrix() const;

                /*!
                 * Retrieves the matrix representing the choices in player 2 states and the associated probability
                 * distributions.
                 *
                 * @return A matrix representing the choices in player 2 states.
                 */
                storm::storage::SparseMatrix<ValueType> const& getPlayer2Matrix() const;
                
                /*!
                 * Retrieves the whether the game has labels attached to the choices in player 1 states.
                 *
                 * @return True if the game has player 1 choice labels.
                 */
                bool hasPlayer1ChoiceLabeling() const;
                
                /*!
                 * Retrieves the labels attached to the choices of player 1 states.
                 *
                 * @return A vector containing the labels of each player 1 choice.
                 */
                std::vector<LabelSet> const& getPlayer1ChoiceLabeling() const;

                /*!
                 * Retrieves whether the game has labels attached to player 2 states.
                 *
                 * @return True if the game has player 2 labels.
                 */
                bool hasPlayer2ChoiceLabeling() const;

                /*!
                 * Retrieves the labels attached to the choices of player 2 states.
                 *
                 * @return A vector containing the labels of each player 2 choice.
                 */
                std::vector<LabelSet> const& getPlayer2ChoiceLabeling() const;
                
            private:
                // A matrix that stores the player 1 choices. This matrix contains a row group for each player 1 node. Every
                // row group contains a row for each choice in that player 1 node. Each such row contains exactly one
                // (non-zero) entry at a column that indicates the player 2 node this choice leads to (which is essentially
                // the index of a row group in the matrix for player 2).
                storm::storage::SparseMatrix<storm::storage::sparse::state_type> player1Matrix;
                
                // An (optional) vector of labels attached to the choices of player 1. Each row of the matrix can be equipped
                // with a set of labels to tag certain choices.
                boost::optional<std::vector<LabelSet>> player1ChoiceLabeling;
                
                // The matrix and labels for player 2 are stored in the superclass.
            };
            
        } // namespace sparse
    } // namespace models
} // namespace storm

#endif /* STORM_MODELS_SPARSE_STOCHASTICTWOPLAYERGAME_H_ */
