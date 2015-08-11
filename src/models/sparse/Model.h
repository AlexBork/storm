#ifndef STORM_MODELS_SPARSE_MODEL_H_
#define STORM_MODELS_SPARSE_MODEL_H_

#include <vector>
#include <boost/container/flat_set.hpp>
#include <boost/optional.hpp>

#include "src/models/ModelBase.h"
#include "src/models/sparse/StandardRewardModel.h"
#include "src/models/sparse/StateLabeling.h"
#include "src/storage/sparse/StateType.h"
#include "src/storage/BitVector.h"
#include "src/storage/SparseMatrix.h"
#include "src/utility/OsDetection.h"

namespace storm {
    namespace models {
        namespace sparse {
            
            // The type used for storing a set of labels.
            typedef boost::container::flat_set<uint_fast64_t> LabelSet;
            
            /*!
             * Base class for all sparse models.
             */
            template<class CValueType, class CRewardModelType = StandardRewardModel<CValueType>>
            class Model : public storm::models::ModelBase {
            public:
                typedef CValueType ValueType;
                typedef CRewardModelType RewardModelType;
                
                Model(Model<ValueType> const& other) = default;
                Model& operator=(Model<ValueType> const& other) = default;
                
#ifndef WINDOWS
                Model(Model<ValueType>&& other) = default;
                Model& operator=(Model<ValueType>&& other) = default;
#endif
                
                /*!
                 * Constructs a model from the given data.
                 *
                 * @param modelType The type of the model.
                 * @param transitionMatrix The matrix representing the transitions in the model.
                 * @param stateLabeling The labeling of the states.
                 * @param rewardModels A mapping of reward model names to reward models.
                 * @param optionalChoiceLabeling A vector that represents the labels associated with the choices of each state.
                 */
                Model(storm::models::ModelType const& modelType,
                      storm::storage::SparseMatrix<ValueType> const& transitionMatrix,
                      storm::models::sparse::StateLabeling const& stateLabeling,
                      std::map<std::string, RewardModelType> const& rewardModels = std::map<std::string, RewardModelType>(),
                      boost::optional<std::vector<LabelSet>> const& optionalChoiceLabeling = boost::optional<std::vector<LabelSet>>());
                
                /*!
                 * Constructs a model by moving the given data.
                 *
                 * @param modelType The type of the model.
                 * @param transitionMatrix The matrix representing the transitions in the model.
                 * @param stateLabeling The labeling of the states.
                 * @param rewardModels A mapping of reward model names to reward models.
                 * @param optionalChoiceLabeling A vector that represents the labels associated with the choices of each state.
                 */
                Model(storm::models::ModelType const& modelType,
                      storm::storage::SparseMatrix<ValueType>&& transitionMatrix,
                      storm::models::sparse::StateLabeling&& stateLabeling,
                      std::map<std::string, RewardModelType>&& rewardModels = std::map<std::string, RewardModelType>(),
                      boost::optional<std::vector<LabelSet>>&& optionalChoiceLabeling = boost::optional<std::vector<LabelSet>>());
                                
                /*!
                 * Retrieves the backward transition relation of the model, i.e. a set of transitions between states
                 * that correspond to the reversed transition relation of this model.
                 *
                 * @return A sparse matrix that represents the backward transitions of this model.
                 */
                storm::storage::SparseMatrix<ValueType> getBackwardTransitions() const;
                
                /*!
                 * Returns an object representing the matrix rows associated with the given state.
                 *
                 * @param state The state for which to retrieve the rows.
                 * @return An object representing the matrix rows associated with the given state.
                 */
                virtual typename storm::storage::SparseMatrix<ValueType>::const_rows getRows(storm::storage::sparse::state_type state) const;
                
                /*!
                 * Returns the number of states of the model.
                 *
                 * @return The number of states of the model.
                 */
                virtual uint_fast64_t getNumberOfStates() const;
                
                /*!
                 * Returns the number of (non-zero) transitions of the model.
                 *
                 * @return The number of (non-zero) transitions of the model.
                 */
                virtual uint_fast64_t getNumberOfTransitions() const;
                
                /*!
                 * Retrieves the initial states of the model.
                 *
                 * @return The initial states of the model represented by a bit vector.
                 */
                storm::storage::BitVector const& getInitialStates() const;
                
                /*!
                 * Returns the sets of states labeled with the given label.
                 *
                 * @param label The label for which to get the labeled states.
                 * @return The set of states labeled with the requested label in the form of a bit vector.
                 */
                storm::storage::BitVector const& getStates(std::string const& label) const;
                
                /*!
                 * Retrieves whether the given label is a valid label in this model.
                 *
                 * @param label The label to be checked for validity.
                 * @return True if the given label is valid in this model.
                 */
                bool hasLabel(std::string const& label) const;
                
                /*!
                 * Retrieves the matrix representing the transitions of the model.
                 *
                 * @return A matrix representing the transitions of the model.
                 */
                storm::storage::SparseMatrix<ValueType> const& getTransitionMatrix() const;
                
                /*!
                 * Retrieves the matrix representing the transitions of the model.
                 *
                 * @return A matrix representing the transitions of the model.
                 */
                storm::storage::SparseMatrix<ValueType>& getTransitionMatrix();

                /*!
                 * Retrieves whether the model has a reward model with the given name.
                 *
                 * @return True iff the model has a reward model with the given name.
                 */
                bool hasRewardModel(std::string const& rewardModelName) const;
                
                /*!
                 * Retrieves the reward model with the given name, if one exists. Otherwise, an exception is thrown.
                 *
                 * @return The reward model with the given name, if it exists.
                 */
                RewardModelType const& getRewardModel(std::string const& rewardModelName) const;

                /*!
                 * Retrieves the labels for the choices of the model. Note that calling this method is only valid if the
                 * model has a choice labeling.
                 *
                 * @return The labels for the choices of the model.
                 */
                std::vector<LabelSet> const& getChoiceLabeling() const;
                
                /*!
                 * Retrieves an optional value that contains the choice labeling if there is one.
                 * 
                 * @return The labels for the choices, if they're saved.
                 */
                boost::optional<std::vector<LabelSet>> const&  getOptionalChoiceLabeling() const;
                
                /*!
                 * Returns the state labeling associated with this model.
                 *
                 * @return The state labeling associated with this model.
                 */
                storm::models::sparse::StateLabeling const& getStateLabeling() const;
                
                /*!
                 * Returns the state labeling associated with this model.
                 *
                 * @return The state labeling associated with this model.
                 */
                storm::models::sparse::StateLabeling& getStateLabeling();
                
                /*!
                 * Retrieves whether this model has a labeling of the choices.
                 *
                 * @return True iff this model has a labeling of the choices.
                 */
                bool hasChoiceLabeling() const;
                
                /*!
                 * Converts the transition rewards of all reward models to state rewards. Note that calling this method
                 * is only valid if the model has transition rewards. Also note that this does not preserve all
                 * properties, but it preserves expected rewards.
                 */
                void convertTransitionRewardsToStateActionRewards();
                
                /*!
                 * Retrieves (an approximation of) the size of the model in bytes.
                 *
                 * @return The size of the internal representation of the model measured in bytes.
                 */
                virtual std::size_t getSizeInBytes() const;
                
                /*!
                 * Prints information about the model to the specified stream.
                 *
                 * @param out The stream the information is to be printed to.
                 */
                virtual void printModelInformationToStream(std::ostream& out) const;
                
                /*!
                 * Exports the model to the dot-format and prints the result to the given stream.
                 *
                 * @param outStream The stream to which the model is to be written.
                 * @param includeLabling If set to true, the states will be exported with their labels.
                 * @param subsystem If not null, this represents the subsystem that is to be exported.
                 * @param firstValue If not null, the values in this vector are attached to the states.
                 * @param secondValue If not null, the values in this vector are attached to the states.
                 * @param stateColoring If not null, this is a mapping from states to color codes.
                 * @param colors A mapping of color codes to color names.
                 * @param finalizeOutput A flag that sets whether or not the dot stream is closed with a curly brace.
                 * @return A string containing the exported model in dot-format.
                 */
                virtual void writeDotToStream(std::ostream& outStream, bool includeLabeling = true, storm::storage::BitVector const* subsystem = nullptr, std::vector<ValueType> const* firstValue = nullptr, std::vector<ValueType> const* secondValue = nullptr, std::vector<uint_fast64_t> const* stateColoring = nullptr, std::vector<std::string> const* colors = nullptr, std::vector<uint_fast64_t>* scheduler = nullptr, bool finalizeOutput = true) const;
                
                /*!
                 * Retrieves the set of labels attached to the given state.
                 *
                 * @param state The state whose labels to retrieve.
                 * @return The set of labels attach to the given state.
                 */
                std::set<std::string> getLabelsOfState(storm::storage::sparse::state_type state) const;
                
                virtual bool isSparseModel() const override;
                
            protected:
                /*!
                 * Sets the transition matrix of the model.
                 *
                 * @param transitionMatrix The new transition matrix of the model.
                 */
                void setTransitionMatrix(storm::storage::SparseMatrix<ValueType> const& transitionMatrix);

                /*!
                 * Sets the transition matrix of the model.
                 *
                 * @param transitionMatrix The new transition matrix of the model.
                 */
                void setTransitionMatrix(storm::storage::SparseMatrix<ValueType>&& transitionMatrix);

                /*!
                 * Prints the information header (number of states and transitions) of the model to the specified stream.
                 *
                 * @param out The stream the information is to be printed to.
                 */
                void printModelInformationHeaderToStream(std::ostream& out) const;
                
                /*!
                 * Prints the information footer (reward models, labels and size in memory) of the model to the
                 * specified stream.
                 *
                 * @param out The stream the information is to be printed to.
                 */
                void printModelInformationFooterToStream(std::ostream& out) const;
                
                /*!
                 * Prints information about the reward models to the specified stream.
                 *
                 * @param out The stream the information is to be printed to.
                 */
                void printRewardModelsInformationToStream(std::ostream& out) const;
                
                /*!
                 * Retrieves the reward models.
                 *
                 * @return A mapping from reward model names to the reward models.
                 */
                std::map<std::string, RewardModelType> const& getRewardModels() const;
                
            private:
                //  A matrix representing transition relation.
                storm::storage::SparseMatrix<ValueType> transitionMatrix;
                
                // The labeling of the states.
                storm::models::sparse::StateLabeling stateLabeling;
                
                // The reward models of the model.
                std::map<std::string, RewardModelType> rewardModels;
                
                // If set, a vector representing the labels of choices.
                boost::optional<std::vector<LabelSet>> choiceLabeling;
            };
            
        } // namespace sparse
    } // namespace models
} // namespace storm

#endif /* STORM_MODELS_SPARSE_MODEL_H_ */
