#ifndef STORM_MODELS_SYMBOLIC_MODEL_H_
#define STORM_MODELS_SYMBOLIC_MODEL_H_

#include <memory>
#include <set>
#include <unordered_map>
#include <boost/optional.hpp>

#include "src/storage/expressions/Expression.h"
#include "src/storage/expressions/Variable.h"
#include "src/storage/dd/DdType.h"
#include "src/models/ModelBase.h"
#include "src/utility/OsDetection.h"

namespace storm {
    namespace dd {
        
        template<storm::dd::DdType Type>
        class Dd;
        
        template<storm::dd::DdType Type, typename ValueType>
        class Add;
        
        template<storm::dd::DdType Type>
        class Bdd;
        
        template<storm::dd::DdType Type>
        class DdManager;
        
    }
    
    namespace adapters {
        template<storm::dd::DdType Type, typename ValueType>
        class AddExpressionAdapter;
    }
    
    namespace models {
        namespace symbolic {
            
            template<storm::dd::DdType Type, typename ValueType>
            class StandardRewardModel;
            
            /*!
             * Base class for all symbolic models.
             */
            template<storm::dd::DdType Type, typename ValueType = double>
            class Model : public storm::models::ModelBase {
            public:
                static const storm::dd::DdType DdType = Type;
                typedef StandardRewardModel<Type, ValueType> RewardModelType;
                
                Model(Model<Type, ValueType> const& other) = default;
                Model& operator=(Model<Type, ValueType> const& other) = default;
                
#ifndef WINDOWS
                Model(Model<Type, ValueType>&& other) = default;
                Model& operator=(Model<Type, ValueType>&& other) = default;
#endif
                
                /*!
                 * Constructs a model from the given data.
                 *
                 * @param modelType The type of the model.
                 * @param manager The manager responsible for the decision diagrams.
                 * @param reachableStates A DD representing the reachable states.
                 * @param initialStates A DD representing the initial states of the model.
                 * @param transitionMatrix The matrix representing the transitions in the model.
                 * @param rowVariables The set of row meta variables used in the DDs.
                 * @param rowExpressionAdapter An object that can be used to translate expressions in terms of the row
                 * meta variables.
                 * @param columVariables The set of column meta variables used in the DDs.
                 * @param columnExpressionAdapter An object that can be used to translate expressions in terms of the
                 * column meta variables.
                 * @param rowColumnMetaVariablePairs All pairs of row/column meta variables.
                 * @param labelToExpressionMap A mapping from label names to their defining expressions.
                 * @param rewardModels The reward models associated with the model.
                 */
                Model(storm::models::ModelType const& modelType,
                      std::shared_ptr<storm::dd::DdManager<Type>> manager,
                      storm::dd::Bdd<Type> reachableStates,
                      storm::dd::Bdd<Type> initialStates,
                      storm::dd::Add<Type, ValueType> transitionMatrix,
                      std::set<storm::expressions::Variable> const& rowVariables,
                      std::shared_ptr<storm::adapters::AddExpressionAdapter<Type, ValueType>> rowExpressionAdapter,
                      std::set<storm::expressions::Variable> const& columnVariables,
                      std::shared_ptr<storm::adapters::AddExpressionAdapter<Type, ValueType>> columnExpressionAdapter,
                      std::vector<std::pair<storm::expressions::Variable, storm::expressions::Variable>> const& rowColumnMetaVariablePairs,
                      std::map<std::string, storm::expressions::Expression> labelToExpressionMap = std::map<std::string, storm::expressions::Expression>(),
                      std::unordered_map<std::string, RewardModelType> const& rewardModels = std::unordered_map<std::string, RewardModelType>());
                
                virtual uint_fast64_t getNumberOfStates() const override;
                
                virtual uint_fast64_t getNumberOfTransitions() const override;
                
                /*!
                 * Retrieves the manager responsible for the DDs that represent this model.
                 *
                 * @return The manager responsible for the DDs that represent this model.
                 */
                storm::dd::DdManager<Type> const& getManager() const;

                /*!
                 * Retrieves the manager responsible for the DDs that represent this model.
                 *
                 * @return The manager responsible for the DDs that represent this model.
                 */
                storm::dd::DdManager<Type>& getManager();
                
                /*!
                 * Retrieves the reachable states of the model.
                 *
                 * @return The reachble states of the model.
                 */
                storm::dd::Bdd<Type> const& getReachableStates() const;
                
                /*!
                 * Retrieves the initial states of the model.
                 *
                 * @return The initial states of the model.
                 */
                storm::dd::Bdd<Type> const& getInitialStates() const;
                
                /*!
                 * Returns the sets of states labeled with the given label.
                 *
                 * @param label The label for which to get the labeled states.
                 * @return The set of states labeled with the requested label in the form of a bit vector.
                 */
                virtual storm::dd::Bdd<Type> getStates(std::string const& label) const;
                
                /*!
                 * Returns the set of states labeled satisfying the given expression (that must be of boolean type).
                 *
                 * @param expression The expression that needs to hold in the states.
                 * @return The set of states satisfying the given expression.
                 */
                virtual storm::dd::Bdd<Type> getStates(storm::expressions::Expression const& expression) const;
                
                /*!
                 * Retrieves whether the given label is a valid label in this model.
                 *
                 * @param label The label to be checked for validity.
                 * @return True if the given label is valid in this model.
                 */
                virtual bool hasLabel(std::string const& label) const;
                
                /*!
                 * Retrieves the matrix representing the transitions of the model.
                 *
                 * @return A matrix representing the transitions of the model.
                 */
                storm::dd::Add<Type, ValueType> const& getTransitionMatrix() const;
                
                /*!
                 * Retrieves the matrix representing the transitions of the model.
                 *
                 * @return A matrix representing the transitions of the model.
                 */
                storm::dd::Add<Type, ValueType>& getTransitionMatrix();

                /*!
                 * Retrieves the matrix qualitatively (i.e. without probabilities) representing the transitions of the
                 * model.
                 *
                 * @return A matrix representing the qualitative transitions of the model.
                 */
                storm::dd::Bdd<Type> getQualitativeTransitionMatrix() const;
                
                /*!
                 * Retrieves the meta variables used to encode the rows of the transition matrix and the vector indices.
                 *
                 * @return The meta variables used to encode the rows of the transition matrix and the vector indices.
                 */
                std::set<storm::expressions::Variable> const& getRowVariables() const;

                /*!
                 * Retrieves the meta variables used to encode the columns of the transition matrix and the vector indices.
                 *
                 * @return The meta variables used to encode the columns of the transition matrix and the vector indices.
                 */
                std::set<storm::expressions::Variable> const& getColumnVariables() const;
                
                /*!
                 * Retrieves the pairs of row and column meta variables.
                 *
                 * @return The pairs of row and column meta variables.
                 */
                std::vector<std::pair<storm::expressions::Variable, storm::expressions::Variable>> const& getRowColumnMetaVariablePairs() const;
                
                /*!
                 * Retrieves an ADD that represents the diagonal of the transition matrix.
                 *
                 * @return An ADD that represents the diagonal of the transition matrix.
                 */
                storm::dd::Add<Type, ValueType> getRowColumnIdentity() const;
                
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
                 * Retrieves the unique reward model, if there exists exactly one. Otherwise, an exception is thrown.
                 *
                 * @return An iterator to the name and the reward model.
                 */
                typename std::unordered_map<std::string, RewardModelType>::const_iterator getUniqueRewardModel() const;
                
                /*!
                 * Retrieves whether the model has a unique reward model.
                 *
                 * @return True iff the model has a unique reward model.
                 */
                bool hasUniqueRewardModel() const;
                
                /*!
                 * Retrieves whether the model has at least one reward model.
                 *
                 * @return True iff the model has a reward model.
                 */
                bool hasRewardModel() const;
                
                /*!
                 * Retrieves the number of reward models associated with this model.
                 *
                 * @return The number of reward models associated with this model.
                 */
                uint_fast64_t getNumberOfRewardModels() const;
                
                virtual std::size_t getSizeInBytes() const override;
                
                virtual void printModelInformationToStream(std::ostream& out) const override;
                
                virtual bool isSymbolicModel() const override;
                
            protected:
                
                /*!
                 * Sets the transition matrix of the model.
                 *
                 * @param transitionMatrix The new transition matrix of the model.
                 */
                void setTransitionMatrix(storm::dd::Add<Type, ValueType> const& transitionMatrix);
                
                /*!
                 * Retrieves the mapping of labels to their defining expressions.
                 *
                 * @returns The mapping of labels to their defining expressions.
                 */
                std::map<std::string, storm::expressions::Expression> const& getLabelToExpressionMap() const;
                
                /*!
                 * Prints the information header (number of states and transitions) of the model to the specified stream.
                 *
                 * @param out The stream the information is to be printed to.
                 */
                void printModelInformationHeaderToStream(std::ostream& out) const;
                
                /*!
                 * Prints the information footer (reward models, labels) of the model to the
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
                 * Prints information about the DD variables to the specified stream.
                 *
                 * @param out The stream the information is to be printed to.
                 */
                virtual void printDdVariableInformationToStream(std::ostream& out) const;
                
            private:
                // The manager responsible for the decision diagrams.
                std::shared_ptr<storm::dd::DdManager<Type>> manager;
                
                // A vector representing the reachable states of the model.
                storm::dd::Bdd<Type> reachableStates;
                
                // A vector representing the initial states of the model.
                storm::dd::Bdd<Type> initialStates;
                
                // A matrix representing transition relation.
                storm::dd::Add<Type, ValueType> transitionMatrix;
                
                // The meta variables used to encode the rows of the transition matrix.
                std::set<storm::expressions::Variable> rowVariables;
                
                // An adapter that can translate expressions to DDs over the row meta variables.
                std::shared_ptr<storm::adapters::AddExpressionAdapter<Type, ValueType>> rowExpressionAdapter;
                
                // The meta variables used to encode the columns of the transition matrix.
                std::set<storm::expressions::Variable> columnVariables;
                
                // An adapter that can translate expressions to DDs over the column meta variables.
                std::shared_ptr<storm::adapters::AddExpressionAdapter<Type, ValueType>> columnExpressionAdapter;
                
                // A vector holding all pairs of row and column meta variable pairs. This is used to swap the variables
                // in the DDs from row to column variables and vice versa.
                std::vector<std::pair<storm::expressions::Variable, storm::expressions::Variable>> rowColumnMetaVariablePairs;
                
                // A mapping from labels to expressions defining them.
                std::map<std::string, storm::expressions::Expression> labelToExpressionMap;
                
                // The reward models associated with the model.
                std::unordered_map<std::string, RewardModelType> rewardModels;
            };
            
        } // namespace symbolic
    } // namespace models
} // namespace storm

#endif /* STORM_MODELS_SYMBOLIC_MODEL_H_ */
